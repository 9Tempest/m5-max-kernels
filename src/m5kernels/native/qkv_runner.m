/* Common host for the grouped-QKV experiment. No upstream shader is copied.
 * Host selection follows antirez/h3.c, commit
 * 8974cc055ea9c02fcd14cc27dfda3e1027c05153 (h3_gpu.m).
 *
 * Copyright (c) 2026 Salvatore Sanfilippo
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    uint32_t rows, input_dim, heads, head_dim, rope_half, rms_mode;
    float epsilon;
    uint32_t padded_rows, candidate_groups, candidate_threads;
} M5QKVConfig;
typedef struct {
    uint32_t rows, input_dim, heads, head_dim, rope_half, head_major;
    float epsilon;
} M5QKVArgs;
typedef struct { uint32_t rows, columns; float clip; } M5QuantArgs;
static _Thread_local char lastError[8192];

static int fail(NSString *message) {
    snprintf(lastError, sizeof(lastError), "%s", message.UTF8String ?: "unknown error");
    return 0;
}
const char *m5_qkv_error(void) { return lastError; }

@interface M5QKVRunner : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) NSArray<id<MTLBuffer>> *buffers;
@property(nonatomic, strong) id<MTLComputePipelineState> quantizer;
@property(nonatomic, strong) id<MTLComputePipelineState> native;
@property(nonatomic, strong) id<MTLComputePipelineState> candidate;
@property(nonatomic) M5QKVConfig config;
@property(nonatomic) double lastGPUSeconds;
@property(nonatomic, copy) NSString *metadata;
@end
@implementation M5QKVRunner
@end

static id<MTLLibrary> compileLibrary(id<MTLDevice> device, const char *path,
                                    NSError **error) {
    NSString *source = [NSString stringWithContentsOfFile:
        [NSString stringWithUTF8String:path] encoding:NSUTF8StringEncoding error:error];
    if (!source) return nil;
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.mathMode = MTLMathModeSafe;
    options.languageVersion = MTLLanguageVersion4_1;
    options.preprocessorMacros = @{ @"H3_METAL_HAS_TENSOR": @"1" };
    return [device newLibraryWithSource:source options:options error:error];
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> device,
    id<MTLLibrary> library, NSString *name, NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) {
        fail([NSString stringWithFormat:@"missing Metal entry point %@", name]);
        return nil;
    }
    id<MTLComputePipelineState> result =
        [device newComputePipelineStateWithFunction:function error:error];
    if (!result) fail((*error).localizedDescription);
    return result;
}

static NSDictionary *pipelineInfo(id<MTLComputePipelineState> pipeline) {
    return @{@"max_threads": @(pipeline.maxTotalThreadsPerThreadgroup),
             @"thread_execution_width": @(pipeline.threadExecutionWidth),
             @"static_threadgroup_memory_bytes": @(pipeline.staticThreadgroupMemoryLength)};
}

void *m5_qkv_create(const char *nativePath, const char *candidatePath,
                    const char *nativeEntry, const char *candidateEntry,
                    const M5QKVConfig *config) {
    @autoreleasepool {
        lastError[0] = 0;
        if (!config || !nativePath || !candidatePath || !nativeEntry || !candidateEntry) {
            fail(@"invalid create arguments"); return NULL;
        }
        M5QKVRunner *runner = [[M5QKVRunner alloc] init];
        runner.config = *config;
        runner.device = MTLCreateSystemDefaultDevice();
        if (!runner.device || !runner.device.hasUnifiedMemory ||
            [runner.device.name rangeOfString:@"M5"].location == NSNotFound) {
            fail(@"UNSUPPORTED: grouped native comparison requires an M5 unified-memory GPU");
            return NULL;
        }
        runner.queue = [runner.device newCommandQueue];
        if (!runner.queue) { fail(@"cannot create Metal command queue"); return NULL; }
        runner.queue.label = @"m5kernels.common-grouped-qkv";
        NSError *error = nil;
        double start = NSProcessInfo.processInfo.systemUptime;
        id<MTLLibrary> nativeLibrary = compileLibrary(runner.device, nativePath, &error);
        if (!nativeLibrary) { fail([@"native shader compilation: "
            stringByAppendingString:error.localizedDescription ?: @"unknown error"]); return NULL; }
        runner.quantizer = pipeline(runner.device, nativeLibrary,
            @"h3_quantize_bf16_int8_rows_scalar", &error);
        if (!runner.quantizer) return NULL;
        runner.native = pipeline(runner.device, nativeLibrary,
            [NSString stringWithUTF8String:nativeEntry], &error);
        if (!runner.native) return NULL;
        double nativeSeconds = NSProcessInfo.processInfo.systemUptime - start;
        start = NSProcessInfo.processInfo.systemUptime;
        id<MTLLibrary> candidateLibrary = compileLibrary(runner.device, candidatePath, &error);
        if (!candidateLibrary) { fail([@"candidate shader compilation: "
            stringByAppendingString:error.localizedDescription ?: @"unknown error"]); return NULL; }
        runner.candidate = pipeline(runner.device, candidateLibrary,
            [NSString stringWithUTF8String:candidateEntry], &error);
        if (!runner.candidate) return NULL;
        double candidateSeconds = NSProcessInfo.processInfo.systemUptime - start;
        if (runner.quantizer.maxTotalThreadsPerThreadgroup < 256 ||
            runner.native.maxTotalThreadsPerThreadgroup < 256 ||
            runner.candidate.maxTotalThreadsPerThreadgroup < config->candidate_threads ||
            runner.quantizer.threadExecutionWidth != 32 ||
            runner.native.threadExecutionWidth != 32 ||
            runner.candidate.threadExecutionWidth != 32) {
            fail(@"UNSUPPORTED: pipeline thread limits or SIMD width"); return NULL;
        }
        const NSUInteger m = config->rows, k = config->input_dim, h = config->heads;
        const NSUInteger p = config->padded_rows, n = h * 3 * 128;
        const NSUInteger sizes[] = {
            p*k, n*k, p*4, n*4, h*m*128*2, h*m*128*2, h*m*128*2,
            128*2, 128*2, m*48*2, m*48*2, m*k*2, n*k*2
        };
        NSArray<NSString *> *names = @[@"input_int8", @"weight_int8", @"input_scales",
            @"weight_scales", @"q", @"k", @"v", @"q_gamma", @"k_gamma", @"cos", @"sin",
            @"input_bf16", @"weight_bf16"];
        NSMutableArray *buffers = [NSMutableArray arrayWithCapacity:13];
        for (NSUInteger slot = 0; slot < 13; slot++) {
            id<MTLBuffer> buffer = [runner.device newBufferWithLength:sizes[slot]
                options:MTLResourceStorageModeShared];
            if (!buffer) { fail([NSString stringWithFormat:@"buffer allocation failed: %@",
                names[slot]]); return NULL; }
            buffer.label = names[slot];
            memset(buffer.contents, 0, sizes[slot]);
            [buffers addObject:buffer];
        }
        runner.buffers = buffers;
        NSDictionary *metadata = @{
            @"device": runner.device.name, @"unified_memory": @(runner.device.hasUnifiedMemory),
            @"native_compile_seconds": @(nativeSeconds),
            @"candidate_compile_seconds": @(candidateSeconds),
            @"math_mode": @"safe", @"language_version": @"Metal 4.1 (explicit, both libraries)",
            @"preprocessor_macros": @{@"H3_METAL_HAS_TENSOR": @"1"},
            @"native_pipeline": pipelineInfo(runner.native),
            @"candidate_pipeline": pipelineInfo(runner.candidate),
            @"quantizer_pipeline": pipelineInfo(runner.quantizer)
        };
        NSData *json = [NSJSONSerialization dataWithJSONObject:metadata options:0 error:&error];
        if (!json) { fail(error.localizedDescription); return NULL; }
        runner.metadata = [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding];
        return (__bridge_retained void *)runner;
    }
}

void m5_qkv_destroy(void *handle) {
    if (handle) { @autoreleasepool { (void)CFBridgingRelease(handle); } }
}
const char *m5_qkv_metadata(void *handle) {
    return handle ? ((__bridge M5QKVRunner *)handle).metadata.UTF8String : NULL;
}
void *m5_qkv_buffer(void *handle, uint32_t slot, uint64_t *length) {
    if (!handle || slot >= 13) { fail(@"invalid buffer slot"); return NULL; }
    M5QKVRunner *runner = (__bridge M5QKVRunner *)handle;
    id<MTLBuffer> buffer = runner.buffers[slot];
    if (length) *length = buffer.length;
    return buffer.contents;
}
int m5_qkv_upload(void *handle, uint32_t slot, const void *data, uint64_t length) {
    uint64_t expected = 0;
    void *target = m5_qkv_buffer(handle, slot, &expected);
    if (!target || !data || expected != length) return fail(@"upload byte length mismatch");
    memcpy(target, data, length);
    return 1;
}
double m5_qkv_last_gpu_seconds(void *handle) {
    return handle ? ((__bridge M5QKVRunner *)handle).lastGPUSeconds : 0;
}

static void encodeQuant(M5QKVRunner *runner, id<MTLCommandBuffer> command, BOOL weights) {
    M5QKVConfig c = runner.config;
    uint32_t rows = weights ? c.heads*3*128 : c.rows;
    uint32_t dispatch = weights ? rows : c.padded_rows;
    M5QuantArgs args = {rows, c.input_dim, 1.0f};
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    encoder.label = weights ? @"native.scalar.weight_quantize" : @"native.scalar.activation_quantize";
    [encoder setComputePipelineState:runner.quantizer];
    [encoder setBuffer:runner.buffers[weights ? 12 : 11] offset:0 atIndex:0];
    [encoder setBuffer:runner.buffers[weights ? 1 : 0] offset:0 atIndex:1];
    [encoder setBuffer:runner.buffers[weights ? 3 : 2] offset:0 atIndex:2];
    [encoder setBytes:&args length:sizeof(args) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(dispatch,1,1)
        threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [encoder endEncoding];
}

/* mode: 0 native, 1 candidate. scope: 0 core, 1 online, 2 quantize, 3 weights. */
int m5_qkv_run(void *handle, uint32_t mode, uint32_t scope) {
    @autoreleasepool {
        if (!handle || mode > 1 || scope > 3) return fail(@"invalid run mode or scope");
        M5QKVRunner *runner = (__bridge M5QKVRunner *)handle;
        M5QKVConfig c = runner.config;
        id<MTLCommandBuffer> command = [runner.queue commandBuffer];
        if (!command) return fail(@"cannot create command buffer");
        command.label = [NSString stringWithFormat:@"grouped_qkv.%@.%u",
            mode ? @"candidate" : @"native", scope];
        if (scope) encodeQuant(runner, command, scope == 3);
        if (scope <= 1) {
            M5QKVArgs args = {c.rows,c.input_dim,c.heads,c.head_dim,c.rope_half,c.rms_mode,c.epsilon};
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            encoder.label = mode ? @"candidate.grouped_qkv_rms_rope" : @"native.grouped_qkv_rms_rope";
            [encoder setComputePipelineState:mode ? runner.candidate : runner.native];
            for (NSUInteger slot = 0; slot < 11; slot++)
                [encoder setBuffer:runner.buffers[slot] offset:0 atIndex:slot];
            [encoder setBytes:&args length:sizeof(args) atIndex:11];
            NSUInteger groups = mode ? c.candidate_groups : (c.padded_rows/128)*c.heads*3;
            NSUInteger threads = mode ? c.candidate_threads : 256;
            [encoder dispatchThreadgroups:MTLSizeMake(groups,1,1)
                threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
            [encoder endEncoding];
        }
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted)
            return fail(command.error.localizedDescription ?: @"Metal command did not complete");
        runner.lastGPUSeconds = command.GPUEndTime - command.GPUStartTime;
        return 1;
    }
}

int m5_qkv_capture(void *handle, const char *path, uint32_t mode, uint32_t scope) {
    @autoreleasepool {
        if (!handle || !path) return fail(@"invalid capture arguments");
        M5QKVRunner *runner = (__bridge M5QKVRunner *)handle;
        MTLCaptureManager *manager = MTLCaptureManager.sharedCaptureManager;
        if (![manager supportsDestination:MTLCaptureDestinationGPUTraceDocument])
            return fail(@"UNSUPPORTED: Metal GPU trace capture; start process with MTL_CAPTURE_ENABLED=1");
        MTLCaptureDescriptor *descriptor = [[MTLCaptureDescriptor alloc] init];
        descriptor.captureObject = runner.queue;
        descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
        descriptor.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        NSError *error = nil;
        if (![manager startCaptureWithDescriptor:descriptor error:&error])
            return fail(error.localizedDescription);
        int result = m5_qkv_run(handle, mode, scope);
        [manager stopCapture];
        return result;
    }
}
