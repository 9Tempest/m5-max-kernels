using namespace metal;
constexpr uint width=$WIDTH, block=$BLOCK, threads=$THREADS;
constexpr uint groups=threads/32, registers=block==1 ? 1 : block/32;
constexpr uint slots=block==1 ? (width+threads-1)/threads : (width/block+groups-1)/groups;
constexpr bool fold=$FOLD;
constexpr float epsilon=$EPSILON;
uint row=threadgroup_position_in_grid.x, tid=thread_position_in_threadgroup.x;
uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
threadgroup float2 partials[groups];
float values[slots][registers], square_sum=0.0f, inv_rms=1.0f;

#pragma unroll
for(uint slot=0;slot<slots;++slot) {
    #pragma unroll
    for(uint reg=0;reg<registers;++reg) {
        uint j=$INDEX;
        float x=j<width ? float(X[row*width+j]) : 0.0f;
        square_sum+=x*x;
        values[slot][reg]=x;
    }
}

// Finish the folded SIMD-local statistic before the Hadamard work. Its
// cross-SIMD reduction still shares the final barrier with the row peak.
if(fold) square_sum=simd_sum(square_sum);

// Baseline waits for inverse RMS before applying gamma and transforming.
// The folded variant omits this entire section at compile time.
if(!fold) {
    square_sum=simd_sum(square_sum);
    if(lane==0) partials[sg]=float2(square_sum,0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    square_sum=simd_sum(lane<groups ? partials[lane].x : 0.0f);
    inv_rms=rsqrt(square_sum/float(width)+epsilon);
    // All reads must finish before the later peak reduction reuses partials.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

float peak=0.0f;
#pragma unroll
for(uint slot=0;slot<slots;++slot) {
    #pragma unroll
    for(uint reg=0;reg<registers;++reg) {
        uint j=$INDEX;
        float v=values[slot][reg];
        if(!fold) v=j<width ? (v*inv_rms)*Gamma[j]*float(Signs[j]) : 0.0f;
        else v=j<width ? v*Gamma[j]*float(Signs[j]) : 0.0f;
        if(block>1) {
            #pragma unroll
            for(uint stride=1;stride<32;stride*=2) {
                float peer=simd_shuffle_xor(v,stride);
                v=(lane&stride) ? peer-v : v+peer;
            }
        }
        values[slot][reg]=v;
    }
    #pragma unroll
    for(uint stride=1;stride<registers;stride*=2) {
        #pragma unroll
        for(uint base=0;base<registers;base+=2*stride) {
            #pragma unroll
            for(uint j=0;j<stride;++j) {
                float a=values[slot][base+j], b=values[slot][base+j+stride];
                values[slot][base+j]=a+b;
                values[slot][base+j+stride]=a-b;
            }
        }
    }
    #pragma unroll
    for(uint reg=0;reg<registers;++reg) {
        values[slot][reg]*=rsqrt(float(block));
        peak=max(peak,abs(values[slot][reg]));
    }
}
peak=simd_max(peak);
if(lane==0) partials[sg]=float2(fold ? square_sum : 0.0f,peak);
// The folded variant has exactly ONE cross-SIMD threadgroup barrier: both
// original-input squared norm and transformed peak travel in this float2.
threadgroup_barrier(mem_flags::mem_threadgroup);
peak=simd_max(lane<groups ? partials[lane].y : 0.0f);
if(fold) {
    square_sum=simd_sum(lane<groups ? partials[lane].x : 0.0f);
    inv_rms=rsqrt(square_sum/float(width)+epsilon);
}
float scale=peak==0.0f ? 1.0f : peak*(fold ? inv_rms : 1.0f)/127.0f;
if(tid==0) S[row]=scale;
#pragma unroll
for(uint slot=0;slot<slots;++slot) {
    #pragma unroll
    for(uint reg=0;reg<registers;++reg) {
        uint j=$INDEX;
        if(j<width) {
            float code=peak==0.0f ? 0.0f : (values[slot][reg]/peak)*127.0f;
            Q[row*width+j]=char(clamp(rint(code),-127.0f,127.0f));
        }
    }
}
