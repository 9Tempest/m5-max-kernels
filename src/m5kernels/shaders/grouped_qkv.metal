// Original grouped-QKV experiment. Native source is compiled separately.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
#define GROUPED_QKV_DEVICE_SCRATCH $DEVICE_SCRATCH

struct GroupedQKVArgs {
    uint rows;
    uint input_dim;
    uint heads;
    uint head_dim;
    uint rope_half;
    uint head_major;
    float epsilon;
};

kernel void grouped_qkv(
    device const int8_t *A [[buffer(0)]],
    device const int8_t *W [[buffer(1)]],
    device const float *AS [[buffer(2)]],
    device const float *BS [[buffer(3)]],
    device bfloat *Q [[buffer(4)]],
    device bfloat *K [[buffer(5)]],
    device bfloat *V [[buffer(6)]],
    device const bfloat *QGamma [[buffer(7)]],
    device const bfloat *KGamma [[buffer(8)]],
    device const bfloat *Cos [[buffer(9)]],
    device const bfloat *Sin [[buffer(10)]],
    constant GroupedQKVArgs &args [[buffer(11)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM=$TM, TN=$TN, GROUPS=$GROUPS, SWIZZLE=$SWIZZLE;
    constexpr uint HEAD_DIM=128, ROPE_HALF=48;
    constexpr bool NATIVE_RMS=$NATIVE_RMS;
    uint column_tiles=args.heads*3, row_tiles=(args.rows+TM-1)/TM;
    uint band=group_id/(column_tiles*SWIZZLE), start=band*SWIZZLE;
    uint height=min(SWIZZLE,row_tiles-start), inner=group_id%(column_tiles*SWIZZLE);
    uint by=start+inner%height, grouped_column=inner/height;
    uint head=grouped_column/3, stream=grouped_column%3;
    uint padded_rows=((args.rows+127)/128)*128;
    uint row_start=by*TM, weight_start=grouped_column*HEAD_DIM;
    device bfloat *output=stream==0 ? Q : (stream==1 ? K : V);
    #if !GROUPED_QKV_DEVICE_SCRATCH
    threadgroup bfloat projected[TM*HEAD_DIM];
    #endif

    // TensorOps only reads these views. Non-const element types satisfy its
    // type traits; the underlying A/W buffers are never written.
    #pragma unroll
    for(uint part=0;part<HEAD_DIM/TN;++part) {
        #if $NATIVE_K
        constexpr uint K_TILE=$K_TILE;
        auto a=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(
            (device int8_t*)A,
            dextents<int32_t,2>(int(args.input_dim),int(padded_rows)));
        auto w=tensor<device int8_t,dextents<int32_t,2>,tensor_inline>(
            (device int8_t*)W,
            dextents<int32_t,2>(int(args.input_dim),int(args.heads*3*HEAD_DIM)));
        auto ta=a.slice<TM,K_TILE>(0,int(row_start));
        auto tw=w.slice<K_TILE,TN>(0,int(weight_start+part*TN));
        constexpr auto desc=matmul2d_descriptor(
            TM,TN,K_TILE,false,true,true,matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc,execution_simdgroups<GROUPS>> op;
        auto acc=op.get_destination_cooperative_tensor<decltype(ta),decltype(tw),int32_t>();
        #pragma unroll
        for(ushort i=0;i<acc.get_capacity();++i) {
            if(acc.is_valid_element(i)) acc[i]=0;
        }
        for(uint k=0;k<args.input_dim;k+=K_TILE) {
            auto ak=a.slice<TM,K_TILE>(int(k),int(row_start));
            auto wk=w.slice<K_TILE,TN>(int(k),int(weight_start+part*TN));
            op.run(ak,wk,acc);
        }
        #else
        auto a=tensor((device int8_t*)A,
            dextents<int,2>(int(args.input_dim),int(padded_rows)),
            array<int,2>{1,int(args.input_dim)});
        auto w=tensor((device int8_t*)W,
            dextents<int,2>(int(args.input_dim),int(args.heads*3*HEAD_DIM)),
            array<int,2>{1,int(args.input_dim)});
        auto ta=a.slice(0,int(row_start));
        auto tw=w.slice(0,int(weight_start+part*TN));
        constexpr auto desc=matmul2d_descriptor(TM,TN,dynamic_length_v<int>,false,true);
        matmul2d<desc,execution_simdgroups<GROUPS>> op;
        auto acc=op.get_destination_cooperative_tensor<decltype(ta),decltype(tw),int>();
        op.run(ta,tw,acc);
        #endif
        #pragma unroll
        for(ushort i=0;i<acc.get_capacity();++i) {
            if(acc.is_valid_element(i)) {
                auto ij=acc.get_multidimensional_index(i);
                uint local_row=uint(ij[1]), dimension=uint(ij[0])+part*TN;
                uint row=row_start+local_row;
                if(row<args.rows) {
                    bfloat rounded=bfloat(float(acc[i])*AS[row]*BS[weight_start+dimension]);
                    #if GROUPED_QKV_DEVICE_SCRATCH
                    output[(ulong(head)*args.rows+row)*HEAD_DIM+dimension]=rounded;
                    #else
                    if(stream==2) {
                        output[(ulong(head)*args.rows+row)*HEAD_DIM+dimension]=rounded;
                    } else {
                        projected[local_row*HEAD_DIM+dimension]=rounded;
                    }
                    #endif
                }
            }
        }
    }
    // Stream is uniform across the whole threadgroup. V is projection-only.
    if(stream==2) return;
    #if GROUPED_QKV_DEVICE_SCRATCH
    threadgroup_barrier(mem_flags::mem_device);
    #else
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #endif
    device const bfloat *gamma=stream==0 ? QGamma : KGamma;
    for(uint local_row=sg;local_row<TM;local_row+=GROUPS) {
        uint row=row_start+local_row;
        if(row<args.rows) {
            ulong destination=(ulong(head)*args.rows+row)*HEAD_DIM;
            #if GROUPED_QKV_DEVICE_SCRATCH
            device const bfloat *projected_row=output+destination;
            #else
            threadgroup const bfloat *projected_row=projected+local_row*HEAD_DIM;
            #endif
            float inverse;
            if(NATIVE_RMS) {
                float sum=0.0f;
                if(lane==0) {
                    for(uint d=0;d<HEAD_DIM;++d) {
                        float value=float(projected_row[d]);
                        sum=fma(value,value,sum);
                    }
                }
                inverse=rsqrt(simd_broadcast(sum,0)/float(HEAD_DIM)+args.epsilon);
            } else {
                float sum=0.0f;
                #pragma unroll
                for(uint d=lane;d<HEAD_DIM;d+=32) {
                    float value=float(projected_row[d]);
                    sum=fma(value,value,sum);
                }
                inverse=rsqrt(simd_sum(sum)/float(HEAD_DIM)+args.epsilon);
            }
            // Each unit owns either a complete rotary pair or one tail
            // channel. This SIMD owns the complete row: its RMS reduction
            // finishes before any output overwrite, and no other pair
            // reads a channel after this unit overwrites it.
            #pragma unroll
            for(uint unit=lane;unit<HEAD_DIM-ROPE_HALF;unit+=32) {
                if(unit<ROPE_HALF) {
                    uint upper=unit+ROPE_HALF;
                    float low=float(projected_row[unit])*inverse*float(gamma[unit]);
                    float high=float(projected_row[upper])*inverse*float(gamma[upper]);
                    float c=float(Cos[ulong(row)*ROPE_HALF+unit]);
                    float s=float(Sin[ulong(row)*ROPE_HALF+unit]);
                    output[destination+unit]=bfloat(low*c-high*s);
                    output[destination+upper]=bfloat(high*c+low*s);
                } else {
                    uint d=unit+ROPE_HALF;
                    output[destination+d]=bfloat(
                        float(projected_row[d])*inverse*float(gamma[d]));
                }
            }
        }
    }
}
