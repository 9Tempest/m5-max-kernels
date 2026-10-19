/* Original MIT-licensed M5MaxLocalBenchmark Q/K tile port.
 * Source: src/m5kernels/shaders/qk_rope_projection.metal + qk_rope_body.metal.
 * TM64/TN128/four SIMD groups/swizzle8; native grouped Q,K,V weight layout.
 * This is a candidate, not an asserted improvement over native grouped QKV.
 * Include inside H3_METAL_HAS_TENSOR after h3_qkv_project_rope_args. */
kernel void h3_m5_qkv_rope_int8_r64_swizzle8(
    device int8_t *input [[buffer(0)]],
    device int8_t *weight [[buffer(1)]],
    device const float *input_scales [[buffer(2)]],
    device const float *weight_scales [[buffer(3)]],
    device bfloat *query [[buffer(4)]],
    device bfloat *key [[buffer(5)]],
    device bfloat *value [[buffer(6)]],
    device const bfloat *q_gamma [[buffer(7)]],
    device const bfloat *k_gamma [[buffer(8)]],
    device const bfloat *rope_cos [[buffer(9)]],
    device const bfloat *rope_sin [[buffer(10)]],
    constant h3_qkv_project_rope_args &args [[buffer(11)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM=64, TN=128, GROUPS=4, SWIZZLE=8, D=128, HALF=48;
    uint row_tiles=(args.rows+TM-1)/TM, columns=args.heads*3;
    uint band=group_id/(columns*SWIZZLE), start=band*SWIZZLE;
    uint height=min(SWIZZLE,row_tiles-start), inner=group_id%(columns*SWIZZLE);
    uint row_start=(start+inner%height)*TM, column_tile=inner/height;
    uint head=column_tile/3, stream=column_tile%3, weight_start=column_tile*D;
    uint padded_rows=(args.rows+127u)&~127u;
    device bfloat *output=stream==0?query:stream==1?key:value;
    threadgroup bfloat projected[TM*D];
    auto a=tensor(input,dextents<int,2>(int(args.input_dim),int(padded_rows)),
                  array<int,2>{1,int(args.input_dim)});
    auto w=tensor(weight,dextents<int,2>(int(args.input_dim),int(columns*D)),
                  array<int,2>{1,int(args.input_dim)});
    auto ta=a.slice(0,int(row_start)), tw=w.slice(0,int(weight_start));
    constexpr auto desc=matmul2d_descriptor(TM,TN,dynamic_length_v<int>,false,true);
    matmul2d<desc,execution_simdgroups<GROUPS>> op;
    auto accum=op.get_destination_cooperative_tensor<decltype(ta),decltype(tw),int>();
    op.run(ta,tw,accum);
    #pragma unroll
    for(ushort i=0;i<accum.get_capacity();++i) {
        if(!accum.is_valid_element(i)) continue;
        auto ij=accum.get_multidimensional_index(i);
        uint local_row=uint(ij[1]), dimension=uint(ij[0]);
        uint row=row_start+local_row;
        if(row<args.rows) {
            // Preserve native left-associated scales and BF16 projection round.
            bfloat rounded=bfloat(float(accum[i])*input_scales[row]*
                                  weight_scales[weight_start+dimension]);
            if(stream==2)
                output[(ulong(head)*args.rows+row)*D+dimension]=rounded;
            else projected[local_row*D+dimension]=rounded;
        }
    }
    // Stream is uniform across the threadgroup. V bypasses normalization/RoPE.
    if(stream==2) return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const bfloat *gamma=stream==0?q_gamma:k_gamma;
    for(uint local_row=sg;local_row<TM;local_row+=GROUPS) {
        uint row=row_start+local_row;
        if(row>=args.rows) continue;
        float square_sum=0.0f;
        #pragma unroll
        for(uint d=lane;d<D;d+=32) {
            float x=float(projected[local_row*D+d]);
            square_sum+=x*x;
        }
        float inverse=rsqrt(simd_sum(square_sum)/float(D)+args.epsilon);
        #pragma unroll
        for(uint d=lane;d<D;d+=32) {
            float y=float(projected[local_row*D+d])*inverse*float(gamma[d]);
            if(d<2*HALF) {
                uint pair=d<HALF?d+HALF:d-HALF, angle=d<HALF?d:d-HALF;
                float paired=float(projected[local_row*D+pair])*inverse*float(gamma[pair]);
                float c=float(rope_cos[ulong(row)*HALF+angle]);
                float s=float(rope_sin[ulong(row)*HALF+angle]);
                y=d<HALF?y*c-paired*s:y*c+paired*s;
            }
            output[(ulong(head)*args.rows+row)*D+d]=bfloat(y);
        }
    }
}
