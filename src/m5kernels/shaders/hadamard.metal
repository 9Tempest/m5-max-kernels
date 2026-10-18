using namespace metal;
constexpr uint block=$BLOCK;
uint tid=thread_position_in_threadgroup.x;
uint offset=threadgroup_position_in_grid.x*block;
uint width=X_shape[1];
float value=float(X[offset+tid])*float(Signs[(offset+tid)%width]);

// Natural-order Sylvester butterflies: upper lanes produce lower - upper.
// Each SIMD group owns 32 contiguous elements.
#pragma unroll
for(uint stride=1;stride<32;stride*=2) {
    float peer=simd_shuffle_xor(value,stride);
    value=(tid&stride) ? peer-value : value+peer;
}

threadgroup float scratch[block];
#pragma unroll
for(uint stride=32;stride<block;stride*=2) {
    scratch[tid]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float peer=scratch[tid^stride];
    value=(tid&stride) ? peer-value : value+peer;
    // All peer loads finish before any group overwrites scratch next stage.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
Y[offset+tid]=value*rsqrt(float(block));
