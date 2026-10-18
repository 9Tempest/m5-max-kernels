using namespace metal;
constexpr uint threads=$THREADS, groups=threads/32;
constexpr bool layer=$LAYER;
constexpr float epsilon=$EPSILON;
uint row=threadgroup_position_in_grid.x, tid=thread_position_in_threadgroup.x;
uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
uint width=X_shape[1];
uint modulation_offset=$MOD_OFFSET;
threadgroup float partials[groups];
float first=0.0f;
for(uint j=tid;j<width;j+=threads) {
    float x=float(X[row*width+j]);
    first+=layer ? x : x*x;
}
first=simd_sum(first);
if(lane==0) partials[sg]=first;
threadgroup_barrier(mem_flags::mem_threadgroup);
first=simd_sum(lane<groups ? partials[lane] : 0.0f)/float(width);
threadgroup_barrier(mem_flags::mem_threadgroup);
float mean=layer ? first : 0.0f, variance=first;
if(layer) {
    variance=0.0f;
    for(uint j=tid;j<width;j+=threads) {
        float centered=float(X[row*width+j])-mean;
        variance+=centered*centered;
    }
    variance=simd_sum(variance);
    if(lane==0) partials[sg]=variance;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    variance=simd_sum(lane<groups ? partials[lane] : 0.0f)/float(width);
}
float inv=rsqrt(variance+epsilon);
for(uint j=tid;j<width;j+=threads) {
    float normalized=(float(X[row*width+j])-mean)*inv*float(Gamma[j]);
    Y[row*width+j]=O(normalized*(1.0f+float(Scale[modulation_offset+j]))
                   +float(Shift[modulation_offset+j]));
}
