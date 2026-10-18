using namespace metal;
constexpr uint threads=$THREADS;
uint row=threadgroup_position_in_grid.x, tid=thread_position_in_threadgroup.x;
uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
uint width=X_shape[1];
float sum=0.0f;
for(uint j=tid;j<width;j+=threads) {float x=float(X[row*width+j]); sum+=x*x;}
sum=simd_sum(sum);
threadgroup float sums[32];
if(lane==0) sums[sg]=sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
sum=simd_sum(lane<threads/32 ? sums[lane] : 0.0f);
float inv=rsqrt(sum/float(width)+1e-5f);
for(uint j=tid;j<width;j+=threads) {
    float value=float(X[row*width+j])*inv*float(W[j]);
    Y[row*width+j]=O(value*(1.0f+float(Scale[j]))+float(Shift[j]));
}
