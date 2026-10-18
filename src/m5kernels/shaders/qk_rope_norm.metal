using namespace metal;
constexpr int HEAD_DIM=128, ROPE_HALF=$ROPE_HALF, GROUPS=$GROUPS;
constexpr float epsilon=$EPSILON;
int M=P_shape[0], N=P_shape[1], heads=N/HEAD_DIM;
uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
int index=int(threadgroup_position_in_grid.x)*GROUPS+int(sg);
if(index<heads*M) {
    int head=index/M, row=index%M;
    $NORM_BODY
}
