using namespace metal;
using namespace mpp::tensor_ops;
constexpr int TM=$TM, TN=$TN, GROUPS=$GROUPS, HEAD_DIM=128, ROPE_HALF=$ROPE_HALF;
constexpr float epsilon=$EPSILON;
int M=A_shape[0], K=A_shape[1], N=B_shape[0], heads=N/HEAD_DIM;
int id=threadgroup_position_in_grid.x, row_tiles=(M+TM-1)/TM;
int band=id/(heads*$SWIZZLE), start=band*$SWIZZLE;
int height=min($SWIZZLE,row_tiles-start), inner=id%(heads*$SWIZZLE);
int by=start+inner%height, head=inner/height;
uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
threadgroup bfloat Projected[TM*HEAD_DIM];
// Generic accumulator indices define ownership; no lane-to-fragment layout
// assumptions are made. Each half-head accumulator dies before the next one.
#pragma unroll
for(int part=0;part<HEAD_DIM/TN;++part) {
    auto a=tensor((device $DTYPE*)A,dextents<int,2>(K,M),array<int,2>{1,K});
    auto b=tensor((device $DTYPE*)B,dextents<int,2>(K,N),array<int,2>{1,K});
    auto ta=a.slice(0,by*TM), tb=b.slice(0,head*HEAD_DIM+part*TN);
    constexpr auto desc=matmul2d_descriptor(TM,TN,dynamic_length_v<int>,false,true);
    matmul2d<desc,execution_simdgroups<GROUPS>> op;
    auto acc=op.get_destination_cooperative_tensor<decltype(ta),decltype(tb),$ACC>();
    op.run(ta,tb,acc);
    #pragma unroll
    for(ushort i=0;i<acc.get_capacity();++i) {
        if(acc.is_valid_element(i)) {
            auto ij=acc.get_multidimensional_index(i);
            int local_row=int(ij[1]), local_col=int(ij[0])+part*TN;
            int row=by*TM+local_row, col=head*HEAD_DIM+local_col;
            if(row<M) Projected[local_row*HEAD_DIM+local_col]=bfloat($VALUE);
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(int local_row=int(sg);local_row<TM;local_row+=GROUPS) {
    int row=by*TM+local_row;
    if(row<M) {
        $NORM_BODY
    }
}
