using namespace metal;
using namespace mpp::tensor_ops;
constexpr int TM=$TM, TN=$TN;
int M=A_shape[0], K=A_shape[1], N=B_shape[0];
int nx=(N+TN-1)/TN, ny=(M+TM-1)/TM;
int id=threadgroup_position_in_grid.x;
// Grouped M ordering improves reuse without assuming a square launch grid.
int band=id/(nx*$SWIZZLE), start=band*$SWIZZLE;
int height=min($SWIZZLE,ny-start), inner=id%(nx*$SWIZZLE);
int by=start+inner%height, bx=inner/height;
$DECL
auto c=tensor(C,dextents<int,2>(N,M),array<int,2>{1,N});
auto ta=a.slice(0,by*TM), tb=b.slice(0,bx*TN);
constexpr auto desc=matmul2d_descriptor(TM,TN,dynamic_length_v<int>,false,true);
matmul2d<desc,execution_simdgroups<$GROUPS>> op;
auto acc=op.get_destination_cooperative_tensor<decltype(ta),decltype(tb),$ACC>();
op.run(ta,tb,acc);
$STORE
