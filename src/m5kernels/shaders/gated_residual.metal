using namespace metal;
uint index=thread_position_in_grid.x;
uint width=X_shape[1];
if(index>=X_shape[0]*width) return;
uint row=index/width, j=index%width;
uint gate_offset=$GATE_OFFSET;
// Keep the gate product in FP32 until after adding the residual. The cast
// here is the single native BF16/FP16 residual rounding boundary.
Y[index]=O(float(X[index])+float(Branch[index])*float(Gate[gate_offset+j]));
