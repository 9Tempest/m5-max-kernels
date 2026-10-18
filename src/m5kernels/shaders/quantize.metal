using namespace metal;
constexpr bool block=$BLOCK, fp8=$FP8;
uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
uint tid=thread_position_in_threadgroup.x, row=threadgroup_position_in_grid.x;
uint width=block ? 32 : X_shape[1], offset=row*width;
float peak=0.0f;
for (uint j=tid;j<width;j+=256) peak=max(peak,abs(float(X[offset+j])));
peak=simd_max(peak);
threadgroup float peaks[8];
if(lane==0) peaks[sg]=peak;
threadgroup_barrier(mem_flags::mem_threadgroup);
peak=simd_max(lane<8 ? peaks[lane] : 0.0f);
float scale=peak==0.0f ? 1.0f : peak/(fp8 ? 448.0f : 127.0f);
int exponent=0;
if(block) {
    exponent=peak==0.0f ? 0 : clamp(int(floor(log2(peak)))-6,-127,127);
    scale=exp2(float(exponent));
}
if(tid==0) S[row]=block ? exponent+127 : scale;
for(uint j=tid;j<width;j+=256) {
    float v=float(X[offset+j])/scale;
    if(fp8) {
        uint sign=v<0.0f ? 128 : 0;
        float av=min(abs(v),448.0f);
        int bits;
        if(av<0.015625f) bits=int(rint(av*512.0f));
        else {
            int e=int(floor(log2(av)));
            int q=int(rint(av/exp2(float(e-3))));
            if(q==16) {q=8; e++;}
            bits=((e+7)<<3)+(q-8);
        }
        Q[offset+j]=uchar(sign|uint(min(bits,126)));
    } else {
        // OCP MXINT8 uses symmetric S1.6 endpoints; /64 is folded into the scale plane.
        Q[offset+j]=char(clamp(rint(v),-127.0f,127.0f));
    }
}
