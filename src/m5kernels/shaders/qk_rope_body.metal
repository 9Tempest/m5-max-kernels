// One SIMD group owns a full row/head. RMS spans all 128 dimensions.
float square_sum=0.0f;
#pragma unroll
for(int dimension=int(lane);dimension<HEAD_DIM;dimension+=32) {
    float value=float($READ);
    square_sum+=value*value;
}
float inverse=rsqrt(simd_sum(square_sum)/float(HEAD_DIM)+epsilon);
#pragma unroll
for(int dimension=int(lane);dimension<HEAD_DIM;dimension+=32) {
    float value=float($READ)*inverse*float(Gamma[dimension]);
    if(dimension<2*ROPE_HALF) {
        int pair=dimension<ROPE_HALF ? dimension+ROPE_HALF : dimension-ROPE_HALF;
        float paired=float($READ_PAIR)*inverse*float(Gamma[pair]);
        int angle=dimension<ROPE_HALF ? dimension : dimension-ROPE_HALF;
        float c=float(Cos[row*ROPE_HALF+angle]), s=float(Sin[row*ROPE_HALF+angle]);
        value=dimension<ROPE_HALF ? value*c-paired*s : value*c+paired*s;
    }
    Y[(head*M+row)*HEAD_DIM+dimension]=bfloat(value);
}
