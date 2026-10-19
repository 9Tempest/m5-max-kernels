/* Independent CPU float64 reference for grouped Q/K/V. MIT. No Metal APIs. */
#ifndef M5_H3_QKV_REFERENCE_H
#define M5_H3_QKV_REFERENCE_H
#include <math.h>
#include <stdint.h>
#include <string.h>

static inline float qvr_f32(uint16_t bits) {
    uint32_t raw=(uint32_t)bits<<16; float value; memcpy(&value,&raw,4); return value;
}
static inline uint16_t qvr_bf16(float value) {
    uint32_t raw; memcpy(&raw,&value,4);
    raw+=0x7fffu+((raw>>16)&1u); return (uint16_t)(raw>>16);
}
/* Reference projection is rounded before statistics. Gamma precedes rotation;
 * no extra BF16 boundary is inserted after norm. V bypasses both operations. */
static inline void qvr_postprocess(const double projection[128],unsigned stream,
    const uint16_t gamma[128],const uint16_t cosine[48],const uint16_t sine[48],
    double epsilon,double output[128]) {
    double rounded[128],normalized[128],sum=0;
    for(unsigned d=0;d<128;d++) {
        rounded[d]=(double)qvr_f32(qvr_bf16((float)projection[d]));
        sum+=rounded[d]*rounded[d];
    }
    if(stream==2) { memcpy(output,rounded,sizeof(rounded)); return; }
    double inverse=1.0/sqrt(sum/128.0+epsilon);
    for(unsigned d=0;d<128;d++) normalized[d]=rounded[d]*inverse*(double)qvr_f32(gamma[d]);
    for(unsigned d=0;d<128;d++) {
        double value=normalized[d];
        if(d<48) value=normalized[d]*(double)qvr_f32(cosine[d])-normalized[d+48]*(double)qvr_f32(sine[d]);
        else if(d<96) value=normalized[d]*(double)qvr_f32(cosine[d-48])+normalized[d-48]*(double)qvr_f32(sine[d-48]);
        output[d]=(double)qvr_f32(qvr_bf16((float)value));
    }
}
/* Independent scalar H256, with an explicit FP32 boundary per butterfly.
 * The GPU lane/register implementation is not shared with this reference. */
static inline void qvr_h256(float *values,unsigned width) {
    for(unsigned block=0;block<width;block+=256) {
        for(unsigned stride=1;stride<256;stride*=2)
            for(unsigned start=0;start<256;start+=2*stride)
                for(unsigned j=0;j<stride;j++) {
                    float a=values[block+start+j],b=values[block+start+j+stride];
                    values[block+start+j]=a+b;
                    values[block+start+j+stride]=a-b;
                }
        for(unsigned j=0;j<256;j++) values[block+j]*=0.0625f;
    }
}
/* A complete head/stream, all K terms. Independent original and decoded dots.
 * Integer dot is exact for K5376, with float64 decoding of supplied FP32 scales. */
static inline void qvr_project(const uint16_t *input,const uint16_t *weights,
    const int8_t *codes,const int8_t *weight_codes,float input_scale,
    const float *weight_scales,unsigned k,double original[128],double decoded[128]) {
    for(unsigned d=0;d<128;d++) {
        double sum=0; int64_t integer_sum=0;
        for(unsigned j=0;j<k;j++) {
            sum+=(double)qvr_f32(input[j])*(double)qvr_f32(weights[(size_t)d*k+j]);
            integer_sum+=(int64_t)codes[j]*(int64_t)weight_codes[(size_t)d*k+j];
        }
        original[d]=sum;
        decoded[d]=(double)integer_sum*(double)input_scale*(double)weight_scales[d];
    }
}
#endif
