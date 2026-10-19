/* Independent CPU reference for the opt-in native projection/gate fusion.
 * Original code, MIT; no Metal implementation is included here. */
#ifndef H3_FUSED_REFERENCE_H
#define H3_FUSED_REFERENCE_H
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline uint16_t ref_bf16(float x) {
    uint32_t bits; memcpy(&bits,&x,4);
    bits+=0x7fffu+((bits>>16)&1u); return (uint16_t)(bits>>16);
}
static inline float ref_f32(uint16_t x) {
    uint32_t bits=(uint32_t)x<<16; float y; memcpy(&y,&bits,4); return y;
}

/* Complete K, one selected row, every output column. Group scales belong
 * to activations; weight scales are always one per output channel. */
static inline void ref_projection(const uint16_t *original_a,
        const uint16_t *original_w,const int8_t *codes_a,const int8_t *codes_w,
        const float *scales_a,const float *scales_w,size_t k,size_t n,
        size_t group_size,double *decoded,double *unquantized) {
    for(size_t c=0;c<n;c++) {
        double quantized_sum=0,original_sum=0;
        for(size_t j=0;j<k;j++) {
            double a=(double)codes_a[j]*(double)scales_a[j/group_size];
            double w=(double)codes_w[c*k+j]*(double)scales_w[c];
            quantized_sum+=a*w;
            original_sum+=(double)ref_f32(original_a[j])*(double)ref_f32(original_w[c*k+j]);
        }
        decoded[c]=quantized_sum; unquantized[c]=original_sum;
    }
}

/* Preserve declared BF16 boundaries. RMS and AdaLN are independently
 * evaluated in float64, rather than reproducing the Metal reduction tree.
 * Gate uses FP32 fmaf and RNE BF16, matching the residual contract. */
static inline int ref_follow(const double *projection,const uint16_t *residual,
        const uint16_t *gate,const uint16_t *norm,const uint16_t *shift,
        const uint16_t *scale,size_t width,uint16_t *branch,uint16_t *hidden,
        uint16_t *adaln,int8_t *quantized,float *output_scale) {
    double sum=0;
    for(size_t c=0;c<width;c++) {
        if(!isfinite(projection[c])||!isfinite(ref_f32(residual[c]))||
            !isfinite(ref_f32(gate[c]))||!isfinite(ref_f32(norm[c]))||
            !isfinite(ref_f32(shift[c]))||!isfinite(ref_f32(scale[c]))) return 0;
        branch[c]=ref_bf16((float)projection[c]);
        hidden[c]=ref_bf16(fmaf(ref_f32(branch[c]),ref_f32(gate[c]),ref_f32(residual[c])));
        if(!isfinite(ref_f32(branch[c]))||!isfinite(ref_f32(hidden[c]))) return 0;
        double value=ref_f32(hidden[c]); sum+=value*value;
    }
    double inverse=1.0/sqrt(sum/(double)width+(double)1e-5f),maximum=0;
    for(size_t c=0;c<width;c++) {
        double value=(double)ref_f32(hidden[c])*inverse*(double)ref_f32(norm[c]);
        value=value*(1.0+(double)ref_f32(scale[c]))+(double)ref_f32(shift[c]);
        adaln[c]=ref_bf16((float)value);
        if(!isfinite(ref_f32(adaln[c]))) return 0;
        double magnitude=fabs((double)ref_f32(adaln[c]));
        if(magnitude>maximum) maximum=magnitude;
    }
    float max_abs=(float)maximum;
    *output_scale=max_abs>0?max_abs/127.0f:1.0f/127.0f;
    float multiplier=max_abs>0?127.0f/max_abs:127.0f;
    for(size_t c=0;c<width;c++) {
        float value=nearbyintf(ref_f32(adaln[c])*multiplier);
        if(value>127) value=127;
        if(value< -127) value= -127;
        quantized[c]=(int8_t)value;
    }
    return 1;
}
#endif
