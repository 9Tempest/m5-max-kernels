/* Paired numerical validation and warmed online timing. Caller MUST own GPU flock.
 * Uses real block-0 weights and deterministic synthetic BF16 activations.
 * Compare original projection + gate/AdaLN/quant with fused projection gate
 * + pre-gated AdaLN/quant. Independent CPU fmaf checks every gated element.
 * Timings exclude weight preparation, reference/readback and residual reset.
 * Timings include online activation packing, projection and next AdaLN/quant.
 * All output files are exclusive. */
#include "h3_gpu.h"
#include "h3_weights.h"
#include "fused_reference.h"
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { N = 5376, SLOTS = 6, MOD_ROWS = 3, HEADS = 56, D = 128 };
static void die(const char *s) { fprintf(stderr, "fused_validate: %s\n", s); exit(1); }
static void *allocate(size_t n) { void *p=calloc(n,1); if(!p) die("out of memory"); return p; }
static uint16_t bf16(float x) {
    uint32_t bits; memcpy(&bits,&x,4); bits+=0x7fffu+((bits>>16)&1u); return (uint16_t)(bits>>16);
}
static float f32(uint16_t x) { uint32_t bits=(uint32_t)x<<16; float y; memcpy(&y,&bits,4); return y; }
static float sample(uint32_t row,uint32_t col,uint32_t seed) {
    uint32_t v=(row+1u)*2654435761u ^ (col+7u)*2246822519u ^ seed;
    v^=v>>13; v*=3266489917u; v^=v>>16;
    return ((float)(v&65535u)/32768.0f-1.0f)*0.5f;
}
static size_t mismatch(const void *left,const void *right,size_t count,size_t width) {
    const unsigned char *a=left,*b=right; size_t n=0;
    for(size_t i=0;i<count;i++) n+=(memcmp(a+i*width,b+i*width,width)!=0);
    return n;
}
static void check_gpu(int ok,h3_gpu *g) { if(!ok) die(h3_gpu_error(g)); }

typedef struct {
    h3_gpu *g;
    h3_gpu_tensor *x,*aq,*as,*wq,*ws,*branch,*original,*fused;
    h3_gpu_tensor *modulation,*norm_modulation,*weight,*row_map,*oq,*fq,*os,*fs;
    uint32_t rows,padded,k,gate_slot,shift_slot,scale_slot;
    int attention,hadamard;
} path_context;

static void encode_path(path_context *p,int candidate) {
    h3_gpu *g=p->g;
    if(candidate) {
        check_gpu(p->attention?h3_gpu_linear_int8_head_major_gate_bf16(g,p->fused,p->aq,p->as,p->x,p->wq,p->ws,p->rows,HEADS,D,N,
            p->fused,p->modulation,p->row_map,SLOTS,p->gate_slot):h3_gpu_fc2_int8_gate_bf16(g,p->fused,p->aq,p->as,p->x,p->wq,p->ws,
            p->rows,p->k,N,p->hadamard?1:0,0,p->fused,p->modulation,p->row_map,SLOTS,p->gate_slot),g);
        check_gpu(h3_gpu_adaln_quantize_int8_pre_gated(g,p->fq,p->fs,p->fused,p->weight,p->norm_modulation,p->row_map,
            p->rows,p->padded,N,SLOTS,p->shift_slot,p->scale_slot,1e-5f),g);
    } else {
        check_gpu(p->attention?h3_gpu_linear_int8_head_major_bf16(g,p->branch,p->aq,p->as,p->x,p->wq,p->ws,p->rows,HEADS,D,N):
            h3_gpu_fc2_int8_bf16(g,p->branch,p->aq,p->as,p->x,p->wq,p->ws,p->rows,p->k,N,p->hadamard?1:0,0),g);
        check_gpu(h3_gpu_gate_adaln_quantize_int8(g,p->original,p->oq,p->os,p->original,p->branch,p->weight,
            p->modulation,p->norm_modulation,p->row_map,p->rows,p->padded,N,SLOTS,p->gate_slot,p->shift_slot,p->scale_slot,1e-5f),g);
    }
}

static double now_seconds(void) {
    struct timespec t; if(clock_gettime(CLOCK_MONOTONIC,&t)) die("clock failure");
    return (double)t.tv_sec+(double)t.tv_nsec*1e-9;
}

typedef struct { double wall,encode,submit_wait,gpu; uint64_t dispatches; } timing;
static timing time_path(path_context *p,int candidate,const uint16_t *residual) {
    /* Previous invocation has completed. Restore exactly the same operand;
     * shared-memory CPU write is outside both wall and device intervals. */
    check_gpu(h3_gpu_tensor_write_bf16(candidate?p->fused:p->original,residual,(size_t)p->rows*N),p->g);
    h3_gpu_stats before,after;
    check_gpu(h3_gpu_get_stats(p->g,&before),p->g);
    double start=now_seconds();
    check_gpu(h3_gpu_begin(p->g),p->g);
    encode_path(p,candidate);
    double encoded=now_seconds();
    check_gpu(h3_gpu_submit(p->g),p->g);
    double end=now_seconds();
    check_gpu(h3_gpu_get_stats(p->g,&after),p->g);
    timing t={end-start,encoded-start,end-encoded,after.gpu_seconds-before.gpu_seconds,
        after.direct_dispatches-before.direct_dispatches};
    if(!isfinite(t.wall)||t.wall<=0||!isfinite(t.gpu)||t.gpu<0) die("invalid timing");
    return t;
}
static void print_timings(FILE *out,const timing *values,unsigned count) {
    fputc('[',out);
    for(unsigned i=0;i<count;i++) {
        if(i) fputc(',',out);
        fprintf(out,"{\"wall_seconds\":%.12g,\"cpu_encode_seconds\":%.12g,\"submit_wait_seconds\":%.12g,\"supplemental_gpu_seconds\":%.12g,\"direct_dispatches\":%llu}",
            values[i].wall,values[i].encode,values[i].submit_wait,values[i].gpu,(unsigned long long)values[i].dispatches);
    }
    fputc(']',out);
}

typedef struct { double squared_error,squared_reference,max_abs; size_t count,nonfinite; } error_metric;
static void add_error(error_metric *metric,double got,double reference) {
    metric->count++;
    if(!isfinite(got)||!isfinite(reference)) { metric->nonfinite++; return; }
    double delta=got-reference;
    metric->squared_error+=delta*delta; metric->squared_reference+=reference*reference;
    if(fabs(delta)>metric->max_abs) metric->max_abs=fabs(delta);
}
static double relative_error(const error_metric *m) {
    if(m->nonfinite||!m->count) return INFINITY;
    return m->squared_reference>0?sqrt(m->squared_error/m->squared_reference):m->squared_error==0?0:INFINITY;
}
static void print_metric(FILE *out,const error_metric *m) {
    double relative=relative_error(m);
    fprintf(out,"{\"count\":%zu,\"nonfinite\":%zu,\"relative_l2\":",m->count,m->nonfinite);
    if(isfinite(relative)) fprintf(out,"%.17g",relative); else fputs("null",out);
    fprintf(out,",\"rmse\":%.17g,\"max_abs\":%.17g}",m->count?sqrt(m->squared_error/(double)m->count):0,m->max_abs);
}

int main(int argc,char **argv) {
    if(argc!=7&&argc!=9) die("usage: fused_validate MODEL_DIT_DIRECTORY SHADER ROWS SEED POLICY NEW_REPORT.json [WARMUPS REPEATS]; POLICY=group1024|h256|attention");
    char *end=NULL; unsigned long m=strtoul(argv[3],&end,10);
    if(!end||*end||m<=2048||m>16384) die("rows must be 2049..16384");
    uint32_t rows=(uint32_t)m,padded=(rows+127u)&~127u;
    unsigned long seed_value=strtoul(argv[4],&end,10);
    if(!end||*end||seed_value>UINT32_MAX) die("seed must fit uint32");
    uint32_t seed=(uint32_t)seed_value;
    unsigned warmups=3,repeats=10;
    if(argc==9) {
        unsigned long value=strtoul(argv[7],&end,10);
        if(!end||*end||value<2||value>100) die("warmups must be 2..100");
        warmups=(unsigned)value;
        value=strtoul(argv[8],&end,10);
        if(!end||*end||value<3||value>1000) die("repeats must be 3..1000");
        repeats=(unsigned)value;
    }
    int attention=!strcmp(argv[5],"attention"), hadamard=!strcmp(argv[5],"h256");
    if(!attention&&!hadamard&&strcmp(argv[5],"group1024")) die("invalid policy");
    uint32_t k=attention?7168u:14336u,gate_slot=attention?2u:5u;
    uint32_t shift_slot=attention?3u:0u,scale_slot=attention?4u:1u;
    size_t elements=(size_t)rows*N,padded_elements=(size_t)padded*N;
    uint16_t *input=allocate((size_t)rows*k*2),*residual=allocate(elements*2);
    uint16_t *mod=allocate((size_t)MOD_ROWS*SLOTS*N*2),*norm_mod=allocate((size_t)MOD_ROWS*SLOTS*N*2),*norm=allocate(N*2);
    uint32_t *maps=allocate((size_t)rows*4);
    for(uint32_t r=0;r<rows;r++) {
        maps[r]=r%MOD_ROWS;
        for(uint32_t c=0;c<k;c++) {
            float v=r==0?0:sample(r,c,seed);
            if(r==rows-1&&c%257u==0) v*=32.0f;
            size_t index=attention?((size_t)(c/D)*rows+r)*D+c%D:(size_t)r*k+c;
            input[index]=bf16(v);
        }
        for(uint32_t c=0;c<N;c++) residual[(size_t)r*N+c]=bf16(r==0?0:sample(r,c,seed+1u));
    }
    for(uint32_t c=0;c<N;c++) norm[c]=bf16(0.75f+0.25f*sample(3,c,seed));
    for(uint32_t r=0;r<MOD_ROWS;r++) for(uint32_t slot=0;slot<SLOTS;slot++)
        for(uint32_t c=0;c<N;c++) {
            float v=sample(r+slot,c,seed+2u);
            if(slot==gate_slot&&c%19u==0) v=0;
            if(r==0&&slot==shift_slot) v=0;
            mod[((size_t)r*SLOTS+slot)*N+c]=bf16(v);
            norm_mod[((size_t)r*SLOTS+slot)*N+c]=bf16(r==0&&slot==shift_slot?0:sample(r+slot,c,seed+3u));
        }
    int fd=open(argv[6],O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd<0) die("report must be a new file");
    FILE *report=fdopen(fd,"w"); if(!report) die("cannot open report");
    char error[1024]; h3_gpu *g=h3_gpu_create(argv[2],error,sizeof(error));
    if(!g) die(error);
    if(!h3_gpu_has_int8_mlp(g)) die("TensorOps unavailable; no fallback");
    h3_weight_store *store=h3_weight_store_open(argv[1],error,sizeof(error));
    if(!store) die(error);
    const char *key=attention?"blocks.0.attn.out_proj.weight":"blocks.0.mlp.fc2.weight";
    uint64_t shape[]={N,k};
    h3_gpu_tensor *w=h3_weight_load_bf16(store,g,key,2,shape,error,sizeof(error));
    if(!w) die(error);
    h3_gpu_tensor *x=h3_gpu_tensor_from_bf16(g,input,(size_t)rows*k);
    h3_gpu_tensor *aq=h3_gpu_tensor_new_i8(g,(size_t)padded*k);
    h3_gpu_tensor *as=h3_gpu_tensor_new_f32(g,(size_t)padded*14);
    h3_gpu_tensor *wq=h3_gpu_tensor_new_i8(g,(size_t)N*k);
    h3_gpu_tensor *ws=h3_gpu_tensor_new_f32(g,N);
    h3_gpu_tensor *branch=h3_gpu_tensor_new_bf16(g,elements);
    h3_gpu_tensor *original=h3_gpu_tensor_from_bf16(g,residual,elements);
    h3_gpu_tensor *fused=h3_gpu_tensor_from_bf16(g,residual,elements);
    h3_gpu_tensor *modulation=h3_gpu_tensor_from_bf16(g,mod,(size_t)MOD_ROWS*SLOTS*N);
    h3_gpu_tensor *norm_modulation=attention?modulation:h3_gpu_tensor_from_bf16(g,norm_mod,(size_t)MOD_ROWS*SLOTS*N);
    h3_gpu_tensor *weight=h3_gpu_tensor_from_bf16(g,norm,N);
    h3_gpu_tensor *row_map=h3_gpu_tensor_from_u32(g,maps,rows);
    h3_gpu_tensor *oq=h3_gpu_tensor_new_i8(g,padded_elements),*fq=h3_gpu_tensor_new_i8(g,padded_elements);
    h3_gpu_tensor *os=h3_gpu_tensor_new_f32(g,padded),*fs=h3_gpu_tensor_new_f32(g,padded);
    if(!x||!aq||!as||!wq||!ws||!branch||!original||!fused||!modulation||!norm_modulation||!weight||!row_map||!oq||!fq||!os||!fs) die(h3_gpu_error(g));
    path_context context={g,x,aq,as,wq,ws,branch,original,fused,modulation,norm_modulation,weight,row_map,oq,fq,os,fs,
        rows,padded,k,gate_slot,shift_slot,scale_slot,attention,hadamard};
    /* Distinct poisons prevent missing stores from passing paired equality. */
    int8_t *poison=allocate(padded_elements); memset(poison,0x80,padded_elements);
    check_gpu(h3_gpu_tensor_write_i8(oq,poison,padded_elements),g);
    memset(poison,0x7f,padded_elements);
    check_gpu(h3_gpu_tensor_write_i8(fq,poison,padded_elements),g); free(poison);
    uint16_t *branch_poison=allocate(elements*2);
    for(size_t i=0;i<elements;i++) branch_poison[i]=0x7fc0;
    check_gpu(h3_gpu_tensor_write_bf16(branch,branch_poison,elements),g); free(branch_poison);
    float *scale_poison=allocate((size_t)padded*4);
    for(uint32_t r=0;r<padded;r++) scale_poison[r]=NAN;
    check_gpu(h3_gpu_tensor_write_f32(os,scale_poison,padded)&&h3_gpu_tensor_write_f32(fs,scale_poison,padded),g);
    free(scale_poison);
    check_gpu(h3_gpu_begin(g),g);
    check_gpu(hadamard?h3_gpu_quantize_h256_int8(g,wq,ws,w,N,N,k):h3_gpu_quantize_weight_int8(g,wq,ws,w,N,k),g);
    check_gpu(h3_gpu_submit(g),g);
    check_gpu(h3_gpu_begin(g),g);
    encode_path(&context,0);
    check_gpu(h3_gpu_submit(g),g);
    check_gpu(h3_gpu_begin(g),g);
    encode_path(&context,1);
    check_gpu(h3_gpu_submit(g),g);
    uint16_t *b=allocate(elements*2),*o=allocate(elements*2),*f=allocate(elements*2);
    int8_t *oc=allocate(padded_elements),*fc=allocate(padded_elements);
    float *osc=allocate((size_t)padded*4),*fsc=allocate((size_t)padded*4);
    check_gpu(h3_gpu_tensor_read_bf16(branch,b,elements)&&h3_gpu_tensor_read_bf16(original,o,elements)&&
        h3_gpu_tensor_read_bf16(fused,f,elements)&&h3_gpu_tensor_read_i8(oq,oc,padded_elements)&&
        h3_gpu_tensor_read_i8(fq,fc,padded_elements)&&h3_gpu_tensor_read_f32(os,osc,padded)&&
        h3_gpu_tensor_read_f32(fs,fsc,padded),g);
    size_t nonfinite=0,cpu_original=0,cpu_fused=0;
    for(size_t i=0;i<elements;i++) {
        uint32_t r=(uint32_t)(i/N),c=(uint32_t)(i%N);
        float gate=f32(mod[((size_t)maps[r]*SLOTS+gate_slot)*N+c]);
        uint16_t reference=bf16(fmaf(f32(b[i]),gate,f32(residual[i])));
        cpu_original+=(reference!=o[i]); cpu_fused+=(reference!=f[i]);
        nonfinite+=!isfinite(f32(b[i]))+!isfinite(f32(o[i]))+!isfinite(f32(f[i]));
    }
    for(uint32_t r=0;r<padded;r++) nonfinite+=!isfinite(osc[r])+!isfinite(fsc[r]);
    size_t hidden_mismatch=mismatch(o,f,elements,2),codes_mismatch=mismatch(oc,fc,padded_elements,1);
    size_t scales_mismatch=mismatch(osc,fsc,padded,4);
    /* Independent full-K decoded and original BF16 dots, three complete
     * hidden rows. No model weights or activation arrays are written out. */
    uint32_t selected_rows[]={0,rows/2,rows-1};
    uint16_t *original_w=allocate((size_t)N*k*2),*original_a=allocate((size_t)k*2);
    int8_t *packed_w=allocate((size_t)N*k),*packed_a=allocate(k);
    float *weight_scales=allocate(N*4),activation_scales[14];
    double decoded[N],unquantized[N];
    uint16_t ref_branch[N],ref_hidden[N],ref_adaln[N]; int8_t ref_codes[N];
    uint16_t original_branch[N],original_hidden[N],original_adaln[N]; int8_t original_codes[N];
    float ref_scale,original_scale;
    error_metric projection_implementation={0},projection_quantization={0},hidden_implementation={0},next_quant_implementation={0};
    error_metric hidden_quantization={0},next_quantization={0};
    error_metric row_metrics[3][4]={0};
    size_t reference_failures=0;
    check_gpu(h3_gpu_tensor_read_bf16(w,original_w,(size_t)N*k)&&
        h3_gpu_tensor_read_i8(wq,packed_w,(size_t)N*k)&&h3_gpu_tensor_read_f32(ws,weight_scales,N),g);
    for(unsigned sample_index=0;sample_index<3;sample_index++) {
        uint32_t r=selected_rows[sample_index];
        size_t groups=(!hadamard&&!attention)?14:1;
        check_gpu(h3_gpu_tensor_read_i8_range(aq,(size_t)r*k,packed_a,k)&&
            h3_gpu_tensor_read_f32_range(as,(size_t)r*groups,activation_scales,groups),g);
        for(uint32_t c=0;c<k;c++) original_a[c]=input[attention?((size_t)(c/D)*rows+r)*D+c%D:(size_t)r*k+c];
        ref_projection(original_a,original_w,packed_a,packed_w,activation_scales,weight_scales,k,N,k/groups,decoded,unquantized);
        const uint16_t *norm_table=attention?mod:norm_mod;
        size_t mod_base=(size_t)maps[r]*SLOTS*N;
        int valid=ref_follow(decoded,residual+(size_t)r*N,mod+mod_base+(size_t)gate_slot*N,norm,
            norm_table+mod_base+(size_t)shift_slot*N,norm_table+mod_base+(size_t)scale_slot*N,N,
            ref_branch,ref_hidden,ref_adaln,ref_codes,&ref_scale);
        valid=valid&&ref_follow(unquantized,residual+(size_t)r*N,mod+mod_base+(size_t)gate_slot*N,norm,
            norm_table+mod_base+(size_t)shift_slot*N,norm_table+mod_base+(size_t)scale_slot*N,N,
            original_branch,original_hidden,original_adaln,original_codes,&original_scale);
        if(!valid) { reference_failures++; continue; }
        for(uint32_t c=0;c<N;c++) {
            size_t i=(size_t)r*N+c;
            add_error(&projection_implementation,f32(b[i]),ref_f32(ref_branch[c]));
            add_error(&projection_quantization,decoded[c],unquantized[c]);
            add_error(&hidden_implementation,f32(f[i]),ref_f32(ref_hidden[c]));
            add_error(&next_quant_implementation,(double)fc[i]*fsc[r],(double)ref_codes[c]*ref_scale);
            add_error(&hidden_quantization,ref_f32(ref_hidden[c]),ref_f32(original_hidden[c]));
            add_error(&next_quantization,(double)ref_codes[c]*ref_scale,(double)original_codes[c]*original_scale);
            add_error(&row_metrics[sample_index][0],f32(b[i]),ref_f32(ref_branch[c]));
            add_error(&row_metrics[sample_index][1],decoded[c],unquantized[c]);
            add_error(&row_metrics[sample_index][2],f32(f[i]),ref_f32(ref_hidden[c]));
            add_error(&row_metrics[sample_index][3],(double)fc[i]*fsc[r],(double)ref_codes[c]*ref_scale);
        }
    }
    size_t zero_errors=0,padding_errors=0;
    for(uint32_t c=0;c<N;c++) zero_errors+=(f32(o[c])!=0)+(f32(f[c])!=0)+(oc[c]!=0)+(fc[c]!=0);
    zero_errors+=(osc[0]!=1.0f/127.0f)+(fsc[0]!=1.0f/127.0f);
    for(size_t i=elements;i<padded_elements;i++) padding_errors+=(oc[i]!=0)+(fc[i]!=0);
    for(uint32_t r=rows;r<padded;r++) padding_errors+=(osc[r]!=1)+(fsc[r]!=1);
    int independent_pass=!reference_failures&&relative_error(&projection_implementation)<=8e-3&&
        relative_error(&hidden_implementation)<=8e-3&&relative_error(&next_quant_implementation)<=8e-3&&
        !projection_quantization.nonfinite;
    for(unsigned r=0;r<3;r++) for(unsigned metric=0;metric<4;metric++)
        if(metric!=1&&relative_error(&row_metrics[r][metric])>8e-3) independent_pass=0;
    int pass=independent_pass&&!nonfinite&&!cpu_original&&!cpu_fused&&!hidden_mismatch&&!codes_mismatch&&!scales_mismatch&&!zero_errors&&!padding_errors;
    timing *native_times=allocate((size_t)repeats*sizeof(timing)),*fused_times=allocate((size_t)repeats*sizeof(timing));
    unsigned initial_order=(seed&4u)?1u:0u;
    if(pass) {
        for(unsigned i=0;i<warmups+repeats;i++) for(unsigned j=0;j<2;j++) {
            unsigned candidate=(i+initial_order+j)&1u;
            timing t=time_path(&context,(int)candidate,residual);
            if(i>=warmups) (candidate?fused_times:native_times)[i-warmups]=t;
        }
    }
    fprintf(report,"{\"schema\":2,\"scope\":\"real_weights_synthetic_activations_paired_online\",\"policy\":\"%s\",\"rows\":%u,\"padded_rows\":%u,\"seed\":%u,\"K\":%u,\"N\":%d,\"weight_key\":\"%s\",\"in_place_residual\":true,\"gate_slot\":%u,\"distinct_next_block_modulation\":%s,\"cpu_reference\":\"FP32 fmaf over original BF16 residual/gate and independently stored original BF16 projection, then RNE BF16\",\"nonfinite\":%zu,\"cpu_original_mismatches\":%zu,\"cpu_fused_mismatches\":%zu,\"hidden_bit_mismatches\":%zu,\"quantized_code_mismatches\":%zu,\"quantized_scale_bit_mismatches\":%zu,\"zero_row_errors\":%zu,\"padding_errors\":%zu,\"passed\":%s,\"timing\":{\"status\":\"%s\",\"scope\":\"online_activation_pack_projection_gate_next_adaln_quant_submit_wait\",\"excluded\":[\"loading\",\"offline_weight_preparation\",\"shader_pipeline_compile_and_warmup\",\"correctness_readback\",\"residual_restore\"],\"warmups_per_variant\":%u,\"repeats_per_variant\":%u,\"paired_order\":\"alternates_native_first_fused_first\",\"first_measured_variant\":\"%s\",\"native\":",
        argv[5],rows,padded,seed,k,N,key,gate_slot,attention?"false":"true",nonfinite,cpu_original,cpu_fused,hidden_mismatch,codes_mismatch,scales_mismatch,zero_errors,padding_errors,pass?"true":"false",pass?"completed":"not_run_incorrect",warmups,repeats,((warmups+initial_order)&1u)?"fused":"native");
    print_timings(report,native_times,pass?repeats:0);
    fputs(",\"fused\":",report); print_timings(report,fused_times,pass?repeats:0);
    fprintf(report,"},\"independent_reference\":{\"rows\":[0,%u,%u],\"columns\":%d,\"full_K\":%u,\"arithmetic\":\"float64 decoded/original dot, BF16 branch, FP32 fmaf gate and BF16 residual, float64 RMS/AdaLN and BF16 boundary, native reciprocal-multiply RNE quantizer\",\"relative_l2_implementation_limit\":0.008,\"reference_failures\":%zu,\"passed\":%s,\"projection_implementation_vs_rounded_decoded\":",
        rows/2,rows-1,N,k,reference_failures,independent_pass?"true":"false");
    print_metric(report,&projection_implementation);
    fputs(",\"projection_quantization_vs_original_bf16\":",report); print_metric(report,&projection_quantization);
    fputs(",\"hidden_implementation\":",report); print_metric(report,&hidden_implementation);
    fputs(",\"next_quantized_implementation\":",report); print_metric(report,&next_quant_implementation);
    fputs(",\"hidden_quantization_vs_original_bf16_path\":",report); print_metric(report,&hidden_quantization);
    fputs(",\"next_quantization_vs_original_bf16_path\":",report); print_metric(report,&next_quantization);
    fputs(",\"per_row\":[",report);
    for(unsigned r=0;r<3;r++) {
        if(r) fputc(',',report);
        fprintf(report,"{\"row\":%u,\"projection_implementation\":",selected_rows[r]); print_metric(report,&row_metrics[r][0]);
        fputs(",\"projection_quantization\":",report); print_metric(report,&row_metrics[r][1]);
        fputs(",\"hidden_implementation\":",report); print_metric(report,&row_metrics[r][2]);
        fputs(",\"next_quantized_implementation\":",report); print_metric(report,&row_metrics[r][3]);
        fputc('}',report);
    }
    fputs("]}}\n",report);
    if(fclose(report)) die("report write failed");
    return pass?0:2;
}
