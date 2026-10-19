/* MIT. Real block-0 grouped QKV versus the independent TM64 native candidate.
 * One process, common packed operands; full-K/complete-head CPU float64 refs.
 * This executable never owns scheduling: its caller must hold the shared GPU lock.
 * Offline W packing and correctness/readback are excluded from warmed intervals. */
#include "h3_gpu.h"
#include "h3_weights.h"
#include "qkv_reference.h"
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

enum { WIDTH=5376, HEADS=56, DIM=128, INNER=HEADS*DIM, OUTPUTS=INNER*3, ROTARY=48, SAMPLES=4 };
static FILE *report;
static void die(const char *why) {
    fprintf(stderr,"qkv_validate: %s\n",why);
    if(report) { fputs("{\"schema\":\"h3_native_qkv_v1\",\"status\":\"failed\",\"reason\":\"native driver failed; see stderr\"}\n",report); fclose(report); }
    exit(1);
}
static void *allocate(size_t bytes) { void *p=calloc(bytes,1); if(!p) die("allocation failed"); return p; }
static void check(int ok,h3_gpu *g) { if(!ok) die(h3_gpu_error(g)); }
static double now(void) { struct timespec ts; if(clock_gettime(CLOCK_MONOTONIC,&ts)) die("clock failed"); return (double)ts.tv_sec+1e-9*(double)ts.tv_nsec; }
static float sample(uint32_t row,uint32_t col,uint32_t seed) {
    uint32_t v=(row+1u)*2654435761u^(col+7u)*2246822519u^seed;
    v^=v>>13;v*=3266489917u;v^=v>>16;
    return ((float)(v&65535u)/32768.0f-1.0f)*0.5f;
}
static unsigned argument(const char *value,unsigned low,unsigned high) {
    char *end; unsigned long x=strtoul(value,&end,10);
    if(!*value||*end||x<low||x>high) die("invalid integer argument"); return (unsigned)x;
}
typedef struct { double error2,reference2,max_abs; size_t count,nonfinite; } metric;
static void add(metric *m,double got,double ref) {
    m->count++; if(!isfinite(got)||!isfinite(ref)) { m->nonfinite++;return; }
    double d=got-ref;m->error2+=d*d;m->reference2+=ref*ref;if(fabs(d)>m->max_abs)m->max_abs=fabs(d);
}
static double relative(const metric *m) {
    if(m->nonfinite||!m->count)return INFINITY;
    return m->reference2>0?sqrt(m->error2/m->reference2):m->error2==0?0:INFINITY;
}
static void print_metric(const metric *m) {
    double r=relative(m);
    fprintf(report,"{\"count\":%zu,\"nonfinite\":%zu,\"relative_l2\":",m->count,m->nonfinite);
    if(isfinite(r))fprintf(report,"%.17g",r);else fputs("null",report);
    fprintf(report,",\"rmse\":%.17g,\"max_abs\":%.17g}",m->count?sqrt(m->error2/(double)m->count):0,m->max_abs);
}
typedef struct {
    h3_gpu *g; h3_gpu_tensor *x,*aq,*as,*wq,*ws,*gamma[2],*cosine,*sine,*output[2][3];
    uint32_t rows,padded; int hadamard;
} context;
static void encode(context *p,int candidate,int online) {
    int quantized=!online;
    if(online && p->hadamard) {
        check(h3_gpu_quantize_h256_int8(p->g,p->aq,p->as,p->x,p->rows,p->padded,WIDTH),p->g);
        quantized=1;
    }
    if(candidate)check(h3_gpu_m5_grouped_qkv_linear_rope_int8(p->g,p->output[1][0],p->output[1][1],p->output[1][2],
        p->aq,p->as,p->x,p->wq,p->ws,p->gamma[0],p->gamma[1],p->cosine,p->sine,p->rows,WIDTH,HEADS,DIM,ROTARY,1e-5f,quantized),p->g);
    else check(h3_gpu_grouped_qkv_linear_rope_int8(p->g,p->output[0][0],p->output[0][1],p->output[0][2],
        p->aq,p->as,p->x,p->wq,p->ws,p->gamma[0],p->gamma[1],p->cosine,p->sine,p->rows,WIDTH,HEADS,DIM,ROTARY,1e-5f,quantized,0,0,0),p->g);
}
typedef struct { double wall,encode,wait,gpu; uint64_t dispatches; } timing;
static timing measure(context *p,int candidate,int online) {
    h3_gpu_stats before,after;check(h3_gpu_get_stats(p->g,&before),p->g);
    double begin=now();check(h3_gpu_begin(p->g),p->g);encode(p,candidate,online);double encoded=now();
    check(h3_gpu_submit(p->g),p->g);double finish=now();check(h3_gpu_get_stats(p->g,&after),p->g);
    timing t={finish-begin,encoded-begin,finish-encoded,after.gpu_seconds-before.gpu_seconds,after.direct_dispatches-before.direct_dispatches};
    if(!isfinite(t.wall)||t.wall<=0||!isfinite(t.gpu)||t.gpu<0||t.dispatches!=(uint64_t)(online?2:1))die("invalid synchronized timing/dispatch count");
    return t;
}
static void print_times(const timing *t,unsigned n) {
    fputc('[',report);for(unsigned i=0;i<n;i++){if(i)fputc(',',report);
        fprintf(report,"{\"wall_seconds\":%.12g,\"cpu_encode_seconds\":%.12g,\"submit_wait_seconds\":%.12g,\"supplemental_gpu_seconds\":%.12g,\"direct_dispatches\":%" PRIu64 "}",t[i].wall,t[i].encode,t[i].wait,t[i].gpu,t[i].dispatches);}
    fputc(']',report);
}
/* Independent row packing screen for the actual BF16 values. The native zero
 * row is Q=0,S=1/127; padded rows are Q=0,S=1. Do not import an alternate policy. */
static size_t packing_errors(const uint16_t *x,const int8_t *q,float scale,unsigned k,int hadamard) {
    float values[WIDTH];for(unsigned j=0;j<k;j++)values[j]=qvr_f32(x[j]);if(hadamard)qvr_h256(values,k);
    float peak=0; for(unsigned j=0;j<k;j++)if(fabsf(values[j])>peak)peak=fabsf(values[j]);
    float want_scale=peak>0?peak/127.0f:hadamard?1.0f:1.0f/127.0f,inverse=peak>0?127.0f/peak:127.0f;
    size_t errors=memcmp(&want_scale,&scale,sizeof(float))!=0;
    for(unsigned j=0;j<k;j++) { float v=nearbyintf(hadamard?values[j]/want_scale:values[j]*inverse); if(v>127)v=127;if(v< -127)v=-127;errors+=(int8_t)v!=q[j]; }
    return errors;
}
int main(int argc,char **argv) {
    if(argc!=7&&argc!=9)die("usage: qkv_validate MODEL_DIT_DIRECTORY SHADER ROWS SEED POLICY NEW_REPORT.json [WARMUPS REPEATS]; POLICY=native|h256");
    unsigned rows=argument(argv[3],1,32768),seed=argument(argv[4],0,UINT32_MAX);
    unsigned warmups=argc==9?argument(argv[7],3,100):3,repeats=argc==9?argument(argv[8],3,1000):10;
    int fd=open(argv[6],O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)die("report must be new");report=fdopen(fd,"w");if(!report)die("report open failed");
    if(rows<128||(strcmp(argv[5],"native")&&strcmp(argv[5],"h256"))) {
        fprintf(report,"{\"schema\":\"h3_native_qkv_v1\",\"status\":\"unsupported\",\"rows\":%u,\"seed\":%u,\"reason\":\"native API requires M>=128; operand policy must be native or h256\"}\n",rows,seed);
        fclose(report);report=NULL;return 2;
    }
    const char *overrides[]={"H3_DISABLE_FUSED_INT8_QKV_ROPE","H3_DISABLE_QKV_LOCAL_SCALES","H3_DISABLE_QKV_FULL_K","H3_DISABLE_VECTOR_QKV_RMS","H3_DISABLE_VECTOR_QKV_ROPE"};
    for(unsigned i=0;i<sizeof(overrides)/sizeof(overrides[0]);i++)if(getenv(overrides[i]))die("inherited native QKV override is forbidden");
    int hadamard=strcmp(argv[5],"h256")==0;
    uint32_t padded=(rows+127u)&~127u;size_t output_elements=(size_t)rows*INNER;
    uint16_t *input=allocate((size_t)rows*WIDTH*2),*cosine=allocate((size_t)rows*ROTARY*2),*sine=allocate((size_t)rows*ROTARY*2);
    for(unsigned r=0;r<rows;r++) {
        for(unsigned j=0;j<WIDTH;j++) {
            float v=r==0?0:sample(r,j,seed);if(r==1)v*=1e-8f;if(r==rows-1&&j%257u==0)v*=32;
            input[(size_t)r*WIDTH+j]=qvr_bf16(v);
        }
        for(unsigned d=0;d<ROTARY;d++) {
            float angle=sample(r,d,seed+3u)*3.0f;
            cosine[(size_t)r*ROTARY+d]=qvr_bf16(cosf(angle));sine[(size_t)r*ROTARY+d]=qvr_bf16(sinf(angle));
        }
    }
    char error[1024];h3_gpu *g=h3_gpu_create(argv[2],error,sizeof(error));if(!g)die(error);if(!h3_gpu_has_int8_mlp(g))die("TensorOps unavailable; no fallback");
    h3_weight_store *store=h3_weight_store_open(argv[1],error,sizeof(error));if(!store)die(error);
    uint64_t shape[]={OUTPUTS,WIDTH},norm_shape[]={DIM};
    h3_gpu_tensor *w=h3_weight_load_bf16(store,g,"blocks.0.attn.qkv_proj.weight",2,shape,error,sizeof(error));if(!w)die(error);
    context p={0};p.g=g;p.rows=rows;p.padded=padded;p.hadamard=hadamard;
    p.gamma[0]=h3_weight_load_bf16(store,g,"blocks.0.attn.q_norm.weight",1,norm_shape,error,sizeof(error));if(!p.gamma[0])die(error);
    p.gamma[1]=h3_weight_load_bf16(store,g,"blocks.0.attn.k_norm.weight",1,norm_shape,error,sizeof(error));if(!p.gamma[1])die(error);
    p.x=h3_gpu_tensor_from_bf16(g,input,(size_t)rows*WIDTH);p.aq=h3_gpu_tensor_new_i8(g,(size_t)padded*WIDTH);p.as=h3_gpu_tensor_new_f32(g,padded);
    p.wq=h3_gpu_tensor_new_i8(g,(size_t)OUTPUTS*WIDTH);p.ws=h3_gpu_tensor_new_f32(g,OUTPUTS);
    p.cosine=h3_gpu_tensor_from_bf16(g,cosine,(size_t)rows*ROTARY);p.sine=h3_gpu_tensor_from_bf16(g,sine,(size_t)rows*ROTARY);
    if(!p.x||!p.aq||!p.as||!p.wq||!p.ws||!p.cosine||!p.sine)die(h3_gpu_error(g));
    uint16_t *poison=allocate(output_elements*2);for(size_t i=0;i<output_elements;i++)poison[i]=0x7fc0;
    for(unsigned path=0;path<2;path++)for(unsigned stream=0;stream<3;stream++){
        p.output[path][stream]=h3_gpu_tensor_from_bf16(g,poison,output_elements);if(!p.output[path][stream])die(h3_gpu_error(g));}
    free(poison);
    double prep_start=now();check(h3_gpu_begin(g),g);if(hadamard)check(h3_gpu_quantize_h256_int8(g,p.wq,p.ws,w,OUTPUTS,OUTPUTS,WIDTH),g);else check(h3_gpu_quantize_weight_int8(g,p.wq,p.ws,w,OUTPUTS,WIDTH),g);check(h3_gpu_submit(g),g);double offline=now()-prep_start;
    check(h3_gpu_begin(g),g);encode(&p,0,1);check(h3_gpu_submit(g),g);
    check(h3_gpu_begin(g),g);encode(&p,1,0);check(h3_gpu_submit(g),g);
    uint16_t *outputs[2][3],*original_w=allocate((size_t)OUTPUTS*WIDTH*2),gamma[2][DIM];
    int8_t *aq=allocate((size_t)padded*WIDTH),*wq=allocate((size_t)OUTPUTS*WIDTH);float *as=allocate((size_t)padded*4),*ws=allocate((size_t)OUTPUTS*4);
    check(h3_gpu_tensor_read_bf16(w,original_w,(size_t)OUTPUTS*WIDTH)&&h3_gpu_tensor_read_i8(p.wq,wq,(size_t)OUTPUTS*WIDTH)&&h3_gpu_tensor_read_f32(p.ws,ws,OUTPUTS)&&
          h3_gpu_tensor_read_i8(p.aq,aq,(size_t)padded*WIDTH)&&h3_gpu_tensor_read_f32(p.as,as,padded),g);
    for(unsigned i=0;i<2;i++)check(h3_gpu_tensor_read_bf16(p.gamma[i],gamma[i],DIM),g);
    size_t nonfinite[2][3]={{0}},bit_mismatch[3]={0},zero_errors=0;metric pair[3]={{0}};
    for(unsigned path=0;path<2;path++)for(unsigned stream=0;stream<3;stream++){
        outputs[path][stream]=allocate(output_elements*2);check(h3_gpu_tensor_read_bf16(p.output[path][stream],outputs[path][stream],output_elements),g);
        for(size_t i=0;i<output_elements;i++)nonfinite[path][stream]+=!isfinite(qvr_f32(outputs[path][stream][i]));
        for(unsigned h=0;h<HEADS;h++)for(unsigned d=0;d<DIM;d++)zero_errors+=qvr_f32(outputs[path][stream][(size_t)h*rows*DIM+d])!=0;
    }
    for(unsigned s=0;s<3;s++)for(size_t i=0;i<output_elements;i++){
        bit_mismatch[s]+=outputs[0][s][i]!=outputs[1][s][i];add(&pair[s],qvr_f32(outputs[1][s][i]),qvr_f32(outputs[0][s][i]));}
    unsigned selected[SAMPLES]={0,1,rows/2,rows-1},heads[]={0,HEADS/2,HEADS-1};
    metric impl[2][SAMPLES][3]={{{{0}}}},total[2][SAMPLES][3]={{{{0}}}},quant[SAMPLES][3]={{{0}}};
    size_t pack_input=0,pack_weight=0,padding_errors=0,epsilon_errors=0,nonfinite_scales=0;
    for(unsigned r=0;r<padded;r++)nonfinite_scales+=!isfinite(as[r])||as[r]<=0;
    for(unsigned c=0;c<OUTPUTS;c++)nonfinite_scales+=!isfinite(ws[c])||ws[c]<=0;
    for(unsigned i=0;i<SAMPLES;i++) {unsigned r=selected[i];pack_input+=packing_errors(input+(size_t)r*WIDTH,aq+(size_t)r*WIDTH,as[r],WIDTH,hadamard);}
    for(unsigned r=rows;r<padded;r++){padding_errors+=as[r]!=1;for(unsigned j=0;j<WIDTH;j++)padding_errors+=aq[(size_t)r*WIDTH+j]!=0;}
    for(unsigned hi=0;hi<3;hi++)for(unsigned s=0;s<3;s++)for(unsigned d=0;d<DIM;d++){
        unsigned c=(heads[hi]*3+s)*DIM+d;pack_weight+=packing_errors(original_w+(size_t)c*WIDTH,wq+(size_t)c*WIDTH,ws[c],WIDTH,hadamard);}
    for(unsigned i=0;i<SAMPLES;i++)for(unsigned hi=0;hi<3;hi++)for(unsigned s=0;s<3;s++) {
        unsigned r=selected[i],h=heads[hi],column=(h*3+s)*DIM;double original[DIM],decoded[DIM],oref[DIM],dref[DIM];
        qvr_project(input+(size_t)r*WIDTH,original_w+(size_t)column*WIDTH,aq+(size_t)r*WIDTH,wq+(size_t)column*WIDTH,as[r],ws+column,WIDTH,original,decoded);
        const uint16_t *gg=gamma[s==1?1:0];qvr_postprocess(original,s,gg,cosine+(size_t)r*ROTARY,sine+(size_t)r*ROTARY,(double)1e-5f,oref);
        qvr_postprocess(decoded,s,gg,cosine+(size_t)r*ROTARY,sine+(size_t)r*ROTARY,(double)1e-5f,dref);
        if(i==1&&s<2){double square=0;for(unsigned d=0;d<DIM;d++)square+=decoded[d]*decoded[d];epsilon_errors+=!(square>0&&square/DIM<1e-5);}
        for(unsigned d=0;d<DIM;d++) {size_t offset=((size_t)h*rows+r)*DIM+d;add(&quant[i][s],dref[d],oref[d]);
            for(unsigned path=0;path<2;path++){double y=qvr_f32(outputs[path][s][offset]);add(&impl[path][i][s],y,dref[d]);add(&total[path][i][s],y,oref[d]);}}
    }
    int passed=!nonfinite_scales&&!pack_input&&!pack_weight&&!padding_errors&&!epsilon_errors&&!zero_errors&&!bit_mismatch[2];
    for(unsigned path=0;path<2;path++)for(unsigned s=0;s<3;s++){
        passed=passed&&!nonfinite[path][s];for(unsigned i=0;i<SAMPLES;i++)passed=passed&&relative(&impl[path][i][s])<=0.008;}
    timing *times[2][2];for(unsigned stage=0;stage<2;stage++)for(unsigned path=0;path<2;path++)times[stage][path]=allocate((size_t)repeats*sizeof(timing));
    if(passed)for(unsigned stage=0;stage<2;stage++){
        for(unsigned i=0;i<warmups;i++){(void)measure(&p,0,(int)stage);(void)measure(&p,1,(int)stage);}
        for(unsigned i=0;i<repeats;i++){unsigned first=(i+(seed==29))%2;times[stage][first][i]=measure(&p,(int)first,(int)stage);times[stage][1-first][i]=measure(&p,(int)(1-first),(int)stage);}
    }
    fprintf(report,"{\"schema\":\"h3_native_qkv_v1\",\"status\":\"%s\",\"seed\":%u,\"operand_policy\":\"%s\",\"shape\":{\"rows\":%u,\"padded_rows\":%u,\"input_dim\":5376,\"heads\":56,\"head_dim\":128,\"rope_half\":48},\"baseline_shader\":\"%s\",\"candidate_shader\":\"h3_m5_qkv_rope_int8_r64_swizzle8\",\"candidate_tile\":[64,128,4,8],\"baseline_slow_flags\":[0,0,0],\"baseline_selection_basis\":\"native source default, runtime overrides rejected\",\"offline_weight_quantization_seconds\":%.12g,\"warmups\":%u,\"repeats\":%u,\"first_candidate_parity\":%u,\"implementation_relative_l2_limit\":0.008,\"packing\":{\"sampled_input_rows\":4,\"sampled_weight_rows\":1152,\"input_errors\":%zu,\"weight_errors\":%zu,\"padding_errors\":%zu,\"nonfinite_scales\":%zu},\"zero_output_errors\":%zu,\"epsilon_dominated_reference_errors\":%zu,\"paired\":[",
        passed?"completed":"incorrect",seed,hadamard?"h256-int8":"native-int8",rows,padded,rows>2048?"h3_qkv_project_split_int8_rope_local_scales_nax_r128_k5376_morton4":"h3_qkv_project_split_int8_rope_nax_r128_k5376_morton4",offline,warmups,repeats,seed==29,pack_input,pack_weight,padding_errors,nonfinite_scales,zero_errors,epsilon_errors);
    for(unsigned s=0;s<3;s++){if(s)fputc(',',report);fprintf(report,"{\"stream\":%u,\"bit_mismatches\":%zu,\"native_nonfinite\":%zu,\"candidate_nonfinite\":%zu,\"metric\":",s,bit_mismatch[s],nonfinite[0][s],nonfinite[1][s]);print_metric(&pair[s]);fputc('}',report);}
    fputs("],\"reference\":[",report);const char *kind[]={"zeros","epsilon_dominated","normal","outliers_tail"};
    for(unsigned i=0;i<SAMPLES;i++){if(i)fputc(',',report);fprintf(report,"{\"row\":%u,\"kind\":\"%s\",\"heads\":[0,28,55],\"streams\":[",selected[i],kind[i]);
        for(unsigned s=0;s<3;s++){if(s)fputc(',',report);fprintf(report,"{\"stream\":%u,\"native_implementation\":",s);print_metric(&impl[0][i][s]);fputs(",\"candidate_implementation\":",report);print_metric(&impl[1][i][s]);fputs(",\"native_total_error\":",report);print_metric(&total[0][i][s]);fputs(",\"candidate_total_error\":",report);print_metric(&total[1][i][s]);fputs(",\"quantization_error\":",report);print_metric(&quant[i][s]);fputc('}',report);}fputs("]}",report);}
    fputs("],\"timings\":{",report);for(unsigned stage=0;stage<2;stage++){if(stage)fputc(',',report);fprintf(report,"\"%s\":{\"native\":",stage?"online_activation_plus_qkv":"prepacked_qkv");print_times(times[stage][0],passed?repeats:0);fputs(",\"candidate\":",report);print_times(times[stage][1],passed?repeats:0);fputc('}',report);}
    fputs("},\"timing_scope\":\"synchronized host command begin/encode/submit/wait; all Q/K/V head-major outputs; online adds policy activation packing and padding; input is already AdaLN BF16, so AdaLN itself is excluded; excludes weights, references and readback; GPU duration supplemental\",\"quality_decision\":\"not_evaluated\"}\n",report);
    fclose(report);report=NULL;return passed?0:3;
}
