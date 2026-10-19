/* Independent CPU reference + synchronized native preparation comparison.
 * Caller owns the shared GPU lock. No projection or model speed is inferred.
 * Inputs exercise zero, epsilon-dominated, outlier, separate-map and tail rows.
 * Frozen reports are exclusive; CPU references/readback/reset are not timed. */
#include "h3_gpu.h"
#include "fused_reference.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

enum { K=5376, SLOTS=6, MODS=3, VARIANTS=3 };
static void die(const char *s) { fprintf(stderr,"prep_validate: %s\n",s); exit(1); }
static void *alloc(size_t n) { void *p=calloc(n,1); if(!p) die("allocation"); return p; }
static void ck(int ok,h3_gpu *g) { if(!ok) die(h3_gpu_error(g)); }
static float sample(uint32_t r,uint32_t c,uint32_t seed) {
    uint32_t x=(r+1u)*2654435761u^(c+7u)*2246822519u^seed;
    x^=x>>13; x*=3266489917u; x^=x>>16;
    return ((float)(x&65535u)/32768.0f-1.0f)*0.5f;
}
static double now(void) { struct timespec t; if(clock_gettime(CLOCK_MONOTONIC,&t)) die("clock"); return t.tv_sec+t.tv_nsec*1e-9; }
typedef struct { double sse,ssr,max; size_t n,bad; } metric;
static void add(metric *m,double a,double b) {
    m->n++; if(!isfinite(a)||!isfinite(b)) {m->bad++;return;}
    double d=a-b; m->sse+=d*d; m->ssr+=b*b; if(fabs(d)>m->max)m->max=fabs(d);
}
static double rel(metric m) { return m.bad?INFINITY:m.ssr?sqrt(m.sse/m.ssr):m.sse?INFINITY:0; }
static void print_metric(FILE *f,metric m) {
    fprintf(f,"{\"count\":%zu,\"nonfinite\":%zu,\"relative_l2\":",m.n,m.bad);
    if(isfinite(rel(m)))fprintf(f,"%.17g",rel(m));else fputs("null",f);
    fprintf(f,",\"rmse\":%.17g,\"max_abs\":%.17g}",sqrt(m.sse/m.n),m.max);
}
static void hadamard(double *x) {
    for(unsigned b=0;b<K;b+=256) {
        for(unsigned stride=1;stride<256;stride*=2)
            for(unsigned start=0;start<256;start+=2*stride)
                for(unsigned j=0;j<stride;j++) {
                    double a=x[b+start+j],v=x[b+start+j+stride];
                    x[b+start+j]=a+v; x[b+start+j+stride]=a-v;
                }
        for(unsigned j=0;j<256;j++)x[b+j]/=16.0;
    }
}
typedef struct {
    h3_gpu *g; h3_gpu_tensor *res[3],*q[3],*sc[3],*scratch[3];
    h3_gpu_tensor *branch,*gamma,*gm,*nm,*gmap,*nmap;
    unsigned rows,padded; int gate,distinct;
} context;
static void encode(context *c,unsigned v,int diagnostic) {
    if(v) ck(h3_gpu_adaln_h256_int8(c->g,c->res[v],c->q[v],c->sc[v],
        v==1||diagnostic?c->scratch[v]:NULL,c->res[v],c->branch,c->gamma,
        c->gm,c->nm,c->gmap,c->nmap,c->rows,c->padded,K,SLOTS,2,3,4,1e-5f,c->gate,v==1),c->g);
    else if(c->gate&&!c->distinct)
        ck(h3_gpu_gate_adaln_quantize_int8(c->g,c->res[0],c->q[0],c->sc[0],
            c->res[0],c->branch,c->gamma,c->gm,c->nm,c->gmap,c->rows,c->padded,K,SLOTS,2,3,4,1e-5f),c->g);
    else {
        if(c->gate)ck(h3_gpu_gate_bf16(c->g,c->res[0],c->res[0],c->branch,c->gm,c->gmap,c->rows,K,SLOTS,2),c->g);
        ck(h3_gpu_adaln_quantize_int8_pre_gated(c->g,c->q[0],c->sc[0],c->res[0],c->gamma,
            c->nm,c->nmap,c->rows,c->padded,K,SLOTS,3,4,1e-5f),c->g);
    }
}
typedef struct { double wall,gpu; uint64_t dispatches; } timing;
static timing measure(context *c,unsigned v,const uint16_t *res) {
    ck(h3_gpu_tensor_write_bf16(c->res[v],res,(size_t)c->rows*K),c->g);
    h3_gpu_stats a,b; ck(h3_gpu_get_stats(c->g,&a),c->g);
    double t=now(); ck(h3_gpu_begin(c->g),c->g); encode(c,v,0); ck(h3_gpu_submit(c->g),c->g);
    timing z={now()-t,0,0}; ck(h3_gpu_get_stats(c->g,&b),c->g);
    z.gpu=b.gpu_seconds-a.gpu_seconds;z.dispatches=b.direct_dispatches-a.direct_dispatches;
    return z;
}
int main(int argc,char **argv) {
    if(argc!=9)die("SHADER ROWS SEED GATE DISTINCT_MAPS NEW_REPORT WARMUPS REPEATS");
    unsigned rows=(unsigned)strtoul(argv[2],NULL,10),seed=(unsigned)strtoul(argv[3],NULL,10);
    int gate=atoi(argv[4]),distinct=atoi(argv[5]);
    unsigned warm=atoi(argv[7]),repeats=atoi(argv[8]),padded=(rows+127u)&~127u;
    if(rows<17||rows>18000||(gate!=0&&gate!=1)||(distinct!=0&&distinct!=1)||warm<3||repeats<3||repeats>1000)die("invalid recipe");
    int fd=open(argv[6],O_CREAT|O_EXCL|O_WRONLY,0600);if(fd<0)die("new report required");
    FILE *out=fdopen(fd,"w");if(!out)die("report");
    size_t n=(size_t)rows*K,pn=(size_t)padded*K,mn=MODS*SLOTS*K;
    uint16_t *res=alloc(n*2),*branch=alloc(n*2),*gamma=alloc(K*2),*gm=alloc(mn*2),*nm=alloc(mn*2);
    uint32_t *gmap=alloc(rows*4),*nmap=alloc(rows*4);
    for(unsigned r=0;r<rows;r++) {
        gmap[r]=r%MODS;nmap[r]=distinct?(2*r)%MODS:gmap[r];
        for(unsigned k=0;k<K;k++) {
            float x=r==0?0:sample(r,k,seed),b=r==0?0:sample(r,k,seed+1);
            if(r==1){x*=1e-5f;b*=1e-5f;}
            if(r==rows-1&&k%257==0){x*=128;b*=64;}
            res[(size_t)r*K+k]=ref_bf16(x);branch[(size_t)r*K+k]=ref_bf16(b);
        }
    }
    for(unsigned k=0;k<K;k++)gamma[k]=ref_bf16(0.75f+sample(0,k,seed+2)*0.25f);
    for(unsigned r=0;r<MODS;r++)for(unsigned s=0;s<SLOTS;s++)for(unsigned k=0;k<K;k++) {
        gm[((size_t)r*SLOTS+s)*K+k]=ref_bf16(k%19?sample(r+s,k,seed+3):0);
        nm[((size_t)r*SLOTS+s)*K+k]=ref_bf16((s==3&&(r==0||r==1))?0:sample(r+s,k,seed+4));
    }
    char error[1024];h3_gpu *g=h3_gpu_create(argv[1],error,sizeof(error));if(!g)die(error);
    if(!h3_gpu_has_int8_mlp(g))die("TensorOps unavailable");
    context c={.g=g,.rows=rows,.padded=padded,.gate=gate,.distinct=distinct};
    c.branch=h3_gpu_tensor_from_bf16(g,branch,n);c.gamma=h3_gpu_tensor_from_bf16(g,gamma,K);
    c.gm=h3_gpu_tensor_from_bf16(g,gm,mn);c.nm=h3_gpu_tensor_from_bf16(g,nm,mn);
    c.gmap=h3_gpu_tensor_from_u32(g,gmap,rows);c.nmap=h3_gpu_tensor_from_u32(g,nmap,rows);
    int8_t *codes[3];float *scales[3];uint16_t *hidden[3],*adaln[3];
    size_t nonfinite=0,gate_errors=0,padding_errors=0,zero_errors=0,code_mismatch=0,scale_mismatch=0,adaln_mismatch=0;
    for(unsigned v=0;v<3;v++) {
        c.res[v]=h3_gpu_tensor_from_bf16(g,res,n);c.q[v]=h3_gpu_tensor_new_i8(g,pn);c.sc[v]=h3_gpu_tensor_new_f32(g,padded);
        c.scratch[v]=h3_gpu_tensor_new_bf16(g,n);
        codes[v]=alloc(pn);scales[v]=alloc(padded*4);hidden[v]=alloc(n*2);adaln[v]=alloc(n*2);
        memset(codes[v],v?0x80:0x7f,pn);for(unsigned r=0;r<padded;r++)scales[v][r]=NAN;
        for(size_t j=0;j<n;j++)adaln[v][j]=0x7fc0;
        ck(h3_gpu_tensor_write_i8(c.q[v],codes[v],pn)&&h3_gpu_tensor_write_f32(c.sc[v],scales[v],padded)&&h3_gpu_tensor_write_bf16(c.scratch[v],adaln[v],n),g);
        ck(h3_gpu_begin(g),g);encode(&c,v,1);ck(h3_gpu_submit(g),g);
        ck(h3_gpu_tensor_read_i8(c.q[v],codes[v],pn)&&h3_gpu_tensor_read_f32(c.sc[v],scales[v],padded)&&h3_gpu_tensor_read_bf16(c.res[v],hidden[v],n),g);
        if(v)ck(h3_gpu_tensor_read_bf16(c.scratch[v],adaln[v],n),g);
        for(size_t j=0;j<n;j++) {
            unsigned r=j/K,k=j%K;
            uint16_t expected=gate?ref_bf16(fmaf(ref_f32(branch[j]),ref_f32(gm[((size_t)gmap[r]*SLOTS+2)*K+k]),ref_f32(res[j]))):res[j];
            gate_errors+=hidden[v][j]!=expected;nonfinite+=!isfinite(ref_f32(hidden[v][j]));
            if(v)nonfinite+=!isfinite(ref_f32(adaln[v][j]));
        }
        for(unsigned r=0;r<padded;r++)nonfinite+=!isfinite(scales[v][r])||scales[v][r]<=0;
        for(size_t j=n;j<pn;j++)padding_errors+=codes[v][j]!=0;
        for(unsigned r=rows;r<padded;r++)padding_errors+=scales[v][r]!=1.0f;
        for(unsigned k=0;k<K;k++)zero_errors+=codes[v][k]!=0;
        if(v)zero_errors+=scales[v][0]!=1.0f;
    }
    for(size_t j=0;j<pn;j++)code_mismatch+=codes[1][j]!=codes[2][j];
    for(unsigned r=0;r<padded;r++)scale_mismatch+=memcmp(&scales[1][r],&scales[2][r],4)!=0;
    for(size_t j=0;j<n;j++)adaln_mismatch+=adaln[1][j]!=adaln[2][j];
    unsigned selected[]={0,1,rows/2,rows-1};metric impl[2]={0},norm_impl[2]={0},quant[3]={0};
    metric row_impl[4][2]={0};
    for(unsigned ri=0;ri<4;ri++) {
        unsigned r=selected[ri];double x[K],rotated[K],sum=0,maximum=0;
        for(unsigned k=0;k<K;k++) {
            size_t j=(size_t)r*K+k;
            uint16_t h=gate?ref_bf16(fmaf(ref_f32(branch[j]),ref_f32(gm[((size_t)gmap[r]*SLOTS+2)*K+k]),ref_f32(res[j]))):res[j];
            x[k]=ref_f32(h);sum+=x[k]*x[k];
        }
        double inv=1.0/sqrt(sum/K+(double)1e-5f);
        for(unsigned k=0;k<K;k++) {
            size_t b=((size_t)nmap[r]*SLOTS)*K+k;
            x[k]=ref_f32(ref_bf16((float)(x[k]*inv*ref_f32(gamma[k])*(1.0+ref_f32(nm[b+4*K]))+ref_f32(nm[b+3*K]))));
            rotated[k]=x[k];
            for(unsigned v=1;v<3;v++)add(&norm_impl[v-1],ref_f32(adaln[v][(size_t)r*K+k]),x[k]);
        }
        hadamard(rotated);for(unsigned k=0;k<K;k++)if(fabs(rotated[k])>maximum)maximum=fabs(rotated[k]);
        float scale=maximum?(float)maximum/127.0f:1.0f;
        for(unsigned v=0;v<3;v++) {
            double decoded[K];
            for(unsigned k=0;k<K;k++) {
                decoded[k]=(double)codes[v][(size_t)r*K+k]*scales[v][r];
                if(v) {
                    float q=nearbyintf((float)rotated[k]/scale);if(q>127)q=127;if(q< -127)q=-127;
                    add(&impl[v-1],decoded[k],(double)q*scale);add(&row_impl[ri][v-1],decoded[k],(double)q*scale);
                }
            }
            if(v)hadamard(decoded);
            for(unsigned k=0;k<K;k++)add(&quant[v],decoded[k],x[k]);
        }
    }
    int passed=!nonfinite&&!gate_errors&&!padding_errors&&!zero_errors&&!code_mismatch&&!scale_mismatch&&!adaln_mismatch;
    for(unsigned v=0;v<2;v++){passed&=rel(impl[v])<=0.008&&rel(norm_impl[v])<=0.008;for(unsigned r=0;r<4;r++)passed&=rel(row_impl[r][v])<=0.008;}
    timing *times[3];for(unsigned v=0;v<3;v++)times[v]=alloc(repeats*sizeof(timing));
    if(passed)for(unsigned i=0;i<warm+repeats;i++)for(unsigned j=0;j<3;j++) {
        unsigned v=(i+((seed>>2)&1)+j)%3;timing t=measure(&c,v,res);if(i>=warm)times[v][i-warm]=t;
    }
    fprintf(out,"{\"schema\":\"h3_prep_validation_v1\",\"rows\":%u,\"padded_rows\":%u,\"K\":5376,\"seed\":%u,\"gate\":%d,\"distinct_maps\":%d,\"passed\":%s,\"nonfinite\":%zu,\"gate_errors\":%zu,\"padding_errors\":%zu,\"zero_errors\":%zu,\"staged_fused_code_mismatches\":%zu,\"staged_fused_scale_mismatches\":%zu,\"staged_fused_adaln_mismatches\":%zu,\"reference\":{\"full_width\":5376,\"implementation_limit\":0.008,\"rows\":[0,1,%u,%u],\"variants\":[",rows,padded,seed,gate,distinct,passed?"true":"false",nonfinite,gate_errors,padding_errors,zero_errors,code_mismatch,scale_mismatch,adaln_mismatch,rows/2,rows-1);
    for(unsigned v=0;v<3;v++) {
        if(v)fputc(',',out);fprintf(out,"{\"name\":\"%s\",\"quantization_vs_original_adaln\":",v==0?"native":v==1?"staged":"fused");print_metric(out,quant[v]);
        if(v){fputs(",\"implementation_vs_cpu_quantized\":",out);print_metric(out,impl[v-1]);fputs(",\"adaln_implementation_vs_fp64\":",out);print_metric(out,norm_impl[v-1]);
            fputs(",\"per_row_implementation\":[",out);for(unsigned r=0;r<4;r++){if(r)fputc(',',out);print_metric(out,row_impl[r][v-1]);}fputc(']',out);}
        fputc('}',out);
    }
    fprintf(out,"]},\"timing\":{\"status\":\"%s\",\"scope\":\"preparation_only_synchronized_online_including_padding\",\"warmups\":%u,\"repeats\":%u,\"rotation_seed_offset\":%u,\"variants\":[",passed?"completed":"not_measured",warm,repeats,(seed>>2)&1);
    for(unsigned v=0;v<3;v++){if(v)fputc(',',out);fprintf(out,"{\"name\":\"%s\",\"samples\":[",v==0?"native":v==1?"staged":"fused");
        if(passed)for(unsigned i=0;i<repeats;i++){if(i)fputc(',',out);fprintf(out,"{\"wall_seconds\":%.12g,\"supplemental_gpu_seconds\":%.12g,\"direct_dispatches\":%llu}",times[v][i].wall,times[v][i].gpu,(unsigned long long)times[v][i].dispatches);}fputs("]}",out);}
    fputs("]}}\n",out);if(fclose(out))die("close report");
    return passed?0:2;
}
