/* Offline prepared DiT weight export. Link against the pinned H256 runtime.
 * The caller owns the repository GPU lock and verifies the original model.
 * Every output is exclusive; a partial/failed export never becomes a manifest.
 */
#include "h3_gpu.h"
#include "h3_weights.h"
#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static void die(const char *s) { fprintf(stderr,"prepared_export: %s\n",s); exit(1); }
static double seconds(void) { struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec*1e-9; }
static void *allocate(size_t bytes) { void *p=malloc(bytes);if(!p)die("out of memory");return p; }
static void sha(const void *data,size_t size,char hex[65]) {
    if(size>UINT32_MAX)die("SHA256 input too large");
    unsigned char digest[32];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    if(!CC_SHA256(data,(CC_LONG)size,digest))die("SHA256 failed");
#pragma clang diagnostic pop
    static const char digits[]="0123456789abcdef";
    for(unsigned i=0;i<32;++i){hex[i*2]=digits[digest[i]>>4];hex[i*2+1]=digits[digest[i]&15];}hex[64]=0;
}
static void save(const char *directory,const char *name,const void *data,size_t bytes) {
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s",directory,name);
    if(n<0||(size_t)n>=sizeof(path))die("path too long");
    int fd=open(path,O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)die("exclusive output open failed");
    FILE *f=fdopen(fd,"wb");if(!f)die("fdopen failed");
    if(fwrite(data,1,bytes,f)!=bytes||fclose(f))die("write failed");
}
int main(int argc,char **argv) {
    if(argc!=4 && argc!=6)die("usage: prepared-export COMPONENT_DIR NEW_OUTPUT_DIR SHADER [FIRST_BLOCK COUNT]");
    unsigned first=0,count=50;
    if(argc==6){char *e=NULL;unsigned long a=strtoul(argv[4],&e,10);if(!e||*e||a>=50)die("bad first block");first=(unsigned)a;a=strtoul(argv[5],&e,10);if(!e||*e||!a||a>50-first)die("bad block count");count=(unsigned)a;}
    char error[1024];
    double init_start=seconds();
    h3_gpu *g=h3_gpu_create(argv[3],error,sizeof(error));if(!g)die(error);
    if(!h3_gpu_has_int8_mlp(g))die("M5 TensorOps required; no fallback");
    h3_weight_store *store=h3_weight_store_open(argv[1],error,sizeof(error));if(!store)die(error);
    double initialize_seconds=seconds()-init_start;
    const char *families[]={"qkv","out","fc1","fc2"};
    const char *suffix[]={"attn.qkv_proj.weight","attn.out_proj.weight","mlp.fc1.weight","mlp.fc2.weight"};
    const unsigned rows[]={21504,5376,28672,5376},columns[]={5376,7168,5376,14336};
    double total_start=seconds();
    for(unsigned block=first;block<first+count;++block)for(unsigned f=0;f<4;++f){
        char tensor[128],stem[64],file[128];
        snprintf(tensor,sizeof(tensor),"blocks.%u.%s",block,suffix[f]);
        snprintf(stem,sizeof(stem),"block%02u-%s",block,families[f]);
        size_t elements=(size_t)rows[f]*columns[f];uint64_t shape[]={rows[f],columns[f]};
        double start=seconds();
        h3_gpu_tensor *w=h3_weight_load_bf16(store,g,tensor,2,shape,error,sizeof(error));if(!w)die(error);
        double load_seconds=seconds()-start;
        uint16_t *original=allocate(elements*2);
        char original_sha[65],codes_sha[65],scales_sha[65];
        start=seconds();
        if(!h3_gpu_tensor_read_bf16(w,original,elements))die("original weight readback failed");
        for(size_t i=0;i<elements;++i)if((original[i]&0x7f80u)==0x7f80u)die("original weight nonfinite");
        sha(original,elements*2,original_sha);free(original);
        double hash_seconds=seconds()-start;
        h3_gpu_tensor *q=h3_gpu_tensor_new_i8(g,elements),*s=h3_gpu_tensor_new_f32(g,rows[f]);
        if(!q||!s)die(h3_gpu_error(g));
        start=seconds();
        if(!h3_gpu_begin(g))die(h3_gpu_error(g));
        int ok=f==3?h3_gpu_quantize_h256_int8(g,q,s,w,rows[f],rows[f],columns[f]):
            h3_gpu_quantize_weight_int8(g,q,s,w,rows[f],columns[f]);
        if(!ok||!h3_gpu_submit(g))die(h3_gpu_error(g));
        double preparation_seconds=seconds()-start;
        start=seconds();
        int8_t *codes=allocate(elements);float *scales=allocate((size_t)rows[f]*4);
        if(!h3_gpu_tensor_read_i8(q,codes,elements)||!h3_gpu_tensor_read_f32(s,scales,rows[f]))die("prepared readback failed");
        for(size_t i=0;i<elements;++i)if(codes[i]==INT8_MIN)die("invalid symmetric INT8 code");
        for(unsigned i=0;i<rows[f];++i)if(!isfinite(scales[i])||scales[i]<=0)die("invalid prepared scale");
        sha(codes,elements,codes_sha);sha(scales,(size_t)rows[f]*4,scales_sha);
        snprintf(file,sizeof(file),"%s.codes.i8",stem);save(argv[2],file,codes,elements);
        snprintf(file,sizeof(file),"%s.scales.f32",stem);save(argv[2],file,scales,(size_t)rows[f]*4);
        double write_seconds=seconds()-start;
        char metadata[2048];
        int bytes=snprintf(metadata,sizeof(metadata),"{\"schema\":1,\"block\":%u,\"family\":\"%s\",\"tensor\":\"%s\",\"rows\":%u,\"columns\":%u,\"original_tensor_sha256\":\"%s\",\"codes_sha256\":\"%s\",\"scales_sha256\":\"%s\",\"weight_load_seconds\":%.9f,\"original_hash_seconds\":%.9f,\"offline_preparation_seconds\":%.9f,\"readback_hash_write_seconds\":%.9f,\"all_values_finite\":true}\n",block,families[f],tensor,rows[f],columns[f],original_sha,codes_sha,scales_sha,load_seconds,hash_seconds,preparation_seconds,write_seconds);
        if(bytes<0||(size_t)bytes>=sizeof(metadata))die("metadata too long");
        snprintf(file,sizeof(file),"%s.export.json",stem);save(argv[2],file,metadata,(size_t)bytes);
        fprintf(stderr,"prepared export: block=%u family=%s load=%.6f prepare=%.6f write=%.6f\n",block,families[f],load_seconds,preparation_seconds,write_seconds);fflush(stderr);
        free(codes);free(scales);h3_gpu_tensor_free(q);h3_gpu_tensor_free(s);h3_gpu_tensor_free(w);
    }
    char done[512];int n=snprintf(done,sizeof(done),"{\"schema\":1,\"first_block\":%u,\"blocks\":%u,\"projections\":%u,\"initialize_seconds\":%.9f,\"export_seconds\":%.9f}\n",first,count,count*4,initialize_seconds,seconds()-total_start);
    if(n<0||(size_t)n>=sizeof(done))die("completion metadata too long");save(argv[2],"export-complete.json",done,(size_t)n);
    h3_weight_store_free(store);h3_gpu_free(g);return 0;
}
