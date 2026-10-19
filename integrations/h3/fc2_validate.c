/* Standalone FC2 numerical readback driver. Run only under the repository GPU
 * lock. It shares the exact native dispatcher with full H3 inference.
 * This driver does not report performance: every policy submits and reads back.
 */
#include "h3_gpu.h"
#include "h3_weights.h"
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { K = 14336, N = 5376, C = 32 };

static void die(const char *s) { fprintf(stderr,"fc2_validate: %s\n",s); exit(1); }
static void *allocate(size_t bytes) { void *p=calloc(bytes,1); if(!p) die("out of memory"); return p; }
static void write_data(const char *dir,const char *name,const void *p,size_t bytes) {
    char path[4096];
    int size=snprintf(path,sizeof(path),"%s/%s",dir,name);
    if(size<0 || (size_t)size>=sizeof(path)) die("path too long");
    int fd=open(path,O_CREAT|O_EXCL|O_WRONLY,0600);
    if(fd<0) die("output must not already exist");
    FILE *f=fdopen(fd,"wb");
    if(!f) die("fdopen failed");
    if(fwrite(p,1,bytes,f)!=bytes || fclose(f)) die("write failed");
}

int main(int argc,char **argv) {
    if(argc!=7) die("usage: fc2_validate MODEL_DIT_DIRECTORY BLOCK INPUT.bf16 OUTPUT_DIRECTORY ROWS SHADER");
    char *end=NULL;
    unsigned long parsed=strtoul(argv[2],&end,10);
    if(!end || *end || parsed>=50) die("invalid block");
    unsigned block=(unsigned)parsed;
    parsed=strtoul(argv[5],&end,10);
    if(!end || *end || parsed<128 || parsed>16384) die("rows must be 128..16384");
    uint32_t rows=(uint32_t)parsed, padded=(rows+127u)&~127u;
    struct stat st;
    if(stat(argv[3],&st) || st.st_size<=0 || st.st_size%(K*2)!=0) die("invalid BF16 input size");
    size_t input_rows=(size_t)st.st_size/(K*2);
    if(input_rows>rows) die("input exceeds selected rows");
    uint16_t *x=allocate((size_t)rows*K*2);
    FILE *input=fopen(argv[3],"rb");
    if(!input || fread(x,2,input_rows*K,input)!=input_rows*K || fclose(input)) die("read input failed");
    /* Real sample rows stay first, followed by exact-zero rows. This also
     * checks padding/zeros while keeping all operands comparable. */
    char error[1024];
    h3_gpu *g=h3_gpu_create(argv[6],error,sizeof(error));
    if(!g) die(error);
    if(!h3_gpu_has_int8_mlp(g)) die("unsupported TensorOps, no fallback");
    h3_weight_store *store=h3_weight_store_open(argv[1],error,sizeof(error));
    if(!store) die(error);
    char name[128]; snprintf(name,sizeof(name),"blocks.%u.mlp.fc2.weight",block);
    uint64_t shape[]={N,K};
    h3_gpu_tensor *w=h3_weight_load_bf16(store,g,name,2,shape,error,sizeof(error));
    if(!w) die(error);
    h3_gpu_tensor *a=h3_gpu_tensor_from_bf16(g,x,(size_t)rows*K);
    h3_gpu_tensor *aq=h3_gpu_tensor_new_i8(g,(size_t)padded*K);
    h3_gpu_tensor *as=h3_gpu_tensor_new_f32(g,(size_t)padded*(K/1024));
    h3_gpu_tensor *wq=h3_gpu_tensor_new_i8(g,(size_t)N*K);
    h3_gpu_tensor *ws=h3_gpu_tensor_new_f32(g,N);
    h3_gpu_tensor *y=h3_gpu_tensor_new_bf16(g,(size_t)rows*N);
    if(!a||!aq||!as||!wq||!ws||!y) die(h3_gpu_error(g));
    uint32_t columns[C];
    for(unsigned c=0;c<C;++c) columns[c]=(uint32_t)(((uint64_t)(N-1)*c)/(C-1));
    uint16_t *w_samples=allocate((size_t)C*K*2);
    for(unsigned c=0;c<C;++c)
        if(!h3_gpu_tensor_read_bf16_range(w,(size_t)columns[c]*K,w_samples+(size_t)c*K,K)) die("weight readback failed");
    write_data(argv[4],"input.bf16",x,(size_t)rows*K*2);
    write_data(argv[4],"weight-samples.bf16",w_samples,(size_t)C*K*2);
    write_data(argv[4],"columns.u32",columns,sizeof(columns));
    uint16_t *output=allocate((size_t)rows*N*2);
    int8_t *codes=allocate((size_t)padded*K);
    int8_t *weight_codes=allocate((size_t)C*K);
    float *scales=allocate((size_t)padded*(K/1024)*sizeof(float));
    float weight_scales[C];
    for(int policy=0;policy<3;++policy) {
        const char *tag=policy==0?"native-group1024":policy==1?"hadamard-row":"native-row";
        if(!h3_gpu_begin(g)) die(h3_gpu_error(g));
        int ok=policy==1 ? h3_gpu_quantize_h256_int8(g,wq,ws,w,N,N,K) :
            h3_gpu_quantize_weight_int8(g,wq,ws,w,N,K);
        if(!ok || !h3_gpu_submit(g)) die(h3_gpu_error(g));
        if(!h3_gpu_begin(g) || !h3_gpu_fc2_int8_bf16(g,y,aq,as,a,wq,ws,rows,K,N,policy,0) ||
           !h3_gpu_submit(g)) die(h3_gpu_error(g));
        size_t scale_count=(size_t)padded*(policy==0?K/1024:1);
        if(!h3_gpu_tensor_read_bf16(y,output,(size_t)rows*N) ||
           !h3_gpu_tensor_read_i8(aq,codes,(size_t)padded*K) ||
           !h3_gpu_tensor_read_f32(as,scales,scale_count)) die("activation/output readback failed");
        for(unsigned c=0;c<C;++c) {
            if(!h3_gpu_tensor_read_i8_range(wq,(size_t)columns[c]*K,weight_codes+(size_t)c*K,K) ||
               !h3_gpu_tensor_read_f32_range(ws,columns[c],weight_scales+c,1)) die("weight pack readback failed");
        }
        char file[128];
#define SAVE(suffix,p,bytes) do { snprintf(file,sizeof(file),"%s.%s",tag,suffix); write_data(argv[4],file,p,bytes); } while(0)
        SAVE("output.bf16",output,(size_t)rows*N*2);
        SAVE("activation.i8",codes,(size_t)padded*K);
        SAVE("activation-scales.f32",scales,scale_count*sizeof(float));
        SAVE("weight-samples.i8",weight_codes,(size_t)C*K);
        SAVE("weight-scales.f32",weight_scales,sizeof(weight_scales));
#undef SAVE
        fprintf(stderr,"fc2_validate: %s complete, rows=%u padded=%u\n",tag,rows,padded);
    }
    char metadata[1024];
    int bytes=snprintf(metadata,sizeof(metadata),"{\"schema\":1,\"scope\":\"numerics_readback_not_timing\",\"block\":%u,\"sample_rows\":%zu,\"rows\":%u,\"padded_rows\":%u,\"K\":%d,\"N\":%d,\"sample_columns\":%d,\"weight_key\":\"%s\",\"weight_quantization\":\"per_output_row\",\"native_group_size\":1024,\"hadamard\":\"normalized_plain_H256_FP32_then_whole_row_INT8\"}\n",block,input_rows,rows,padded,K,N,C,name);
    if(bytes<0 || (size_t)bytes>=sizeof(metadata)) die("metadata too long");
    write_data(argv[4],"driver.json",metadata,(size_t)bytes);
    h3_gpu_tensor_free(y); h3_gpu_tensor_free(ws); h3_gpu_tensor_free(wq);
    h3_gpu_tensor_free(as); h3_gpu_tensor_free(aq); h3_gpu_tensor_free(a); h3_gpu_tensor_free(w);
    h3_weight_store_free(store); h3_gpu_free(g);
    free(x); free(w_samples); free(output); free(codes); free(weight_codes); free(scales);
    return 0;
}
