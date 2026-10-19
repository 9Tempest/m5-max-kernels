/* Compile/load the pinned runtime library without loading model weights.
 * Run only while holding m5kernels.locking.gpu_lock in the caller. */
#include "h3_gpu.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    setenv("H3_NAX", "qkv-attn", 1);
    setenv("H3_NAX_DIAGNOSTIC", "1", 1);
    char error[4096] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "%s\n", error); return 1; }
    int m5 = h3_gpu_is_m5(gpu);
    int int8 = h3_gpu_has_int8_mlp(gpu);
    printf("{\"m5\":%d,\"int8_mlp_api_available\":%d,"
           "\"scope\":\"shader compile/load only; no numerical or performance validation\"}\n",
           m5, int8);
    h3_gpu_free(gpu);
    return m5 && int8 ? 0 : 1;
}
