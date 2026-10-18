/* CPU-only probe of the pinned native tokenizer; never creates a Metal device. */
#include "h3_tokenizer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    FILE *f = fopen(argv[2], "rb");
    if (!f || fseek(f, 0, SEEK_END)) return 2;
    long length = ftell(f);
    if (length <= 0 || length > 1024 * 1024 || fseek(f, 0, SEEK_SET)) return 2;
    char *prompt = calloc((size_t)length + 1, 1);
    if (!prompt || fread(prompt, 1, (size_t)length, f) != (size_t)length ||
        fclose(f) || strlen(prompt) != (size_t)length) return 2;
    char error[512] = {0};
    h3_tokenizer *tokenizer = h3_tokenizer_load(argv[1], error, sizeof(error));
    uint32_t *ids = NULL, *prefix_ids = NULL;
    size_t count = 0, prefix_count = 0;
    if (!tokenizer || !h3_tokenizer_encode(tokenizer, prompt, 0, &ids, &count,
            error, sizeof(error)) ||
        !h3_tokenizer_encode(tokenizer, "<Picture 1>: ", 0, &prefix_ids,
            &prefix_count, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    char *decoded = h3_tokenizer_decode(tokenizer, ids, count, error, sizeof(error));
    int exact = decoded && strcmp(decoded, prompt) == 0;
    printf("{\"schema\":1,\"prompt_bytes\":%ld,\"prompt_tokens\":%zu,"
           "\"image_prefix_tokens\":%zu,\"roundtrip_exact\":%s}\n",
           length, count, prefix_count, exact ? "true" : "false");
    free(decoded); free(prompt);
    h3_tokenizer_ids_free(ids); h3_tokenizer_ids_free(prefix_ids);
    h3_tokenizer_free(tokenizer);
    return exact ? 0 : 1;
}
