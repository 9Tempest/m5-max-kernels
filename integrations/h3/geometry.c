/* CPU-only request geometry using the pinned upstream tokenizer/layout API. */
#include "h3_host.h"
#include "h3_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: geometry tokenizer.json prompt\n");
        return 2;
    }
    char error[512];
    h3_tokenizer *tokenizer = h3_tokenizer_load(argv[1], error, sizeof(error));
    if (!tokenizer) { fprintf(stderr, "%s\n", error); return 1; }
    uint32_t *ids = NULL;
    size_t count = 0;
    if (!h3_tokenizer_encode(tokenizer, argv[2], 1, &ids, &count, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        h3_tokenizer_free(tokenizer);
        return 1;
    }
    h3_temporal_shape temporal = h3_temporal(120);
    printf("{\"evidence\":\"CPU pinned tokenizer + temporal API; not a GPU trace\","
           "\"requested_frames\":120,\"aligned_frames\":%d,\"fps\":24,"
           "\"width\":512,\"height\":512,\"video_t\":%d,\"audio_t\":%d,"
           "\"text_tokens\":%zu,\"video_tokens\":%d,\"stereo_audio_tokens\":%d,"
           "\"packed_tokens\":%zu,\"token_ids\":[",
           temporal.frame_count, temporal.video_t, temporal.audio_t, count,
           temporal.video_t * 256, temporal.audio_t * 2,
           count + (size_t)temporal.video_t * 256 + (size_t)temporal.audio_t * 2);
    for (size_t i = 0; i < count; i++) printf("%s%u", i ? "," : "", ids[i]);
    puts("]}");
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    return 0;
}
