/* Original host-only cache experiment for the pinned h3.c public API.
 * Build from repository root (no GPU execution during compilation):
 * clang -std=c11 -O2 -Wall -Wextra -Werror -D_DARWIN_C_SOURCE -I.cache/h3.c \
 *   integrations/h3/session.c .cache/h3.c/libh3.a -framework Foundation \
 *   -framework Metal -framework MetalPerformanceShaders \
 *   -framework MetalPerformanceShadersGraph -framework Accelerate -licucore -lm \
 *   -o .build/h3-session
 * Run from .cache/h3.c, where the original h3_shaders.metal is located:
 * ../../.build/h3-session MODE MODEL_DIR NEW_OUTPUT_DIR [WIDTH HEIGHT STEPS SEED]
 * MODE is bf16 or upstream-int8. Defaults: 512 512 20 42. 124 source frames.
 * The driver holds the same flock file used by m5kernels.locking.gpu_lock.
 * Records are exclusive. One warmup precedes three same-seed timed generations.
 * Audio VAE creates h3_gpu and loads stage weights on EVERY generation; these
 * are cached deployment latencies, not compilation-free kernel measurements.
 * No upstream implementation or model weights are copied into this file.
 */

#include "h3.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

static const char *prompt =
    "A red fox walks through fresh snow in a pine forest. Medium tracking shot, "
    "natural winter light, realistic fur, soft footsteps and wind.";

static double now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) abort();
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void json_string(FILE *file, const char *text) {
    fputc('"', file);
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p == '"' || *p == '\\') fprintf(file, "\\%c", *p);
        else if (*p < 32) fprintf(file, "\\u%04x", *p);
        else fputc(*p, file);
    }
    fputc('"', file);
}

static void cache_json(FILE *file, const h3_cache_info *info) {
    fprintf(file, "{\"embedding_entries\":%zu,\"embedding_bytes\":%zu,"
            "\"prepared_dit\":%d,\"video_decoder\":%d}",
            info->embedding_entries, info->embedding_bytes,
            info->prepared_dit, info->video_decoder);
}

static int exclusive_log(const char *directory, const char *name) {
    char path[PATH_MAX];
    if (snprintf(path, sizeof(path), "%s/%s", directory, name) >= (int)sizeof(path))
        return -1;
    return open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
}

/* Python tempfile.gettempdir() checks these variables in this order. */
#ifndef H3_SESSION_EXPERIMENT
static int lock_gpu(void) {
    const char *keys[] = {"TMPDIR", "TEMP", "TMP"};
    const char *directory = NULL;
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i) {
        const char *candidate = getenv(keys[i]);
        struct stat info;
        if (candidate && *candidate && stat(candidate, &info) == 0 &&
            S_ISDIR(info.st_mode) && access(candidate, W_OK | X_OK) == 0) {
            directory = candidate;
            break;
        }
    }
    if (!directory) directory = "/tmp";
    char path[PATH_MAX];
    if (snprintf(path, sizeof(path), "%s/m5-max-kernels-%u.lock", directory,
                 (unsigned)getuid()) >= (int)sizeof(path)) return -1;
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0) return fd;
    if (fd >= 0) close(fd);
    return -1;
}
#endif

typedef struct {
    double start;
    double denoise_start;
    double denoise_end;
    int read_fd;
    int fallback;
    char tail[64];
} progress_state;

static int detect_fallback(progress_state *state) {
    char block[4096 + 64];
    size_t prefix = strlen(state->tail);
    memcpy(block, state->tail, prefix);
    ssize_t count;
    fflush(stderr);
    while ((count = read(state->read_fd, block + prefix, 4096)) > 0) {
        size_t length = prefix + (size_t)count;
        block[length] = '\0';
        if (strstr(block, "TensorOps compile failed:")) state->fallback = 1;
        prefix = length < 63 ? length : 63;
        memmove(block, block + length - prefix, prefix);
    }
    memcpy(state->tail, block, prefix);
    state->tail[prefix] = '\0';
    return state->fallback;
}

static int progress(const char *phase, int completed, int total, void *opaque) {
    progress_state *state = opaque;
    double timestamp = now();
    if (!strncmp(phase, "denoise", 7) && completed == 0 && state->denoise_start < 0)
        state->denoise_start = timestamp;
    if (!strcmp(phase, "denoise") && completed == total)
        state->denoise_end = timestamp;
    fprintf(stderr, "session progress: elapsed=%.6fs phase=%s completed=%d total=%d\n",
            timestamp - state->start, phase, completed, total);
    return detect_fallback(state);
}

static uint64_t integer(const char *text, uint64_t min, uint64_t max) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || text[0] == '-' || !*text || !end || *end || value < min || value > max) {
        fprintf(stderr, "invalid integer argument: %s\n", text);
        exit(2);
    }
    return (uint64_t)value;
}

int main(int argc, char **argv) {
#ifdef H3_SESSION_EXPERIMENT
    /* The experimental Python wrapper owns the GPU lock through preflight,
     * rebuild, execution and validation. Compile this entry only through it. */
    if (argc != 9 || (strcmp(argv[1], "bf16") &&
        strcmp(argv[1], "upstream-int8") && strcmp(argv[1], "hadamard-fc2"))) {
        fprintf(stderr, "Usage: %s MODE MODEL_DIR NEW_OUTPUT_DIR WIDTH HEIGHT STEPS SEED "
                "none|fc2|attention|both\n", argv[0]);
        return 2;
    }
    const char *fusion = argv[8];
    if (strcmp(fusion, "none") && strcmp(fusion, "fc2") &&
        strcmp(fusion, "attention") && strcmp(fusion, "both")) return 2;
    if (strcmp(fusion, "none") && strcmp(argv[1], "hadamard-fc2")) return 2;
#else
    if ((argc != 4 && argc != 8) ||
        (strcmp(argv[1], "bf16") && strcmp(argv[1], "upstream-int8"))) {
        fprintf(stderr, "Usage: %s bf16|upstream-int8 MODEL_DIR NEW_OUTPUT_DIR "
                "[WIDTH HEIGHT STEPS SEED]\n", argv[0]);
        return 2;
    }
#endif
    for (char **entry = environ; *entry; ++entry) {
        if (!strncmp(*entry, "H3_", 3)) {
            fprintf(stderr, "Unset inherited H3_* overrides before this fixed experiment\n");
            return 2;
        }
    }
    h3_params params = H3_PARAMS_DEFAULT;
    params.width = argc >= 8 ? (int)integer(argv[4], 256, 4096) : 512;
    params.height = argc >= 8 ? (int)integer(argv[5], 256, 4096) : 512;
    params.steps = argc >= 8 ? (int)integer(argv[6], 2, 1000) : 20;
    params.seed = argc >= 8 ? integer(argv[7], 0, UINT64_MAX) : 42;
    if (params.width % 32 || params.height % 32) {
        fprintf(stderr, "Dimensions must be multiples of 32\n");
        return 2;
    }
    params.frames = 124;
    params.use_reference_rope = 1;
    params.use_slower_bf16_mlp = !strcmp(argv[1], "bf16");
    params.use_slower_bf16_qkv = params.use_slower_bf16_mlp;
    params.use_slower_bf16_attention_output = params.use_slower_bf16_mlp;
    if (mkdir(argv[3], 0755) != 0) {
        perror("exclusive output directory");
        return 2;
    }
    int record_fd = exclusive_log(argv[3], "session.jsonl");
    FILE *records = record_fd < 0 ? NULL : fdopen(record_fd, "w");
    if (!records) return 2;
    setvbuf(records, NULL, _IOLBF, 0);
    fprintf(records, "{\"event\":\"recipe\",\"mode\":\"%s\",\"width\":%d,"
            "\"height\":%d,\"frames\":124,\"fps\":24,\"steps\":%d,\"seed\":%" PRIu64 ","
            "\"layers\":50,\"reuse\":1,\"core_reuse\":1,\"use_reference_rope\":true,"
            "\"warmups\":1,\"repetitions\":3,\"prompt\":", argv[1], params.width,
            params.height, params.steps, params.seed);
    json_string(records, prompt);
#ifdef H3_SESSION_EXPERIMENT
    fprintf(records, ",\"fusion\":\"%s\"", fusion);
#endif
    fprintf(records, ",\"upstream_revision\":\"8974cc055ea9c02fcd14cc27dfda3e1027c05153\","
            "\"model_revision\":\"42ed227ee7df40d41602854ae760620d6eb651fe\","
            "\"timing_scope\":\"h3_generate including AV export; audio VAE shader "
            "compilation API and weight loading repeat; not compilation-free\","
            "\"profile_caveat\":\"cached DiT wall marks include time since previous "
            "generation denoise mark; use fresh callback interval for denoise host timing\","
            "\"denoise_callback_scope\":\"first denoise enqueue after input packing/copy "
            "through synchronized GPU completion and latent unpack; excludes initial input "
            "packing/copy, includes final latent copy/unpack\"}\n");
#ifndef H3_SESSION_EXPERIMENT
    int lock = lock_gpu();
    if (lock < 0) {
        fprintf(records, "{\"event\":\"failure\",\"error\":\"GPU lock unavailable\"}\n");
        fclose(records);
        return 1;
    }
#else
    setenv("H3_HADAMARD_FC2", !strcmp(argv[1], "hadamard-fc2") ? "1" : "0", 1);
    if (strcmp(fusion, "none")) {
        setenv("H3_FUSE_FC2_GATE",
               !strcmp(fusion, "fc2") || !strcmp(fusion, "both") ? "1" : "0", 1);
        setenv("H3_FUSE_ATTN_OUT_GATE",
               !strcmp(fusion, "attention") || !strcmp(fusion, "both") ? "1" : "0", 1);
    }
#endif
    setenv("H3_PROFILE", "1", 1);
    setenv("H3_NAX", "qkv-attn", 1);
    setenv("H3_NAX_DIAGNOSTIC", "1", 1);
    int log_fd = exclusive_log(argv[3], "startup.log");
    fflush(stderr);
    if (log_fd < 0 || dup2(log_fd, STDERR_FILENO) < 0) return 1;
    close(log_fd);
    double start = now();
    h3_ctx *ctx = h3_load_dir(argv[2]);
    fprintf(records, "{\"event\":\"startup\",\"wall_seconds\":%.9f,\"status\":\"%s\"}\n",
            now() - start, ctx ? "completed" : "failed");
    int status = 1;
    if (!ctx) {
        fprintf(records, "{\"event\":\"failure\",\"error\":");
        json_string(records, h3_last_error(NULL));
        fprintf(records, "}\n");
        goto done;
    }
    const h3_device_info *device = h3_device(ctx);
    fprintf(records, "{\"event\":\"device\",\"name\":");
    json_string(records, device->name);
    fprintf(records, ",\"physical_memory_bytes\":%" PRIu64 "}\n", device->physical_memory);
    if (!strstr(device->name, "M5")) {
        fprintf(records, "{\"event\":\"unsupported\",\"error\":\"M5 required\"}\n");
        goto done;
    }
    h3_cache_set_enabled(ctx, 1);
    params.on_progress = progress;
    for (int run = 0; run < 4; ++run) {
        char name[64], output_path[PATH_MAX], log_path[PATH_MAX];
        snprintf(name, sizeof(name), "%s-%d", run ? "timed" : "warmup", run);
        int n = snprintf(output_path, sizeof(output_path), "%s/%s.mp4", argv[3], name);
        int m = snprintf(log_path, sizeof(log_path), "%s/%s.log", argv[3], name);
        if (n >= (int)sizeof(output_path) || m >= (int)sizeof(log_path)) goto done;
        char log_name[80];
        snprintf(log_name, sizeof(log_name), "%s.log", name);
        log_fd = exclusive_log(argv[3], log_name);
        fflush(stderr);
        if (log_fd < 0 || dup2(log_fd, STDERR_FILENO) < 0) goto done;
        close(log_fd);
        progress_state state = {0};
        state.denoise_start = -1;
        state.denoise_end = -1;
        state.read_fd = open(log_path, O_RDONLY);
        if (state.read_fd < 0) goto done;
        params.callback_opaque = &state;
        params.output_path = output_path;
        h3_cache_info before, after;
        h3_cache_get_info(ctx, &before);
        time_t started_unix = time(NULL);
        state.start = now();
        h3_result *result = h3_generate(ctx, prompt, &params);
        double elapsed = now() - state.start;
        detect_fallback(&state);
        close(state.read_fd);
        h3_cache_get_info(ctx, &after);
        int valid = result && !state.fallback && result->width == params.width &&
                    result->height == params.height && result->frames == 124 && result->fps == 24;
        fprintf(records, "{\"event\":\"generation\",\"index\":%d,\"warmup\":%s,"
                "\"started_unix_seconds\":%lld,\"wall_seconds\":%.9f,"
                "\"status\":\"%s\",\"output\":\"%s.mp4\","
                "\"profile_log\":\"%s.log\",\"cache_before\":", run,
                run ? "false" : "true", (long long)started_unix, elapsed,
                state.fallback ? "unsupported" : valid ? "completed" : "failed", name, name);
        cache_json(records, &before);
        fprintf(records, ",\"cache_after\":");
        cache_json(records, &after);
        fprintf(records, ",\"denoise_callback_seconds\":");
        if (state.denoise_start >= 0 && state.denoise_end >= state.denoise_start)
            fprintf(records, "%.9f", state.denoise_end - state.denoise_start);
        else fprintf(records, "null");
        if (!valid) {
            fprintf(records, ",\"error\":");
            json_string(records, state.fallback ? "TensorOps compile fallback rejected" :
                        result ? "unexpected output structure" : h3_last_error(ctx));
        }
        fprintf(records, "}\n");
        fflush(records);
        fsync(fileno(records));
        printf("%s %s %.6f seconds\n", name, valid ? "completed" : "failed", elapsed);
        fflush(stdout);
        h3_result_free(result);
        if (!valid) goto done;
        if (!after.prepared_dit || !after.video_decoder || !after.embedding_entries) {
            fprintf(records, "{\"event\":\"failure\",\"error\":\"requested cache not retained\"}\n");
            goto done;
        }
    }
    status = 0;
done:
    log_fd = exclusive_log(argv[3], "teardown.log");
    if (log_fd >= 0) {
        fflush(stderr);
        dup2(log_fd, STDERR_FILENO);
        close(log_fd);
    }
    h3_free(ctx);
    fprintf(records, "{\"event\":\"finished\",\"exit_code\":%d}\n", status);
    fclose(records);
#ifndef H3_SESSION_EXPERIMENT
    flock(lock, LOCK_UN);
    close(lock);
#endif
    return status;
}
