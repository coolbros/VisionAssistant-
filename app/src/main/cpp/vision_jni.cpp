#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <android/bitmap.h>
#include <mutex>
#include <memory>
#include <chrono>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-helper-common.h"
#include "common.h"
#include "sampling.h"

#define TAG "LocalVisionEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static void llama_log_callback(ggml_log_level level, const char * text, void * user_data) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        __android_log_print(ANDROID_LOG_ERROR, "llama.cpp", "%s", text);
    } else if (level == GGML_LOG_LEVEL_WARN) {
        __android_log_print(ANDROID_LOG_WARN, "llama.cpp", "%s", text);
    } else {
        __android_log_print(ANDROID_LOG_INFO, "llama.cpp", "%s", text);
    }
}

static std::mutex g_mutex;

struct LocalVisionContext {
    llama_model * model = nullptr;
    llama_context * lctx = nullptr;
    const llama_vocab * vocab = nullptr;
    common_sampler * smpl = nullptr;
    mtmd_context * ctx_vision = nullptr;

    ~LocalVisionContext() {
        if (smpl) common_sampler_free(smpl);
        if (ctx_vision) mtmd_free(ctx_vision);
        if (lctx) llama_free(lctx);
        if (model) llama_free_model(model);
    }
};

static std::unique_ptr<LocalVisionContext> g_ctx;

// Utility to get time in ms
static int64_t get_time_ms() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_visionassistant_LocalVisionEngine_nativeInitialize(JNIEnv *env, jobject thiz, jstring model_path, jstring mmproj_path, jint n_threads, jstring native_lib_dir) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ctx) {
        LOGI("Already initialized");
        return JNI_TRUE;
    }

    int64_t t_start = get_time_ms();
    LOGI("LOCAL_LFM2_INIT_START");

    llama_log_set(llama_log_callback, nullptr);

    const char * model_str = env->GetStringUTFChars(model_path, 0);
    const char * mmproj_str = env->GetStringUTFChars(mmproj_path, 0);
    const char * native_lib_dir_str = env->GetStringUTFChars(native_lib_dir, 0);

    ggml_backend_load_all_from_path(native_lib_dir_str);
    llama_backend_init();

    auto ctx = std::make_unique<LocalVisionContext>();

    llama_model_params mparams = llama_model_default_params();
    ctx->model = llama_load_model_from_file(model_str, mparams);
    if (!ctx->model) {
        LOGE("Failed to load LLaMA model");
        return JNI_FALSE;
    }

    ctx->vocab = llama_model_get_vocab(ctx->model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 4096;
    cparams.n_batch = 2048;
    ctx->lctx = llama_new_context_with_model(ctx->model, cparams);
    
    common_params params;
    params.sampling.temp = 0.1f;
    ctx->smpl = common_sampler_init(ctx->model, params.sampling);

    mtmd_context_params mmparams = mtmd_context_params_default();
    mmparams.n_threads = n_threads;
    ctx->ctx_vision = mtmd_init_from_file(mmproj_str, ctx->model, mmparams);
    
    env->ReleaseStringUTFChars(model_path, model_str);
    env->ReleaseStringUTFChars(mmproj_path, mmproj_str);

    g_ctx = std::move(ctx);

    int64_t t_end = get_time_ms();
    LOGI("LOCAL_LFM2_INIT_END (took %lld ms)", (long long)(t_end - t_start));
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_visionassistant_LocalVisionEngine_nativeInfer(JNIEnv *env, jobject thiz, jobject bitmap, jstring prompt) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ctx) {
        LOGE("Engine not initialized");
        return env->NewStringUTF("ERROR");
    }

    int64_t t_start = get_time_ms();
    LOGI("LOCAL_LFM2_INFER_START");

    AndroidBitmapInfo info;
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0 || AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        LOGE("Failed to get bitmap pixels");
        return env->NewStringUTF("ERROR");
    }

    // Prepare mtmd_bitmap (RGB 8-bit). Android ARGB_8888 -> RGB
    std::vector<unsigned char> rgb_data(info.width * info.height * 3);
    uint32_t* src = (uint32_t*)pixels;
    for (size_t i = 0; i < info.width * info.height; ++i) {
        uint32_t p = src[i];
        // Android ARGB_8888 (in Java it's AARRGGBB, but in memory via NDK it's strictly ordered RGBA on some platforms or ABGR on others)
        // Actually, AndroidBitmap_getInfo FORMAT_RGBA_8888 is ABGR in little endian memory. 
        // byte 0 = R, byte 1 = G, byte 2 = B, byte 3 = A.
        uint8_t* p_byte = (uint8_t*)&src[i];
        rgb_data[i*3 + 0] = p_byte[0]; // R
        rgb_data[i*3 + 1] = p_byte[1]; // G
        rgb_data[i*3 + 2] = p_byte[2]; // B
    }
    AndroidBitmap_unlockPixels(env, bitmap);

    int64_t t_prep = get_time_ms();
    LOGI("LOCAL_LFM2_IMAGE_PREPROCESS_MS: %lld", (long long)(t_prep - t_start));

    mtmd_bitmap * mtmd_bmp = mtmd_bitmap_init(info.width, info.height, rgb_data.data());

    // Prompt preparation
    const char * prompt_str = env->GetStringUTFChars(prompt, 0);
    std::string full_prompt = std::string(mtmd_default_marker()) + prompt_str;
    env->ReleaseStringUTFChars(prompt, prompt_str);

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text txt = {full_prompt.c_str(), full_prompt.size(), false, true};
    const mtmd_input_part part1 = {nullptr, mtmd_bmp};
    const mtmd_input_part part2 = {&txt, nullptr};
    const mtmd_input_part* parts[] = {&part1, &part2};

    int32_t res = mtmd_tokenize_from_parts(g_ctx->ctx_vision, chunks, parts, 2, true);
    if (res != 0) {
        LOGE("Failed to tokenize parts");
        mtmd_bitmap_free(mtmd_bmp);
        mtmd_input_chunks_free(chunks);
        return env->NewStringUTF("ERROR");
    }

    llama_pos n_past = 0;
    llama_memory_clear(llama_get_memory(g_ctx->lctx), true);

    mtmd_batch * mbatch = mtmd_batch_init(g_ctx->ctx_vision);
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    
    // Batch encode image
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            mtmd_batch_add_chunk(mbatch, chunk);
        }
    }
    mtmd_batch_encode(mbatch);
    
    int64_t t_vision = get_time_ms();
    LOGI("LOCAL_LFM2_VISION_MS: %lld", (long long)(t_vision - t_prep));

    // Decode chunks
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            llama_pos new_n_past = n_past;
            mtmd_helper_eval_chunk_single(g_ctx->ctx_vision, g_ctx->lctx, chunk, n_past, 0, 2048, i == n_chunks - 1, &new_n_past);
            n_past = new_n_past;
        } else {
            float * embd = mtmd_batch_get_output_embd(mbatch, chunk);
            llama_pos new_n_past = n_past;
            mtmd_helper_decode_image_chunk(g_ctx->ctx_vision, g_ctx->lctx, chunk, embd, n_past, 0, 2048, &new_n_past, nullptr, nullptr);
            n_past = new_n_past;
        }
    }

    int64_t t_prompt = get_time_ms();
    LOGI("LOCAL_LFM2_PROMPT_MS: %lld", (long long)(t_prompt - t_vision));

    // Generate output
    std::string result = "";
    common_batch cbatch = common_batch(g_ctx->lctx);
    common_sampler_reset(g_ctx->smpl);

    for (int i = 0; i < 10; i++) {
        llama_token token_id = common_sampler_sample(g_ctx->smpl, g_ctx->lctx, -1);
        common_sampler_accept(g_ctx->smpl, token_id, true);

        if (llama_vocab_is_eog(g_ctx->vocab, token_id)) {
            break;
        }
        
        std::string piece = common_token_to_piece(g_ctx->lctx, token_id);
        result += piece;
        
        cbatch.clear();
        cbatch.add(token_id, n_past++, 0, true);
        if (llama_process(g_ctx->lctx, LLAMA_PROCESS_TYPE_DECODE, cbatch.get())) {
            break;
        }
    }

    int64_t t_decode = get_time_ms();
    LOGI("LOCAL_LFM2_DECODE_MS: %lld", (long long)(t_decode - t_prompt));
    LOGI("LOCAL_LFM2_TOTAL_MS: %lld", (long long)(t_decode - t_start));
    LOGI("LOCAL_LFM2_RESULT: %s", result.c_str());

    mtmd_batch_free(mbatch);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(mtmd_bmp);

    // Trim result
    size_t first = result.find_first_not_of(" \n\r\t");
    size_t last = result.find_last_not_of(" \n\r\t");
    if (first == std::string::npos) result = "";
    else result = result.substr(first, (last - first + 1));

    return env->NewStringUTF(result.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_visionassistant_LocalVisionEngine_nativeClose(JNIEnv *env, jobject thiz) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_ctx.reset();
    LOGI("Engine closed");
}
