// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

//#define MTMD_AUDIO_DEBUG

#define MINIAUDIO_IMPLEMENTATION
#ifndef MTMD_AUDIO_DEBUG
#   define MA_NO_ENCODING
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#ifdef LLAMA_MTMD_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#endif

#ifdef MTMD_INTERNAL_HEADER
#error "mtmd-helper is a public library outside of mtmd. it must not include internal headers"
#endif

//
// internal logging functions
//

struct mtmd_helper_logger {
    ggml_log_callback default_callback = [](ggml_log_level level, const char * text, void * user_data) {
        (void) level;
        (void) user_data;
        fputs(text, stderr);
        fflush(stderr);
    };

    ggml_log_callback log_callback = default_callback;
    void * log_callback_user_data;

    void log_v(enum ggml_log_level level, const char * format, va_list args) {
        if (format == NULL) {
            return;
        }
        va_list args_copy;
        va_copy(args_copy, args);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            log_callback(level, buffer, log_callback_user_data);
        } else {
            char * buffer2 = (char *) calloc(len + 1, sizeof(char));
            vsnprintf(buffer2, len + 1, format, args_copy);
            buffer2[len] = 0;
            log_callback(level, buffer2, log_callback_user_data);
            free(buffer2);
        }
        va_end(args_copy);
    }

    void log(enum ggml_log_level level, const char * format, ...) {
        va_list args;
        va_start(args, format);
        log_v(level, format, args);
        va_end(args);
    }
} g_logger;

#define LOG_INF(...) g_logger.log(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) g_logger.log(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) g_logger.log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data) {
    if (log_callback == nullptr) {
        log_callback = g_logger.default_callback;
    }
    g_logger.log_callback = log_callback;
    g_logger.log_callback_user_data = user_data;
    mtmd_log_set(log_callback, user_data);
}

//
// helper functions
//

size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks) {
    size_t n_tokens = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_tokens += mtmd_input_chunk_get_n_tokens(chunk);
    }
    return n_tokens;
}

llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks) {
    llama_pos n_pos = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_pos += mtmd_input_chunk_get_n_pos(chunk);
    }
    return n_pos;
}

// helper struct to make working with embd batch easier
// note: this will be removed after llama_batch_ext refactoring
struct decode_embd_batch {
    int n_pos_per_embd;
    int n_mmproj_embd;
    std::vector<llama_pos>      pos;
    std::vector<llama_pos>      pos_view; // used by mrope
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, int n_pos_per_embd, int n_mmproj_embd) : n_pos_per_embd(n_pos_per_embd), n_mmproj_embd(n_mmproj_embd) {
        pos     .resize(n_tokens * n_pos_per_embd);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
    }

    void set_position_normal(llama_pos pos_0, llama_seq_id seq_id) {
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    // M-RoPE for image
    void set_position_mrope_2d(llama_pos pos_0, int nx, int ny, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        seq_id_0[0] = seq_id;
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                int i = y * nx + x;
                pos[i                     ] = pos_0;
                pos[i + batch.n_tokens    ] = pos_0 + y;
                pos[i + batch.n_tokens * 2] = pos_0 + x;
                pos[i + batch.n_tokens * 3] = 0; // last pos dim is unused
            }
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    // M-RoPE for audio
    void set_position_mrope_1d(llama_pos pos_0, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            pos[i                     ] = pos_0 + i;
            pos[i + batch.n_tokens    ] = pos_0 + i;
            pos[i + batch.n_tokens * 2] = pos_0 + i;
            pos[i + batch.n_tokens * 3] = 0; // last pos dim is unused
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    llama_batch get_view(int offset, int n_tokens) {
        llama_pos * pos_ptr;
        pos_view.clear();
        pos_view.reserve(n_tokens * n_pos_per_embd);
        if (n_pos_per_embd > 1) {
            // mrope
            // for example, with layout of src: 1234...1234...1234...1234...
            //       offset 2 will give us dst: 34...34...34...34...
            for (int i = 0; i < n_pos_per_embd; i++) {
                // assume n_tokens is less than or equal to batch.n_tokens
                // batch.n_tokens is number of **total** tokens
                // n_tokens is number of viewed token
                size_t src_idx = i * batch.n_tokens + offset;
                pos_view.insert(pos_view.end(),
                    pos.data() + src_idx,
                    pos.data() + src_idx + n_tokens);
            }
            pos_ptr = pos_view.data();
        } else {
            // normal
            pos_ptr = pos.data() + offset;
        }
        return {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ batch.embd     + offset * n_mmproj_embd,
            /*pos            =*/ pos_ptr,
            /*n_seq_id       =*/ batch.n_seq_id + offset,
            /*seq_id         =*/ batch.seq_id   + offset,
            /*logits         =*/ batch.logits   + offset,
        };
    }
};

// Helper function for decoding an image whose embeddings have already been calculated
int32_t mtmd_helper_decode_image_chunk(
        mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        float * encoded_embd,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        llama_pos * new_n_past) {
    auto chunk_type = mtmd_input_chunk_get_type(chunk);
    const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_ERR("failed to decode chunk: input chunk not of image/audio type\n");
        return -1;
    }

    const llama_model * model = llama_get_model(lctx);
    int n_mmproj_embd = llama_model_n_embd_inp(model);
    int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;

    int32_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    int32_t i_batch = 0;
    int32_t n_img_batches = GGML_PAD(n_tokens, n_batch) / n_batch;
    decode_embd_batch batch_embd(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);

    if (mtmd_decode_use_mrope(ctx)) {
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            if (!image_tokens) {
                LOG_ERR("failed to decode chunk: image tokens are null\n");
                return -1;
            }
            const int nx = mtmd_image_tokens_get_nx(image_tokens);
            const int ny = mtmd_image_tokens_get_ny(image_tokens);
            batch_embd.set_position_mrope_2d(n_past, nx, ny, seq_id);
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            batch_embd.set_position_mrope_1d(n_past, seq_id);
        } else {
            GGML_ABORT("invalid chunk type for M-RoPE");
        }
    } else {
        batch_embd.set_position_normal(n_past, seq_id);
    }

    if (mtmd_decode_use_non_causal(ctx)) {
        llama_set_causal_attn(lctx, false);
        // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
    }

    while (i_batch < n_img_batches) { // split into batches
        int pos_offset = i_batch*n_batch;
        int n_tokens_batch = std::min(n_batch, n_tokens - pos_offset);
        llama_batch batch_embd_view = batch_embd.get_view(pos_offset, n_tokens_batch);

        LOG_INF("decoding %s batch %d/%d, n_tokens_batch = %d\n", name, i_batch+1, n_img_batches, n_tokens_batch);

        int64_t t1 = ggml_time_ms();
        int32_t ret = llama_decode(lctx, batch_embd_view);
        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            llama_set_causal_attn(lctx, true); // restore causal attn
            return ret;
        }

        LOG_INF("%s decoded (batch %d/%d) in %" PRId64 " ms\n", name, i_batch+1, n_img_batches, ggml_time_ms() - t1);

        i_batch++;
    }

    n_past += mtmd_input_chunk_get_n_pos(chunk);
    *new_n_past = n_past;

    if (mtmd_decode_use_non_causal(ctx)) {
        llama_set_causal_attn(lctx, true);
    }
    return 0;
}

int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        bool logits_last,
        llama_pos * new_n_past) {
    int32_t ret;
    llama_batch text_batch = llama_batch_init(n_batch, 0, 1);
    auto chunk_type = mtmd_input_chunk_get_type(chunk);

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_tokens;
        const auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        // LOG_INF("decoding text chunk, n_tokens = %zu\n", n_tokens);
        size_t i = 0;
        while (i < n_tokens) { // split into batches
            text_batch.n_tokens = 0; // clear the batch
            for (; i < n_tokens && text_batch.n_tokens < n_batch; i++) {
                int32_t j = text_batch.n_tokens;
                text_batch.token   [j]    = tokens[i];
                text_batch.pos     [j]    = n_past++;
                text_batch.n_seq_id[j]    = 1;
                text_batch.seq_id  [j][0] = seq_id;
                text_batch.logits  [j]    = false;

                text_batch.n_tokens++;
            }
            bool is_last_token = (i == n_tokens);
            if (logits_last && is_last_token) {
                text_batch.logits[text_batch.n_tokens - 1] = true;
            }
            ret = llama_decode(lctx, text_batch);
            if (ret != 0) {
                LOG_ERR("failed to decode text\n");
                llama_batch_free(text_batch);
                return ret;
            }
            *new_n_past += text_batch.n_tokens;
        }

    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
        int64_t t0 = ggml_time_ms();

        LOG_INF("encoding %s slice...\n", name);

        ret = mtmd_encode_chunk(ctx, chunk);
        if (ret != 0) {
            LOG_ERR("failed to encode %s slice\n", name);
            llama_batch_free(text_batch);
            return ret;
        }

        LOG_INF("%s slice encoded in %" PRId64 " ms\n", name, ggml_time_ms() - t0);

        float * embd = mtmd_get_output_embd(ctx);
        ret = mtmd_helper_decode_image_chunk(ctx, lctx, chunk, embd, n_past, seq_id, n_batch, new_n_past);
        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            llama_batch_free(text_batch);
            return ret;
        }
    } else {
        GGML_ABORT("chunk type not supported");
    }

    llama_batch_free(text_batch);
    return 0;
}

int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
                                struct llama_context * lctx,
                                const mtmd_input_chunks * chunks,
                                llama_pos n_past,
                                llama_seq_id seq_id,
                                int32_t n_batch,
                                bool logits_last,
                                llama_pos * new_n_past) {
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        LOG_WRN("no chunks to eval\n");
        return 0;
    }

    for (size_t i = 0; i < n_chunks; i++) {
        bool chunk_logits_last = (i == n_chunks - 1) && logits_last;
        auto chunk = mtmd_input_chunks_get(chunks, i);

        int32_t res = mtmd_helper_eval_chunk_single(ctx, lctx, chunk, n_past, seq_id, n_batch, chunk_logits_last, &n_past);
        if (res != 0) {
            LOG_ERR("failed to eval chunk %zu\n", i);
            return res;
        }
        *new_n_past = n_past;
    }

    return 0;
}

namespace audio_helpers {

static bool is_audio_file(const char * buf, size_t len) {
    if (len < 12) {
        return false;
    }

    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    bool is_wav = memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WAVE", 4) == 0;
    bool is_mp3 = len >= 3 && (
        memcmp(buf, "ID3", 3) == 0 ||
        // Check for MPEG sync word (simplified check)
        ((unsigned char)buf[0] == 0xFF && ((unsigned char)buf[1] & 0xE0) == 0xE0)
    );
    bool is_flac = memcmp(buf, "fLaC", 4) == 0;

    return is_wav || is_mp3 || is_flac;
}

// returns true if the buffer is a valid audio file
static bool decode_audio_from_buf(const unsigned char * buf_in, size_t len, int target_sampler_rate, std::vector<float> & pcmf32_mono) {
    ma_result result;
    const int channels = 1;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, target_sampler_rate);
    ma_decoder decoder;

    result = ma_decoder_init_memory(buf_in, len, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        return false;
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    pcmf32_mono.resize(frame_count);
    result = ma_decoder_read_pcm_frames(&decoder, pcmf32_mono.data(), frame_count, &frames_read);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

#ifdef MTMD_AUDIO_DEBUG
    // save audio to wav file
    ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, target_sampler_rate);
    ma_encoder encoder;
    ma_encoder_init_file("output.wav", &config, &encoder);
    ma_encoder_write_pcm_frames(&encoder, pcmf32_mono.data(), pcmf32_mono.size(), &frames_read);
    ma_encoder_uninit(&encoder);
#endif

    ma_decoder_uninit(&decoder);
    return true;
}

} // namespace audio_helpers

#ifdef LLAMA_MTMD_FFMPEG
namespace video_helpers {

struct avformat_input_deleter {
    void operator()(AVFormatContext * ctx) const {
        if (ctx != nullptr) {
            avformat_close_input(&ctx);
        }
    }
};

struct avcodec_context_deleter {
    void operator()(AVCodecContext * ctx) const {
        avcodec_free_context(&ctx);
    }
};

struct avframe_deleter {
    void operator()(AVFrame * frame) const {
        av_frame_free(&frame);
    }
};

struct avpacket_deleter {
    void operator()(AVPacket * packet) const {
        av_packet_free(&packet);
    }
};

struct swscale_context_deleter {
    void operator()(SwsContext * ctx) const {
        sws_freeContext(ctx);
    }
};

struct swresample_context_deleter {
    void operator()(SwrContext * ctx) const {
        swr_free(&ctx);
    }
};

using avformat_input_ptr = std::unique_ptr<AVFormatContext, avformat_input_deleter>;
using avcodec_context_ptr = std::unique_ptr<AVCodecContext, avcodec_context_deleter>;
using avframe_ptr = std::unique_ptr<AVFrame, avframe_deleter>;
using avpacket_ptr = std::unique_ptr<AVPacket, avpacket_deleter>;
using swscale_context_ptr = std::unique_ptr<SwsContext, swscale_context_deleter>;
using swresample_context_ptr = std::unique_ptr<SwrContext, swresample_context_deleter>;

double stream_duration_seconds(const AVFormatContext * fmt_ctx, const AVStream * stream) {
    if (stream->duration > 0) {
        return stream->duration * av_q2d(stream->time_base);
    }
    if (fmt_ctx->duration > 0) {
        return fmt_ctx->duration / (double) AV_TIME_BASE;
    }
    return 0.0;
}

double frame_timestamp_seconds(const AVFrame * frame, const AVStream * stream, int64_t decoded_index) {
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE) {
        return frame->best_effort_timestamp * av_q2d(stream->time_base);
    }
    AVRational fps = stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate : stream->r_frame_rate;
    if (fps.num > 0 && fps.den > 0) {
        return decoded_index / av_q2d(fps);
    }
    return (double) decoded_index;
}

bool add_rgb_frame(mtmd::bitmaps & bitmaps, SwsContext * sws_ctx, AVFrame * src_frame, const std::string & fname, size_t frame_idx) {
    const int width = src_frame->width;
    const int height = src_frame->height;
    const int rgb_buf_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, width, height, 1);
    if (rgb_buf_size <= 0) {
        return false;
    }

    std::vector<uint8_t> rgb_buf(rgb_buf_size);
    uint8_t * dst_data[4] = { nullptr, nullptr, nullptr, nullptr };
    int dst_linesize[4] = { 0, 0, 0, 0 };
    if (av_image_fill_arrays(dst_data, dst_linesize, rgb_buf.data(), AV_PIX_FMT_RGB24, width, height, 1) < 0) {
        return false;
    }

    if (sws_scale(sws_ctx, src_frame->data, src_frame->linesize, 0, height, dst_data, dst_linesize) <= 0) {
        return false;
    }

    mtmd::bitmap bmp(width, height, rgb_buf.data());
    bmp.set_id((fname + "#frame-" + std::to_string(frame_idx)).c_str());
    bitmaps.entries.push_back(std::move(bmp));
    return true;
}

size_t decode_video_frames(const std::string & fname, mtmd::bitmaps & bitmaps) {
    AVFormatContext * raw_fmt_ctx = nullptr;
    if (avformat_open_input(&raw_fmt_ctx, fname.c_str(), nullptr, nullptr) < 0) {
        return 0;
    }
    avformat_input_ptr fmt_ctx(raw_fmt_ctx);
    if (avformat_find_stream_info(fmt_ctx.get(), nullptr) < 0) {
        return 0;
    }

    const int video_stream_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx < 0) {
        return 0;
    }

    AVStream * stream = fmt_ctx->streams[video_stream_idx];
    const AVCodec * codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (codec == nullptr) {
        return 0;
    }

    avcodec_context_ptr codec_ctx(avcodec_alloc_context3(codec));
    if (!codec_ctx) {
        return 0;
    }
    if (avcodec_parameters_to_context(codec_ctx.get(), stream->codecpar) < 0) {
        return 0;
    }
    if (avcodec_open2(codec_ctx.get(), codec, nullptr) < 0) {
        return 0;
    }

    swscale_context_ptr sws_ctx(sws_getContext(
        codec_ctx->width,
        codec_ctx->height,
        codec_ctx->pix_fmt,
        codec_ctx->width,
        codec_ctx->height,
        AV_PIX_FMT_RGB24,
        SWS_BILINEAR,
        nullptr,
        nullptr,
        nullptr));
    if (!sws_ctx) {
        return 0;
    }

    const double duration_sec = stream_duration_seconds(fmt_ctx.get(), stream);
    const size_t max_frames = 8;
    size_t target_frames = 1;
    if (duration_sec > 0.0) {
        target_frames = std::max<size_t>(1, std::min<size_t>(max_frames, (size_t) std::llround(duration_sec)));
    }
    const double sample_interval = target_frames > 1 ? duration_sec / (double) (target_frames - 1) : 0.0;
    double next_sample = sample_interval;

    avpacket_ptr packet(av_packet_alloc());
    avframe_ptr frame(av_frame_alloc());
    if (!packet || !frame) {
        return 0;
    }

    size_t loaded = 0;
    int64_t decoded_index = 0;

    auto handle_frame = [&](AVFrame * cur_frame) {
        const double ts_sec = frame_timestamp_seconds(cur_frame, stream, decoded_index++);
        const bool should_take = loaded == 0 || sample_interval <= 0.0 || ts_sec + 1e-6 >= next_sample;
        if (!should_take) {
            return true;
        }
        if (!add_rgb_frame(bitmaps, sws_ctx.get(), cur_frame, fname, loaded)) {
            return false;
        }
        ++loaded;
        if (sample_interval > 0.0) {
            next_sample += sample_interval;
        }
        return loaded < target_frames;
    };

    while (av_read_frame(fmt_ctx.get(), packet.get()) >= 0) {
        if (packet->stream_index != video_stream_idx) {
            av_packet_unref(packet.get());
            continue;
        }
        if (avcodec_send_packet(codec_ctx.get(), packet.get()) < 0) {
            av_packet_unref(packet.get());
            return loaded;
        }
        av_packet_unref(packet.get());
        while (avcodec_receive_frame(codec_ctx.get(), frame.get()) >= 0) {
            if (!handle_frame(frame.get())) {
                return loaded;
            }
            av_frame_unref(frame.get());
        }
    }

    avcodec_send_packet(codec_ctx.get(), nullptr);
    while (avcodec_receive_frame(codec_ctx.get(), frame.get()) >= 0) {
        if (!handle_frame(frame.get())) {
            break;
        }
        av_frame_unref(frame.get());
    }

    return loaded;
}

bool decode_audio_samples(const std::string & fname, int sample_rate, std::vector<float> & pcmf32) {
    AVFormatContext * raw_fmt_ctx = nullptr;
    if (avformat_open_input(&raw_fmt_ctx, fname.c_str(), nullptr, nullptr) < 0) {
        return false;
    }
    avformat_input_ptr fmt_ctx(raw_fmt_ctx);
    if (avformat_find_stream_info(fmt_ctx.get(), nullptr) < 0) {
        return false;
    }

    const int audio_stream_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (audio_stream_idx < 0) {
        return false;
    }

    AVStream * stream = fmt_ctx->streams[audio_stream_idx];
    const AVCodec * codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (codec == nullptr) {
        return false;
    }

    avcodec_context_ptr codec_ctx(avcodec_alloc_context3(codec));
    if (!codec_ctx) {
        return false;
    }
    if (avcodec_parameters_to_context(codec_ctx.get(), stream->codecpar) < 0) {
        return false;
    }
    if (avcodec_open2(codec_ctx.get(), codec, nullptr) < 0) {
        return false;
    }

    AVChannelLayout out_layout;
    av_channel_layout_default(&out_layout, 1);
    SwrContext * raw_swr_ctx = nullptr;
    if (swr_alloc_set_opts2(
            &raw_swr_ctx,
            &out_layout,
            AV_SAMPLE_FMT_FLT,
            sample_rate,
            &codec_ctx->ch_layout,
            codec_ctx->sample_fmt,
            codec_ctx->sample_rate,
            0,
            nullptr) < 0) {
        av_channel_layout_uninit(&out_layout);
        return false;
    }
    swresample_context_ptr swr_ctx(raw_swr_ctx);
    av_channel_layout_uninit(&out_layout);
    if (!swr_ctx || swr_init(swr_ctx.get()) < 0) {
        return false;
    }

    avpacket_ptr packet(av_packet_alloc());
    avframe_ptr frame(av_frame_alloc());
    if (!packet || !frame) {
        return false;
    }

    auto append_frame = [&](AVFrame * cur_frame) {
        const int dst_nb_samples = av_rescale_rnd(
            swr_get_delay(swr_ctx.get(), codec_ctx->sample_rate) + cur_frame->nb_samples,
            sample_rate,
            codec_ctx->sample_rate,
            AV_ROUND_UP);
        std::vector<float> buffer(dst_nb_samples);
        std::vector<const uint8_t *> input(cur_frame->ch_layout.nb_channels);
        for (int i = 0; i < cur_frame->ch_layout.nb_channels; ++i) {
            input[i] = cur_frame->extended_data[i];
        }
        uint8_t * out[] = { reinterpret_cast<uint8_t *>(buffer.data()) };
        const int out_samples = swr_convert(
            swr_ctx.get(),
            out,
            dst_nb_samples,
            input.data(),
            cur_frame->nb_samples);
        if (out_samples > 0) {
            pcmf32.insert(pcmf32.end(), buffer.begin(), buffer.begin() + out_samples);
        }
        return out_samples >= 0;
    };

    while (av_read_frame(fmt_ctx.get(), packet.get()) >= 0) {
        if (packet->stream_index != audio_stream_idx) {
            av_packet_unref(packet.get());
            continue;
        }
        if (avcodec_send_packet(codec_ctx.get(), packet.get()) < 0) {
            av_packet_unref(packet.get());
            return !pcmf32.empty();
        }
        av_packet_unref(packet.get());
        while (avcodec_receive_frame(codec_ctx.get(), frame.get()) >= 0) {
            if (!append_frame(frame.get())) {
                return !pcmf32.empty();
            }
            av_frame_unref(frame.get());
        }
    }

    avcodec_send_packet(codec_ctx.get(), nullptr);
    while (avcodec_receive_frame(codec_ctx.get(), frame.get()) >= 0) {
        if (!append_frame(frame.get())) {
            return !pcmf32.empty();
        }
        av_frame_unref(frame.get());
    }

    if (swr_get_delay(swr_ctx.get(), codec_ctx->sample_rate) > 0) {
        const int dst_nb_samples = av_rescale_rnd(
            swr_get_delay(swr_ctx.get(), codec_ctx->sample_rate),
            sample_rate,
            codec_ctx->sample_rate,
            AV_ROUND_UP);
        std::vector<float> buffer(dst_nb_samples);
        uint8_t * out[] = { reinterpret_cast<uint8_t *>(buffer.data()) };
        const int out_samples = swr_convert(swr_ctx.get(), out, dst_nb_samples, nullptr, 0);
        if (out_samples > 0) {
            pcmf32.insert(pcmf32.end(), buffer.begin(), buffer.begin() + out_samples);
        }
    }

    return !pcmf32.empty();
}

size_t append_video_file(mtmd_context * ctx, const std::string & fname, mtmd::bitmaps & out) {
    size_t loaded = 0;
    if (mtmd_support_vision(ctx)) {
        loaded += decode_video_frames(fname, out);
    }
    if (mtmd_support_audio(ctx)) {
        const int sample_rate = mtmd_get_audio_bitrate(ctx);
        const int chunk_len = mtmd_get_audio_chunk_len(ctx);
        if (sample_rate > 0) {
            std::vector<float> pcmf32;
            if (decode_audio_samples(fname, sample_rate, pcmf32)) {
                if (chunk_len > 0) {
                    const size_t max_samples = (size_t) sample_rate * (size_t) chunk_len;
                    if (pcmf32.size() > max_samples) {
                        pcmf32.resize(max_samples);
                    }
                }
                mtmd::bitmap bmp(mtmd_bitmap_init_from_audio(pcmf32.size(), pcmf32.data()));
                if (bmp.ptr) {
                    bmp.set_id((fname + "#audio").c_str());
                    out.entries.push_back(std::move(bmp));
                    ++loaded;
                }
            }
        }
    }
    return loaded;
}

} // namespace video_helpers
#endif

mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len) {
    if (audio_helpers::is_audio_file((const char *)buf, len)) {
        std::vector<float> pcmf32;
        int bitrate = mtmd_get_audio_bitrate(ctx);
        if (bitrate < 0) {
            LOG_ERR("This model does not support audio input\n");
            return nullptr;
        }
        if (!audio_helpers::decode_audio_from_buf(buf, len, bitrate, pcmf32)) {
            LOG_ERR("Unable to read WAV audio file from buffer\n");
            return nullptr;
        }
        return mtmd_bitmap_init_from_audio(pcmf32.size(), pcmf32.data());
    }

    // otherwise, we assume it's an image
    mtmd_bitmap * result = nullptr;
    {
        int nx, ny, nc;
        auto * data = stbi_load_from_memory(buf, len, &nx, &ny, &nc, 3);
        if (!data) {
            LOG_ERR("%s: failed to decode image bytes\n", __func__);
            return nullptr;
        }
        result = mtmd_bitmap_init(nx, ny, data);
        stbi_image_free(data);
    }
    return result;
}

mtmd_bitmap * mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname) {
    std::vector<unsigned char> buf;
    FILE * f = fopen(fname, "rb");
    if (!f) {
        LOG_ERR("Unable to open file %s: %s\n", fname, strerror(errno));
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf.resize(file_size);

    size_t n_read = fread(buf.data(), 1, file_size, f);
    fclose(f);
    if (n_read != (size_t)file_size) {
        LOG_ERR("Failed to read entire file %s", fname);
        return nullptr;
    }

    return mtmd_helper_bitmap_init_from_buf(ctx, buf.data(), buf.size());
}

size_t mtmd::helper_bitmaps_append_from_file(mtmd_context * ctx, const char * fname, mtmd::bitmaps & out) {
    mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx, fname));
    if (bmp.ptr) {
        out.entries.push_back(std::move(bmp));
        return 1;
    }

#ifdef LLAMA_MTMD_FFMPEG
    return video_helpers::append_video_file(ctx, fname, out);
#else
    return 0;
#endif
}
