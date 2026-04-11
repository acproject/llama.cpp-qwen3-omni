#include "arg.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-tts.h"

#include <cmath>
#include <memory>
#include <vector>
#include <string>
#include <limits.h>
#include <cinttypes>

#ifdef LLAMA_MTMD_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#endif

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

// volatile, because of signal being an interrupt
static volatile bool g_is_generating = false;
static volatile bool g_is_interrupted = false;

/**
 * Please note that this is NOT a production-ready stuff.
 * It is a playground for trying multimodal support in llama.cpp.
 * For contributors: please keep this code simple and easy to understand.
 */

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Experimental CLI for multimodal\n\n"
        "Usage: %s [options] -m <model> --mmproj <mmproj> --image <image> --audio <audio> --video <video> -p <prompt>\n\n"
        "  -m and --mmproj are required\n"
        "  -hf user/repo can replace both -m and --mmproj in most cases\n"
        "  --image, --audio, --video and -p are optional, if NOT provided, the CLI will run in chat mode\n"
        "  to disable using GPU for mmproj model, add --no-mmproj-offload\n\n"
        "TTS options (Qwen3-Omni):\n"
        "  --tts-model <path>   Path to Talker model for speech synthesis\n"
        "  --tts-output <path>  Default output WAV file (default: output.wav)\n"
        "  --speak              Auto-generate speech in non-interactive mode\n"
        "  Use /speak command in chat mode to generate speech from last response\n",
        argv[0]
    );
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_generating) {
            g_is_generating = false;
        } else {
            console::cleanup();
            if (g_is_interrupted) {
                _exit(1);
            }
            g_is_interrupted = true;
        }
    }
}
#endif

static void append_media_markers(std::string & text, size_t n_markers) {
    for (size_t i = 0; i < n_markers; ++i) {
        text = mtmd_default_marker() + text;
    }
}

#ifdef LLAMA_MTMD_FFMPEG
namespace {

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

static double ffmpeg_stream_duration_seconds(const AVFormatContext * fmt_ctx, const AVStream * stream) {
    if (stream->duration > 0) {
        return stream->duration * av_q2d(stream->time_base);
    }
    if (fmt_ctx->duration > 0) {
        return fmt_ctx->duration / (double) AV_TIME_BASE;
    }
    return 0.0;
}

static double ffmpeg_frame_timestamp_seconds(const AVFrame * frame, const AVStream * stream, int64_t decoded_index) {
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE) {
        return frame->best_effort_timestamp * av_q2d(stream->time_base);
    }
    AVRational fps = stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate : stream->r_frame_rate;
    if (fps.num > 0 && fps.den > 0) {
        return decoded_index / av_q2d(fps);
    }
    return (double) decoded_index;
}

static bool ffmpeg_add_rgb_frame(mtmd::bitmaps & bitmaps, SwsContext * sws_ctx, AVFrame * src_frame, const std::string & fname, size_t frame_idx) {
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

static size_t ffmpeg_decode_video_frames(const std::string & fname, mtmd::bitmaps & bitmaps) {
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

    const double duration_sec = ffmpeg_stream_duration_seconds(fmt_ctx.get(), stream);
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
        const double ts_sec = ffmpeg_frame_timestamp_seconds(cur_frame, stream, decoded_index++);
        const bool should_take = loaded == 0 || sample_interval <= 0.0 || ts_sec + 1e-6 >= next_sample;
        if (!should_take) {
            return true;
        }
        if (!ffmpeg_add_rgb_frame(bitmaps, sws_ctx.get(), cur_frame, fname, loaded)) {
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

static bool ffmpeg_decode_audio_samples(const std::string & fname, int sample_rate, std::vector<float> & pcmf32) {
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

static size_t ffmpeg_load_video_media(mtmd_context * ctx, const std::string & fname, mtmd::bitmaps & bitmaps) {
    size_t loaded = 0;
    if (mtmd_support_vision(ctx)) {
        loaded += ffmpeg_decode_video_frames(fname, bitmaps);
    }
    if (mtmd_support_audio(ctx)) {
        const int sample_rate = mtmd_get_audio_bitrate(ctx);
        const int chunk_len = mtmd_get_audio_chunk_len(ctx);
        if (sample_rate > 0) {
            std::vector<float> pcmf32;
            if (ffmpeg_decode_audio_samples(fname, sample_rate, pcmf32)) {
                if (chunk_len > 0) {
                    const size_t max_samples = (size_t) sample_rate * (size_t) chunk_len;
                    if (pcmf32.size() > max_samples) {
                        pcmf32.resize(max_samples);
                    }
                }
                mtmd::bitmap bmp(mtmd_bitmap_init_from_audio(pcmf32.size(), pcmf32.data()));
                if (bmp.ptr) {
                    bmp.set_id((fname + "#audio").c_str());
                    bitmaps.entries.push_back(std::move(bmp));
                    ++loaded;
                }
            }
        }
    }
    return loaded;
}

}
#endif

struct mtmd_cli_context {
    mtmd::context_ptr ctx_vision;
    common_init_result_ptr llama_init;

    llama_model       * model;
    llama_context     * lctx;
    const llama_vocab * vocab;
    common_sampler    * smpl;
    llama_batch         batch;
    int                 n_batch;

    mtmd::bitmaps bitmaps;

    // chat template
    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;
    bool use_jinja = false;
    // TODO: support for --system-prompt with /clear command

    // support for legacy templates (models not having EOT token)
    llama_tokens antiprompt_tokens;

    // TTS support (Qwen3-Omni)
    llama_model       * tts_model = nullptr;
    mtmd_tts_context  * tts_ctx = nullptr;
    std::string         tts_output_path = "output.wav";
    std::string         last_response;  // for /speak command

    int n_threads    = 1;
    llama_pos n_past = 0;

    mtmd_cli_context(common_params & params) : llama_init(common_init_from_params(params)) {
        model = llama_init->model();
        lctx = llama_init->context();
        vocab = llama_model_get_vocab(model);
        smpl = common_sampler_init(model, params.sampling);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(1, 0, 1); // batch for next token generation
        n_batch = params.n_batch;

        if (!model || !lctx) {
            exit(1);
        }

        if (!llama_model_chat_template(model, nullptr) && params.chat_template.empty()) {
            LOG_ERR("Model does not have chat template.\n");
            LOG_ERR("  For old llava models, you may need to use '--chat-template vicuna'\n");
            LOG_ERR("  For MobileVLM models, use '--chat-template deepseek'\n");
            LOG_ERR("  For Mistral Small 3.1, use '--chat-template mistral-v7'\n");
            exit(1);
        }

        tmpls = common_chat_templates_init(model, params.chat_template);
        use_jinja = params.use_jinja;
        chat_history.clear();
        LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(tmpls.get(), params.use_jinja, params.default_template_kwargs).c_str());

        init_vision_context(params);

        // load antiprompt tokens for legacy templates
        if (params.chat_template == "vicuna") {
            antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
        } else if (params.chat_template == "deepseek") {
            antiprompt_tokens = common_tokenize(lctx, "###", false, true);
        }
    }

    ~mtmd_cli_context() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
        if (tts_ctx) {
            mtmd_tts_free(tts_ctx);
        }
        if (tts_model) {
            llama_model_free(tts_model);
        }
    }

    bool init_tts(const std::string & tts_model_path, int n_gpu_layers = 99) {
        if (tts_model_path.empty()) {
            return true;  // TTS not requested
        }

        LOG_INF("Loading TTS model: %s\n", tts_model_path.c_str());

        llama_model_params tts_mparams = llama_model_default_params();
        tts_mparams.n_gpu_layers = n_gpu_layers;

        tts_model = llama_model_load_from_file(tts_model_path.c_str(), tts_mparams);
        if (!tts_model) {
            LOG_ERR("Failed to load TTS model\n");
            return false;
        }

        // Check if TTS is supported
        if (!mtmd_tts_supported(tts_model)) {
            LOG_ERR("TTS model does not have required tensors (Code2Wav)\n");
            llama_model_free(tts_model);
            tts_model = nullptr;
            return false;
        }

        mtmd_tts_params tts_params = mtmd_tts_params_default();
        tts_params.verbose = false;

        tts_ctx = mtmd_tts_init(model, tts_model, tts_params);
        if (!tts_ctx) {
            LOG_ERR("Failed to initialize TTS context\n");
            llama_model_free(tts_model);
            tts_model = nullptr;
            return false;
        }

        LOG_INF("TTS initialized successfully\n");
        return true;
    }

    bool generate_speech(const std::string & output_path) {
        if (!tts_ctx) {
            LOG_ERR("TTS not initialized. Use --tts-model to enable.\n");
            return false;
        }

        if (last_response.empty()) {
            LOG_ERR("No response to speak. Generate a response first.\n");
            return false;
        }

        LOG_INF("Generating speech for: \"%s\"\n", last_response.c_str());

        // Allocate output buffer
        int max_samples = mtmd_tts_estimate_samples(500);  // Max 500 codec tokens
        std::vector<float> audio_samples(max_samples);

        // Generate speech using convenience function (handles tokenization and Thinker inference)
        int n_samples = mtmd_tts_generate_from_text(tts_ctx, lctx, last_response.c_str(),
                                                    audio_samples.data(), max_samples);

        if (n_samples <= 0) {
            LOG_ERR("Failed to generate speech\n");
            return false;
        }

        // Write WAV file
        if (!mtmd_tts_write_wav(output_path.c_str(), audio_samples.data(), n_samples, 24000)) {
            LOG_ERR("Failed to write WAV file\n");
            return false;
        }

        LOG_INF("Saved speech to: %s (%.2f sec)\n", output_path.c_str(), n_samples / 24000.0f);
        return true;
    }

    void init_vision_context(common_params & params) {
        const char * clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = params.mmproj_use_gpu;
        mparams.print_timings    = true;
        mparams.n_threads        = params.cpuparams.n_threads;
        mparams.flash_attn_type  = params.flash_attn_type;
        mparams.warmup           = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        ctx_vision.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
            exit(1);
        }
    }

    bool check_antiprompt(const llama_tokens & generated_tokens) {
        if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
            return false;
        }
        return std::equal(
            generated_tokens.end() - antiprompt_tokens.size(),
            generated_tokens.end(),
            antiprompt_tokens.begin()
        );
    }

    size_t load_media(const std::string & fname) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), fname.c_str()));
        if (!bmp.ptr) {
#ifdef LLAMA_MTMD_FFMPEG
            size_t n_loaded = ffmpeg_load_video_media(ctx_vision.get(), fname, bitmaps);
            if (n_loaded > 0) {
                return n_loaded;
            }
#endif
            return 0;
        }
        bitmaps.entries.push_back(std::move(bmp));
        return 1;
    }
};

static int generate_response(mtmd_cli_context & ctx, int n_predict) {
    llama_tokens generated_tokens;
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating || g_is_interrupted) {
            LOG("\n");
            break;
        }

        llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id) || ctx.check_antiprompt(generated_tokens)) {
            LOG("\n");
            break; // end of generation
        }

        LOG("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
        fflush(stdout);

        if (g_is_interrupted) {
            LOG("\n");
            break;
        }

        // eval the token
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
    }

    std::string generated_text = common_detokenize(ctx.lctx, generated_tokens);
    common_chat_msg msg;
    msg.role    = "assistant";
    msg.content = generated_text;
    ctx.chat_history.push_back(std::move(msg));

    // Save for TTS /speak command
    ctx.last_response = generated_text;

    return 0;
}

static std::string chat_add_and_format(mtmd_cli_context & ctx, common_chat_msg & new_msg) {
    LOG_DBG("chat_add_and_format: new_msg.role='%s', new_msg.content='%s'\n",
        new_msg.role.c_str(), new_msg.content.c_str());
    auto formatted = common_chat_format_single(ctx.tmpls.get(), ctx.chat_history,
        new_msg, new_msg.role == "user",
        ctx.use_jinja);
    ctx.chat_history.push_back(new_msg);
    return formatted;
}

static int eval_message(mtmd_cli_context & ctx, common_chat_msg & msg) {
    bool add_bos = ctx.chat_history.empty();
    auto formatted_chat = chat_add_and_format(ctx, msg);
    LOG_DBG("formatted_chat.prompt: %s\n", formatted_chat.c_str());

    mtmd_input_text text;
    text.text          = formatted_chat.c_str();
    text.add_special   = add_bos;
    text.parse_special = true;

    if (g_is_interrupted) return 0;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx.bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx.ctx_vision.get(),
                        chunks.ptr.get(), // output
                        &text, // text
                        bitmaps_c_ptr.data(),
                        bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERR("Unable to tokenize prompt, res = %d\n", res);
        return 1;
    }

    ctx.bitmaps.entries.clear();

    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(ctx.ctx_vision.get(),
                ctx.lctx, // lctx
                chunks.ptr.get(), // chunks
                ctx.n_past, // n_past
                0, // seq_id
                ctx.n_batch, // n_batch
                true, // logits_last
                &new_n_past)) {
        LOG_ERR("Unable to eval prompt\n");
        return 1;
    }

    ctx.n_past = new_n_past;

    LOG("\n");

    return 0;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    // Pre-parse TTS arguments and filter them from argv (not handled by common_params)
    std::string tts_model_path;
    std::string tts_output_path = "output.wav";
    bool auto_speak = false;  // Auto-generate speech in non-interactive mode
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tts-model" && i + 1 < argc) {
            tts_model_path = argv[++i];
        } else if (arg == "--tts-output" && i + 1 < argc) {
            tts_output_path = argv[++i];
        } else if (arg == "--speak") {
            auto_speak = true;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    int filtered_argc = (int)filtered_argv.size();

    common_params params;

    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    common_init();
    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.mmproj.path.empty()) {
        show_additional_info(argc, argv);
        LOG_ERR("ERR: Missing --mmproj argument\n");
        return 1;
    }

    mtmd_cli_context ctx(params);
    LOG_INF("%s: loading model: %s\n", __func__, params.model.path.c_str());

    // Initialize TTS if model specified
    ctx.tts_output_path = tts_output_path;
    if (!tts_model_path.empty()) {
        if (!ctx.init_tts(tts_model_path, params.n_gpu_layers)) {
            LOG_WRN("TTS initialization failed, /speak command will be unavailable\n");
        }
    }

    // Single-turn mode: prompt + (media OR auto-speak TTS)
    // This allows text-only TTS when --speak is provided without image/audio
    bool is_single_turn = !params.prompt.empty() && (!params.image.empty() || auto_speak);

    int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

    // Ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (g_is_interrupted) return 130;

    auto eval_system_prompt_if_present = [&] {
        if (params.system_prompt.empty()) {
            return 0;
        }

        common_chat_msg msg;
        msg.role = "system";
        msg.content = params.system_prompt;
        return eval_message(ctx, msg);
    };

    LOG_WRN("WARN: This is an experimental CLI for testing multimodal capability.\n");
    LOG_WRN("      For normal use cases, please use the standard llama-cli\n");

    if (eval_system_prompt_if_present()) {
        return 1;
    }

    if (is_single_turn) {
        g_is_generating = true;
        size_t n_loaded_media = 0;
        for (const auto & image : params.image) {
            size_t n_loaded = ctx.load_media(image);
            if (n_loaded == 0) {
                return 1; // error is already printed by libmtmd
            }
            n_loaded_media += n_loaded;
        }
        if (params.prompt.find(mtmd_default_marker()) == std::string::npos) {
            append_media_markers(params.prompt, n_loaded_media);
        }

        common_chat_msg msg;
        msg.role = "user";
        msg.content = params.prompt;
        if (eval_message(ctx, msg)) {
            return 1;
        }
        if (!g_is_interrupted && generate_response(ctx, n_predict)) {
            return 1;
        }

        // Auto-speak if --speak flag was provided
        if (auto_speak && ctx.tts_ctx && !g_is_interrupted) {
            ctx.generate_speech(tts_output_path);
        }

    } else {
        LOG("\n Running in chat mode, available commands:");
        if (mtmd_support_vision(ctx.ctx_vision.get())) {
            LOG("\n   /image <path>    load an image");
        }
        if (mtmd_support_audio(ctx.ctx_vision.get())) {
            LOG("\n   /audio <path>    load an audio");
        }
#ifdef LLAMA_MTMD_FFMPEG
        if (mtmd_support_vision(ctx.ctx_vision.get()) || mtmd_support_audio(ctx.ctx_vision.get())) {
            LOG("\n   /video <path>    load a video");
        }
#endif
        if (ctx.tts_ctx) {
            LOG("\n   /speak [path]    generate speech from last response (default: output.wav)");
        }
        LOG("\n   /clear           clear the chat history");
        LOG("\n   /quit or /exit   exit the program");
        LOG("\n");

        std::string content;

        while (!g_is_interrupted) {
            g_is_generating = false;
            LOG("\n> ");
            console::set_display(DISPLAY_TYPE_USER_INPUT);
            std::string line;
            console::readline(line, false);
            if (g_is_interrupted) break;
            console::set_display(DISPLAY_TYPE_RESET);
            line = string_strip(line);
            if (line.empty()) {
                continue;
            }
            if (line == "/quit" || line == "/exit") {
                break;
            }
            if (line == "/clear") {
                ctx.n_past = 0;
                ctx.chat_history.clear();
                llama_memory_clear(llama_get_memory(ctx.lctx), true);
                if (eval_system_prompt_if_present()) {
                    return 1;
                }
                LOG("Chat history cleared\n\n");
                continue;
            }
            g_is_generating = true;

            // TTS: /speak command
            bool is_speak = line == "/speak" || line.find("/speak ") == 0;
            if (is_speak) {
                std::string output_path = ctx.tts_output_path;
                if (line.size() > 7) {
                    output_path = line.substr(7);
                }
                ctx.generate_speech(output_path);
                continue;
            }

            bool is_image = line == "/image" || line.find("/image ") == 0;
            bool is_audio = line == "/audio" || line.find("/audio ") == 0;
            bool is_video = line == "/video" || line.find("/video ") == 0;
            if (is_image || is_audio || is_video) {
                if (line.size() < 8) {
                    LOG_ERR("ERR: Missing media filename\n");
                    continue;
                }
                std::string media_path = line.substr(7);
                size_t n_loaded = ctx.load_media(media_path);
                if (n_loaded > 0) {
                    const char * media_type = is_image ? "image" : (is_audio ? "audio" : "video");
                    LOG("%s %s loaded\n", media_path.c_str(), media_type);
                    append_media_markers(content, n_loaded);
                }
                // else, error is already printed by libmtmd
                continue;
            } else {
                content += line;
            }
            common_chat_msg msg;
            msg.role = "user";
            msg.content = content;
            int ret = eval_message(ctx, msg);
            if (ret) {
                return 1;
            }
            if (g_is_interrupted) break;
            if (generate_response(ctx, n_predict)) {
                return 1;
            }
            content.clear();
        }
    }
    if (g_is_interrupted) LOG("\nInterrupted by user\n");
    LOG("\n\n");
    llama_perf_context_print(ctx.lctx);
    return g_is_interrupted ? 130 : 0;
}
