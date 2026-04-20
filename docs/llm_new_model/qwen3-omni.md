# Qwen3-Omni 多模态推理能力实现总结

## 1. 项目概述

本项目在 llama.cpp 基础上实现了 Qwen3-Omni 的真实多模态推理能力，支持文本、图像、音频、视频等多种模态输入，并实现了语音输出（TTS）功能。

### 核心功能

- **多模态输入支持**：文本、图像、音频、视频
- **语音输出**：Text-to-Speech (TTS) 功能
- **HTTP 服务**：OpenAI 兼容的 REST API
- **命令行工具**：`llama-mtmd-cli` 用于本地推理

### 技术栈

- **llama.cpp**：C++ 实现的 LLM 推理引擎
- **libmtmd**：多模态处理库（vision/audio encoder）
- **GGUF**：llama.cpp 的模型格式
- **FFmpeg**：音视频处理库
- **Python**：模型转换和工具脚本

## 2. 环境配置

### Python 环境

使用虚拟环境 `/home/acproject/workspace/python_projects/HEX`：

```bash
source /home/acproject/workspace/python_projects/HEX/bin/activate
```

### 系统依赖

- **FFmpeg**：已安装，用于视频解码和音频提取
- **CMake 3.14+**：构建系统
- **C++17 编译器**：GCC 13.3+ 或 Clang

### 模型文件

模型路径：`/home/acproject/gguf_models/Qwen3-omni-30B-Q8_0`

包含以下 GGUF 文件：
- `qwen3-omni-30B-Q8_0.gguf`：主模型（thinker）
- `mmproj-qwen3-omni-30B-F16-fixed.gguf`：多模态投影器
- `talker-qwen3-omni-30B-F16-from-hf.gguf`：语音合成模型（TTS）

## 3. 构建与安装

### 基础构建

```bash
cmake -B build
cmake --build build --config Release -j $(nproc)
```

### 启用 FFmpeg（视频支持）

```bash
cmake -B build -DFFMPEG_LIBRARIES="avcodec;avformat;avutil;swscale;swresample"
cmake --build build --config Release -j $(nproc)
```

### 构建结果

构建完成后，可执行文件位于 `build/bin/`：
- `llama-mtmd-cli`：多模态 CLI 工具
- `llama-server`：HTTP 服务

## 4. 模型转换

### Qwen3-Omni 模型转换

从 Hugging Face 下载的原始模型转换为 GGUF 格式：

```bash
python convert_hf_to_gguf.py /path/to/hf/model \
  --outfile /path/to/output.gguf \
  --outtype Q8_0
```

### Talker (TTS) 模型转换

```bash
python convert_hf_to_gguf.py /path/to/hf/model \
  --talker \
  --outfile /path/to/talker.gguf \
  --outtype f16
```

### mmproj 转换

```bash
python convert_hf_to_gguf.py /path/to/hf/model \
  --mmproj \
  --outfile /path/to/mmproj.gguf \
  --outtype f16
```

## 5. 命令行工具使用

### 基础文本推理

```bash
./build/bin/llama-mtmd-cli \
  -m /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/qwen3-omni-30B-Q8_0.gguf \
  -mmproj /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/mmproj-qwen3-omni-30B-F16-fixed.gguf \
  -p "Hello, how are you?"
```

### 图像推理

```bash
./build/bin/llama-mtmd-cli \
  -m /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/qwen3-omni-30B-Q8_0.gguf \
  -mmproj /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/mmproj-qwen3-omni-30B-F16-fixed.gguf \
  -img /path/to/image.jpg \
  -p "Describe this image."
```

### 音频推理

```bash
./build/bin/llama-mtmd-cli \
  -m /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/qwen3-omni-30B-Q8_0.gguf \
  -mmproj /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/mmproj-qwen3-omni-30B-F16-fixed.gguf \
  -aud /home/acproject/workspace/test_data/intro-disclaimers/LibriSpeech/intro/14/14-212.flac \
  -p "Transcribe this audio."
```

### 视频推理

```bash
./build/bin/llama-mtmd-cli \
  -m /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/qwen3-omni-30B-Q8_0.gguf \
  -mmproj /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/mmproj-qwen3-omni-30B-F16-fixed.gguf \
  -video /home/acproject/workspace/test_data/videos/1.mp4 \
  -p "Describe this video."
```

### 语音输出（TTS）

```bash
./build/bin/llama-mtmd-cli \
  -m /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/qwen3-omni-30B-Q8_0.gguf \
  -mmproj /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/mmproj-qwen3-omni-30B-F16-fixed.gguf \
  --talker /home/acproject/gguf_models/hf_to_gguf/talker-qwen3-omni-30B-F16-from-hf.gguf \
  -p "This is a text to speech test." \
  --out-audio /path/to/output.wav
```

## 6. HTTP 服务使用

### 启动服务

```bash
./build/bin/llama-server \
  -m /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/qwen3-omni-30B-Q8_0.gguf \
  -mmproj /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/mmproj-qwen3-omni-30B-F16-fixed.gguf \
  --model-vocoder /home/acproject/gguf_models/hf_to_gguf/talker-qwen3-omni-30B-F16-from-hf.gguf \
  -c 8192 \
  --port 18083
```

### 语音输出接口

#### 独立 TTS 接口

```bash
curl http://127.0.0.1:18083/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a text to speech test.",
    "speaker_id": 2301
  }' \
  --output speech.wav
```

#### Chat 接口 + 语音输出（speech 对象）

```bash
curl http://127.0.0.1:18083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "speech": {
      "response_format": "wav",
      "speaker_id": 2301
    }
  }'
```

#### Chat 接口 + 语音输出（modalities）

```bash
curl http://127.0.0.1:18083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "modalities": ["text", "audio"],
    "audio": {
      "format": "wav",
      "speaker_id": 2302
    }
  }'
```

### 多模态输入接口

#### 图像输入

```bash
curl http://127.0.0.1:18083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}},
          {"type": "text", "text": "Describe this image."}
        ]
      }
    ]
  }'
```

#### 音频输入

```bash
curl http://127.0.0.1:18083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "audio_url", "audio_url": {"url": "file:///path/to/audio.flac"}},
          {"type": "text", "text": "Transcribe this audio."}
        ]
      }
    ]
  }'
```

#### 视频输入

```bash
curl http://127.0.0.1:18083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "video_url", "video_url": {"url": "file:///path/to/video.mp4"}},
          {"type": "text", "text": "Describe this video."}
        ]
      }
    ]
  }'
```

## 7. Speaker ID 列表

| Speaker ID | Voice Name | Gender |
|------------|------------|--------|
| 2301       | chelsie    | female |
| 2302       | ethan      | male   |
| 2303       | aiden      | male   |

## 8. 技术实现细节

### 多模态处理流程

1. **输入预处理**：
   - 文本：分词（tokenizer）
   - 图像：CLIP vision encoder → projection
   - 音频：CLIP audio encoder → projection
   - 视频：逐帧图像处理 + 音频提取

2. **特征融合**：
   - 将多模态特征与文本特征拼接
   - 输入到 thinker transformer

3. **推理**：
   - Thinker transformer 生成文本响应
   - TTS 模型生成语音输出（可选）

### TTS 实现

#### 模型组件

1. **Text Projection MLP**：文本特征投影
2. **Talker**：20 层 MoE transformer（128 个专家）
3. **Code Predictor**：5 层 transformer + 15 个 lm heads
4. **Code2Wav**：HiFi-GAN vocoder

#### 量化支持

修复了量化 tensor 读取问题，支持 Q8_0 等量化格式：

```cpp
static bool read_tensor_row_as_float(const ggml_tensor * tensor, int64_t row, int64_t ncols, float * dst) {
    if (ggml_is_quantized(tensor->type)) {
        const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
        if (traits && traits->to_float) {
            traits->to_float(raw.data(), dst, ncols);
            return true;
        }
    }
    // ... F32/F16 handling
}
```

### WAV 格式封装

```cpp
std::string build_wav_audio(const float * samples, int n_samples, int sample_rate) {
    // RIFF/WAVE container format
    // Header: RIFF + size + WAVE + fmt + data
}
```

## 9. 文件修改清单

### 核心文件

- **tools/mtmd/mtmd-tts.cpp**：添加量化 tensor 读取支持
- **tools/server/server-context.h**：添加 `has_out_audio` 字段
- **tools/server/server-context.cpp**：实现 TTS 接口
- **tools/server/server.cpp**：注册语音输出路由
- **tools/mtmd/mtmd.h/.cpp**：添加 `mtmd_get_audio_chunk_len()` 函数

### 文档文件

- **docs/llm_new_model/qwen3-omni.md**：本文档
- **docs/qwen3-omni-server-tts-summary.md**：TTS 实现详细总结

## 10. 常见问题

### Q1: 如何验证模型是否正确转换？

```bash
python - <<'PY'
from gguf import GGUFReader
p = '/path/to/model.gguf'
r = GGUFReader(p)
print('fields:', len(r.fields), 'tensors:', len(r.tensors))
for k in list(r.fields)[:80]:
    print(k)
PY
```

### Q2: 如何处理长视频？

长视频会被自动截断为模型允许的最大音频长度（通过 `mtmd_get_audio_chunk_len()` 控制）。

### Q3: 如何切换语音风格？

使用不同的 `speaker_id`：
- 2301: chelsie (female)
- 2302: ethan (male)
- 2303: aiden (male)

### Q4: 如何禁用语音输出？

不传递 `--model-vocoder` 参数，或在请求中不包含 `speech`/`modalities: ["audio"]`。

## 11. 下一步工作

- [ ] 支持更多语音输出格式（如 mp3）
- [ ] 支持流式语音输出
- [ ] 优化长文本 TTS 性能
- [ ] 添加更多 speaker voice 选项
- [ ] 支持语音识别（ASR）输入

## 12. 参考资源

- [llama.cpp 官方文档](https://github.com/ggerganov/llama.cpp)
- [Qwen3-Omni Hugging Face](https://huggingface.co/Qwen/Qwen3-Omni)
- [GGUF 格式规范](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
