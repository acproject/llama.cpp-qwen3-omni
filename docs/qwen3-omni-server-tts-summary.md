# Qwen3-Omni Server TTS 集成总结

## 已完成

- 已使用 HEX 虚拟环境将 Qwen3-Omni 的 talker/vocoder 从 HF 权重成功转换成 GGUF：
  - `/home/acproject/gguf_models/hf_to_gguf/talker-qwen3-omni-30B-F16-from-hf.gguf`
- `llama-server` 现已支持通过 `--model-vocoder` 加载 talker 模型，并提供可用的语音输出能力。
- 已补齐 `/v1/chat/completions` 的“回复自动转语音”联动能力。

## 转换结果

- 使用仓库内已有的 `--talker` 转换入口完成导出。
- 源目录：
  - `/home/acproject/gguf_models/hf_to_gguf`
- 输出文件：
  - `/home/acproject/gguf_models/hf_to_gguf/talker-qwen3-omni-30B-F16-from-hf.gguf`
- 实际转换命令：

```bash
source /home/acproject/workspace/python_projects/HEX/.venv/bin/activate
python convert_hf_to_gguf.py /home/acproject/gguf_models/hf_to_gguf \
  --talker \
  --outfile /home/acproject/gguf_models/hf_to_gguf/talker-qwen3-omni-30B-F16-from-hf.gguf \
  --outtype f16
```

## 代码改动

### Server 侧

- server 启动时支持加载 `--model-vocoder`，并初始化专用 TTS thinker context。
- `/props` 增加 `modalities.speech`，用于标识当前实例是否支持语音输出。
- 新增独立接口：
  - `POST /audio/speech`
  - `POST /v1/audio/speech`
- 新增 chat 自动转语音联动，支持两种请求方式：
  - `speech` 对象
  - `modalities: ["text", "audio"] + audio`

### TTS 兼容修复

- 修复 Q8_0 thinker 主模型在 TTS 路径下无法读取量化 `tok_embd` 的问题。
- 新增量化 embedding 行读取与反量化逻辑，使 Qwen3-Omni Q8_0 thinker 可以直接参与 TTS。

## 使用方法

### 启动 server

```bash
./build/bin/llama-server \
  -m /home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/qwen3-omni-30B-Q8_0.gguf \
  --mmproj /home/acproject/gguf_models/hf_to_gguf/mmproj-qwen3-omni-30B-F16-from-hf.gguf \
  --model-vocoder /home/acproject/gguf_models/hf_to_gguf/talker-qwen3-omni-30B-F16-from-hf.gguf \
  --host 127.0.0.1 --port 18083 \
  --ctx-size 4096 \
  --media-path /home/acproject/workspace/test_data/
```

### 文本转语音

```bash
curl http://127.0.0.1:18083/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "你好，这是一个语音输出测试。",
    "voice": "chelsie",
    "response_format": "wav"
  }' \
  --output speech.wav
```

### Chat 回复自动带语音

#### 方式一：使用 `speech`

```bash
curl http://127.0.0.1:18083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "请用一句话介绍你自己。"}
    ],
    "max_tokens": 24,
    "stream": false,
    "speech": {
      "voice": "ethan",
      "response_format": "wav"
    }
  }'
```

#### 方式二：使用 `modalities + audio`

```bash
curl http://127.0.0.1:18083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "请用一句话打个招呼。"}
    ],
    "max_tokens": 16,
    "stream": false,
    "modalities": ["text", "audio"],
    "audio": {
      "voice": "aiden",
      "format": "wav"
    }
  }'
```

## 验证结果

- `/v1/audio/speech` 实测返回 `200`
- 生成 WAV 文件属性：
  - mono
  - 24000 Hz
  - 约 4.891 秒
- `/v1/chat/completions + speech` 实测返回 `200`
  - 返回文本内容
  - 同时返回 `message.audio.data` 的 base64 WAV
  - 解码后音频约 7.907 秒
- `/v1/chat/completions + modalities:["text","audio"]` 实测返回 `200`

## 构建与测试

- 构建通过：

```bash
cmake --build . --config Release -j $(nproc) --target mtmd llama-server llama-mtmd-cli
cmake --build . --config Release -j $(nproc) --target llama-server
```

- CTest 已执行。当前可见失败项为网络下载相关测试：
  - `test-eval-callback`
  - `test-state-restore-fragmented`
- 失败原因是测试环境无法从 Hugging Face 拉取测试模型，不是本次改动引入的回归。

## 当前状态

- 当前 server 已在 `http://127.0.0.1:18083` 启动并可用。
- 当前 multimodal 能力：
  - `vision`
  - `audio`
  - `video`
  - `speech`

## 备注

- 本次改动涉及 `tools/server` 与 TTS 相关实现。
- 如果后续发 PR，建议按仓库要求在提交信息中披露 AI 参与，例如：

```text
[AI] add Qwen3-Omni server TTS and chat speech output
```
