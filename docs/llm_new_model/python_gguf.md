### 使用 python 分析 GGUF 模型

GGUF (GGML Universal Format) 是 llama.cpp 项目使用的模型文件格式。通过 Python 的 `gguf` 库，我们可以读取和分析 GGUF 模型文件的结构和内容。

#### GGUFReader 类简介

`GGUFReader` 是 `gguf` 库提供的主要类，用于读取 GGUF 文件：
- `r.fields`: 字典，包含模型的元数据字段（如模型类型、维度、词表大小等）
- `r.tensors`: 列表，包含模型的所有权重张量
- `f.parts`: 字段的数据部分，用于读取具体的元数据值

#### 基本使用示例

```py
python - <<'PY'
from gguf import GGUFReader
p='/home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/mmproj-qwen3-omni-30B-F16-fixed.gguf'
r=GGUFReader(p)
print('fields:', len(r.fields), 'tensors:', len(r.tensors))
for k in list(r.fields)[:80]:
    f=r.fields[k]
    try:
        vals=f.parts[f.data[0]:f.data[-1]+1] if hasattr(f,'data') and f.data is not None and len(f.data) else None
    except Exception:
        vals=None
    print(k)
print('first tensors:', [t.name for t in r.tensors[:40]])
PY
```

**代码说明：**
- 第 3-4 行：导入 `GGUFReader` 类并指定 GGUF 文件路径
- 第 5 行：创建读取器实例，加载整个 GGUF 文件
- 第 6 行：输出元数据字段数量和权重张量数量
- 第 7-13 行：遍历前 80 个元数据字段，尝试提取字段值并打印字段名
- 第 14 行：输出前 40 个权重张量的名称

#### 简化版本

```py
python3 - <<'PY'
import sys
sys.path.insert(0, '.')
from gguf import GGUFReader
p='/home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/mmproj-qwen3-omni-30B-F16-fixed.gguf'
r=GGUFReader(p)
print('fields:', len(r.fields), 'tensors:', len(r.tensors))
for k in list(r.fields)[:60]:
    print(k)
print('first tensors:', [t.name for t in r.tensors[:40]])
PY
```

**代码说明：**
- 第 2-3 行：将当前目录添加到 Python 路径，确保能找到本地安装的 `gguf` 模块
- 第 8-10 行：简化版，直接打印前 60 个元数据字段名
- 这个版本省略了字段值的提取逻辑，更适合快速查看模型结构

#### 指定 gguf-py 路径版本

```py
python3 - <<'PY'
import sys
sys.path.insert(0, '/home/acproject/workspace/cpp_projects/llama.cpp-qwen3-omni/gguf-py')
from gguf import GGUFReader
p='/home/acproject/gguf_models/Qwen3-omni-30B-Q8_0/mmproj-qwen3-omni-30B-F16-fixed.gguf'
r=GGUFReader(p)
print('fields:', len(r.fields), 'tensors:', len(r.tensors))
print('first 80 keys:')
for k in list(r.fields)[:80]:
    print(k)
print('sample tensors:')
for t in r.tensors[:80]:
    print(t.name)
PY
```

**代码说明：**
- 第 3 行：明确指定 `gguf-py` 库的路径，适用于未安装到系统 Python 的情况
- 第 9-11 行：打印前 80 个元数据字段名
- 第 12-14 行：逐行打印前 80 个权重张量的名称，便于查看完整的模型结构

#### 典型输出示例

运行上述代码后，你会看到类似以下的输出：

```
fields: 45 tensors: 320
general.architecture
general.name
general.file_type
llama.embedding_length
llama.block_count
llama.feed_forward_length
llama.attention.head_count
llama.attention.layer_norm_rms_epsilon
tokenizer.ggml.tokens
tokenizer.ggml.scores
...
first tensors: ['token_embd.weight', 'blk.0.attn_q.weight', 'blk.0.attn_k.weight', 'blk.0.attn_v.weight', 'blk.0.attn_output.weight', ...]
```

#### 实用技巧

1. **查看模型架构信息**：
   ```python
   print(r.fields['general.architecture'])  # 如 "llama"
   print(r.fields['llama.embedding_length'])  # 如 5120
   ```

2. **查看词表信息**：
   ```python
   tokens = r.fields['tokenizer.ggml.tokens']
   print(f"词表大小：{len(tokens)}")
   ```

3. **查看所有张量形状和类型**：
   ```python
   for t in r.tensors:
       print(f"{t.name}: shape={t.shape}, type={t.tensor_type}")
   ```

4. **检查多模态模型**：对于像 `mmproj-*.gguf` 这样的多模态投影模型，可以查看视觉相关的张量：
   ```python
   for t in r.tensors:
       if 'vision' in t.name or 'projector' in t.name:
           print(t.name)
   ```
