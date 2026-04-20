### 使用python分析GGUF模型
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