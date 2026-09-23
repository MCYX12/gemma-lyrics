# gemma-lyrics

基于 [Gemma 4 e4b-it](https://huggingface.co/google/gemma-4-e4b-it) 的中文歌词生成模型，使用 MLX LoRA 在 Apple Silicon 上微调，支持 Ollama 本地部署。

---

## 特性

- 生成叙事抒情、画面感强、语言克制的中文原创歌词
- 支持标签式输入：`风格：... 情绪：... 场景：...`
- 支持**数字简谱**输出（含调号、拍号、旋律行）
- 两种推理方式：Ollama（更快）和 MLX 直接推理

---

## 环境要求

- Apple Silicon Mac（M 系列芯片）
- Python 3.10+
- [Ollama](https://ollama.com)（如需 Ollama 推理）
- [llama.cpp](https://github.com/ggerganov/llama.cpp)（如需自行量化 GGUF）

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/MCYX12/gemma-lyrics.git
cd gemma-lyrics
```

### 2. 创建虚拟环境并安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install mlx-lm
```

### 3. 下载基座模型

从 HuggingFace 下载 Gemma 4 e4b-it，放到 `gemma-4-e4b-it/` 目录：

```bash
huggingface-cli download google/gemma-4-e4b-it --local-dir gemma-4-e4b-it
```

---

## 推理

### 方式一：Ollama（推荐）

需要先用 llama.cpp 将模型量化为 GGUF，然后注册到 Ollama：

```bash
# 量化完成后执行
bash scripts/build_ollama.sh
```

之后即可生成歌词：

```bash
# 普通提示词
bash scripts/run_ollama.sh "写一首关于离别的歌"

# 标签式输入
bash scripts/run_ollama.sh "风格：冷淡克制 情绪：孤独 场景：夜晚城市"

# 数字简谱
bash scripts/run_ollama.sh "写一段简谱，风格：民谣"
```

### 方式二：MLX 直接推理

需要将微调后的模型转换为 MLX 格式，放到 `model/gemma4_mlx/`。

```bash
# 普通歌词
bash scripts/generate.sh "风格：后摇 情绪：告别 场景：车站"

# 数字简谱
bash scripts/generate.sh "简谱，风格：民谣 场景：田野"
```

---

## 训练

如需在自己的数据集上重新微调：

### 1. 准备数据

将训练样本放入 `data/train_original.jsonl`，格式为：

```json
{"messages": [{"role": "user", "content": "写一段..."}, {"role": "assistant", "content": "歌词正文..."}]}
```

### 2. 数据预处理 + 训练

```bash
bash scripts/train.sh
```

训练参数（在 `scripts/train.sh` 中修改）：

| 参数 | 默认值 |
|------|--------|
| LoRA rank | 8 |
| 训练轮数 | 300 iters |
| 学习率 | 5e-5 |
| batch size | 1 |
| max seq length | 512 |
| 微调层数 | 16 |

训练完成后，适配器权重保存在 `adapter/lora/`。

---

## 文件结构

```
gemma-lyrics/
├── Modelfile                   # Ollama 模型配置（含 system prompt）
├── adapter/lora/
│   ├── adapters.safetensors    # 训练好的 LoRA 权重
│   └── adapter_config.json     # LoRA 超参数配置
├── data/
│   ├── train_original.jsonl    # 原始训练数据
│   ├── train.jsonl             # 处理后训练集
│   ├── valid.jsonl             # 验证集
│   └── test.jsonl              # 测试集
└── scripts/
    ├── train.sh                # 数据预处理 + MLX LoRA 训练
    ├── generate.sh             # MLX 直接推理
    ├── run_ollama.sh           # Ollama 推理
    ├── build_ollama.sh         # 注册 Ollama 模型
    ├── generate_lyrics.py      # MLX 推理主程序
    ├── lyrics_to_jianpu.py     # 歌词转数字简谱
    └── prepare_mlx_data.py     # 数据集预处理
```

---

## 示例输出

**输入：** `风格：冷淡克制 情绪：孤独 场景：夜晚城市`

```
楼道的灯一闪一闪
像你走后没说完的话
雨水顺着窗缝爬进来
把床单一点点泡凉

我听见夜里很远的车声
像谁把名字叫到一半
整座城市都关了门
只剩我还站在回声里面
```

---

## License

本项目代码采用 [MIT License](LICENSE)。模型权重基于 [Gemma Terms of Use](https://ai.google.dev/gemma/terms)。
