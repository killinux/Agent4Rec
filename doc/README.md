# Agent4Rec — GLM 适配 + 腾讯云复现记录

基于 [LehengTHU/Agent4Rec](https://github.com/LehengTHU/Agent4Rec) fork,适配智谱 GLM API,在腾讯云 CPU 环境上完成全量 1000 avatar 仿真 + KS 检验。

## 改动清单

### 1. GLM/Zhipu API 适配
| 文件 | 改动 |
|---|---|
| `glm_setup.py` (新建) | 配置 `openai.api_base` 指向 Zhipu OpenAI 兼容端点,设置 API key |
| `main.py` | 顶部 `import glm_setup`(必须在所有其他 import 前) |
| `simulation/memory.py` | `gpt-3.5-turbo` → `glm-4-plus`(4 处) |
| `simulation/avatar.py` | `gpt-3.5-turbo` → `glm-4-plus` + `ChatOpenAI(model_name="glm-4-plus")` |

### 2. Embedding 本地化
| 文件 | 改动 |
|---|---|
| `simulation/avatar.py:74` | `OpenAIEmbeddings(1536d)` → `HuggingFaceEmbeddings("all-MiniLM-L6-v2", 384d)` |
| `simulation/avatar.py:75` | `embedding_size = 1536` → `384` |
| `simulation/avatar.py:75-77` | Embedding model 改为类级别单例(`Avatar._shared_embeddings`),init 从 75min 压到 60s |

### 3. CPU 兼容(无 GPU 环境)
所有 `.cuda(self.device)` → `.to(self.device)`,`torch.device(args.cuda)` 加 CPU fallback。影响文件:
- `recommenders/data.py`
- `recommenders/models/{LightGCN,MF,InfoNCE,MultVAE}.py`
- `recommenders/models/base/{abstract_model,abstract_RS}.py`
- `simulation/base/abstract_arena.py`

### 4. Serial 模式 bug 修复
| 位置 | 问题 | 修复 |
|---|---|---|
| `arena.py:396` | `self.movie_detail[idx]` 把 int 当列名 | → `.loc[idx]` |
| `arena.py:407` | `reaction_to_recommended_items()` 缺 `current_page` 参数 | 补参数 + `i` 计数器 |
| `arena.py:407` | 直接传 list 而不是格式化 string | 构造 `recommended_items_str` |

### 5. 并发控制
`simulation/arena.py:88,206` 的 `max_workers` 从 100/500 降到 8/16(防 Zhipu API 限流)。

## 环境要求

```bash
# 系统依赖
dnf install -y python3-devel gcc-c++

# Python venv(独立,不污染系统)
python3 -m venv agent4rec_venv
source agent4rec_venv/bin/activate
pip install openai==0.27.8 langchain==0.0.265 sentence-transformers faiss-cpu \
            termcolor tqdm pandas matplotlib seaborn reckit colorlog wandb Cython

# Cython 编译
cd recommenders && python setup.py build_ext --inplace && cd ..

# 国内 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/hf_cache
```

## 运行

```bash
# GLM API key(默认写在 glm_setup.py,也可环境变量覆盖)
export GLM_API_KEY=<your-zhipu-key>

# 烟雾测试(3 avatar,约 20s)
python -u main.py --n_avatars 3 --max_pages 1 --execution_mode serial

# 全量 1000 avatar(约 57min,约 6.74 元)
python -u main.py --n_avatars 1000 --max_pages 5 --execution_mode parallel --simulation_name plus_full
```

## 全量实验结果(GLM-4-plus, 1000 avatar x 5 page)

| 维度 | 值 |
|---|---|
| 总时间 | 57 分钟 |
| Token 总量 | 3.74M |
| 成本 | 6.74 元 |
| Avg recall / precision | 0.413 / 0.299 |
| Avg exit page | 1.726 |
| Click rate | 20.26% |

### KS 检验(评分分布对齐)

```
                         1星      2星      3星      4星      5星
模拟 (GLM-4-plus)        0.00%   0.00%    3.83%   52.44%   43.73%
真实 (MovieLens 1000)     5.78%   10.80%   26.11%   34.79%   22.52%

KS D = 0.3887, p = 0 → 不通过
模拟均值 4.40 vs 真实均值 3.58(高 0.82 星)
```

关键发现:实证了 OmniBehavior (2026) 定义的 utopian bias — LLM 模拟 avatar 完全不打 1-2 星,4-5 星合占 96%。论文用 GPT-3.5 报告 KS 通过;GLM-4-plus 跑出 D=0.39,说明换更强模型不能修复结构性偏差。

## 数据目录(需自行下载/生成,不在 git 里)

```
datasets/ml-1m/          # MovieLens-1M 数据,从 grouplens.org 下载
recommenders/weights/    # LightGCN checkpoint,运行 recommenders/training.sh 生成
storage/                 # 仿真产物,运行 main.py 生成
```

## 相关

- 论文: [On Generative Agents in Recommendation](https://arxiv.org/abs/2310.10108) (SIGIR 2024)
- 原始代码: [LehengTHU/Agent4Rec](https://github.com/LehengTHU/Agent4Rec)
