# Agent4Rec — 用 1000 个 LLM Agent 模拟推荐系统用户

基于论文 [On Generative Agents in Recommendation](https://arxiv.org/abs/2310.10108) (SIGIR 2024, NUS + 清华 + USTC)，fork 自 [LehengTHU/Agent4Rec](https://github.com/LehengTHU/Agent4Rec)，适配智谱 GLM API，在腾讯云 CPU 环境上完成全量 1000 avatar 仿真 + KS 检验。

---

## 一、论文核心思想

把 Stanford Smallville (Generative Agents, Park et al. 2023) 的 **Profile / Memory / Reflection / Action** 架构搬到推荐场景：每个真实 MovieLens-1M 用户变成一个 LLM Agent，在翻页式推荐中自主决定 **看不看 / 评几分 / 是否退出**。

核心问题：**LLM 模拟的用户行为分布能否与真实人群一致？** 如果能，就可以用 LLM Agent 做离线 A/B 测试，替代真人实验。

## 二、架构总览

```
                    +--------------------------+
                    | LightGCN (预训练推荐器)    |  <- 离线训练,输出全量 user*item 排序
                    +------------+-------------+
                                 |  每页 Top-4 电影
                                 v
                    +--------------------------+
                    | Avatar (LLM Agent)        |
                    | +--------+  +----------+ |
                    | |Profile |->| Memory   | |  <- FAISS VectorStore + Reflection
                    | |3 trait |  | (历史行为)| |
                    | +--------+  +-----+----+ |
                    |                   v      |
                    |              +--------+  |
                    |              | Action |  |  <- LLM: ALIGN / WATCH / RATING / EXIT
                    |              +--------+  |
                    +------------+-------------+
                                 |  行为写入 Memory -> 反思满意度 -> 决定翻页或退出
                                 v
                        下一页 (直到 EXIT 或 max_pages)
```

## 三、三维 Social Trait (论文最被后续借鉴的设计)

每个 Agent 有三维 trait，每维 3 档，共 **3 x 3 x 3 = 27 个 prototype** 覆盖 1000 用户：

| Trait | 含义 | 3 档 | 在 prompt 中的作用 |
|---|---|---|---|
| **Activity** | 看片频率 | Elusive -> Occasional -> Active | 决定翻几页才退出 |
| **Conformity** | 评分跟随历史均分的程度 | Dedicated Follower -> Balanced -> Independent | 决定 1-5 星怎么打 |
| **Diversity** | 口味宽窄 | Selective -> Curious -> Adventurous | 决定 ALIGN yes/no 的宽容度 |

每个 trait 都有一段详细的 persona 描述直接写进 system prompt，例如 Activity=1：

> "An Incredibly Elusive Occasional Viewer, so seldom attracted by movie recommendations that it's almost a legendary event when you do watch a movie. And you will exit the recommender system immediately even if you just feel little unsatisfied."

这种 **"行为后果直接硬编进 persona"** 的设计是 Agent4Rec 控制成本的关键——不靠复杂 reflection 链来涌现退出行为，而是 prompt 里直接说明。

## 四、每页交互流程

每一页推荐 4 部电影，Avatar 做一次 LLM 调用，输出三段式结构化回答：

```
(1) ALIGN 判定 (对每部电影)
   MOVIE: Babe (1995);          ALIGN: yes; REASON: 动画+冒险,符合我偏好
   MOVIE: Groundhog Day (1993); ALIGN: no;  REASON: 不在我的5种偏好类型里

(2) 选择要看的 (基于 activity + diversity trait)
   NUM: 3; WATCH: Babe, Aladdin, Shakespeare in Love; REASON: ...

(3) 打分 + 感受
   MOVIE: Babe;               RATING: 4; FEELING: charming and delightful
   MOVIE: Shakespeare in Love; RATING: 5; FEELING: perfect blend of romance
```

然后 Memory 写入 -> Reflection 触发 (每 3 条 memory 问一次"你满不满意") -> 决定 [EXIT] 或 [NEXT PAGE]。

## 五、Memory 模块

基于 LangChain VectorStore (FAISS) + sentence-transformers 本地嵌入：

- **写入**：每页一条结构化记忆 ("我在 page 1 看了 [Babe, Aladdin]，评 [4, 5]，不喜欢 [Groundhog Day]")
- **检索**：用语义相似度找相关记忆 (代码里 Smallville 原版的 recency/importance 被旁路，实际只用 relevance)
- **反思**：`reflection_threshold=3`，每 3 条 memory 触发一次满意度评估 ("[satisfied/unsatisfied] because...")

## 六、KS 检验 (论文核心验证方法)

**Kolmogorov-Smirnov test**：非参数检验，比较两个样本的累积分布函数 (CDF) 的最大差距：

```
D = max | F_real(x) - F_sim(x) |
p > 0.05 -> 不能拒绝"分布相同" -> 视为通过
p < 0.05 -> 分布有显著差异 -> 不通过
```

**为什么选 KS 不选其他检验**：

| 候选 | 缺点 |
|---|---|
| t-test | 要求正态分布，评分是 1-5 整数远离正态 |
| chi-squared | 对小样本敏感 |
| Wasserstein | 解释成本高 |
| **KS** | 不依赖分布形式，一行 `scipy.stats.ks_2samp` 搞定 |

论文用 GPT-3.5 跑 1000 用户，报告 KS **通过** (评分分布形状与真实 MovieLens 吻合)。

## 七、论文 vs 代码的"温差"

| 论文宣传 | 代码实际 |
|---|---|
| Memory 用 recency x importance x relevance 三因素 | 只用了 relevance (retriever.py 里 recency 硬编为 1, importance 没接) |
| 3 种 reflection 机制 | 默认只跑 1 种 (满意度反思)，另外两种被注释掉 |
| Serial + Parallel 两种模式 | Serial 路径有 3 个 bug (缺参数/传错类型/索引方式错误)，实际只用 parallel |

## 八、本 fork 的改动

### GLM/Zhipu API 适配
- 新建 `glm_setup.py`：配置 `openai.api_base` 指向 Zhipu OpenAI 兼容端点
- `simulation/memory.py` + `avatar.py`：`gpt-3.5-turbo` -> `glm-4-plus`

### Embedding 本地化
- `OpenAIEmbeddings(1536d)` -> `HuggingFaceEmbeddings("all-MiniLM-L6-v2", 384d)`
- Embedding model 做成类级别单例，init 从 75min 压到 60s

### CPU 兼容
- 所有 `.cuda(self.device)` -> `.to(self.device)`，`torch.device` 加 CPU fallback (7 个文件)

### Serial 模式修复
- `arena.py:396`：`.movie_detail[idx]` -> `.loc[idx]`
- `arena.py:407`：补 `current_page` 参数 + 计数器 + 构造 `recommended_items_str`

### 并发控制
- `max_workers` 从 100/500 降到 8/16 (防 Zhipu API 限流)

## 九、环境搭建

```bash
# 系统依赖
dnf install -y python3-devel gcc-c++    # 或 apt install python3-dev g++

# 独立 venv
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

数据需自行准备 (不在 git 里)：
```
datasets/ml-1m/          # MovieLens-1M -> grouplens.org 下载
recommenders/weights/    # LightGCN checkpoint -> 运行 recommenders/training.sh 生成
```

## 十、运行

```bash
# 设置 GLM API key (或修改 glm_setup.py 里的默认值)
export GLM_API_KEY=<your-zhipu-key>

# 烟雾测试 (3 avatar, 约 20s)
python -u main.py --n_avatars 3 --max_pages 1 --execution_mode serial

# 全量 1000 avatar (约 57min, 约 6.74 元)
python -u main.py --n_avatars 1000 --max_pages 5 --execution_mode parallel --simulation_name plus_full
```

产出在 `storage/ml-1m/LightGCN/<simulation_name>/`：
- `behavior/{0..999}.pkl` — 每个 avatar 的结构化行为
- `running_logs/{0..999}.txt` — 每个 avatar 的完整对话
- `metrics.txt` — 全局指标

## 十一、全量实验结果 (GLM-4-plus, 1000 avatar x 5 page)

| 维度 | 值 |
|---|---|
| 总时间 | 57 分钟 |
| Token 总量 | 3.74M |
| 成本 | 6.74 元 |
| Avg recall / precision | 0.413 / 0.299 |
| Avg exit page | 1.726 |
| Click rate | 20.26% |

### KS 检验结果

```
                         1星      2星      3星      4星      5星
模拟 (GLM-4-plus)        0.00%   0.00%    3.83%   52.44%   43.73%
真实 (MovieLens 1000)     5.78%   10.80%   26.11%   34.79%   22.52%

KS D = 0.3887, p = 0 -> 不通过
模拟均值 4.40 vs 真实均值 3.58 (高 0.82 星)
```

**关键发现**：
- 模拟 avatar **完全不打 1-2 星**，4-5 星合占 96% (真实只有 57%)
- 实证了 OmniBehavior (2026) 定义的 **utopian bias**
- 论文用 GPT-3.5 报告 KS 通过；我们用 GLM-4-plus (更强模型) 跑出 D=0.39 -> **换更强模型不能修复结构性偏差**
- 下一步方向：用 [UserMirrorer](https://github.com/Joinn99/UserMirrorer) SFT/DPO 微调来修复

## 相关

- 论文：[On Generative Agents in Recommendation](https://arxiv.org/abs/2310.10108) (SIGIR 2024)
- 原始代码：[LehengTHU/Agent4Rec](https://github.com/LehengTHU/Agent4Rec)
- 详细改动日志：[doc/README.md](doc/README.md)
