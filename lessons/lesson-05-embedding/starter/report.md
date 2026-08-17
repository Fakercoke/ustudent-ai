# 作业 5 报告 · 向量索引

环境：chromadb 0.5.23，默认 embedding `all-MiniLM-L6-v2`（384 维，本地 CPU，不用 API key）。
索引落在 `./chroma_db/`，collection `handbook`，距离度量显式设为 `hnsw:space=cosine`（0=相同，2=相反）。

---

## Part 1 · 手搓 cosine

```
$ python index_handbook.py demo
sim(a, b) = 0.463
sim(a, c) = 0.000
```

- a = "drop the course before week two"
- b = "withdraw from the class by week two"
- c = "today's weather is wonderful"

**为什么 a-b 比 a-c 高？**
a 和 b 共享 `the / week / two` 三个词，点积不为 0；a 和 c 一个词都不重合，点积为 0，余弦直接是 0。

**但 0.463 其实很低**，而 a、b 的意思几乎一样。原因是词袋只比字面：它不知道 `drop ≈ withdraw`、`course ≈ class`。
用真 embedding 跑同一对句子（`drop the course` vs `withdraw from the class`）：词袋 0.289，embedding 0.524 —— **同样的输入，语义模型给出接近两倍的相似度**。这就是 Part 2 存在的理由。

---

## Part 2 · Chroma 索引

```
$ python index_handbook.py build
  data/handbook.md             ->   19 chunks
  data/faq.md                  ->    9 chunks
  data/courses-catalog.md      ->   12 chunks
Indexed 40 chunks
```

### 三个必答查询（top-3）

| 查询 | top-1 距离 | top-1 命中 | 是否正确 |
|---|---|---|---|
| How many credits do I need to graduate? | 0.293 | `data/faq.md::0`（含 Q1 完整答案：120 学分 + GPA 2.0） | ✅ |
| Can I take CS101 and MATH101 together? | 0.509 | `data/faq.md::1`（含 Q3 完整答案：时间冲突，不能同时选） | ✅ |
| How is GPA calculated? | 0.288 | `data/handbook.md::13`（GPA computation 小节） | ✅ |

**第 2 个查询距离偏高（0.509）值得单独说。** 它命中的内容是对的，但 `faq.md::1` 这一块同时装了 Q2、Q3、Q4 三个问题：

```
Q2. How many courses can I take in one semester? ...
### Q3. Can I take CS101 and MATH101 at the same time? ...   ← 我要的答案在中间
### Q4. Can a first-year student take CS201? ...
```

向量是整块内容的**平均语义**，Q2 和 Q4 稀释了 Q3 的信号，所以距离被拉高。这个现象在下面的对照实验里会放大成主要结论。

---

## 参数对照实验 · chunk size 100 / 600 / 3000

> 实现说明：starter 的签名是 `overlap: int = 100`。直接跑 `size=100` 会 `step = 100 - 100 = 0`，
> 抛 `ValueError: range() arg 3 must not be zero`。改成 `overlap` 默认取 `size // 6`
> —— 在默认 `size=600` 时正好等于 100（和原 starter 完全一致），同时让对照实验里三组的重叠比例保持一致，实验才公平。

```
$ python index_handbook.py compare

top-1 distance by chunk size (lower = better match)
query                                                100       600      3000
------------------------------------------------------------------------------
How many credits do I need to graduate?            0.260     0.293     0.327
Can I take CS101 and MATH101 together?             0.139     0.509     0.576
How is GPA calculated?                             0.256     0.288     0.614
What happens if I fail a required course?          0.462     0.479     0.513
How late can I withdraw without hurting my GPA?    0.384     0.352     0.473
How do I get a parking permit?  (无答案)            0.680     0.752     0.816
------------------------------------------------------------------------------
chunks                                               233        40         9
```

### ⚠️ 数字有陷阱：size=100 距离最低，但检索质量最差

同一个查询在三种 size 下**实际取回的内容**：

```
size=100   distance=0.260   长度 99
  "I need to graduate? You need a minimum of 120 credits and a cumulative GPA of at least 2.0 to grad"
   ↑ 开头缺半句                                                                              ↑ 结尾被砍断

size=600   distance=0.293   长度 600
  "# Frequently Asked Questions ... ### Q1. How many credits do I need to graduate?
   You need a minimum of 120 credits and a cumulative GPA of at least 2.0 to graduate.
   You also need to complete every required course ..."      ← 答案完整

size=3000  distance=0.327   长度 3000
  同上开头，但后面还塞了 Q2~Q8 六七个不相干的问答
```

**所以「距离低」和「答案好」不是一回事。** 小块之所以距离低，是因为块里几乎只有跟问题相关的字，噪音少、向量方向纯；但它**切断了答案本身**，把这段文本交给 lesson 7 的 LLM，它拿不到完整信息。

### 结论（约 150 字）

**chunk 太小（100）**：距离数字反而最漂亮，因为块内几乎没有噪音，向量方向很纯。但代价有三个：一是答案被物理切断，"至少 2.0 才能毕" 这种半句话没法用；二是块数从 40 涨到 233，建库和检索开销翻 6 倍；三是有答案与无答案的距离差被压缩（0.260 vs 0.680，差 0.42），兜底阈值更难划。

**chunk 太大（3000）**：一块混进 6~7 个话题，向量是整块的平均语义，目标信号被无关内容稀释，距离全线升高（GPA 那题从 0.288 恶化到 0.614，翻了一倍多）。而且 9 块里每块 3000 字符，塞进 prompt 既贵又稀释 LLM 的注意力。

**最终选 600。** 理由：它是唯一能**同时**保证「取回的块含有完整答案」和「距离仍然可区分」的档位。有答案 0.29~0.35 / 无答案 0.75，差距 0.4 以上，阈值好定。size=100 虽然距离更低，但内容残缺，对下游 RAG 是净损失——**检索的目标不是让距离数字最小，是让 LLM 拿到能回答问题的完整上下文。**

---

## 自己的三个查询（其中一个故意无答案）

```
$ python index_handbook.py extra
```

| 查询 | top-1 距离 | 结果 |
|---|---|---|
| What happens if I fail a required course? | 0.479 | ❌ **检索失败**，详见下节 |
| How late can I withdraw without hurting my GPA? | 0.352 | ✅ 命中 `faq.md::3`，正确（Week 8 前退课记 W，不影响 GPA） |
| **How do I get a parking permit?** | **0.752** | ❌ 语料里根本没有停车证。但 Chroma **照样返回了 3 条**，top-3 距离 0.752 / 0.807 / 0.816 |

### ⚠️ 意外收获：一次「有答案却检索不到」的失败

`What happens if I fail a required course?` 这题**语料里明明有答案**：

- `data/handbook.md:141` — "A student who receives a D or **F** grade may **repeat** the course once..."
- `data/faq.md` Q8 — "I got a D in MATH101. Can I **retake** it?"

但 size=600 的 top-1 是 `handbook.md::5`（选课流程里讲先修课的段落），距离 0.479，**完全不相干**。size=100 也一样错。

把问题换个说法再查：

| 问法 | size=100 top-1 | size=600 top-1 |
|---|---|---|
| What happens if I **fail** a required course? | 0.462 `handbook.md::50` ❌ | 0.479 `handbook.md::5` ❌ |
| Can I **retake** a course I got an **F** in? | **0.339 `faq.md::27`（Q8）✅** | 0.385 `faq.md::4`（GPA 计算）❌ |

**说明两件事：**

1. **向量检索也有词汇鸿沟。** 语料用的词是 `repeat / retake / D or F grade`，我问的是 `fail`。MiniLM 认识 `drop ≈ withdraw`，但 `fail → repeat the course` 这种「问题→措施」的跨越，它跨不过去。**embedding 解决的是同义词，不是推理。**
2. **这不是靠调 chunk size 能修的。** 三个 size 全错。要修得靠 lesson 6/7 的手段：按 markdown 标题切块（让「Repeating a course」整节成为一块）、查询改写（先让 LLM 把用户问题扩写成几个同义问法）、或混合检索（向量 + 关键词 BM25）。

**这一条是整个作业里最有价值的发现**——它标出了纯向量检索的能力边界，正好是下一课要解决的问题。

### 这条无答案查询的意义

**Chroma 永远返回 `n_results` 条，它不会说「没有」。** 它只是返回「最不离谱的几条」，哪怕全部不相关。唯一的信号是距离数值：

```
有答案的查询   top-1 距离  0.26 ~ 0.48
无答案的查询   top-1 距离  0.752
                          ↑ 明显高出一档
```

**lesson 7 的兜底阈值建议取 0.65**（在 0.48 和 0.752 之间，留一点余量）：

```python
if top1_distance > 0.65:
    return "抱歉，学生手册里没有查到相关内容，建议联系教务处。"
```

不做这一步，RAG 就会拿着「学术咨询办公室审批 GPA 3.5」那段完全不相关的文字，一本正经地回答停车证问题——这就是幻觉的典型来源。

---

## 附：踩到并修掉的坑

| 坑 | 现象 | 处理 |
|---|---|---|
| `size=100` + `overlap=100` | `ValueError: range() arg 3 must not be zero` | `overlap` 默认改为 `size // 6`，并对 `overlap >= size` 显式抛带说明的 ValueError |
| 重复 build 堆数据 | `add()` 是追加不是覆盖，跑三次库里三份 | `build_index` 开头先 `delete_collection` |
| 距离看不懂（1.2、1.5） | Chroma 默认 squared L2 | 建 collection 时显式 `metadata={"hnsw:space": "cosine"}`，距离落到 [0,2] |
| ID 冲突 | `str(i)` 会让不同文件的同序号块互相覆盖 | 用 `f"{rel}::{i}"` |
| telemetry 报错刷屏 | chromadb 0.5.x 已知 bug | 不影响功能，忽略；或设 `ANONYMIZED_TELEMETRY=False` |
