# 作业 10 · 提交说明

## Part 1 · 评估

**入口**：`lessons/lesson-10-eval-safety/starter/eval.py`

```bash
python lessons/lesson-10-eval-safety/starter/eval.py golden   # 课程验收集，8 题
python lessons/lesson-10-eval-safety/starter/eval.py dev      # 自建调参集，22 题
python lessons/lesson-10-eval-safety/starter/eval.py dev --show
```

| 评测集 | 得分 |
|---|---|
| golden（留出，不参与调参） | **8/8 · 100%** |
| dev（自建，用于调参） | **20/22 · 91%** |

**可复现性**：judge 与被测系统**都**跑 `temperature=0`。
只把 judge 设成 0 不够 —— 确定性的裁判去评一个非确定性的答案，
分数仍然会在两次运行之间漂移。两端都设 0 之后，`app/llm.py` 的磁盘缓存
让重跑既免费又逐字一致。

---

### 100 字总结（作业要求）

> 我改的是 prompt 里的完整性约束。诊断依据是 `sources`：
> 中文提问「毕业需要多少学分」时，top-1 已经命中 § Graduation，
> 五个条件全在检索结果里，但答案只给了「120 学分」——
> **资料对、答案漏，属于生成问题，不是检索问题**，所以不动 chunk 和阈值。
> 把「列出每一项条件」改写成显式规则并声明跨语言同样适用后，
> 该题从漏答 1 条变为完整列出 5 条。
> dev 集 19/22 → 20/22，golden 集保持 8/8。
> 过程中我先改坏过一次：新规则替换掉了原有的通用约束，导致英文的 g1 反而漏掉 GPA，
> 由全量评测立刻发现并回补。

---

### 过程记录：三次「指标本身是 bug」

这轮最大的收获不在系统，在**评测方法**。三次失败的根因都在评测代码里：

| 现象 | 真实原因 |
|---|---|
| d9 被判错，但答案更完整 | 我把 `must_contain`（检索关键词）当成了 judge 的参考答案，裁判嫌它「多说了」 |
| d16 被判**通过**，但系统实际拒答了一道有答案的题 | 同上。参考答案是 `"CS101"`，裁判无从判断，把拒答当成了正确 ⚠️ **假通过最危险** |
| d5 被判错 | 我写的参考答案本身是错的（把 academic warning 的 1.5–2.0 写成了 probation 的阈值），系统答对了 |

修法：`must_contain` 只用于**检索**判定，另立 `ref` 作为**答案**判定的完整标准答案。
修完 dev 从虚高的 21/22 掉到真实的 19/22。**分数变低是修对了的信号。**

### 仍未通过的一条

**d16「CS101 一共有多少学分」** — 语料里明确写着 `CS101 ... Credits: 3`，
但检索捞回的全是 CS201 相关内容。`CS101` 与 `CS201` 只差一个字符，
稠密向量分辨不出。属于**检索问题**，且不是调参能解决的：
需要混合检索（向量 + BM25 关键词）。记为已知局限，未在本课范围内实现。

---

## Part 2 · 安全加固

**两项都做了**，实现在 `app/safety.py`（放在 `app/` 下才会进 Docker 镜像；
`lessons/` 目录不在 `COPY` 范围内）。

**演示**：`python lessons/lesson-10-eval-safety/starter/safety.py`

### 选项 A · PII 脱敏

```
原文     : Student z1234567 emailed jane@uplus.edu about CS201, call 0412 345 678.
脱敏后   : Student [REDACTED_ID] emailed [REDACTED_EMAIL] about CS201, call [REDACTED_PHONE].
日志只写 : pii detected — email=1, student_id=1, phone=1
```

三件套：邮箱、学号（`z` + 7 位，或独立的 8 位数字）、澳洲手机（`04` / `+614`）。

两个实现细节：

- **手机必须先于学号匹配**，否则手机号中间的 8 位连号会被学号规则吃掉，剩余部分明文留存
- **日志只写计数不写值**（`redaction_summary`）—— 把匹配到的原始 PII 写进日志，
  正是这个函数要防止的那件事

接进管线：生成的答案在返回与记录之前都过一遍 `redact()`。
手册当前不含 PII，但「不应该有」不是控制措施；语料未来会扩展。

### 选项 B · Prompt injection 防护

六类规则：覆盖指令、忽略指令、新指令、系统提示词、角色改写、**边界伪造**。

**关键设计：分风险等级处理，而不是一刀切。**

| 路径 | 处理 | 理由 |
|---|---|---|
| **检索回来的语料** | 命中即中止，**不调用模型**，`blocked=True` | 这些文本会被当作权威材料引用给模型，风险最高 |
| 用户输入 | 不硬拦 | 会误伤正常提问（"What should I do? Ignore the previous policy..."）。靠检索为空 + prompt 约束兜住 |

实测：把注入语句直接当问题问，`used_fallback=True`，模型正常拒答；
把同样的文本放进检索结果，`blocked=True`，模型根本没被调用。

**边界防伪**：`wrap_untrusted()` 会先剥掉文本里已有的 marker。
否则攻击者写一个 `<<<END_UNTRUSTED_DATA>>>`，就能提前闭合围栏，
后面的内容重新被当成指令。

**新增第三个状态字段 `blocked`**，与 `used_fallback`、`degraded` 并列：

```
used_fallback=true, blocked=false, degraded=false   手册里没有答案（正常业务）
used_fallback=true, blocked=false, degraded=true    有答案，但生成失败（故障）
used_fallback=true, blocked=true,  degraded=false   拒绝处理（攻击信号）
```

合并成一个字段，线上就会把攻击埋没在正常流量里。

---

## 诚实的边界

- 正则 PII 检测会漏掉非常规格式。生产应使用专用识别器（如 Microsoft Presidio）
- 模式匹配的注入检测可被改写轻易绕过
- 二者都是**一层**，不是防御本身。价值在于抬高最简单攻击的成本，
  以及给审计日志留下记录

## 测试

`tests/test_safety.py` — 33 个用例，覆盖三类 PII、手机/学号的匹配顺序、
课程代码不被误判、日志摘要不泄露原值、六类注入、边界伪造剥离、
被污染 chunk 在调用模型前被拦截、答案 PII 脱敏。

全仓 **143 passed**。
