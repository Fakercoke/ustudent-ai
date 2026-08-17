# 作业 10 · 运行输出存档

作业要求提交「eval 输出截图」和「安全 demo 截图」。
这里存的是同样内容的**文本原件** —— 可复制、可 diff、可追溯到具体一次运行，
比截图更有用。要交截图的话，打开这些文件截图即可。

| 文件 | 内容 |
|---|---|
| `01-eval-golden.txt` | LLM-as-judge · 课程留出集 8 题 |
| `02-eval-dev.txt` | LLM-as-judge · 自建调参集 22 题 |
| `03-safety-demo.txt` | PII 脱敏 + 注入检测 + 管线拦截 + 真实请求 |
| `04-pytest.txt` | 全仓测试 |

## 复现

```bash
source .venv/bin/activate && export PYTHONPATH=.
python lessons/lesson-10-eval-safety/starter/eval.py golden --show
python lessons/lesson-10-eval-safety/starter/eval.py dev --show
python lessons/lesson-10-eval-safety/starter/safety.py
pytest -q
```

judge 与被测系统都跑 `temperature=0`，`app/llm.py` 的磁盘缓存使重跑逐字一致。

## 关于 dev 集分数的波动

两次记录分别为 20/22 与 21/22，差异出在 **d3**：某次运行时生成调用被限流，
走了降级路径（返回手册原文而非 AI 总结），裁判据此判为不通过。

这不是评估不可复现，而是**外部依赖的可用性波动被如实计入了分数** ——
降级本身正是设计要求的行为。真正不可复现的情形（同一输入、同一状态、不同分数）
已经通过两端 `temperature=0` 消除。

稳定不通过的只有 **d16**（`CS101 一共有多少学分`）：
语料写明 `Credits: 3`，但 `CS101` 与 `CS201` 仅差一字符，稠密向量无法区分，
检索始终捞回 CS201 相关内容。属检索问题，需混合检索（向量 + BM25）解决，
不在本课范围内，已记入已知局限。
