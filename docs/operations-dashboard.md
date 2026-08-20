# UStudent AI 运营后台：使用与排障

这不是用来改课程、学生或选课记录的业务管理后台。它是 AI 服务的“体检报告”，回答三个问题：

1. 有多少请求进来，响应是否稳定？
2. RAG 是在哪一步失败的？
3. 大模型用了多少 Token，固定评测集是否退步？

## 1. 开启后台

在 `.env` 中至少配置：

```bash
OPS_ADMIN_USERNAME=admin
OPS_ADMIN_PASSWORD=请使用足够长的随机密码
OPS_HASH_SALT=请使用另一个随机字符串
OPS_DB_PATH=data/ops.sqlite3
```

`OPS_ADMIN_PASSWORD` 为空或仍使用默认盐时，`/ops` 返回 503，后台默认不开启。生产环境应使用 HTTPS；当前只有 HTTP IP 时，代码会拒绝公网后台登录，请通过 SSH 隧道访问，不要在公共网络直接输入管理密码。

启动服务后打开：

```text
http://127.0.0.1:8000/ops
```

浏览器会要求输入上述用户名和密码。

腾讯云当前没有 HTTPS，可以先在自己电脑执行：

```bash
ssh -i ~/.ssh/tencent_ustudent -L 8000:127.0.0.1:8000 ubuntu@49.235.155.82
```

保持该窗口打开，再访问 `http://127.0.0.1:8000/ops`。密码只经过加密的 SSH 通道，不会通过公网 HTTP 发送。

## 2. 一次请求怎样进入后台

```text
学生请求
  ↓
FastAPI middleware 创建 request ID，并开始计时
  ↓
RAG 记录 Top-1 距离、来源章节、fallback / degraded / blocked
  ↓
Agent 记录本轮实际调用的工具
  ↓
LLM 客户端记录模型、Token、缓存命中和错误
  ↓
响应完成后只写一条 SQLite 记录
```

问题文本在写入前会进行有限的正则脱敏（邮箱、项目格式学生号、澳洲手机号），最多保留 200 字预览；它不是完整匿名化，姓名、地址和其他格式的号码仍可能出现，所以默认只保留 90 天。原始请求体和原始 IP 不会进入数据库。健康检查、后台自身请求、Swagger 文件和浏览器 favicon 不计入业务请求。

React 页面访问量来自隐私最小化的 Nginx JSON 日志，只统计配置在 `OPS_WEB_ROUTES` 中的真实页面（默认 `/`、`/login`、`/dashboard`、`/ai-chat`），因此不会把 `/wp-admin` 一类扫描器路径算成用户。统计会读取当前和轮转后的压缩日志；若总日志超过 `OPS_WEB_LOG_MAX_BYTES`，后台会明确标记“统计不完整”，不会悄悄把下限当成完整 PV。数据库过期记录会在写入新请求或读取后台时清理。

“匿名请求来源”是把来源地址加盐哈希后的近似值，不是网页登录用户数。腾讯 Nginx 只把它实际看到的 `$remote_addr` 交给 AI 服务，AI 服务也只在直连端属于受信 Docker 网段时读取该值；公网直连请求伪造 `X-Forwarded-For` 不会生效。

腾讯云首次部署还要安装宿主机日志轮转（实际部署流程已执行这一步）：

```bash
bash deploy/install-ops-host.sh
```

它把配置安装到 `/etc/logrotate.d/uplus-nginx`，按天或单文件超过 20 MB 时轮转，最多保留 90 天。后台最多每 15 秒重读一次访问日志，并按**解压后的真实字节数**执行读取上限；日志轮转刚好发生在读取期间时，本次统计会标记为不完整，而不会拖垮或中断主业务。

## 3. RAG 指标怎样读

| 后台诊断 | 真实含义 | 先检查哪里 |
|---|---|---|
| `retrieval_empty` | 向量库没有返回任何 chunk | 建库、持久化、collection |
| `distance_gate` | Top-1 距离超过 0.75，第一层直接拒答 | 查询归一化、embedding、切块和召回 |
| `model_abstention` | 距离通过，但模型认为 sources 不足 | 人工看 sources；资料够就调 Prompt，不够仍调检索 |
| `generation_failure` | 检索成功，但模型返回不可用内容 | 模型接口、JSON 格式、Prompt |
| `llm_error` | Key、额度、网络、模型名或限流出错 | LLM 配置与服务日志 |
| `security_block` | 召回资料命中提示词注入规则 | 检查语料，不要直接关闭规则 |
| `agent_tool_used` | Agent 调用了工具 | 再看 tool trace 的工具名与结果 |

`rag_answered` 只表示系统生成了回答，不表示回答一定正确。线上问题没有标准答案，不能用“距离低”或“没有 fallback”冒充正确率。

## 4. 真正的 RAG 质量从哪里来

运行：

```bash
python scripts/eval_rag.py both
```

脚本会分别运行 dev 和课程验收/回归集，并把最新结果写进后台。后者参与过早期设计分析，因此不能包装成严格未见过的盲测集：

- 拒答准确率：有答案时是否回答、没答案时是否拒绝；
- 检索命中率：正确事实是否进入 sources；
- 答案落地率：正确事实是否最终进入 answer。

三个指标分别对应决策、检索和生成，不能混成一个分数。

Docker 部署后要让评测写进同一份持久化数据库，请在服务器运行：

```bash
docker exec ustudent-ai python scripts/eval_rag.py both
```

运营 SQLite 位于 `/app/runtime/ops.sqlite3` 并挂载到 Docker volume `ustudent-ai-ops`，替换镜像或容器不会清空历史数据。

## 5. Token 与费用

Token 会从 OpenAI-compatible 返回值和 LangChain 的 `AIMessage` 中记录。Agent 只统计本轮消息，避免把 thread memory 中的历史 Token 重复计算。

费用单价不写死在代码里，因为供应商会改价。把当前模型的每百万 Token 单价填入 `.env`：

```bash
LLM_COST_CURRENCY=CNY
LLM_INPUT_COST_PER_MILLION=0
LLM_CACHED_INPUT_COST_PER_MILLION=0
LLM_OUTPUT_COST_PER_MILLION=0
```

单价为 0 时后台仍显示 Token，但明确标记“费用未配置”，不会给出虚假的成本数字。

## 6. 汇报

后台右上角提供：

- “打印周报”：浏览器打印或保存为 PDF；
- “下载 Markdown”：保存最近 1、7、30 或 90 天的核心指标与问题分层。

汇报时应说“RAG 兜底率、失败分层和固定评测结果”，不要把 API 请求量说成完整网站 PV。腾讯云部署配置已经把最外层 Nginx 的隐私最小化日志只读挂载给 AI 服务；其他部署若未配置 `OPS_WEB_ACCESS_LOG_PATH`，后台会明确显示“网页访问尚未接入”。
