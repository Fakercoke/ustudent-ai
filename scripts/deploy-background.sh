#!/usr/bin/env bash
# 后台部署 —— 下载慢或 SSH 断开都不会中断。
#
#   bash scripts/deploy-background.sh          # 启动，立刻返回
#   tail -f ~/deploy.log                       # 看进度
#   bash scripts/deploy-background.sh status   # 查状态
#
# 为什么需要这个：Chroma 的 embedding 模型托管在 AWS S3，从中国大陆下载
# 只有 15-50 KB/s，80 MB 要 30-60 分钟。Chroma 自带的下载器没有重试，
# 连接一挂就无限等待；而任何前台 SSH 会话都撑不到那么久。
#
# 所以：用 curl 断点续传把模型先下到本地，再让构建直接复用它。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$HOME/deploy.log"
PIDFILE="$HOME/deploy.pid"
VENDOR="$REPO/vendor/chroma-onnx/all-MiniLM-L6-v2"
MODEL_URL="https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"

# ─────────────────────────────────────────────── status
if [ "${1:-}" = "status" ]; then
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "  运行中 (pid $(cat "$PIDFILE"))"
  else
    echo "  未运行"
  fi
  [ -f "$VENDOR/onnx.tar.gz" ] && \
    printf "  模型已下 %s / 79 MB\n" "$(du -h "$VENDOR/onnx.tar.gz" | cut -f1)"
  echo "  镜像 $(docker images ustudent-ai --format '{{.Size}}' 2>/dev/null | head -1 || echo 未生成)"
  echo "  容器 $(docker ps -a --filter name=ustudent-ai --format '{{.Status}}' 2>/dev/null || echo 未启动)"
  echo "  --- 日志末尾 ---"
  tail -6 "$LOG" 2>/dev/null | sed 's/^/  /' || echo "  （无日志）"
  exit 0
fi

# ─────────────────────────────────────────────── guard
if [ "${1:-}" != "_worker" ] && \
   [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "已经在跑了 (pid $(cat "$PIDFILE"))。看进度：tail -f $LOG"
  exit 0
fi

# ─────────────────────────────────────────────── worker
run() {
  set -x
  mkdir -p "$VENDOR"

  # 断点续传。--speed-limit/--speed-time 让 curl 在传输停滞时主动断开重连，
  # 这正是 Chroma 自带下载器缺的那一步：它挂住之后会永远等下去。
  until [ -s "$VENDOR/onnx.tar.gz" ] && \
        [ "$(stat -c%s "$VENDOR/onnx.tar.gz")" -ge 82000000 ]; do
    curl -fL --continue-at - \
         --retry 50 --retry-delay 5 --retry-all-errors \
         --speed-limit 3000 --speed-time 30 \
         --connect-timeout 20 \
         -o "$VENDOR/onnx.tar.gz" "$MODEL_URL" || true
    sleep 3
  done
  echo "模型下载完成：$(du -h "$VENDOR/onnx.tar.gz" | cut -f1)"

  tar -xzf "$VENDOR/onnx.tar.gz" -C "$VENDOR"
  ls -la "$VENDOR/onnx" | head

  cd "$REPO"
  DOCKER_BUILDKIT=1 docker build \
    --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    -t ustudent-ai:latest -f Dockerfile.vendored .

  docker rm -f ustudent-ai 2>/dev/null || true
  docker run -d --name ustudent-ai --restart unless-stopped \
    --env-file "$REPO/.env" -e OPS_DB_PATH=/app/runtime/ops.sqlite3 \
    -v ustudent-ai-ops:/app/runtime -p 8000:8000 --memory 1500m \
    ustudent-ai:latest

  for i in $(seq 1 60); do
    curl -fsS -m 3 http://localhost:8000/health >/dev/null 2>&1 && break
    sleep 3
  done

  echo "=== 冒烟测试 ==="
  for p in /health / /docs; do
    printf '%-10s %s\n' "$p" "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:8000$p")"
  done
  printf '%-10s %s\n' /rag-ask "$(curl -s -o /dev/null -w '%{http_code}' -X POST \
    http://localhost:8000/rag-ask -H 'Content-Type: application/json' \
    -d '{"question":"How many credits do I need to graduate?"}')"
  echo "=== 部署完成 ==="
}

# Re-enter this script in the background instead of serialising only the
# function definition into a fresh shell.  The previous approach lost the
# REPO/VENDOR/MODEL_URL variables, so curl received an empty URL and retried
# forever with "URL rejected: Malformed input".
if [ "${1:-}" = "_worker" ]; then
  run
  exit 0
fi

: > "$LOG"
nohup bash "$0" _worker >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"

cat <<INFO
  已在后台启动 (pid $(cat "$PIDFILE"))

  这个窗口可以关掉，任务不会中断。

  看进度：  tail -f ~/deploy.log
  查状态：  bash ~/ustudent-ai/scripts/deploy-background.sh status
INFO
