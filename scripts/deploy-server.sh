#!/usr/bin/env bash
# 在服务器上构建并启动服务。改完代码重跑这一条即可。
#
#   cd ~/ustudent-ai && git pull && bash scripts/deploy-server.sh
#
# 构建在服务器上进行，索引也在构建阶段生成 —— 镜像因此是自洽的，
# 不依赖任何人本地的文件。
set -euo pipefail

IMAGE=ustudent-ai:latest
NAME=ustudent-ai
PORT=${PORT:-8000}

log() { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

cd "$(dirname "$0")/.."

log "1/5 · 检查 .env"
if [ ! -f .env ]; then
  die "找不到 .env。先复制模板并填入你的 API key：
       cp .env.example .env && nano .env"
fi
grep -q '^LLM_API_KEY=.\+' .env || die ".env 里的 LLM_API_KEY 是空的"
echo "  .env 就位（内容不会被打印）"
grep -E '^LLM_(BASE_URL|MODEL)=' .env | sed 's/^/  /'

log "2/5 · 构建镜像"
# pip 走清华源；大陆直连 pypi.org 慢到会超时。
# --platform 显式指定，避免在 ARM 机器上构建出 ECS 跑不了的镜像。
DOCKER_BUILDKIT=1 docker build \
  --platform linux/amd64 \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t "$IMAGE" . || die "构建失败。若卡在拉取基础镜像，检查镜像加速：sudo docker info | grep -A3 'Registry Mirrors'"

log "3/5 · 替换旧容器"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --env-file .env \
  -e OPS_DB_PATH=/app/runtime/ops.sqlite3 \
  -v ustudent-ai-ops:/app/runtime \
  -p "${PORT}:8000" \
  --memory 1500m \
  --health-cmd 'curl -fsS http://localhost:8000/health || exit 1' \
  --health-interval 30s --health-timeout 5s --health-start-period 90s --health-retries 3 \
  "$IMAGE" >/dev/null
echo "  容器已启动（--restart unless-stopped，重启服务器会自动拉起）"

log "4/5 · 等待就绪"
# 冷启动要加载 80MB 的 embedding 模型，给足 90 秒。
for i in $(seq 1 45); do
  if curl -fsS -m 3 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "  就绪（等了约 $((i * 2)) 秒）"
    ok=1; break
  fi
  sleep 2
done
[ "${ok:-}" = 1 ] || { docker logs --tail 40 "$NAME"; die "启动超时，日志见上"; }

log "5/5 · 冒烟测试"
printf '  /health      %s\n' "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health)"
printf '  /            %s\n' "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/)"
printf '  /docs        %s\n' "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/docs)"
printf '  /rag-ask     %s\n' "$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  http://localhost:${PORT}/rag-ask -H 'Content-Type: application/json' \
  -d '{"question":"How many credits do I need to graduate?"}')"

IP=$(curl -s -m 5 https://ifconfig.me 2>/dev/null || echo '<你的公网IP>')
cat <<NEXT

  上线了：

    http://${IP}:${PORT}/        演示页
    http://${IP}:${PORT}/docs    API 文档

  常用命令：
    docker logs -f ${NAME}          看日志
    docker restart ${NAME}          重启
    docker stats --no-stream        看内存占用

  访问不了的话，检查腾讯云控制台的防火墙是否放行了 ${PORT} 端口。
NEXT
