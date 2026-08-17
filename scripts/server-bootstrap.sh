#!/usr/bin/env bash
# 服务器初始化 —— 在一台全新的 Ubuntu 上跑一次就够。
#
#   curl -fsSL https://raw.githubusercontent.com/<你的用户名>/ustudent-ai/main/scripts/server-bootstrap.sh | bash
#   或者：把本文件内容粘进网页终端执行
#
# 做三件事：装 Docker、配国内镜像加速、开机自启。
# 大陆机器直连 Docker Hub 通常很慢或超时，所以镜像加速不是优化而是必需。
set -euo pipefail

log() { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }

log "1/4 · 系统信息"
. /etc/os-release && echo "  $PRETTY_NAME  $(uname -m)"
echo "  内存: $(free -h | awk '/^Mem:/{print $2}')　磁盘: $(df -h / | awk 'NR==2{print $4}') 可用"

if command -v docker >/dev/null 2>&1; then
  log "2/4 · Docker 已安装，跳过"
  docker --version | sed 's/^/  /'
else
  log "2/4 · 安装 Docker（用清华源，比官方源快很多）"
  sudo apt-get update -qq
  sudo apt-get install -y -qq ca-certificates curl gnupg
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  docker --version | sed 's/^/  /'
fi

log "3/4 · 配置镜像加速"
# mirror.ccs.tencentyun.com 只在腾讯云内网可用，命中率最高；后面几个是公共兜底。
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<'JSON'
{
  "registry-mirrors": [
    "https://mirror.ccs.tencentyun.com",
    "https://docker.m.daocloud.io",
    "https://dockerproxy.com"
  ],
  "log-driver": "json-file",
  "log-opts": { "max-size": "10m", "max-file": "3" }
}
JSON
sudo systemctl daemon-reload
sudo systemctl restart docker
sudo systemctl enable docker >/dev/null 2>&1
echo "  已配置，并限制日志大小（否则容器日志会慢慢吃满 40GB 磁盘）"

log "4/4 · 免 sudo 使用 docker"
sudo usermod -aG docker "$USER"
echo "  已把 $USER 加入 docker 组"

log "完成"
cat <<'NEXT'
  下一步：

    1) 退出并重新登录终端（让 docker 组生效）
    2) 验证：  docker run --rm hello-world
    3) 部署：  bash scripts/deploy-server.sh

  如果 hello-world 拉不下来，说明镜像加速没生效，检查：
    sudo docker info | grep -A3 "Registry Mirrors"
NEXT
