#!/usr/bin/env bash
# 把 Chroma 的默认 embedding 模型取到 vendor/，供 Dockerfile.vendored 使用。
#
#   bash scripts/fetch-model.sh
#
# 为什么不用 Chroma 自己下载：它从 AWS S3 取一个 79 MB 的 tar.gz，
# 中国大陆实测 15-50 KB/s（一小时以上），而且它的下载器没有重试，
# 连接一挂就无限等待。
#
# 这里改为逐个文件取。默认走 hf-mirror.com（国内可满速），
# 失败时自动回落到 huggingface.co。文件内容与 Chroma 的 tar 包一致 ——
# 都源自 sentence-transformers/all-MiniLM-L6-v2 的 onnx 导出。
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/vendor/chroma-onnx/all-MiniLM-L6-v2/onnx"
REPO_PATH="sentence-transformers/all-MiniLM-L6-v2/resolve/main"
MIRRORS=("https://hf-mirror.com" "https://huggingface.co")

# 文件名 期望字节数（用于校验，宁可重下也不要半截文件进镜像）
FILES=(
  "model.onnx 90405214"
  "tokenizer.json 711661"
  "vocab.txt 231508"
  "config.json 612"
  "tokenizer_config.json 350"
  "special_tokens_map.json 125"
)

mkdir -p "$DEST"
printf '  目标目录 %s\n\n' "$DEST"

for entry in "${FILES[@]}"; do
  name="${entry%% *}"
  want="${entry##* }"
  out="$DEST/$name"
  # Hugging Face stores only the ONNX weights under /onnx. Tokenizer and
  # configuration files live at the repository root.
  remote_path="$name"
  [ "$name" != "model.onnx" ] || remote_path="onnx/$name"

  if [ -f "$out" ] && [ "$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out")" -ge "$((want * 95 / 100))" ]; then
    printf '  ✓ %-24s 已存在 %s\n' "$name" "$(du -h "$out" | cut -f1)"
    continue
  fi

  ok=
  for base in "${MIRRORS[@]}"; do
    printf '  ↓ %-24s from %s\n' "$name" "${base#https://}"
    # --continue-at 断点续传；--speed-time 让停滞的连接主动断开重连，
    # 而不是像 Chroma 的下载器那样挂住不动。
    if curl -fL --continue-at - \
            --retry 20 --retry-delay 3 --retry-all-errors \
            --speed-limit 2000 --speed-time 30 --connect-timeout 15 \
            --progress-bar -o "$out" "$base/$REPO_PATH/$remote_path"; then
      got=$(stat -c%s "$out" 2>/dev/null || stat -f%z "$out")
      if [ "$got" -ge "$((want * 95 / 100))" ]; then ok=1; break; fi
      printf '    大小不对（%s，期望约 %s），换下一个源\n' "$got" "$want"
    fi
  done
  [ -n "$ok" ] || { echo "  ✗ $name 下载失败"; exit 1; }
done

echo
echo "  完成："
du -sh "$DEST" | sed 's/^/    /'
ls -la "$DEST" | tail -n +4 | awk '{printf "    %-26s %s\n", $9, $5}'
