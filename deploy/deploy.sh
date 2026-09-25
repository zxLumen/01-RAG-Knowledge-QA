#!/usr/bin/env bash
# 在服务器上的 deploy/ 目录内运行:从 GHCR 拉取新镜像并滚动更新 RAG 容器。
# 服务器不构建镜像(CI 负责构建推送)。
#
# 用法:
#   RAG_IMAGE_TAG=<git-sha> ./deploy.sh     # 指定版本(推荐,可回滚)
#   ./deploy.sh                             # 不传则用 .env 的 RAG_IMAGE_TAG / latest
set -euo pipefail
cd "$(dirname "$0")"

IMAGE_TAG="${1:-${RAG_IMAGE_TAG:-latest}}"
[[ "${EUID}" -eq 0 ]] || SUDO="sudo -n"
echo "[rag-deploy] RAG_IMAGE_TAG=${IMAGE_TAG} (工作目录: $(pwd))"

[[ -f .env ]] || { echo "[rag-deploy] 缺少 .env,请先 cp .env.example .env 并填写"; exit 1; }

# sudo 默认 env_reset 会清掉变量,用 env 显式传回 compose
${SUDO:-} env RAG_IMAGE_TAG="${IMAGE_TAG}" docker compose pull rag
${SUDO:-} env RAG_IMAGE_TAG="${IMAGE_TAG}" docker compose up -d
# 仅清理悬空镜像(无标签层)。共享宿主上不要用 -a 全局清理,以免误删其他项目
# 未运行但仍在用的镜像;旧版本 RAG 镜像请按需手动清理。
${SUDO:-} docker image prune -f

${SUDO:-} docker compose ps
echo "[rag-deploy] 完成;健康检查: docker exec rag curl -fsS http://localhost:8000/api/version"
echo "[rag-deploy] 公网入口(需主页 Caddy 已配置 rag.<域名>): https://rag.<域名>/api/version"
