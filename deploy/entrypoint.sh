#!/bin/sh
# 容器入口:修复挂载卷属主,清理 qdrant 残留锁,然后以非 root 运行。
set -e

mkdir -p /app/qdrant_data /app/data/visitors
rm -f /app/qdrant_data/.lock

# /app/data 顶层也要可写:应用会在其中创建 collection_order.json /
# collection_aliases.json。samples 子目录为镜像内只读内容,故不递归。
# 具名卷首次挂载属主为 root,这里改给 appuser(非递归:首次 chown 顶层即可,
# 之后新文件都由 appuser 创建)。
chown appuser:appuser /app/data /app/qdrant_data /app/data/visitors

exec gosu appuser env HOME=/home/appuser python run_server.py
