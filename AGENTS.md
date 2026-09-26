# AGENTS.md

本文件记录本仓库的本地开发/验证约定，供 AI 助手与协作者遵循。

## 本地开发与验证

- **本地服务由 launchd 托管**：`com.rag-knowledge-qa.server`（`~/Library/LaunchAgents/com.rag-knowledge-qa.server.plist`）。
  - 工作目录=仓库根；命令 `/usr/bin/python3 run_server.py`；端口 `8000`；`KeepAlive=true`。
  - 环境：`DEMO_MODE=true`、`ADMIN_TOKEN=demo123`。
- **改了代码必须重启本地服务**：`run_server.py` 使用 `reload=False`，不重启则新路由/改动不生效（典型表现：新接口返回 404/405）。
  ```bash
  launchctl kickstart -k gui/$(id -u)/com.rag-knowledge-qa.server
  curl -s localhost:8000/api/version   # 确认已起来
  ```
- **本地验证不要用 Docker**（构建耗时）；直接拉起服务即可。需要隔离环境时，用临时目录 + `uvicorn` 指定其它端口，避免动到真实 `qdrant_data/`。
- **本地管理员**：`admin.json` 存在时 `ADMIN_TOKEN` 不再用于登录；忘记密码用后门重置即可
  `POST /api/admin/reset`（body `{admin_token, new_password}`，密码仅支持 ASCII），或登录页「忘记密码？用 ADMIN_TOKEN 重置」填 `demo123`。

## 测试与检查

```bash
pytest -q
ruff check src tests
```

## 上线与部署

- **默认只动线下（本地）环境**：除非用户当次明确说「上线/部署」，否则不要 push、不要部署服务器。
  「本地验证」不等于「可以上线」；需要上线必须得到当次显式指令（不要沿用之前的授权）。
- 上线流程：`git push`（CI 构建镜像推 GHCR）→ 服务器 `cd /home/ubuntu/rag && ./deploy.sh <sha>`。
  `deploy.sh` 会按 tag 同步 `docker-compose.yml`，所以**服务器 `.env` 的 `RAG_IMAGE_TAG` 不会自动更新**，
  手工 `docker compose up -d` 前务必先改 `.env`，否则会回滚到旧镜像。
- 服务器：`ubuntu@43.161.241.32`（2026-09 由 124.156.168.32 迁来），目录 `/home/ubuntu/rag`，
  域名 `rag.zxlumen.cn`，复用主页 Caddy + `docker_web` 网络。
  - `rag` 只 `expose 8000`（未发布端口），宿主机 `curl localhost:8000` 连不上；
    排查用 `docker exec rag curl -fsS http://localhost:8000/api/version` 或走公网域名。
  - 服务器专用改动写在 `docker-compose.override.yml`（当前挂载 `data/go`、`data/python-docs`），
    不会被 `deploy.sh` 的 compose 同步覆盖。
- **向量库已是独立 Qdrant 服务**（`qdrant/qdrant:v1.19.0`，容器 `rag-qdrant`）：
  - 只挂 `rag_rag-internal` 私有网络、不发布端口；`rag` 通过 `QDRANT_URL=http://qdrant:6333` 访问。
  - 集合创建时 dense/sparse/payload 全部 `on_disk`；`rag` 内存降到 ~160m（原先内嵌需 ~900m）。
  - 迁移/回滚：旧内嵌数据仍完整保留在 `rag_rag-data` 卷（`collection/`，188M）。
    再迁回去只需把 `.env` 的 `QDRANT_URL` 改回 `./qdrant_data` 并 `docker compose up -d rag`；
    反向迁移用 `python -m src.migrate_server --src <内嵌路径> --dst http://qdrant:6333`（需先停 `rag`）。

## 访客配额（2026-09 调整）

- 线上取值：`VISITOR_MAX_POINTS=5000`、`VISITOR_GLOBAL_MAX_POINTS=150000`、`VISITOR_MINT_PER_HOUR=20`。
- **为什么是 5000**：共享库里最大的单文件 `09-新特性.md` 有 3004 分块，`go` 全库 3102，
  python 全库 11373。3000 会让「挑任意单个文件导入」都失败（用户实际踩到的坑）；
  5000 让任意单文件都能导入，且单访客仍只用掉全局预算的 1/30。
  批量选整个目录仍会被拒，这是有意的，别再往上调。
- 磁盘换算：约 12KB/分块。全局 150000 分块用满 ≈ Qdrant 磁盘 1.8GB、Qdrant anon 50–80MB。
- 铸身份限流**落盘**在 `qdrant_data/mint_hits.json`（在 `rag_rag-data` 卷里）。
  早前是进程内存计数，每次重新部署都被清零，等于白送绕过窗口，别再改回内存实现。
- 访客没有「数量」上限：闲置 7 天会连身份、上传、集合一起删；另有 2GB 上传全局配额。

## 宿主机内存（3.7G，与多个项目共用）

- 排查用 `~/bin/mem-watch.sh`（= 仓库 `deploy/mem-watch.sh`），cron 每 10 分钟跑一次，
  日志 `~/logs/mem-watch.log`（超 8MB 自截断）。**看内存必须看 anon，不要看容器总量**：
  容器的 file cache 是可回收的，把两者相加会得到「占了 4G」这种假象。
- 容器内存上限（已在线生效 + 写进各自 compose，重建后仍生效）：
  - `mailserver` 1g —— compose 在 `/home/ubuntu/mail/docker-compose.yml`
  - 监控栈 512m/512m/256m/128m（grafana/alloy/cadvisor/node-exporter）——
    compose 在 `/home/ubuntu/zxLumen-Blog/docker/docker-compose.yml`，且属 `monitoring` profile，
    改完校验要用 `docker compose --profile monitoring config`，不带 profile 看不到这几个服务。
  - `QDRANT_MEM_LIMIT` 保持 800m：Qdrant anon 只有 ~34M，降限收益小、风险大，别动。
- 在线加/改内存上限用 `docker update --memory`（不重建容器、零中断），
  但**不会持久化**，必须同步写进对应 compose 文件。
- 共享宿主：只允许 `docker image prune -f`，绝不用 `-a`（会误删其它项目在用但暂未运行的镜像）。
  悬空镜像一般已被 `deploy.sh` 清掉，通常回收为 0B，别把磁盘当瓶颈。

## 约定

- 提交用 Conventional Commits + 中文 subject；只在明确要求时提交。
