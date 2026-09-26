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
- 上线流程：`git push`（CI 构建镜像推 GHCR）→ 服务器 `cd /home/ubuntu/rag && RAG_IMAGE_TAG=<sha> ./deploy.sh`。
- 服务器：`ubuntu@124.156.168.32`，域名 `rag.zxlumen.cn`，复用主页 Caddy + `docker_web` 网络。

## 约定

- 提交用 Conventional Commits + 中文 subject；只在明确要求时提交。
