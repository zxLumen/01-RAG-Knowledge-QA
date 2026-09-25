# RAG Knowledge QA

本地运行的知识库检索问答应用（RAG）。默认全部能力本地闭环：dense + sparse 混合向量、Qdrant（本地持久化）存储与检索、Ollama 提供大模型推理。**也可一键切换到云端 API**（OpenAI 兼容 / 嵌入 API），便于部署成公开 demo。

## 特性

- **混合检索**：dense（bge-m3 向量相似度）+ 关键词（Text 匹配加权）双路召回
- **文件级去重**：top-k 检索保证「不同文件数 ≤ k」，同一文件最多保留 `chunks_per_file` 个高分 chunk，避免单个长文档垄断上下文
- **多厂商模型管理**：内置 OpenAI / DeepSeek / 智谱 / 通义 / Kimi / 硅基流动 / OpenRouter / **OpenCode Zen / OpenCode Go** / 本地 Ollama 预设；**对话模型与嵌入模型都是多档案模式**（新建/编辑/删除/激活切换）
- **模型名下拉选择**：从厂商实时拉取（Ollama `/api/tags` 或 OpenAI `GET /models`），带筛选框；**不回退预设**，点「获取模型列表」实际请求，失败才红字提示（如需 Key 会提示认证失败），`custom` 自定义厂商手填
- **流式问答**：`/api/query` SSE 流式输出，回答强制带 `[来源: 文件名]` 引用
- **停止回答**：生成中可随时中断（`/api/query/cancel`），保留已生成的部分内容并释放模型算力
- **增量导入**：按文件内容哈希去重（重复导入秒过），支持删除已删除文件、全量重建（recreate）
- **上传文件（演示模式）**：访客可上传 md/txt/pdf（单文件/单次/单访客均 2M），勾选后导入
- **访客隔离（演示模式）**：按 cookie 隔离——每人独立目录与 Qdrant 集合，只能看到自己的文件 + 只读样例；一周不访问自动清理，全局占满 2G 时从最旧访客回收
- **导入进度**：导入/嵌入进度 SSE 轮询、取消导入
- **集合管理**：多知识库切换、重命名、删除（本地 sqlite 持久化）
- **Web UI**：对话/导入/导入记录/状态/知识库管理/管理面板六个页面（知识库管理仅管理员可见），Tab 与会话栏可拖拽排序（顺序本地持久化），明暗主题
- **会话存档与管理**：对话记录自动同步到服务端（`qdrant_data/chats.db`），管理面板「会话管理」可查看、软删除（可恢复）、彻底删除；恢复后对话页面刷新即拉回
- **可部署 demo**：配置走环境变量或本地 JSON，写操作可用 `ADMIN_TOKEN` 保护，支持按 IP 限流

## 技术栈

| 组件 | 说明 |
| --- | --- |
| Python 3.9+ | 运行环境 |
| FastAPI + Uvicorn | Web 服务（SSE 流式） |
| Qdrant | 向量库（本地 `path=` 模式，sqlite 持久化） |
| bge-m3（Ollama 或 API） | dense (1024 维) + FastEmbed bm25 sparse 混合向量 |
| Ollama / OpenAI 兼容 API | LLM 与 Embedding 推理（可切换） |
| LangChain Text Splitters | 文档分块 |
| PyMuPDF | PDF 解析 |
| Alpine.js | 前端交互 |

## 目录结构

```
├── run_server.py            # 启动入口（uvicorn，端口 8000）
├── rag.py                   # CLI 入口（typer，等价于 -m src.main）
├── pyproject.toml           # 依赖与 pytest/ruff 配置
├── .env                     # 运行配置
├── src/
│   ├── api/                 # FastAPI 路由与 schema（routes.py, app.py）
│   ├── ingest/              # 文档加载 / 分块 / 导入管线 / 进度
│   ├── retrieval/hybrid.py  # 混合检索与文件级去重
│   ├── qa/                  # chain（上下文组装+问答）、llm（统一调用）、
│   │                        # llm_config（档案/密钥）、providers（厂商预设）、cancel（停止）
│   ├── vectorstore/         # store（写入/集合操作）、embedder、naming
│   └── config.py            # 配置（.env 驱动）
├── static/                  # 前端（index.html + app.css，Alpine.js CDN）
├── data/                    # 知识库源文件（运行数据目录，被 gitignore）
├── samples/                 # 随镜像内置的只读样例文档
├── qdrant_data/             # 本地向量库持久化（storage, imports.db, chats.db, 别名）
├── deploy/                  # 服务器部署（docker-compose / deploy.sh / entrypoint / .env.example）
├── scripts/                 # 数据构建工具（Go 标准库文档等）
└── tests/                   # pytest
```

## 安装

```bash
python3 -m pip install -e ".[dev]"
```

> macOS 自带 `/usr/bin/python3`（3.9.6）即可；如使用 Homebrew 的更新版本 Python 需重装依赖（`qdrant-client`、`fastembed`、`pymupdf` 等）。

### 前置依赖

- **Ollama**（本地模式）：`ollama serve` 后 `ollama pull qwen3:8b`（对话）与 `ollama pull bge-m3`（嵌入）；也可用云端 API 免本地依赖
- 稀疏检索用 FastEmbed `Qdrant/bm25`，首次会自动下载模型（需联网一次）

## 配置（.env）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `QDRANT_URL` | `http://localhost:6333` | 本地模式填目录路径（如 `./qdrant_data`），内存模式填 `:memory:`，也可连远程 qdrant server |
| `QDRANT_COLLECTION` | `knowledge_base` | 当前激活集合（存储名，中文会自动映射为安全文件名） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | 分块参数 |
| `TOP_K` | `3` | 检索目标：不同的文件数上限 |
| `CHUNKS_PER_FILE` | `3` | 每个选中文件最多保留的 chunk 数 |
| `RERANK_TOP_K` / `RRF_K` | `5` / `60` | 预留的粗排参数 |

**LLM / Embedding（env 为默认，`qdrant_data/llm_config.json` 覆盖）**

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama` / `openai` / `deepseek` / `zhipuai` / `dashscope` / `moonshot` / `siliconflow` / `openrouter` / `opencode` / `opencode-go` / `custom` |
| `LLM_BASE_URL` | 按 provider 预置 | 如 `https://api.deepseek.com/v1` |
| `LLM_API_KEY` | 空 | 远程 Key（也支持 `${ENV_VAR}` 引用） |
| `LLM_MODEL` | `OLLAMA_MODEL` | 模型名 |
| `LLM_TEMPERATURE` | `0` | 采样温度 |
| `EMBEDDING_PROVIDER` | `ollama` | 嵌入厂商（同上） |
| `EMBEDDING_BASE_URL` | 按 provider 预置 | 嵌入接口地址 |
| `EMBEDDING_API_KEY` | 空 | 嵌入 Key |
| `EMBEDDING_MODEL` | `DENSE_EMBEDDING_MODEL` | 嵌入模型 |
| `EMBEDDING_DIM` | `1024` | 向量维度（换模型若维度变化需重建索引） |
| `ADMIN_TOKEN` | 空 | 管理员初始密码；设置后写操作（模型配置、集合变更）需携带 `X-Admin-Token`。可在「管理面板」内改密码，改后以 `qdrant_data/admin.json`（哈希）为准 |
| `RATE_LIMIT_PER_MINUTE` | `0` | 每 IP 每分钟 `/api/query` 次数上限（0=不限） |
| `DEMO_MODE` | `false` | 开启访客隔离与上传（公开 demo 用） |

> 切换集合时集合名经 `storage_name()` 映射为 Qdrant 合法存储名，显示名与存储名通过集合别名文件关联，重启后保持。
> `OLLAMA_MODEL` / `OLLAMA_BASE_URL` / `DENSE_EMBEDDING_MODEL` 保留为旧配置兜底。

## 启动

```bash
/usr/bin/python3 run_server.py
```

打开 http://localhost:8000 即可使用。

服务重启期间的注意事项（本地模式）：

```bash
kill -9 $(lsof -tnP -iTCP:8000 -sTCP:LISTEN)   # 停旧进程
rm -f qdrant_data/.lock                         # 清残留锁
nohup /usr/bin/python3 run_server.py > /tmp/rag_server.log 2>&1 &
```

> `run_server.py` 用 uvicorn `reload=False`：**改了代码必须重启进程**，否则新路由/页面不会生效。重启后用 `curl -s localhost:8000/api/version` 确认服务已是新构建（`APP_BUILD`，CI 为 git sha）；镜像部署由 Watchtower 自动拉取 `:latest`。

## 使用

### Web UI

- **💬 对话**：提问 → 流式回答，来源为 `[文件名]`；左侧会话多开、可重命名、拖拽排序
- **📥 导入**：选择目录/文件导入；同文件重复导入自动跳过；打开「重建」可全量重建当前集合；勾选文件后可「删除所选」批量删除自己上传的文件（同时移除索引），目录与只读样例不可删；**空目录也会显示**（删除文件不会删除目录）
- **📚 导入记录**：历史导入会话详情
- **📊 状态**：集合、点数与模型信息
- **📖 知识库管理**：文件列表、集合（切换/重命名/删除）、向量点数；集合按创建时间**从新到旧**排列；**仅管理员登录后可见**（访客不显示该 Tab，接口也会拒绝）
- **🔑 管理面板**：登录后为多厂商配置档案的新建/编辑/删除/激活、嵌入模型配置；含「修改密码」「主题配置」（勾选开放给访客的主题并排序、设默认）

右上角下拉快速切换激活档案；⚙ 进入模型页；🔑 登录/退出管理员；右上角按钮切换明暗主题。

### 模型管理与 API 调用

- 进入「🔑 管理面板」新建档案：选厂商（自动带出 `base_url`）、填名称/Key/模型/温度
- **模型列表按用途分流**：对话档案只列对话模型、嵌入档案只列嵌入模型；点「获取模型列表」时会实际请求，成功填充、失败（如缺 Key/认证失败）才在红字显示原因；`custom` 自定义厂商手填模型名
- 「名称」留空自动取「厂商 + 模型」（编辑时留空保持原名）；「温度」为采样随机性（知识库问答建议 0）
- **对话模型**与**嵌入模型**各自是多档案：可新建多个、点「启用」切换当前使用项
- **嵌入厂商下拉只列支持嵌入的厂商**（deepseek/moonshot/openrouter/opencode 无 embeddings 接口，不出现）
- Embedding 档案含「向量维度」；**改动后需到「导入」页「重建」重新索引**（维度或向量空间变化会导致检索不匹配）
- 服务端设置了 `ADMIN_TOKEN` 时，模型配置写操作需管理员令牌；右上角 🔑 输入令牌登录（保存在浏览器本机）
- **OpenCode Zen / Go**：厂商预设 `OpenCode Zen`（`https://opencode.ai/zen/v1`）与 `OpenCode Go`（`https://opencode.ai/zen/go/v1`），OpenAI 兼容；在档案里手动粘贴 `OPENCODE_API_KEY` 即可调用其模型

### API 要点

| 方法 & 路径 | 说明 |
| --- | --- |
| `POST /api/query` | 流式问答（SSE：`{"type":"chunk","text":...}` / 结束时 `{"type":"done","stopped":bool,...}`），body: `{question, top_k?, collection?, session_id?}`；`session_id` 用于定位/取消本次生成 |
| `POST /api/query/cancel` | 停止正在生成中的回答，body: `{session_id}`，返回 `{cancelled: bool}`；保留已输出的部分内容 |
| `POST /api/ingest` | 导入，body: `{path?, paths?, recreate?, delete_missing?}`（阻塞至完成，返回统计） |
| `GET /api/ingest/progress` / `POST /api/ingest/cancel` | 导入进度 / 取消 |
| `GET /api/status` / `GET /api/config` | 状态 / 配置 |
| `GET /api/llm/providers` | 内置厂商预设（含 `embeddings` 标记）与已知嵌入维度 |
| `GET /api/llm/config` | 对话/嵌入 激活档案与档案列表（Key 已脱敏） |
| `POST /api/llm/profiles` · `DELETE /api/llm/profiles/{id}` | 对话档案 新建/更新 · 删除 |
| `POST /api/llm/active` | 切换激活对话档案 |
| `POST /api/llm/embedding/profiles` · `DELETE /api/llm/embedding/profiles/{id}` | 嵌入档案 新建/更新 · 删除 |
| `POST /api/llm/embedding/active` | 切换激活嵌入档案 |
| `POST /api/llm/test` | 连通性并返回模型列表，`{target: "chat"\|"embedding"}` 分流 |
| `POST /api/upload` | 上传文件（演示模式），multipart `files`，返回已保存相对路径 |
| `GET /api/status` | 状态（演示模式返回访客自己的集合） |
| `GET|POST /api/models` | 兼容旧接口（模型列表 / 切换当前模型） |
| `GET /api/files` / `GET /api/files/content` | 已索引文件 / 内容 |
| `POST /api/files/delete` | 批量删除文件（磁盘 + 索引），body: `{rels: [...]}`；访客仅限自己的上传目录，管理员可删 `data/` 下任意文件；目录与 `data/samples/` 样例跳过并在 `skipped` 返回 |
| `GET /api/version` | 服务版本与构建标识（`APP_BUILD`，CI 注入 git sha），用于确认服务是否已更新 |
| `POST /api/collections/switch` · `/rename` · `/delete` | 集合切换 / 重命名 / 删除 |
| `GET /api/imports` · `GET /api/imports/{id}` | 导入记录列表 / 详情 |
| `POST /api/chat/sync` | 对话存档同步（每次问答结束自动调用），body: `{session_id, title?, collection?, messages?, deleted?}` |
| `GET /api/chat/sessions` | 当前访客/管理员自己的会话列表（未删除） |
| `GET /api/chat/admin/list` · `POST /api/chat/admin/delete` · `POST /api/chat/admin/restore` · `POST /api/chat/admin/hard-delete` | 管理员会话管理：列表 / 软删除 / 恢复 / 彻底删除，body: `{rowid}` |

## 检索与问答原理

1. **召回**：`top_k` 目标文件数，候选池扩至 `max(top_k*4, 24)`；
   密集检索（bge-m3 余弦）+ 关键词匹配（`MatchText`，命中加权 `×1.1`）合并
2. **文件级去重**：按分数贪心先定 `top_k` 个**不同文件**（每个文件以其最佳 chunk 代表参选）；
   再从候选里为每个选中文件保留至多 `chunks_per_file` 个高分 chunk
3. **上下文组装**（`src/qa/chain.py`）：chunk 按分数降序拼入，总字符预算 `MAX_CONTEXT_CHARS=4000` 截断（约 2000 token，防超 `num_ctx=8192`）；`chunks_used` 反映实际喂入数
4. **回答**：仅基于检索到的 chunk 片段回答（不读整文件），`temperature=0`，强制标注 `[来源: 文件名]`

### 停止回答

- 回答生成期间，输入框发送按钮变为红色「⏹ 停止」；点击后前端调用 `POST /api/query/cancel`
- 服务端维护生成注册表（`src/qa/cancel.py`）：每路生成（按 `session_id`）进入时就登记，拿到流式响应后挂载连接句柄；取消即置位「已取消」事件并关闭上游连接
- 上游在客户端断开流时立即中止生成（不白白算完），前端收到终帧 `{"type":"done","stopped":true}` 后保留已生成的部分文本并标注「已手动停止」
- 请求由 daemon 线程发出、主生成器轮询取消事件，因此 prefill 阶段点停止也能即时生效
- 生成结束/异常时注册表自动释放；取消不存在的生成返回 `cancelled:false`

> 说明：当前回复只使用检索到的 chunk 片段，不使用整文件内容。

## 部署（公开 demo / 服务器）

服务器上通常没有本地 Ollama，建议全部走云端 API（OpenAI 兼容）：

```bash
# 例：嵌入用硅基流动 bge-m3，对话用 DeepSeek；写操作需管理员密码
export EMBEDDING_PROVIDER=openai
export EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
export EMBEDDING_MODEL=Pro/BAAI/bge-m3        # 或免费版 BAAI/bge-m3
export EMBEDDING_API_KEY=sk-...
export LLM_PROVIDER=openai
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_MODEL=deepseek-chat
export LLM_API_KEY=sk-...
export ADMIN_TOKEN=your-secret
export RATE_LIMIT_PER_MINUTE=20
/usr/bin/python3 run_server.py
```

- **密钥**：仅通过环境变量或本地 `qdrant_data/llm_config.json`（`.gitignore` 已忽略）保存，接口一律脱敏返回；也可在管理面板登录后控制写权限
- **嵌入切换**：换模型/维度后需重新导入（导入页「重建」），否则检索维度不匹配
- **并发**：本地 Qdrant `path=` 模式适合单实例 demo；高并发可改用 Qdrant server 模式
- 提示：DeepSeek 等对话 API **没有 embeddings 接口**，嵌入需另选厂商（OpenAI / 智谱 / 通义 / 硅基流动等）

### Docker 部署（推荐，配合已有反向代理）

镜像由 CI 构建推送 GHCR（`ghcr.io/aleiq/01-rag-knowledge-qa:<sha>` / `:latest`，命名空间须与仓库 owner 一致），服务器只 `pull` 不构建。`deploy/` 是独立编排，复用同机已有的 Caddy（按子域名反代）：

```bash
# 服务器：把 deploy/ 下的 docker-compose.yml、deploy.sh、.env.example 放到 ~/rag
cd ~/rag
cp .env.example .env && vim .env         # DEMO_MODE / ADMIN_TOKEN / 模型 key / RAG_MEM_LIMIT 等
RAG_IMAGE_TAG=<git-sha> ./deploy.sh      # 拉取镜像并启动（不传则用 latest）
```

- **首次启动即用云端模型**：`.env` 里设 `LLM_PROVIDER=opencode-go`（+ `LLM_API_KEY` 等）与 `EMBEDDING_PROVIDER=siliconflow`（+ `EMBEDDING_API_KEY` 等），首次启动会据此自动生成档案；provider id 保留，故 `x-opencode-session` 等厂商适配照常生效。
- 容器加入主页 Caddy 所在的外部网络（默认 `docker_web`，可用 `WEB_NETWORK` 改）；在主页 Caddyfile 加：
  `rag.<域名> { reverse_proxy rag:8000 }`，即可访问 `https://rag.<域名>`。
- **内存上限**：默认 `RAG_MEM_LIMIT=512m`（2C2G 机器友好）；峰值可结合 `docker stats` 调大，或依赖 swap 兜底。
- **持久化卷**：`rag-data`（Qdrant 向量、`imports.db`、`chats.db`、`admin.json`、`llm_config.json`、`ui_config.json`、访客状态）、`rag-uploads`（访客上传）；只读样例随镜像内置（仓库 `samples/`）。
- **非 root 运行**：入口脚本修复卷属主后以 `appuser` 运行；健康检查走 `/api/version`（`/api/status` 在演示模式下会新建访客，故不用）。
- **反代下限流**：`run_server.py` 已开启 `proxy_headers`，限流按真实客户端 IP（容器仅在内网可达，信任转发头安全）。
- **更新**：`git push` → CI 构建镜像 → 服务器 `RAG_IMAGE_TAG=<sha> ./deploy.sh`。
- **备份**：备份 `rag-data` 卷即可（建议纳入主页的备份脚本）。

### 演示模式（访客隔离）

公开 demo 建议开启访客隔离：

```bash
export DEMO_MODE=true
export ADMIN_TOKEN=your-secret   # 可选：保护模型配置等写操作
export RATE_LIMIT_PER_MINUTE=20
/usr/bin/python3 run_server.py
```

- 首次访问自动种 cookie（`rag_visitor`，7 天），此后文件、集合、状态都限定在该访客
- 上传限制：单文件 / 单次 / 单访客均 **2M**，仅 `.md/.txt/.pdf`；全局占用上限 **2G**
- 只读样例放在 `data/samples/`，所有访客可见；现有 `data/` 其它内容在演示模式下不可见
- 超过 7 天未访问的访客会被惰性清理（删集合 + 目录）；全局超 2G 时从最旧访客开始回收
- 访客集合名为 `visitor_<id>`（每人一个），记录在 `qdrant_data/visitors.json`

### 管理员视角（演示模式下）

演示模式默认所有请求都按访客隔离。要切到管理员视角（查看共享知识库与全部文件）：

1. 服务端设置 `ADMIN_TOKEN` 启动（作为初始密码）：`ADMIN_TOKEN=your-secret DEMO_MODE=true /usr/bin/python3 run_server.py`
2. 打开「🔑 管理面板」Tab：未登录显示登录页，输入密码（`POST /api/admin/verify` 校验）；登录后即为模型/嵌入配置页
3. 之后请求自动带 `X-Admin-Token`，后端识别为管理员：`/api/status`、`/api/files`、`/api/query`、`/api/imports` 等都返回共享/管理员数据；点右上角 ⏻「退出管理员」恢复访客视角

> **修改密码**：管理面板内「修改密码」卡片（`POST /api/admin/password`），改后存于 `qdrant_data/admin.json`（PBKDF2 哈希，文件优先于 `ADMIN_TOKEN`）。删除该文件即回退到环境变量密码。
> 该密码用于保护模型配置写操作与集合变更（`/api/collections/switch|rename|delete`）。未设置时无法进入管理员视角。
> **访客上传**走自己的隔离目录，无需管理员令牌（演示模式即可用）。
> 「🔑 管理面板」对所有人可见，但未登录只显示登录页。「📖 知识库管理」Tab 仅管理员登录后可见。访客提问/记录始终走自己的集合，客户端传来的集合名会被忽略；集合切换/重命名/删除需管理员令牌，未登录一律拒绝。
> **主题配置**：管理员在「主题配置」中开放的默认主题对新访客生效（首屏由服务端注入，无主题闪烁）；老访客若本地保存的主题仍在开放列表内则保留自己的选择。

## 数据与存储

- `data/`：源文档（支持 `.md` / `.txt` / `.pdf`），导入时按目录递归发现
- `qdrant_data/`：本地 qdrant 持久化（每集合一个 sqlite）、`imports.db`（导入历史/文档哈希）、`collection_aliases.json`（显示名 ↔ 存储名）、`llm_config.json`（模型档案/密钥）、`llm_model.json`（旧版单模型，已兼容迁移）
- PDF 按页切分，chunk 携带 `source`（相对路径）、`filename`、`page`、`chunk_index` 元数据

## 测试与检查

```bash
/usr/bin/python3 -m pytest -q    # 69 passed
ruff check src/
```

日常修改后建议两者都跑一遍。

## 已知实现细节

- 本地 sqlite 模式（macOS sqlite 编译为 `THREADSAFE=2`）跨线程写入曾触发 `check_same_thread` 报错；`store.py` 已对所有写操作开启 `force_disable_check_same_thread=True` 并用模块级 `RLock` 串行化，导入与集合操作可并发触发
- 大模型推理串行排队：同一时间只有一个会话在生成，其余会话的输入不阻塞（每会话瞬态 `busy` 状态）
- 本机 qwen3:8b 首个 token 延迟约 20+ 秒（prefill 慢），回答生成流畅后取消可即时中断
- 新建集合的向量维度取自嵌入配置（`llm_config.embedding.dim`），换维需重建
- 注册表/取消逻辑有独立单元测试（`tests/test_cancel.py`），配置存储/厂商预设/SSE 解析亦有测试

## 脚本

- `scripts/build_stdlib.py`：抓取 Go 标准库中文文档（studygolang pkgdoc + `go doc -all` 英文补洞），生成 `data/go/go-docs/00-标准库/`
- `scripts/fetch_go_docs.py`：抓取 golang.ac.cn 中文文档，生成可导入的 markdown
- `export_notes.py`：通过 AppleScript 把 Apple 备忘录按文件夹导出为 markdown 到 `data/备忘录/`