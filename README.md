# PRVolve

**自进化 Agent Runtime Harness 的 PR 审查与修复智能体**

PRVolve 面向研发流程中的 Pull Request，将本地规则、仓库级工具和多 Agent 协作组织成可恢复的审查链路。系统只把有位置、有证据且通过复核的结论发布为正式 Finding；证据不足的模型判断会留在 Suggestion 区等待人工确认。

它覆盖从风险发现、证据复核、安全修复、结果验证到反馈学习和版本回滚的完整闭环，同时如实记录模型调用、工具证据、降级状态和门禁结果。

> PRVolve 是当前产品名称。为保持已有部署兼容，Python 包名、环境变量前缀、Docker Compose 项目组和修复分支前缀仍沿用 `evoagent` / `EVOAGENT_*`。

## 核心能力

- **证据驱动的多 Agent 审查**：Lead 负责任务拆分与综合，Security 和 Correctness/Reliability 并行分析，Critic 独立寻找反例并复核结论。
- **真实仓库上下文**：按角色授权全仓搜索、文件读取、符号与调用关系、测试定位、AST、Git 历史、静态扫描和管理员配置的项目检查。
- **单一事实源的可恢复执行**：一条 append-only 的 `checkpoints` 日志记录全部节点进度、模型调用与工具调用；任务状态、Run Trace、成本账本和报告都是它的投影，没有第二套恢复机制。断点续跑精确到单个 Worker Assignment。
- **受控修复**：在线修复使用 LLM Unified Patch；补丁经过路径、AST/CST、编译和测试门禁后，只写入独立分支和 Draft PR。代码库同时保留确定性 SafeFixer，用于有限规则的安全转换。
- **Prompt / Skill 进化**：将确认后的误报、漏报、坏修复和执行异常回流为候选版本，通过 Validation、Holdout、安全与非退化门禁后再激活或回滚。

## 工作流程

```text
PR Webhook / JSON API
          │
          ▼
   ReviewService                                     serving/
          │
          ▼
   ReviewHarness ── AgentRuntime                      review/
   （节点调度 · 执行预算 · 节点重试 · 任务取消 · Checkpoint 续跑）
          │
┌─────────▼───────────────────────────────────────┐
│ 1. planning                          PLANNING   │
│    ├─ 解析 Diff                                  │
│    ├─ Scanner 确定性扫描                          │
│    ├─ 准备仓库 / Context                          │
│    └─ Lead 拆任务生成 Assignment                  │
└─────────┬───────────────────────────────────────┘
┌─────────▼───────────────────────────────────────┐
│ 2. executing                        EXECUTING   │
│    ├─ Worker 并行执行（各自一个子节点）            │
│    ├─ 收集 WorkerResult                          │
│    ├─ Lead assess                               │
│    └─ 必要时最多 N 轮 revision                    │
└─────────┬───────────────────────────────────────┘
┌─────────▼───────────────────────────────────────┐
│ 3. reviewing                        REVIEWING   │
│    ├─ Critic 独立复核（隐藏候选来源）              │
│    ├─ Lead final / arbitrate                    │
│    ├─ FindingGate                               │
│    └─ 生成 ReviewReport                          │
└─────────┬───────────────────────────────────────┘
          ▼
       SUCCESS

  每个节点 · 每次模型调用 · 每次工具调用 ──▶ checkpoints（append-only）
                                              │
                     TaskState · Run Trace · 成本账本 · 报告 = 它的投影
```

**节点边界的划分规则**：*参与收敛循环的工作属于拥有该循环的节点；循环之外的一次性判定属于下一个节点。* 这就是 revision 在 `executing`、Critic 在 `reviewing` 的原因，也是三个节点都承载真实成本的原因——checkpoint 只有落在有成本的地方才有意义。

单次评审的实测分布（同一组 fake 模型，共 8 次调用）：

| 节点 | TaskState | 模型调用 | 工具调用 | 子节点 |
| --- | --- | ---: | ---: | --- |
| `planning` | PLANNING | 1 | 2 | `scan`、`delegate` |
| `executing` | EXECUTING | 5 | 0 | `work:<assignment>`、`assess-N` |
| `reviewing` | REVIEWING | 2 | 1 | `critic`、`arbitrate` |

Lead 在三个节点里各出现一次，分别负责**打开**（delegate）、**收敛**（assess）、**关闭**（finalize）工作。节点是恢复维度，角色是协作维度，两者正交：每次 Lead 激活都是一个全新的无状态 `AgentLoop`，上下文显式传入，因此节点边界干净地落在两次激活之间，恢复时不需要序列化任何对话状态。

## 模块结构

| 目录 | 职责 | 依赖方向 |
| --- | --- | --- |
| `core/` | 领域类型与纯函数（Finding、Diff 解析、门禁、Rule ID 策略） | 叶子，不依赖任何内部模块 |
| `session/` | append-only Checkpoint 日志与全部投影（状态、Trace、成本账本） | 只依赖 `core/` |
| `llm/` | 模型客户端与上下文预算管理 | `core/`、`session/` |
| `tools/` | 工具目录与参数校验、仓库工具套件、Agent Skill | `core/`、`session/` |
| `agents/` | 与领域无关的 Agent 循环、角色 Prompt、输出解析、记忆 | `llm/`、`tools/`、`session/` |
| `review/` | 编排：ReviewHarness、AgentRuntime、Agentic 协议、合并与发布判定 | 以上全部 |
| `store/` | SQLite / PostgreSQL 持久化，不含业务规则 | `core/`、`session/` |
| `serving/` | HTTP、任务队列、租户、GitHub 接入、可观测性 | 以上全部 |
| `evolution/`、`eval/` | 离线进化与评测，主链路的消费者 | 以上全部 |

**分层规则**：依赖严格单向向下；`agents/loop.py` 不认识 `Finding`、`task_id` 或 `store`；`session/` 是唯一写入运行进度的地方；`store/` 只做持久化。

### Finding 与 Suggestion

- `findings` 是可发布结论。普通模型发现必须经过 Lead 采纳、Critic 复核、位置校验，并引用与结论匹配的仓库或工具证据。
- `suggestions` 保存可能有价值但证据尚不完整的判断，不会自动写入 PR 评论，也不计入正式 Finding 指标。
- 本地确定性规则提供最低能力边界；模型、角色或工具失败会明确记录为降级，不会伪装成完整的多 Agent 执行。

只提交 Unified Diff 时，系统能分析增量代码，但无法可靠证明跨文件调用、配置约束和测试影响。审查隐藏逻辑时，应同时提供已检出到目标提交的仓库路径：

```json
{
  "repository": "owner/repository",
  "repository_root": "D:\\work\\repository",
  "mode": "agentic",
  "diff": "<unified diff>"
}
```

Docker 部署时，`repository_root` 必须是容器内可读路径，推荐通过只读卷挂载被审仓库。批量评测会在模型调用前校验 checkout HEAD，避免使用错误版本的源码作为证据。

## 当前评测结果

当前推荐基线使用固定实现版本、固定仓库 SHA 和相同模型配置完成两轮全量执行；两轮之间没有修改代码、Prompt 或发布门禁，也没有复用模型缓存。模型输出全部保留，人工复核仅用于处理标签别名以及额外 Finding 的 `required`、`optional`、`invalid`、`duplicate` 判定。

| 指标 | 结果 | 含义 |
| --- | ---: | --- |
| Precision | **89.29%** | 正式 Finding 中属于必修问题的比例 |
| Recall | **96.15%** | 目标缺陷被正式 Finding 命中的比例 |
| F1 | **92.59%** | Precision 与 Recall 的调和平均 |
| 高风险问题召回率 | **100%** | 高风险目标缺陷的命中比例 |
| 已修复 PR 无误报率 | **96.15%** | 正确修复方向未收到错误正式 Finding 的比例 |
| 证据准确率 | **100%** | 命中 Finding 的证据与结论一致比例 |
| 精确行定位率 | **96.00%** | 命中 Finding 定位到目标新增行的比例 |
| 任务执行成功率 | **100%** | 评测任务进入完成状态的比例 |

完整原始报告、复核报告、SHA-256、逐项裁决与限制记录在 [双轮仓库上下文基线](benchmarks/python_all_13_repo_canary_v1_baseline.json)。

这组基线刻意选择了具有公开修复证据的明显缺陷，适合验证“主要问题能否被稳定发现”，不能外推为任意生产 PR 的准确率。两轮仍出现一次真实漏检、少量无效或重复评论和子角色降级，因此项目不宣称达到完全稳定。**已修复 PR 无误报率不是自动修复成功率**；自动补丁能力尚未建立同口径的成功率基线。

<details>
<summary>历史评测与机器可读记录</summary>

- [确定性规则与人工确认基线](benchmarks/pr_diff_100_v2_baselines.json)
- [Python 安全漏洞配对基线](benchmarks/python_security_pairs_v1_baseline.json)
- [Python 稳定性回归基线](benchmarks/python_interview_canary_20_v1_baseline.json)
- [仓库上下文扩展基线](benchmarks/python_three_repo_extension_v1_baseline.json)
- [明显严重缺陷基线](benchmarks/obvious_severe_smoke_v2_baseline.json)

历史结果用于解释能力演进和失败原因。不同提交、不同仓库上下文或混合缓存产生的数字不能直接横向比较，也不能替代当前统一基线。

</details>

## 快速开始

### 本地运行

要求 Python 3.11。下面示例使用 PowerShell：

```powershell
git clone https://github.com/XKYuanii/PRVolve.git
Set-Location PRVolve
python -m pip install -r requirements.txt

$authBytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($authBytes)
$env:EVOAGENT_AUTH_REQUIRED = 'true'
$env:EVOAGENT_AUTH_SECRET = [Convert]::ToBase64String($authBytes)
$env:EVOAGENT_BOOTSTRAP_ADMIN_USERNAME = 'admin'
$env:EVOAGENT_BOOTSTRAP_ADMIN_PASSWORD = '<至少 10 个字符的密码>'

$env:EVOAGENT_LLM_PROVIDER = 'deepseek'
$env:EVOAGENT_DEEPSEEK_API_KEY = '<你的 API Key>'

python -m evoagent
```

服务默认监听 `http://127.0.0.1:8080`。打开该地址即可进入管理台。

Bootstrap 管理员只在用户名尚不存在时创建，重启不会覆盖已有密码。服务可以在未配置模型时启动并提供健康检查，但创建 Agentic 审查任务前必须配置模型。

### Docker Compose

复制 [.env.example](.env.example) 为 `.env`，至少设置以下值：

```env
EVOAGENT_POSTGRES_PASSWORD=<数据库强密码>
EVOAGENT_AUTH_SECRET=<至少 32 字节的随机密钥>
EVOAGENT_BOOTSTRAP_ADMIN_USERNAME=admin
EVOAGENT_BOOTSTRAP_ADMIN_PASSWORD=<至少 10 个字符的密码>
EVOAGENT_LLM_PROVIDER=deepseek
EVOAGENT_DEEPSEEK_API_KEY=<你的 API Key>
```

启动服务：

```powershell
docker compose -p evoagent up --build -d
docker compose -p evoagent ps
Invoke-RestMethod http://127.0.0.1:8080/health
```

Compose 使用 PostgreSQL、Redis 和 PRVolve 服务，并统一归入 `evoagent` 项目组。默认只绑定本机 `8080` 端口。

### 其他模型端点

PRVolve 支持 OpenAI Chat Completions 兼容端点：

```powershell
$env:EVOAGENT_LLM_PROVIDER = 'custom'
$env:EVOAGENT_LLM_BASE_URL = 'https://example.com/v1'
$env:EVOAGENT_LLM_API_KEY = '<token>'
$env:EVOAGENT_LLM_MODEL = '<model-name>'
python -m evoagent
```

全部配置项及 OpenRouter 预设见 [.env.example](.env.example)。API Key 只应通过环境变量或本地 `.env` 注入，不要写入代码或提交到仓库。

四角色模式的 `EVOAGENT_AGENT_TOKEN_BUDGET` 是每个角色的总预算。预算过低可能导致 Worker 已取得证据但无法完成结构化输出，报告会把这种情况记录为降级执行。

## 创建审查任务

登录并创建异步任务：

```powershell
$session = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/auth/login `
  -ContentType 'application/json' `
  -Body (@{
    username = 'admin'
    password = '<你的密码>'
  } | ConvertTo-Json)

$headers = @{Authorization = "Bearer $($session.access_token)"}
$payload = @{
  repository = 'demo/service'
  pull_request = 12
  mode = 'agentic'
  repository_root = 'D:\work\service'
  diff = "diff --git a/service.py b/service.py`n--- a/service.py`n+++ b/service.py`n@@ -10,2 +10,3 @@`n+result = len(value)"
} | ConvertTo-Json

$task = Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8080/v1/reviews?async=true' `
  -Headers $headers `
  -ContentType 'application/json' `
  -Body $payload

Invoke-RestMethod -Headers $headers "http://127.0.0.1:8080/v1/tasks/$($task.task_id)"
```

获取 Markdown 报告：

```powershell
Invoke-WebRequest -Headers $headers `
  "http://127.0.0.1:8080/v1/tasks/$($task.task_id)/report"
```

## 安全修复

PRVolve 包含两类修复实现：

- 在线 `/v1/tasks/{id}/fix` 在模型可用时使用 `VerifiedPatchFixer` 生成 Unified Patch，并在隔离 checkout 中执行路径边界、AST/CST、编译和管理员配置的测试门禁；未配置模型时只返回修复建议。
- `SafeFixer` 是面向少量确定性规则的受限转换实现，当前不作为在线 `/fix` 的默认选择。

修复通过后写入新的 `evoagent/fix-pr-*` 分支并创建 Draft PR，不直接修改原 PR 分支。设置 `EVOAGENT_REPAIR_TEST_COMMAND` 可以加入项目级验证命令；未通过前后测试对比的补丁不会发布。

## Prompt 与 Skill 进化

确认反馈可以标记为误报、漏报、坏修复或执行异常。进化引擎会：

1. 按租户和仓库隔离失败案例；
2. 生成并去重 Prompt 或 Skill 候选；
3. 在版本化 Validation 与 Holdout 上回放；
4. 检查安全性、指标提升和非退化条件；
5. 激活候选，或在灰度错误预算触发时回滚。

动态 Skill 还需通过 manifest、签名、权限和隔离进程检查。调用方提交的回归分数不会直接用于激活。

## GitHub 接入

Webhook 地址为：

```text
https://<公网域名>/webhooks/github
```

推荐只订阅 `Pull requests`，PRVolve 处理 `opened`、`reopened` 和 `synchronize`。Webhook 使用 HMAC-SHA256 签名验证：

```powershell
$webhookBytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($webhookBytes)
$env:EVOAGENT_GITHUB_WEBHOOK_SECRET = [Convert]::ToBase64String($webhookBytes)
$env:EVOAGENT_GITHUB_TOKEN = '<fine-grained PAT>'
$env:EVOAGENT_AUTO_POST_REVIEW = 'true'
```

最小权限按功能配置：

- 读取私有仓库 Diff：`Contents: Read`、`Pull requests: Read`；
- 回写审查评论：`Pull requests: Read and write`；
- 创建修复分支和 Draft PR：`Contents: Read and write`、`Pull requests: Read and write`。

公网部署必须保持登录认证，使用随机 `EVOAGENT_AUTH_SECRET`。推荐通过反向代理仅公开 `/webhooks/github`，不要直接暴露整个管理台。

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/v1/auth/login` | 登录并获取 Bearer Token |
| `POST` | `/v1/reviews` | 创建同步审查任务 |
| `POST` | `/v1/reviews?async=true` | 创建异步审查任务 |
| `GET` | `/v1/tasks/{id}` | 查询状态、Trace 与结构化报告 |
| `GET` | `/v1/tasks/{id}/report` | 获取 Markdown 报告 |
| `POST` | `/v1/tasks/{id}/fix` | 创建受控修复 |
| `POST` | `/v1/tasks/{id}/feedback` | 回流误报、漏报或坏修复 |
| `POST` | `/v1/tasks/{id}/cancel` | 请求取消任务 |
| `POST` | `/v1/tasks/{id}/resume` | 从最近 Checkpoint 续跑 |
| `POST` | `/webhooks/github` | 接收 PR Webhook |
| `GET` | `/metrics` | Prometheus 指标 |

Skill 版本、进化评测、灰度发布、告警、审计与死信队列接口可在管理台中使用，对应路由实现在 [evoagent/serving/api.py](evoagent/serving/api.py)。

## 持久化、队列与可观测性

- 运行进度写入 `checkpoints` 一张 append-only 表，主键 `(task_id, seq)`——是追加的事实，不是可覆盖的每节点一行；`tasks` 表的 `state` / `report_json` 是同事务维护的读模型，仅服务列表与看板查询。
- 断点续跑读取该日志并跳过已完成节点。节点名分层（`executing.work:security-1`），因此 Harness 与 Agentic Reviewer 用同一套机制在各自粒度上续跑，不存在第二套 Checkpoint 格式。
- 本地模式使用 SQLite 和进程内任务执行。
- 生产模式使用 PostgreSQL 与 Redis Streams，支持 ACK、Worker Lease、指数退避、重试和死信队列。
- Working、Episodic、Semantic、Procedural 四类记忆按租户和仓库隔离，并支持归档、召回和过期清理。
- OpenTelemetry 记录调用链，Prometheus 暴露运行指标，告警与管理审计持久化保存。

## 测试

```powershell
python -m unittest discover -s tests -v
```

评测脚本位于 [scripts](scripts)，版本化数据集和人工裁决位于 [benchmarks](benchmarks)。历史报告应优先使用缓存重评分脚本处理标签修订，避免在人工裁决阶段意外产生新的模型调用。

Agentic 模式中的规则 scanner 是不向 Lead/Worker 暴露结果的并行发布兜底，不是 Agent 的提示或任务来源。评测必须固定 scanner catalog（报告记录其 SHA-256），并分别报告 Worker 独立形成 Finding 的召回、Worker 独立可发布召回、scanner 独有发现、scanner 发布兜底和最终产品召回；不得把同一数据集的漏检改写成 scanner 规则后，再把新的产品召回宣称为模型能力。

缺陷定位与 CWE 分类分开评分。`target_detection_recall` 表示已覆盖已知审查位置，`taxonomy_accuracy_on_detected_targets` 表示这些命中中 CWE 也正确的比例；原 `precision` / `recall` / `f1` 继续保留为位置与 CWE 都一致的严格兼容指标。目标位置命中不等于完整语义裁决，也不能用于估算未穷举标签数据集的精确率。

## 已知边界

- 当前主要评测语言是 Python；仓库检索和通用 Diff 分析可用于其他语言，但语言专用 AST、规则和修复能力尚未建立同等基线。
- 仓库上下文能提升调用链与契约判断，但第三方库隐含语义、跨服务状态和缺失测试仍可能造成漏检。
- 严重等级和 CWE 分类的稳定性低于问题定位，不应只依赖模型等级决定是否阻断合并。
- 自动修复只适用于门禁允许的有限范围；任何补丁都应在 Draft PR 中接受人工复核。
- 当前公开评测用于回归和能力证明，不代表生产环境中的总体准确率。
