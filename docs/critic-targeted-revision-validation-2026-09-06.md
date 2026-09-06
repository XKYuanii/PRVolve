# Critic 定向补证协议验证（2026-09-06）

## 目标

在不增加角色、不强制返工、不加入仓库特例的前提下，让 Critic 只指出缺失的事实前提，Lead 只在值得追查且最多缺一两个独立前提时自主选择一次返工，Worker 返工时沿用候选、六项证明状态和证据 ID，仅补缺口。

## 实现

- Critic 为六项发布义务保留 `proof_state`，每项保存状态、所需证明和证据 ID。
- Critic 另行保存独立的 `missing_premises`。一个可达性事实即使同时影响四项发布义务，也只算一个补证任务。
- 有候选且允许返工时，Critic 先给 Lead 结构化缺口；没有返工时，该结果直接复用于发布判断，不再重复一次 Critic 审查。
- Lead 只能选择或推迟已有的结构化目标；返工仍是可选项且上限一轮。
- Worker 收到既有候选、证明状态、缺失前提和原证据 ID，只调查缺失前提。
- 首轮通用取证只额外携带一个最高优先级改动位置的修改前后内容；提示优先看旧保护条件、一个调用入口或输入契约、一个已有测试正文。

关键提交：

- `453764a`：Critic 证明状态与可选定向返工主流程。
- `bf27617`：按独立事实前提而非派生发布字段计数。
- `5e41cb3`：由 Lead 判断风险，不用严重级别硬过滤。
- `fff601a`：约束 Lead 的返工理由与结构化动作一致。

## 验证

单元测试：`python -m unittest discover -s tests`，227 项全部通过。

### 四个既往漏报与四个配对静默例

报告：`output/python-100-repo-benchmark-v1/agentic/critic-premise-4-plus-4-clean/report.json`

SHA-256：`5da251e6c69570b50dc004624e10d3eb11f0e22044b0137b37f94ef9412b0eba`

- Worker 正式目标 Finding：4/4。
- Worker 通过发布链路：1/4。
- 严格 benchmark TP：1/4。
- 配对修复例静默：4/4。
- benchmark 记为 FP：1；它是 Sonar 同一回归中的第二条发布 Finding，未在该用例目标标签中，尚未人工裁定。
- 57 次模型调用，461,018 tokens，平均每例 7.125 次、57,627 tokens。
- scanner finding：0；本组结果完全来自 Worker/Critic/Lead 链路。

四个风险例：

- Sonar：两条候选被 Critic 直接验证并发布，其中一条命中目标；没有返工。
- ERPNext：Critic 得到三个独立缺失前提，超过一次定向补证的范围，未返工、未发布。
- SkyRL：正式候选存在，但缺少可达的无 `mlp` layer 事实，未发布。
- stargazing：正式候选存在，但缺少 `pollution_info` 可为 `None` 的生产者契约，未发布。

### 路由缺陷修复后的定向复跑

报告：`output/python-100-repo-benchmark-v1/agentic/critic-premise-followup-2-plus-2-clean/report.json`

SHA-256：`88b3f582f924bd176507c3c7d87dc4e2f4ebe61fbedc818228634e6c183233ac`

- SkyRL 与 stargazing 风险例：0/2 发布。
- 两个配对修复例：2/2 静默。
- SkyRL 本轮没有形成正式 Finding，因此没有 Critic 补证目标。
- stargazing 形成正式 Finding，Critic 将四个派生状态缺口归并为一个独立事实前提；Lead 看到了 `critic:0:src/gis_service/parsers.py:236`，但结构化动作选择 `defer`。

Lead 动作一致性提示后的 stargazing 最终复跑见 `output/python-100-repo-benchmark-v1/agentic/critic-premise-action-followup-1-plus-1-clean/report.json`，SHA-256 为 `14613cfd5fb8258c0ebc8cc79fbd9034ffc6e2469e68f182aecfb3f46a0c51e6`。风险例仍未发布，修复例保持静默；Lead 的结构化动作仍为 `defer`。

## 结论边界

协议能力已经具备，且没有以降低门禁或强制工具调用换取结果；四个静默对照没有被污染。但这组实跑不能证明“定向返工提高了 recall”，因为没有一次 `critic:*` 目标被 Lead 正式选择执行。1 个恢复的 TP 是 Worker 本轮取得了充分证据后由 Critic 直接验证，不是返工救回。

剩余上限仍是复杂前提的发现与证明：候选没有形成、独立缺口超过两个，或者 Lead 判断补证价值不足时，系统会保守不发布。继续提升这部分需要更稳定的模型取证，而不应通过放松发布门禁实现。

## 另一组历史漏报回归检查

最终代码未再调整。另选 AutoGIS、bobobobo、godon 和 Glasshouse 四个在冻结 200 PR 报告中严格漏报、但 Worker 曾形成目标 Finding 的风险例，并配对同一 PR 的四个修复版本。

报告：`output/python-100-repo-benchmark-v1/agentic/final-regression-check-4-plus-4-clean/report.json`

SHA-256：`bc560e550fd915fe75ee079e7d7fb9662bab67cc37c2af08f46862c7e7cd133f`

- 旧冻结结果（相同八例）：严格风险 TP 0/4、Worker 正式目标 4/4、Worker 发布目标 0/4、修复例静默 4/4。
- 当前结果：严格风险 TP 1/4、Worker 正式目标 3/4、Worker 发布目标 1/4、修复例静默 4/4、FP 0。
- Scanner finding 为 0；本组结果不依赖 scanner。
- 69 次模型调用，619,039 tokens，平均每例 8.625 次、77,380 tokens。
- 共执行五个 Worker 返工任务：Glasshouse 风险例两个、Glasshouse 修复例一个、godon 风险例一个、AutoGIS 风险例一个。Glasshouse 风险例在返工后形成并发布目标 Finding；Glasshouse 修复例的返工没有产生 Finding。
- 这些返工均由 Worker 假设或证据交接触发，没有 `critic:*` 目标被 Lead 选择返工。
- bobobobo 本轮没有形成正式 Finding；godon 和 AutoGIS 形成 Finding 后仍分别缺少完整 SQL 行结构、capability registry/异常可达性证明，未发布。
- godon-fix 与 AutoGIS-fix 内部曾形成非目标候选，但均被发布链路拦截，最终四个修复例全部静默。

这组定向困难样本没有显示最终误报或发布结果退化：相同样本的严格 TP 从 0 增至 1，静默仍为 4/4。不过 Worker 正式目标从 4 降至 3，仍显示模型发现阶段存在运行间波动。可选返工带来了一个恢复结果，也显著增加了调用和 token；不应从这 4 个历史漏报样本外推整体 recall。
