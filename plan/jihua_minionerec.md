# MiniOneRec reproduction plan (CN below)

# 28 天：LLM + 推荐 + MiniOneRec 复现计划（6–8h/天，单卡 16GB）

面向目标：能理解并完成 MiniOneRec 复现，把结果“作品化”写进简历，并能应对大模型/推荐相关岗位面试追问。

---

## 0. 最终交付（必须有）

- 复现报告：环境/数据/配置/指标表（按项目口径：HR@K、NDCG@K、合法率/覆盖率等）
- 对比实验：至少 2 组（例如 SFT-only vs SFT+RL；不同 SID/约束策略/超参）
- 一键脚本：`prepare → train → rl → evaluate`（单机小规模也可）
- Demo：输入用户历史 → 输出 Top-K 推荐 + 可读 item 信息（截图/录屏）

## 1. 每天 6–8 小时建议拆分

- 1h：概念学习 + 笔记（只学“能用到项目上的”）
- 4–6h：项目推进（跑通/改代码/做实验）
- 1h：复盘（记录：命令、配置、结果、坑、明日 TODO）

## 2. 把 Step 1–9 映射到 MiniOneRec

- Step 1（PyTorch）：读懂数据流、训练 loop、日志与断点；能定位 loss/梯度/显存瓶颈
- Step 2（Transformer）：理解自回归 LM、mask、token 生成；能看懂项目里 LLM 前向与生成
- Step 3（NTP/SFT）：理解 label shift + CE loss；能自己改 prompt/模板并复现实验
- Step 4（MoE）：可选项（非必需），有余力再补
- Step 5（LoRA/省显存）：单卡 16GB 重点掌握 LoRA/冻结/混精/梯度累积；分布式/ZeRO 概念了解即可
- Step 6（调参闭环）：固定评估集/seed；一次只动 1–2 个变量；形成对比表与结论
- Step 7（RL/RLHF）：能解释 reward/KL/优势函数；跑通项目 RL 阶段（按 repo 实现：PPO/DPO/GRPO 等）
- Step 8（推理）：理解 KV cache、prefill/decode；（可选）用 vLLM 做推理/吞吐对比
- Step 9（作品化）：README、结果表、复现脚本、讲稿与追问题库

---

## 3. 28 天里程碑（主线不跑偏）

- M0（Day 1–3）：环境 + evaluate 跑通 + tiny 数据端到端跑通
- M1（Day 4–10）：数据与 SID 流程确认 + SFT 跑出第一版指标
- M2（Day 11–18）：SFT 对比实验 + 约束生成/合法性检查稳定
- M3（Day 19–24）：RL 跑通 + 对比 SFT-only vs SFT+RL
- M4（Day 25–28）：打磨交付（报告/脚本/demo/简历/面试）

---

## 4. 周计划（按周推进 + 每天可验收）

### Week 1（Day 1–7）：跑通基线（Step 1–2）

- Day 1：装环境（CUDA/PyTorch/依赖），项目能启动；记录环境信息（GPU/驱动/CUDA）
- Day 2：跑 `evaluate`/推理最小例子；确认输出可复现（固定 seed）
- Day 3：走读数据入口：能在本地打印 3 条样本（user 历史、target、负样本等）
- Day 4：走读训练入口：定位 loss 计算与保存路径；tiny 子集跑 50–200 step
- Day 5：补 Transformer 必要知识：QKV/mask/自回归；对照代码把模块对上
- Day 6：建立评估口径：固定 Top-K 与指标；做一个固定评估集（≥50 条）
- Day 7：画一张“项目结构图”（数据→SID→SFT→RL→评估→推理），写成 1 页笔记

### Week 2（Day 8–14）：SID + SFT 复现（Step 3–5）

- Day 8：读 README/论文：搞清 SID 如何构建/使用；列出关键文件与入口函数
- Day 9：跑通 SID（构建或加载）；验证：同一 item 映射稳定；统计 SID 覆盖率
- Day 10：SFT 正式跑通（单卡）：优先“冻结大模型，只训新增 embedding/adapter”拿到结果
- Day 11：可选增强：加 LoRA；做对比 1（embedding-only vs LoRA）
- Day 12：约束生成/合法性：统计非法率；必要时加/修约束逻辑与日志
- Day 13：省显存与稳定性：混精、梯度累积、梯度裁剪；把显存峰值记录下来
- Day 14：周复盘：SFT 结果表 + 失败案例 3–5 个 + 下一周假设（要验证什么）

### Week 3（Day 15–21）：调参与 RL（Step 6–7）

- Day 15：调参闭环模板：固定 seeds/评估集；输出对比表（配置→指标→结论）
- Day 16：RL 基础补齐：reward/KL/优势函数；对照项目代码逐项“对上”
- Day 17：RL tiny 跑通：目标不是效果，而是“稳定跑完 + 指标不崩”
- Day 18：做对比 2：SFT-only vs SFT+RL（记录指标、合法率、生成样例）
- Day 19：调 reward/采样/超参：一次只动 1–2 个变量，写清因果结论
- Day 20：补推荐基础（面试向）：序列推荐/离线评估常见坑（泄漏、负采样、分布偏移）
- Day 21：整理“踩坑合集”：显存/不收敛/非法生成/评估不一致（≥5 条，含解决方案）

### Week 4（Day 22–28）：推理 + 作品化 + 面试（Step 8–9）

- Day 22：做 demo：输入用户历史 → 输出 Top-K（含可读 item），保存截图/录屏
- Day 23：（可选）vLLM 推理：记录吞吐/延迟；能解释 KV cache 与 prefill/decode
- Day 24：写复现报告初稿：环境、数据、配置、结果表、关键 ablation、失败分析
- Day 25：可复现检查：从空环境按 README 跑到结果（或验证关键步骤可复现）
- Day 26：写简历条目：3–5 条 bullet（做了什么/解决了什么/指标如何/算力如何）
- Day 27：面试讲稿：3 分钟 + 10 分钟；准备追问（SID/约束生成/RL/评估）
- Day 28：归档打包：代码/配置/结果/文档，准备投递

---

## 5. 面试必讲点（推荐 + 大模型交叉）

- 为什么 SID 能把推荐问题转成“生成问题”（item 离散化、序列到序列）
- 约束生成如何保证合法 item（以及如何排查非法率/约束失败）
- SFT vs RL 阶段分别在优化什么（reward/KL/稳定性）
- 离线评估怎么做、指标怎么解释、有哪些坑（泄漏、负采样、分布偏移）
- 单卡 16GB 怎么把训练跑起来（冻结/LoRA/混精/梯度累积/子集）

