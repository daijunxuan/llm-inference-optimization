# 项目说明

这是一个可复现的 LLM 推理性能实验项目，使用 Qwen2.5-0.5B-Instruct，对比 Hugging Face、批处理、量化和 vLLM。

主 README 和代码使用英文，方便作为 GitHub 作品展示。真实数据保存在 `results/local/run-01/`，可从每批耗时重新计算表格。

## 已实现

- Hugging Face FP32 基线与 CPU 动态 INT8 量化。
- batch size 1 / 4 本地实验矩阵。
- NVIDIA GPU 上的 HF FP16、vLLM FP16、HF NF4 对照流程，batch size 1 / 4 / 16。
- 固定模型 commit、相同输入 token、固定输出长度、预热和同步计时。
- P50/P95 批次延迟、输出 token 吞吐、进程内存和可选 GPU 显存采样。
- 原始 JSON、CSV 和自动更新 README 的 Markdown 表格。
- CPU 单元/集成测试、GitHub Actions 配置和 Dockerfile。

## 展示时应讲清楚

1. 本地 CPU 数字来自实际运行；GPU 行是待测项，不能说已经得到 vLLM 加速倍数。
2. 表格中的 latency 是整批完成时间，不是首 token 延迟，也不是完整在线服务延迟。
3. INT8 不一定更快；小模型、硬件内核、量化开销和批大小都会影响结果。
4. RSS 不等于模型权重大小，也不等于 GPU 显存。指标定义比漂亮数字更重要。
5. Docker 配置和 CUDA 路径仍需在 NVIDIA 主机实测验收。

## 简历描述

可以如实写：实现基于 Qwen 的可复现推理基准框架，支持 Transformers/vLLM 适配、批大小扫描和量化实验，自动产出原始数据及性能对照报告；完成本地 CPU 基线与 INT8 实测。

完成 GPU 实测后，再添加有对应原始数据支持的吞吐提升或显存变化幅度。
