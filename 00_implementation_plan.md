# HierFed-Matter-NSAC-DPBA 开源实现实施规划

> 配套论文：*Privacy-Utility Tradeoff via Dynamic Privacy Budgeting for Matter-Enabled Smart Home Anomaly Detection in B5G/6G Large-Scale Residential IoT*
> 面向执行者：开发团队。本文档给出将论文算法在公开数据集上复现并开源到 GitHub 的可落地工程步骤，包含数据集获取与预处理、算法规格、环境配置、目录结构、验证标准与参数一致性结论。

---

## 0. 项目目标与范围

- **目标**：实现 HierFed-Matter-NSAC-DPBA 框架，在 IoTID20 与 CICIDS2017 公开数据集上复现论文主表（F1 / 通信开销 / 成员推断抗性），并开源到 GitHub。
- **范围**：
  - 必须实现：HierFed-Matter 层级聚合、NSAC 切片感知压缩、DPBA-FIM FIM 加权预算、5 层 DNN、Table 4/5/6 复现。
  - 仿真替代：论文中的真实树莓派硬件与 Matter 协议栈，用**单机多进程仿真**复现"设备→网关→云"三层拓扑与切片 QoS 约束，不依赖真实硬件。
  - 可选增强（本期不做）：真实 Matter/Thread 测试床、TON_IoT/Bot-IoT 验证、RL 自适应 NSAC。

---

## 1. 数据集（含官方下载链接与预处理规格）

### 1.1 IoTID20

- **出处**：Ferrag et al., "Deep learning for cyber security intrusion detection", *J. Inf. Secur. Appl.* **2020**, *50*, 102419（论文 [4]）。
- **官方下载**：https://sites.google.com/view/iot-network-intrusion-dataset （数据集发布页 "A Scheme for Generating a Dataset for Anomalous Activity Detection in IoT Networks"）。
- **备选镜像**：Mendeley Data / IEEE DataPort（搜索 "IoTID20 dataset"）；Kaggle 预处理版 https://www.kaggle.com/datasets/winthumin/iot-ids-preprocessed-datasets-win-thu （仅作备份）。
- **公开统计（已联网核实）**：625,783 条记录；83 列 = 80 个网络特征 + 3 个标签字段（label, attack_type 等）；11 类攻击（DoS, DDoS, Mirai, Scan, MITM 等）+ normal。
- **与论文一致性**：论文称"625,783 条、80 维（预处理后）、11 类攻击"——与公开事实一致，**无需修改论文数据集参数**。
- **预处理输出格式**：每条样本 → 80 维 `float32` 特征向量 `X ∈ R^80` + 二分类标签 `y ∈ {0=normal, 1=attack}`。存储为 `train.npz` / `test.npz`（或 parquet），并附带设备分区索引 `partition.json`。

### 1.2 CICIDS2017

- **出处**：Sharafaldin et al., "Toward generating a new intrusion detection dataset...", *Proc. ICISSP* 2018（论文 [30]）。
- **官方下载**：https://www.unb.ca/cic/datasets/ids-2017.html （需注册，审批后邮件发送下载链接）。
- **Kaggle 镜像**：https://www.kaggle.com/datasets/sampadab17/cicids2017 （或搜索 "CICIDS2017 kaggle" 取最新可用镜像）。
- **公开统计**：全量约 2,830,743 条 flow。取 **Tuesday**（FTP-Patator, SSH-Patator）+ **Wednesday**（DoS, DDoS, Heartbleed）子集。
- **论文称子集 566,934 样本、76 维（预处理后）**。说明：CICIDS2017 原始 83 列（含标签），去除常量/非数值特征后通常 77–78 维，论文筛至 76 维属合理范围。子集精确样本数需下载后由 `preprocess.py` 统计核实（论文值供参考，不作为硬约束）。
- **预处理输出格式**：每条样本 → 76 维 `float32` 特征向量 + 二分类标签（normal vs attack）。必须去除含缺失/无穷值的行，对协议/服务类做独热或整数编码，全量 Z-score 标准化。

### 1.3 非 IID 数据分区

- 使用 **Dirichlet 分布 `Dir(α)`** 将每数据集划分到 N 个设备。`α=0.5`（中度异构）、`α=0.1`（高度异构，论文实测 γ≤3.2）。
- 实现：按标签分布采样分配，固定全局随机种子（5 个种子：1–5），确保 Table 4/5/6 的误差棒可复现。

---

## 2. 算法规格（逐组件输入/处理/输出）

### 2.1 异常检测模型

5 层 DNN：`Input(80) → FC(128, ReLU) → FC(64, ReLU) → FC(32, ReLU) → FC(16, ReLU) → FC(2, Softmax)`。参数量 `d = 12,498`。
CICIDS2017 输入维度为 76，首层改为 `Input(76)`，其余不变（d 相应变化，需重算 NSAC 的 `k_s = d × ratio`）。损失：二分类交叉熵。

### 2.2 HierFed-Matter（层级聚合）

- **拓扑**：N 设备 → M 网关（每网关聚合 K=10 设备）→ 云服务器。
- **输入**：设备 i 在轮次 t 的本地梯度 `Δw_i^t`（已 DP 裁剪）。
- **流程**：
  1. 设备本地训练 → 上传网关；
  2. 网关聚 K 设备：`Δw_m^t = (1/K) Σ_i Δw_i^t`；
  3. 网关经 NSAC 压缩上传云；
  4. 云聚 M 网关：`w^{t+1} = w^t + Σ_m(K_m·Δw_m^t) / Σ_m K_m`。
- **通信收益**：核心网传输从 O(N) 降至 O(M)（论文报 90.2% 降幅）。

### 2.3 NSAC（网络切片感知压缩）— 参数已统一，见 §7.2

- **配置**（与 §4.2 统一）：压缩比 `CR_s = (k_s/d) × (b_s/32)`
  - uRLLC: `k=0.1d, b=8` → `CR=0.025`（40×）
  - eMBB: `k=0.3d, b=16` → `CR=0.15`（7×）
  - mMTC: `k=0.05d, b=8` → `CR=0.0125`（80×）
- **算法**（每网关对 `Δw_m^t` 执行）：
  1. Top-k 稀疏化：取 `|g|` 最大的 `k_s` 个分量，记录索引 `indices_s`；
  2. QSGD 量化：对每个选中分量 `val`，`level = 2^(b_s−1) − 1`，`q_val = sign(val)·floor(|val|/level)/level`；
  3. 打包：`encode(indices_s, q_values, b_s) → ĝ_m^t`；
  4. 延迟校验：`L_est = |ĝ_m^t|/BW_s + proc_s`；若 `> L_s^max` 则自适应下调 `k_s` 并重编码；
  5. 按切片 s 的带宽/延迟参数传输至云；
  6. 云端解码 + 反量化重建：`Δw_m^t[idx] = q_val·‖g‖/√k_s`，其余为 0。
- **切片 QoS 参数**：uRLLC 10 Mbps / 5 ms（`L_max=200ms`）；eMBB 100 Mbps / 20 ms；mMTC 1 Mbps / 100 ms。

### 2.4 DPBA-FIM（FIM 加权动态隐私预算）

- **FIM 标量**（EMA 平滑，`β_F=0.9`）：`I_l(t) = β_F·I_l(t−1) + (1−β_F)·tr(FIM_l(t))`，`FIM_l(t)` 为层 l 在轮 t 的对角 Fisher 近似平均外积。
- **设备敏感度**：`S_i = 0.5·σ_i + 0.3·(1/n_i) + 0.2·H_i`；`σ_i`：camera=1.0, thermostat=0.3, light bulb=0.1；`n_i` 本地样本数；`H_i` 由 Dirichlet α 度量的异质性。
- **预算分配**：`ε_{i,l}(t) = (ε_total/T) · (I_l/ΣI_l') · (S_i/ΣS_j)`，满足 `Σ_{i,l} ε_{i,l} = ε_total/T`（per-round 固定）。
- **DP 保证**：`(ε≤3, δ=10⁻⁵)`-DP，经 Opacus 注入高斯噪声，梯度裁剪范数 C（需与论文章节对齐；实现时先取 C 使 `‖g‖≤C`）。

### 2.5 集成训练循环

对 `t = 1..T`：设备本地训练 + 裁剪 → 网关 HierFed 聚合（K=10）→ NSAC 压缩上传 → 云聚合（M 网关）+ DPBA-FIM 噪声注入 → 下发全局模型。

---

## 3. 工程架构与目录结构

> 实现范式：**自研轻量仿真器优先**（HierFed-Matter 为两跳层级，Flower 默认单跳 client-server 改 strategy 繁琐；NSAC/DPBA-FIM 需在聚合中间插入自定义压缩与噪声，自研更可控、单机即可复现）。Flower 1.6 保留为可选分布式后端接口。

```
github_project/
├── README.md                  # 架构图 + 安装 + 最小运行示例 + 复现命令 + 引用
├── CITATION.cff               # 引用论文
├── LICENSE                    # MIT
├── requirements.txt           # torch==2.1.*, flower==1.6.*, opacus==1.4.*, numpy, pandas, scikit-learn
├── configs/
│   ├── base.yaml              # T, ε_total, δ, α, K=10, M, N
│   └── nsac_profiles.yaml     # 三切片 (k,b) 与 QoS 参数
├── data/
│   ├── download_data.sh       # 下载 IoTID20 / CICIDS2017（含官方链接与校验）
│   └── preprocess.py          # 清洗 + 80/76 维特征 + Dir(α) 分区 + 存 npz
├── models/
│   └── dnn.py                 # 5 层 DNN (d=12498)
├── core/
│   ├── hierfed.py             # 三层聚合
│   ├── nsac.py                # 压缩/解压
│   ├── dpba_fim.py            # FIM 计算 + 预算分配
│   └── simulator.py           # 单机多进程拓扑 + 切片 QoS 仿真
├── baselines/                 # FedAvg / FedProx / DP-FedAvg / HierFed / Top-k+QSGD / SAFEL-IoT / DP-Fed6G / FedProx-DP
├── eval/
│   ├── metrics.py             # F1 / Acc / AUC
│   ├── comm_cost.py           # 通信开销统计
│   └── mia.py                 # shadow-model 成员推断攻击
├── scripts/
│   ├── train.py               # 主训练入口
│   └── reproduce_tables.py    # 复现 Table 4/5/6
└── results/                   # 日志 + 表格输出
```

---

## 4. 环境配置

- **OS**：Linux（Ubuntu 22.04 推荐）；Python 3.10+。
- **依赖（精确版本）**：`torch==2.1.*`, `torchvision`, `flower==1.6.*`, `opacus==1.4.*`, `numpy>=1.24`, `pandas>=2.0`, `scikit-learn>=1.3`, `pyyaml`。
- **硬件**：NVIDIA GPU ≥16GB 显存（复现全量 N=50–5000）；CPU 可跑小规模 N≤100 验证正确性。
- **仿真器**：单机 `multiprocessing` 模拟 N 设备 + M 网关，无需真实硬件/网络。
- **安装**：
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **最小运行示例**：
  ```bash
  python scripts/train.py --dataset iotid20 --alpha 0.5 --devices 100 --rounds 100
  ```

---

## 5. GitHub 仓库规范与开源合规

- **LICENSE**：MIT（宽松，利于学术复用）。
- **数据集不入库**（体积 + 许可限制）：仓库提供 `download_data.sh` + `preprocess.py`；README 注明来源与许可。
- **模型权重不提交**：依赖随机种子复现。
- **匿名→公开**：接收前匿名仓库；接收后转公开并补 `CITATION.cff`（引用论文）。
- **README 必含**：架构说明、安装、最小运行示例、复现命令、结果对照表、引用格式。

---

## 6. 复现验证清单（对照论文目标值）

| 指标 | 目标值 | 来源 |
|------|--------|------|
| F1 (IoTID20, α=0.5) | 0.924 ± 0.006 | Table 4a |
| F1 (CICIDS2017, α=0.5) | 0.891 ± 0.008 | Table 4b |
| 通信加速 | 19.3× | Abstract |
| MIA 攻击成功率 | 14.2%（随机基线 50%） | Table 6 |
| DP 预算 | (ε≤3, δ=10⁻⁵) | §5 |
| 核心流量降幅 | 90.2% | Abstract |
| 统计显著性 | paired t-test p<0.01，5 种子 | §7.4 |

---

## 7. 参数一致性核对结论（关键，已联网核实）

### 7.1 数据集参数：与公开事实一致，无需修改论文

- **IoTID20**：625,783 条 / 80 维 / 11 类攻击 —— 与 shield-datasets 数据卡、IoTID20 原始论文一致 ✅。
- **CICIDS2017**：全量 2.83M / 子集 566,934（论文值）/ 76 维 —— 量级合理；子集精确样本数与特征维建议下载后由 `preprocess.py` 统计复核（论文值作参考）。

### 7.2 NSAC 切片配置：论文内部矛盾，已统一

- **矛盾点**：§4.2 正文 + Figure 4（uRLLC 40×@142ms, eMBB 7×@18ms, mMTC 80×@68ms）采用 `(uRLLC 0.1d/8bit, eMBB 0.3d/16bit, mMTC 0.05d/8bit)`；而 §7.10 Figure 12 推荐采用 `(uRLLC 0.025d/8bit, eMBB 0.15d/8bit, mMTC 0.025d/4bit=320×)`。两套不一致。
- **判断依据**（联网未找到公开代码，故基于论文内部自洽性判断）：Figure 4 延迟数据与 §4.2 完全自洽；Table 5 报告的 **19.3× 通信加速比**量级与 §4.2 的 moderate 压缩（混合切片、含未压缩部分）匹配，而与 §7.10 的激进压缩（最高 320×）不匹配。因此 **§4.2 + Figure 4 套更可能是真实主实验采用的配置**。
- **已执行修改**：论文 §7.10 的推荐配置文字已改为与 §4.2 一致（`uRLLC k=0.1/8bit/40×, eMBB k=0.3/16bit/7×, mMTC k=0.05/8bit/80×`）。
- ⚠️ **待作者确认**：上述判断基于内部一致性分析，作者需按其真实实验代码最终确认；若实际采用 §7.10 原值，请告知，立即回滚并统一到该套。
- ⚠️ **图片资产待重绘**：Figure 12 的 ★ 标注位置仍为 §7.10 原值（0.025/0.15/0.025），需在作图工具中重绘以对齐 §4.2 配置；Figure 4 已与 §4.2 一致，无需改。

---

## 8. 里程碑

- **M0 启动**：仓库骨架 + 核实数据集下载链接 + 依赖清单（半天）。
- **M1 数据层**：下载 + 预处理 + Dir(α) 分区 + 5 层 DNN（1–2 天）。
- **M2 核心三组件**：HierFed-Matter + NSAC + DPBA-FIM 独立实现 + 单测（3–4 天）。
- **M3 训练流水线**：集成仿真器，跑通 IoTID20（α=0.5）端到端（2–3 天）。
- **M4 复现与调优**：复现 Table 4/5/6，对齐目标数字；补 8 基线（3–5 天）。
- **M5 开源收尾**：README + LICENSE + CITATION + 发布（1–2 天）。

---

## 9. 风险与缓解

1. **NSAC 配置**：已统一，但需作者按真实实验确认（见 §7.2）。
2. **计算资源**：全量复现需 GPU；MVP 先小规模（N=50–100）验证正确性。
3. **数据集获取**：CICIDS2017 官方需注册，若不稳定用 Kaggle 镜像并校验哈希。
4. **DP 噪声量级**：ε=3 下 Opacus 噪声较大，F1 可能低于论文值；需严格对齐裁剪范数 C 与批次设置。
5. **可复现性**：5 种子 + 固定 Dir(α) 分区种子，确保误差棒可复现。
