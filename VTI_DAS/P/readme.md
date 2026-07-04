# P：固定事件位置的 qP-only VTI Bayesian 反演

## 实验目的

该文件夹用于 **固定微地震事件位置** 条件下的 qP-only VTI 参数反演。代码将事件位置固定在合成真值位置，只反演层状 VTI 模型参数和层界面深度。因此它适合用于检查：仅使用 qP 相邻通道差分到时，是否能够恢复或约束 VTI 参数，以及 qP-only 数据导致的参数非唯一性。

该实验对应论文 synthetic test 中的 qP-only fixed-event calibration 思路：事件位置不参与反演，主要观察 qP DAS moveout 对 `alpha0, beta0, epsilon, gamma, delta` 和界面深度的约束能力。

## 本文件夹的关键设置

```python
MCMC_USE_WAVES = "P"
MCMC_OBJECTIVE = "diff-p-adjacent"
INVERT_SOURCES = False
FIX_SOURCE_TO_TRUE = True
INVERT_DEPTHS = True
FIX_LAST_LAYER = True
```

含义如下：

- 只使用 qP 走时；
- 使用相邻 DAS 通道 qP 差分到时作为似然函数数据；
- 不反演事件位置；
- `04-initial/output/geo.dat` 直接由 `01-input/output/geometry.dat` 复制得到，因此初始事件位置等于真值事件位置；
- 反演层界面深度；
- 固定最后一行半空间参数。

注意：`INIT_GRID_X_MIN`、`INIT_GRID_X_MAX`、`INIT_GRID_Z_MIN`、`INIT_GRID_Z_MAX` 等网格搜索参数在当前默认设置下不会被使用，因为 `FIX_SOURCE_TO_TRUE=True` 已经绕过了事件初始位置搜索。它们只是作为以后改成事件定位测试时的备用参数。


## 正演与反演原理

### 1. 层状 VTI 模型

速度模型由 `vel.dat` 保存。每一行对应一个深度节点，并包含六列：

```text
dep, alpha0, beta0, epsilon, gamma, delta
```

其中：

- `dep`：层顶或深度节点，单位为 m；
- `alpha0`：垂向 qP 速度，单位为 m/s；
- `beta0`：垂向 qS 速度，单位为 m/s；
- `epsilon, gamma, delta`：Thomsen VTI 各向异性参数。

默认真值模型为四个有效层，界面位于 300 m、400 m 和 500 m。最后一行深度为 1000 m，并重复最底层参数，主要用于表示模型底部边界或半空间。若 `FIX_LAST_LAYER=True`，这最后一行不参与反演。

### 2. DAS 几何

`01-input/direct.py` 生成 25 个合成微地震事件和 L 形 DAS 阵列：

- 事件：近似 5×5 分布，x 约为 500–900 m，z 约为 440–780 m，并加入小的确定性扰动；
- 垂直 DAS 段：x=200 m，z=0–640 m，间隔 10 m，共 65 个通道；
- 水平 DAS 段：z=650 m，x=200–1190 m，间隔 10 m，共 100 个通道；
- 总通道数为 165。

几何文件 `geometry.dat` 和 `geo.dat` 的格式为：先写事件坐标 `sx, sz`，再写接收器/DAS 通道坐标 `rx, rz`。

### 3. qx 直接波射线追踪

`02-forward/vti_direct.py` 和 `05-inversion/vti_joint_mcmc_dram.py` 使用相同的 qx 直接波正演核心。对于每个事件–通道对，代码先根据源–检波点深度差计算每层穿过的厚度 `Z[k]`，再求解水平慢度参数。

核心思想是：在每个 VTI 层中，由 Christoffel 方程得到 qP、qSV、qSH 的慢度面，并写成

```text
pz^2 = g(px)
```

其中 `px` 是水平慢度，`pz` 是垂向慢度。给定水平距离 `H` 后，代码求解水平偏移方程：

```text
sum_k layer_dx[k] = H
```

其中

```text
layer_dx[k] = -Z[k] * 0.5 * g1 / pz
```

`g1` 是 `g(px)` 对 `px` 的一阶导数。求得 `px` 后，代码计算每层垂向群速度 `Vz`，并累加得到走时：

```text
t = sum_k Z[k] / Vz[k]
```

为了提高稳定性，代码使用 bracketed Newton / bisection 混合求根。当接近临界慢度面、导数不稳定或 Newton 步失败时，会退回到更稳健的 bracketed 求解。

### 4. station-pair 差分数据

默认反演目标函数为：

```python
MCMC_OBJECTIVE = "diff-p-adjacent"
```

即使用相邻 DAS 通道的走时差，而不是绝对到时。这样可以减少事件发震时刻等公共项的影响。若 `sigma_mode="absolute"`，代码不会把差分误差简单当成独立同分布噪声，而是使用由绝对到时误差传递得到的非对角协方差：

```text
C_d = sigma^2 A A^T
```

其中 `A` 是相邻通道差分矩阵，`sigma` 来自 `OBS_NOISE_STD`。如果改成 `sigma_mode="objective-iid"`，则表示差分后的数据本身被看作独立噪声，这通常只适用于你已经直接给差分数据加噪声的情况。

### 5. Bayesian DRAM MCMC

`05-inversion/vti_joint_mcmc_dram.py` 实现固定层数的 Bayesian 反演。反演变量包括层界面深度、VTI 参数，以及在联合定位实验中还包括事件坐标。每次 MCMC 只更新一个标量参数。

基本流程为：

1. 从 `04-initial/output/geo.dat` 和 `vel.dat` 读取初始模型；
2. 从 `prior.dat` 读取均匀先验范围和相对初始模型的最大允许扰动；
3. 从 `prop.dat` 读取每类参数的 proposal 标准差；
4. 对某个标量参数提出第一阶段 Gaussian proposal；
5. 若第一阶段被拒绝，则使用更小步长 `MCMC_DR_SCALE` 进行 delayed-rejection 第二阶段 proposal；
6. 根据当前模型和新模型的 misfit 差值进行 Metropolis 接受/拒绝；
7. 在 burn-in 之后计算 posterior mean、posterior standard deviation 和 best sample。

misfit 省略了固定的归一化常数和 log-determinant，因为在当前设置中 `sigma` 和协方差矩阵固定，不影响 Metropolis 接受率。



## 目录结构与文件作用

```text
01-input/
  direct.py                      生成真值几何、真值层状 VTI 模型、控制参数
  output/                        运行后生成 geometry.dat, vel.dat, vel1.dat, control.dat 及模型图

02-forward/
  vti_direct.py                  基于真值模型计算 qP/qSV/qSH 正演走时
  plot_forward_validation.py     正演结果检查与可视化

03-output/                       由 02-forward 自动生成
  ttime.dat                      无噪声 qP/qSV/qSH 走时
  qx.dat                         对应 qx/px 解
  diagnostics.dat                qx 收敛、误差、迭代次数等诊断信息
  layer_contributions.dat        每层走时和水平偏移贡献

04-initial/
  vel_add.py                     对真值 VTI 参数和界面深度加入扰动，生成初始模型
  noise_add.py                   对正演走时加入 Gaussian picking noise
  geo_grid_search.py             用初始速度模型和 qP 到时进行事件初始位置网格搜索
  geo_add.py                     旧版：直接扰动事件位置生成初始位置
  prior_prop.py                  生成 prior.dat 和 prop.dat
  output/                        运行后生成 vel.dat, geo.dat, nobs.dat, prior.dat, prop.dat

05-inversion/
  vti_joint_mcmc_dram.py         Bayesian DRAM MCMC 主程序

06-result/
  result_analysis.py             后验统计、misfit 和接受率分析
  loc_plot.py                    事件位置后验图和位置误差统计
  vel_plot.py                    VTI 参数后验剖面与直方图
  forward_compare_mean_best.py   posterior mean / best 模型正演数据与观测数据对比
  vti_plot_utils.py              绘图和文件读取辅助函数
```

## 一键运行

在当前文件夹运行：

```bash
python shell_simple.py
```

`shell_simple.py` 会自动依次执行：

```text
1. 01-input/direct.py
2. 02-forward/vti_direct.py
3. 04-initial/vel_add.py
4. 04-initial/noise_add.py
5. 04-initial/geo_grid_search.py 或复制真值事件位置
6. 04-initial/prior_prop.py
7. 05-inversion/vti_joint_mcmc_dram.py
8. 06-result/result_analysis.py
9. 06-result/loc_plot.py
10. 06-result/vel_plot.py
11. 06-result/forward_compare_mean_best.py
```

建议主要修改 `shell_simple.py` 顶部的统一参数区，不要在多个子脚本中重复改同一个参数。


## 当前 qP-only 反演变量

在当前默认设置下，MCMC 参数向量主要包括：

```text
1. dep[1], dep[2], dep[3], ...              层界面深度，dep[0] 固定
2. alpha0[k]                                活动层 qP 垂向速度
3. beta0[k]                                 活动层 qS 垂向速度
4. epsilon[k]
5. gamma[k]
6. delta[k]
```

由于 `FIX_LAST_LAYER=True`，最后一行半空间参数不参与反演。由于 `INVERT_SOURCES=False`，`sx` 和 `sz` 不进入 MCMC 参数向量。


## `shell_simple.py` 中的重要参数

### 随机数与 qx 求解

```python
MASTER_SEED = 0
QX_STOP = 1.0e-6
QX_MAX_ITER = 20
```

- `MASTER_SEED` 控制几何扰动、初始模型扰动、噪声和 MCMC 随机数的可重复性；
- `QX_STOP` 是 qx 偏移方程的收敛阈值；
- `QX_MAX_ITER` 是输出记录使用的最大迭代数，内部求解在必要时会使用更稳健的扩展迭代。

### 噪声

```python
OBS_NOISE_STD = 0.001000
```

该值表示绝对到时 picking noise 的标准差，单位为秒。`noise_add.py` 会用它给 `ttime.dat` 加噪声，同时 `prior_prop.py` 会把同一个值写入 `prop.dat`。因此改变噪声时建议只改这里。

### 初始模型扰动

```python
INIT_DEP_STD = 20.0
INIT_ALPHA_STD = 500.0
INIT_BETA_STD = 300.0
INIT_EPSILON_STD = 0.2
INIT_GAMMA_STD = 0.2
INIT_DELTA_STD = 0.2
```

这些参数控制 `vel_add.py` 对真值模型加入的均匀扰动半宽。它们影响初始模型离真值的距离。

### 先验范围

```python
PRIOR_DEPTH_MAX_DEV = 2.0 * INIT_DEP_STD
PRIOR_ALPHA_MAX_DEV = 5.0 * INIT_ALPHA_STD
PRIOR_BETA_MAX_DEV = 5.0 * INIT_BETA_STD
PRIOR_EPSILON_MAX_DEV = 2.0 * INIT_EPSILON_STD
PRIOR_GAMMA_MAX_DEV = 2.0 * INIT_GAMMA_STD
PRIOR_DELTA_MAX_DEV = 2.0 * INIT_DELTA_STD
PRIOR_SOURCE_X_MAX_DEV = 2.0 * INIT_SOURCE_X_STD
PRIOR_SOURCE_Z_MAX_DEV = 2.0 * INIT_SOURCE_Z_STD
```

这些值控制 MCMC 参数相对于初始模型的最大可移动范围。若反演经常碰到 prior boundary，可适当增大；若 posterior 太发散，可适当缩小。

### Proposal 标准差

```python
PROP_DEPTH_STD = 10.0
PROP_ALPHA_STD = 100.0
PROP_BETA_STD = 50.0
PROP_EPSILON_STD = 0.005
PROP_GAMMA_STD = 0.005
PROP_DELTA_STD = 0.005
PROP_SOURCE_X_STD = 25.0
PROP_SOURCE_Z_STD = 25.0
```

这些值是 MCMC 第一阶段 proposal 的初始标准差。若接受率过低，说明 proposal 可能过大；若链移动很慢，说明 proposal 可能过小。代码会在 `MCMC_ADAPT_START` 之后按经验标准差进行自适应调整。

### MCMC 控制参数

```python
MCMC_ITERATIONS = 100000
MCMC_BURNIN = 20000
MCMC_PRINT_EVERY = 100
MCMC_ADAPT_START = 5000
MCMC_ADAPT_INTERVAL = 1000
MCMC_ADAPT_STOP = 0
MCMC_DR_SCALE = 0.2
MCMC_UPDATE_ORDER = "cyclic"
MCMC_OBJECTIVE = "diff-p-adjacent"
MCMC_SIGMA_MODE = "absolute"
INVERT_DEPTHS = True
FIX_LAST_LAYER = True
```

- `MCMC_ITERATIONS`：总迭代步数；
- `MCMC_BURNIN`：计算 posterior mean/std 时丢弃的前期样本数；
- `MCMC_ADAPT_START`：从第几步开始根据链的经验方差调整 proposal；
- `MCMC_ADAPT_INTERVAL`：每隔多少步调整一次 proposal；
- `MCMC_ADAPT_STOP=0`：表示自适应在 burn-in 结束时停止；
- `MCMC_DR_SCALE=0.2`：第二阶段 delayed-rejection proposal 标准差为第一阶段的 0.2 倍；
- `MCMC_UPDATE_ORDER="cyclic"`：按参数顺序循环更新。若改为 `"random"`，每步随机选择一个参数；
- `INVERT_DEPTHS=True`：反演层界面深度；
- `FIX_LAST_LAYER=True`：固定最后一行半空间参数。


## qP-only 实验需要特别注意的地方

1. qP 差分数据对 `alpha0` 和部分 qP 敏感的各向异性参数更直接；
2. `beta0`、`gamma` 等主要与 S 波相关的参数在 qP-only 目标函数中通常更不唯一；
3. 因为事件位置被固定，位置–速度 trade-off 被人为消除，更适合作为 VTI 参数可辨识性测试；
4. 若想改成联合定位实验，不建议只改 `INVERT_SOURCES=True`。更推荐使用 `P_event_VTI` 文件夹，因为它已经包含事件初始位置网格搜索和事件参数更新逻辑。


## 主要输出

运行结束后，最重要的结果在 `06-result/output/` 中：

```text
chain.npz       完整 MCMC 链、misfit、接受状态、参数名、posterior mean/best 等
mean.dat        burn-in 后 posterior mean 模型和事件位置
best.dat        misfit 最小样本对应的模型和事件位置
misfit.png      MCMC misfit 随迭代变化曲线
```

后处理结果包括：

```text
06-result/output/analysis_figures/
  posterior_summary.csv
  location_summary.csv
  analysis_report.json
  misfit_trace.png
  acceptance_trace.png

06-result/output/location_figures/
  all_events.png
  event_XXX.png
  posterior_event_XXX.png
  location_errors.csv
  location_posterior_summary.csv

06-result/output/velocity_figures/
  velocity_model_comparison.csv
  velocity_posterior_summary.csv
  posterior_*_hist.png
  posterior_*_interval.png

06-result/output/forward_compare/
  forward_compare_summary.csv
  forward_compare_by_source.csv
  mean_predicted_ttime.dat
  best_predicted_ttime.dat
  mean_forward_comparison.npz
  best_forward_comparison.npz
  arrival_plots/
```

如果只想查看反演结果，优先看 `chain.npz`、`mean.dat`、`best.dat`、`velocity_posterior_summary.csv` 和 `location_errors.csv`。


## 推荐修改方式

- 改噪声水平：只改 `shell_simple.py` 中的 `OBS_NOISE_STD`；
- 改 MCMC 长度：改 `MCMC_ITERATIONS` 和 `MCMC_BURNIN`；
- 改 qP-only / 三相：本文件夹建议保持 qP-only；三相测试请使用 `P_SV_SH`；
- 改真值模型或几何：修改 `01-input/direct.py` 中的事件坐标、DAS 几何和 VTI 参数；
- 改先验宽度：修改 `PRIOR_*_MAX_DEV` 和 `PRIOR_*_MARGIN`；
- 改 proposal：修改 `PROP_*_STD`。
