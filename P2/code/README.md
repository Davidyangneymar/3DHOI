# B 可协作代码快照

版本：2026-10-11。当前正式真实错误试标处于第 2/6 步，条件/来源尚未冻结。此目录供 A/B/C 阅读代码、讨论接口和在合法持有外部资产后运行。

## 代码导航

| 用途 | 入口 | 当前使用范围 |
|---|---|---|
| 原始 OMOMO 接触与显示 | [omomo_contact.py](omomo_contact.py)、[batch_omomo_contact.py](batch_omomo_contact.py)、[plot_omomo_alignment.py](plot_omomo_alignment.py) | 24 关节、手 22/23、5 cm 最近顶点代理 |
| 文本结构审计 | [audit_omomo_text.py](audit_omomo_text.py) | 在本地合法取得的 annotation ZIP 与运动记录间核对 |
| 三类反事实与筛选 | [counterfactual/build_v1.py](counterfactual/build_v1.py) | 查看/复现构造算法；当前不重建 B 的冻结历史数据 |
| 合成样本人工审核 | [counterfactual/prepare_review.py](counterfactual/prepare_review.py)、[定向首审/二审](counterfactual/prepare_review_targeted.py)、[audit_reviews.py](counterfactual/audit_reviews.py) | 无分数审核、独立二审、证据与预算检查 |
| Critic 与规则定义 | [models/interaction_critic.py](models/interaction_critic.py) | 模型结构、固定规则、历史训练实现；当前不启动默认训练入口 |
| 受限 CPU 推理 | [counterfactual/critic_api.py](counterfactual/critic_api.py) | 使用外部可信冻结 checkpoint；当前格式例不能提前评分 |
| A raw 输出适配 | [counterfactual/generator_adapter.py](counterfactual/generator_adapter.py)、[geometry_features.py](counterfactual/geometry_features.py) | 冻结 13 通道，严格单位/坐标/网格/时间核对 |
| 正式真实错误试标 | [counterfactual/real_error_pilot.py](counterfactual/real_error_pilot.py) | lock-plan → prepare → 标签归档 → evaluate |
| 评估与来源协议 | [benchmark.py](counterfactual/benchmark.py)、[protocol_checks.py](counterfactual/protocol_checks.py) | 同批 Critic/规则比较、源分组、组合协议门槛 |
| 已审核子集与训练门槛 | [reviewed_subset.py](counterfactual/reviewed_subset.py)、[train_reviewed.py](counterfactual/train_reviewed.py) | 当前 training_ready=false；不得绕过或修改划分补类 |
| 独立几何候选 | [geometry_candidate_v2.py](counterfactual/geometry_candidate_v2.py)、[contact_surface_v2.py](counterfactual/contact_surface_v2.py) | 17 通道、顶点/三角面距离诊断；不能输入旧 13 通道模型 |

诊断脚本以及通用 calibration/paired_quality 等源代码保留供讨论与后续实验。当前首次真实小批不使用校准、重排或新的生成器训练；CPU 接口不是可微生成器 loss。

## 从哪里运行

克隆后进入 `P2/code`，使这里的 counterfactual、models 和根脚本处于 Python 导入路径内。依赖版本见 [requirements.txt](requirements.txt) 和 [environment_observed.json](environment_observed.json)，是已观察环境，不代表全新机器安装已验收。

```powershell
cd P2/code
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
python -m pip install -r requirements.txt
python -m counterfactual.real_error_pilot --help
```

仅运行无外部资产的临时夹具测试：

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/test_review_form.js tests/test_real_pilot_form.js tests/test_real_pilot_app.js
```

这些测试不是人体动作标签或识错性能证据。资产相关真实试标集成测试单独位于 [tests/integration](tests/integration)，需要可信旧权重和合法持有的物体网格；不属于上面的默认发现目录。

## 外部资产位置与自检

- OMOMO raw 文件：`data/train_diffusion_manip_seq_joints24.p`、`data/test_diffusion_manip_seq_joints24.p`。
- 原始物体网格：`data/captured_objects/`；mop/vacuum 包含上下两个部件。
- 旧 checkpoint 可放 `models/interaction_critic_v1.pt` 或通过正式入口的 `--checkpoint` 指向外部路径。
- 人工票据、冻结 NPZ、身体模型和生成输出都不随源代码发布。
- 旧权重身份与通道说明见 [checkpoint_descriptor.json](checkpoint_descriptor.json)。

只使用程序生成的无标签特征检查接口，不使用 A 的格式例：

```powershell
python -B interface_smoke.py --checkpoint <可信旧权重路径> --output <代码目录之外的新报告.json>
```

该入口先核对 [package_manifest.json](package_manifest.json)，再做受限 checkpoint 载入、分数方向、批量一致性和非法输入拒绝。它不是迁移性能实验；修改快照后须明确记录版本，原清单自检会拒绝文件哈希变化。

## 正式批量流程

正式条件先在 [P2 协作提案](../handoff/real_pilot_v1/README.md) 中确认，再填写 [plan 模板](protocols/real_pilot_plan.template.json)。模板不是已冻结计划。

```powershell
python -m counterfactual.real_error_pilot lock-plan <draft.json> --checkpoint <可信旧权重路径> --output <新的plan.json>
python -m counterfactual.real_error_pilot prepare <plan.json> <delivery.jsonl> --checkpoint <可信旧权重路径> --output <新的盲审目录>
# 首审票据归档后执行：
python -m counterfactual.real_error_pilot evaluate <盲审目录> <首审.jsonl> --checkpoint <可信旧权重路径> --output <新的结果目录>
```

全部成功/失败槽位保留，失败不自动补样；页面隐藏模型和规则分数。真实输入 FPS 须声明，旧 OMOMO 参考 30 FPS 是历史推断。权重、13 通道标准化及旧阈值保持固定；不在该批改阈值或重训。旧特征的 9/10 通道采用原生网格局部单位/帧，并非米/帧；公开副本保持其原定义，不能静默换成米制再输入旧 checkpoint。

## 上传副本与本地冻结代码的差异

仅 `build_v1.py` 的本机附件路径改为环境变量 `COUNTERHOI_SOURCE_REVIEW`，默认 `reviews/source_review.csv`。其输入 CSV 需有 sequence/split/object/status 等原字段，confirmed 行才进入构造。算法、强度与规则未改变。

该路径改动会改变 builder 文件哈希，因此本公开副本不能直接通过 B 历史冻结 manifest 的 builder 哈希检查；没有修改本地历史源码、数据或 manifest。读取历史算法与核对旧归档时须保留这一差别。默认构造/旧训练命令会写各自输出目录，当前阶段不运行它们重建或覆盖旧资产。

逐文件原始与上传 SHA256、路径改动记录见 [publication_manifest.json](publication_manifest.json)。测试与导入验证结果另见 [VALIDATION.md](VALIDATION.md)。几何与模型接口定义均保留当前科学实现；公开源码存在不等于整体实验完成。

## 来源与许可记录

OMOMO 原始表示、24 关节 FK 与物体变换参照 [官方项目](https://github.com/lijiaman/omomo_release)，附 [官方 MIT 声明](third_party/OMOMO_LICENSE.txt) 保留来源署名。它适用于上游软件，不覆盖运动数据、网格、权重或身体模型。此快照没有为未确认的资产新增再分发许可。
