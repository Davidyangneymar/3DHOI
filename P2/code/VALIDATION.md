# 本次代码副本验证

2026-10-11；在独立 P2/code 导入路径内运行，没有借用原项目模块来替代上传副本。

| 检查 | 结果 |
|---|---|
| Python 临时夹具测试（12 个文件） | 80 通过，0 失败 |
| JavaScript 审核表单/回放状态测试 | 32 通过，0 失败 |
| CPU 接口自检 | 外部可信旧 checkpoint，4 个程序生成的无标签特征夹具；单条/批量最大差 0 |
| 接口拒绝检查 | 非法输入及 pending C 协议被拒绝 |
| 本地 Python 模块依赖 | 静态 AST 依赖闭合检查通过 |

见 [validation_summary.json](validation_summary.json) 与 [environment_observed.json](environment_observed.json)。这些结果不证明真实生成错误识别、泛化或生成质量提升；没有产生 A 样本分数、实际人体语义标签或新模型训练。

资产相关集成测试另在 tests/integration，需接收者合法持有旧权重和原始网格，本次不计入 80 个无外部资产测试。新机器依赖安装尚未验收。

文件均按 LF 换行发布并附 .gitattributes，避免 Git 的平台换行转换使文件哈希改变。清单为 package_manifest.json，另有 publication_manifest.json 记录原源码与公开副本差异；改动后须记录新版本。
