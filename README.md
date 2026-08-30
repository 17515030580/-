# MTEGDRP FastAPI 后端

该项目按照当前 `MTEGDRP` 药敏模型的真实输入方式实现，不改变原模型代码。服务接收同一患者的表达、突变和甲基化CSV，两个分支分别使用各自保存的特征清单和KPCA对象。药敏分支把患者三组学嵌入与预存药物图组合，批量预测全部候选药物；亚型分支已预留通用适配器。

## 1. 项目结构

```text
MTEGDRP_FastAPI_Backend/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── api/routes.py
│   ├── adapters/
│   │   ├── mtegdrp_adapter.py
│   │   └── subtype_adapter.py
│   ├── services/
│   │   ├── input_parser.py
│   │   ├── omics_preprocessor.py
│   │   ├── model_registry.py
│   │   ├── prediction_service.py
│   │   └── result_store.py
│   └── core/
├── models/
│   ├── MTEGDRP.py
│   └── egnn_pytorch.py
├── subtype_models/
│   └── model.py
├── artifacts/
│   ├── drug_response/v1.0.0/
│   └── subtype/v1.0.0/
├── runtime_results/
├── tests/
├── scripts/verify_artifacts.py
├── .env.example
├── requirements.txt
└── run.py
```

`models/MTEGDRP.py` 与原文件保持不变。

## 2. 当前是否适合你的模型

适合，但有两个边界：

1. 药敏模型可以按现有代码直接接入，前提是先运行修改后的 `Data_encoding.py` 和 `Model_training.py`，生成权重、KPCA、特征清单和药物图。
2. 亚型模型目前尚未提供，所以默认 `SUBTYPE_ENABLED=false`。此时 `/predict` 仍返回统一结构，但亚型字段为 `pending_model`。补齐亚型模型后即可真正并行调用两个模型。

两个模型接收的是同一组三个原始CSV，但允许分别使用自己的特征列表、KPCA和模型参数，避免错误地假设两个模型的预处理完全相同。

## 3. 放置药敏产物

推荐直接设置服务器路径：

```bash
export DRUG_ARTIFACT_DIR=/home/public_data/jlu/MTEGDRP-main/backend_artifacts/drug_response/v1.0.0
```

也可以将该目录完整复制到：

```text
artifacts/drug_response/v1.0.0/
```

运行检查：

```bash
python scripts/verify_artifacts.py
```

## 4. 安装与启动

在已有MTEGDRP Conda环境中安装API依赖：

```bash
pip install -r requirements.txt
```

复制配置：

```bash
cp .env.example .env
```

服务会自动读取项目根目录下的`.env`，也可以使用shell或部署平台环境变量覆盖。

启动：

```bash
python run.py
```

开发启动：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

接口文档：

```text
http://localhost:8000/docs
```

## 5. POST /predict

字段名必须是：

```text
expression_file
mutation_file
methylation_file
patient_id（可选）
```

示例：

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "patient_id=P001" \
  -F "expression_file=@expression.csv" \
  -F "mutation_file=@mutation.csv" \
  -F "methylation_file=@methylation.csv"
```

推荐CSV格式为一行一个样本：

```csv
patient_id,TP53,EGFR,KRAS
P001,7.2,3.1,2.0
```

也支持两列长表：

```csv
feature,value
TP53,7.2
EGFR,3.1
KRAS,2.0
```

三个文件必须属于同一患者，并且三种组学必须全部上传。

## 6. 统一响应

```json
{
  "success": true,
  "prediction_id": "PRED_...",
  "patient_id": "P001",
  "subtype": {
    "status": "pending_model"
  },
  "drug_response": {
    "status": "success",
    "total_drugs": 250,
    "top10": [],
    "all_drugs": []
  },
  "quality_control": {},
  "model_status": {},
  "downloads": {}
}
```

亚型模型启用后，`subtype` 会自动包含预测亚型、所有类别概率、前两类概率差和预测熵。

## 7. 并行推理

`PARALLEL_INFERENCE=true` 时，后端通过线程池并发调度两个模型。每个模型在启动时仅加载一次，并由独立锁保护推理过程。

两个模型都放在同一块GPU时，并发不一定更快，而且显存占用会叠加。显存不足时可采用：

```bash
PARALLEL_INFERENCE=false
```

或者把亚型模型放到CPU或另一块GPU。

## 8. 亚型模型接入

将真实模型类放入 `subtype_models/model.py`，或把 `SUBTYPE_CLASS_PATH` 指向其他模块。补齐产物后设置：

```bash
SUBTYPE_ENABLED=true
REQUIRE_BOTH_MODELS=true
```

推荐在 `metadata/model_manifest.json` 中明确指定：

```json
{
  "class_path": "subtype_models.model:SubtypeClassifier",
  "model_init_kwargs": {"num_classes": 4},
  "forward_mode": "three_args",
  "output_mode": "logits"
}
```

适配器支持常见输入形式：三个位置参数、三个关键字参数、拼接张量、字典或包含 `target_ge/target_mut/target_meth` 的对象。

## 9. 文件大小与错误处理

- `MAX_FILE_SIZE_MB` 控制单个组学文件大小。
- `MAX_REQUEST_SIZE_MB` 控制整个multipart请求大小。
- 请求头和实际接收字节都会检查。
- 所有错误都返回统一JSON，并带 `X-Request-ID`。

## 10. 结果文件

默认写入：

```text
runtime_results/<prediction_id>/
├── prediction_result.json
├── quality_control.json
├── all_drug_predictions.csv
├── top10_drug_predictions.csv
├── subtype_probabilities.csv
└── download_manifest.json
```

默认不保存患者上传的原始组学文件，只保存模型结果和质控摘要。

## 11. 其他接口

```text
GET /health
GET /models/status
GET /results/{prediction_id}/download/{filename}
GET /drugs/{drug_id}/structure
```
