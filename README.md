# LLaMA-Factory 全参数 SFT 模板

基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的训练模板，默认使用 Qwen3 对话模板；训练其他模型时需修改 `template`。

## 主要文件

- `qwen3_full_sft.yaml`：模型、数据和训练参数。
- `dataset_info.json`：数据集文件与字段映射。
- `deepspeed/`：DeepSpeed ZeRO-3 配置。
- `start.sh`：校验环境并按可见 GPU 数量启动训练。

## 快速开始

1. **安装环境**（Python 3.10+，CUDA 12.4）

   ```bash
   pip install -r requirements.txt
   ```

2. **准备数据**：创建 `data/train.jsonl`，每行一条指令微调样本，例如：

   ```json
   {"instruction": "请用一句话介绍机器学习。", "input": "", "output": "机器学习是让计算机从数据中学习规律的方法。", "system": "你是一个简洁、准确的助手。", "history": []}
   ```

   `instruction` 是用户指令，`system` 是系统提示词；`input` 可填写补充上下文，没有时填空字符串。`history` 是历史对话列表，多轮时填写 `[["上一轮用户提问", "上一轮助手回答"]]`，单轮时填 `[]`。不需要系统提示或历史对话时可省略对应字段。使用其他字段或文件名时，同步修改 `dataset_info.json` 中的映射；使用多个数据集时，也要修改配置中的 `dataset`。

3. **修改配置**：在 `qwen3_full_sft.yaml` 中填写本地 `model_name_or_path`；使用非 Qwen3 模型时，将 `template` 改为对应的 LLaMA-Factory 模板。

4. **启动训练**（至少一张可用的 CUDA GPU）：

   ```bash
   bash start.sh
   ```

   脚本校验模型路径、数据文件和 DeepSpeed 配置后，按当前可见 GPU 数量启动。可用 `CUDA_VISIBLE_DEVICES=0,1 bash start.sh` 选择设备。

## 主要配置

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `finetuning_type` | `full` | 全参数微调；显存不足可考虑 LoRA |
| `template` | `qwen3_nothink` | 须与实际基座模型匹配 |
| `cutoff_len` | `4096` | 单条样本最大长度，显存不足时可调小 |
| `per_device_train_batch_size` | `1` | 每张 GPU 的微批量 |
| `learning_rate` / `num_train_epochs` | `1.0e-5` / `5` | 学习率和训练轮数 |
| `fp16` | `true` | 默认使用 FP16；支持 BF16 时可改为 `bf16: true` 并关闭 `fp16` |

## 输出

训练产物保存在 `./saves/full/sft`（含 checkpoint 与 loss 曲线）。

## 常见问题

- **显存不足**：减小 `cutoff_len`，或改用 `deepspeed/ds_z3_config_offload.json`（CPU offload，更慢）。全参数微调的显存需求仍取决于模型大小和 GPU 容量。
- **数据未被加载**：检查 `dataset_info.json` 的 `file_name` 是否指向非空的 `data/train.jsonl`，以及 `dataset` 是否为 `train`。
- **模板不匹配**：`template` 必须与基座模型一致，否则收敛异常。
