import torch
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    TrainerCallback
)
from datasets import load_from_disk, ClassLabel
import swanlab
import functools
import pandas as pd
import os
os.environ["SWANLAB_MODE"] = "offline"


# ==========================================
# 1. 配置路径
# ==========================================
model_path = ""
output_dir = ""
run_name = ""

os.environ["SWANLAB_PROJECT"] = "Qwen3-VL-8B-Instruct-SFT"
os.environ["SWANLAB_RUN_NAME"] = run_name


class TargetedGateMonitorCallback(TrainerCallback):
    def __init__(self):
        # 您指定的参数列表
        # 注意：在 Trainer 回调中，model 对象本身就是根节点
        # 如果您得到的路径是以 "model." 开头的，代码会自动去除这个前缀以适配相对路径
        self.target_param_names = [
            "model.language_model.layers.8.pvm_block.gate_alpha",
            "model.language_model.layers.16.pvm_block.gate_alpha",
            "model.language_model.layers.24.pvm_block.gate_alpha",
        ]

    def _rgetattr(self, obj, attr, *args):
        """
        递归获取属性的辅助函数。
        例如：_rgetattr(model, 'language_model.layers.8') 
        等价于 model.language_model.layers[8]
        """
        def _getattr(obj, attr):
            # 处理列表/ModuleList索引的情况 (例如 layers.8)
            if attr.isdigit(): 
                return obj[int(attr)]
            return getattr(obj, attr, *args)
            
        return functools.reduce(_getattr, attr.split('.'), obj)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        # 只有主进程 (Rank 0) 负责上传日志，其他进程直接跳过
        if not state.is_world_process_zero:
            return

        if model is None:
            return

        # 1. 解包 DDP/DeepSpeed 封装 (如果有)
        if hasattr(model, "module"):
            real_model = model.module
        else:
            real_model = model

        gate_logs = {}

        # 2. 定点狙击：直接获取指定参数的值
        for name in self.target_param_names:
            # 清理前缀：如果名字以 "model." 开头，但我们在 model 对象内部，需要去掉它
            # 例如 "model.language_model..." -> "language_model..."
            rel_name = name
            if name.startswith("model."):
                rel_name = name.split("model.", 1)[1]
            
            try:
                # 递归查找参数对象
                param = self._rgetattr(real_model, rel_name)
                
                # 只有当它是 Tensor 时才取值
                if hasattr(param, "item"):
                    gate_logs[f"train/{name}"] = param.item()
            except (AttributeError, IndexError, TypeError) as e:
                # 如果因为 LoRA 等原因导致路径变了，打印一次警告方便调试
                # 为了不刷屏，可以只在第一步打印
                if state.global_step <= args.logging_steps:
                    print(f"⚠️ SwanLab Warning: 无法找到参数 '{name}'。可能模型经过了 LoRA 封装导致路径变更。")

        # 3. 发送数据
        if gate_logs:
            swanlab.log(gate_logs, step=state.global_step)


# ==========================================
# 2. 加载模型与处理器
# ==========================================

print(f"Loading model from {model_path}...")
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# ==========================================
# 3. 冻结参数 (Freeze)
# ==========================================
print("🥶 Freezing base model parameters...")
trainable_params = 0
all_params = 0

for name, param in model.named_parameters():
    all_params += param.numel()
    # 仅训练包含 "pvm_block" 的参数
    if "pvm" in name:
        param.requires_grad = True
        trainable_params += param.numel()
    else:
        param.requires_grad = False

print(f"📊 可训练参数 (PVM): {trainable_params / 1e6:.2f} M")
print(f"   占比: {100 * trainable_params / all_params:.4f}%")

train_dataset = load_from_disk('') 

print(f"训练集大小: {len(train_dataset)}")


# ==========================================
# 5. 配置 Trainer
# ==========================================

training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1,
    logging_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    report_to="swanlab",
    completion_only_loss=True,
    warmup_ratio=0.1,
    run_name=run_name,
    max_length=None,
    shuffle_dataset=True,
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[TargetedGateMonitorCallback()]
)


print("🚀 Starting training...")
trainer.train()

print(f"Saving model to {output_dir}...")
trainer.save_model(output_dir)
print("✅ Training finished!")
