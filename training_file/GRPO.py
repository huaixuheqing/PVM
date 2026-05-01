import os
import re
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)
from trl import GRPOConfig, GRPOTrainer
from huggingface_hub import notebook_login, login

from mathruler.grader import grade_answer

from accelerate import Accelerator

os.environ["SWANLAB_MODE"] = "offline"


model_name = "" 


print("正在加载数据集...")
dataset_id = ''
train_dataset = load_from_disk(dataset_id)
train_dataset = train_dataset.shuffle(seed=42)
val_size = 100
val_dataset = train_dataset.select(range(val_size))
train_dataset = train_dataset.select(range(val_size, len(train_dataset)))


print(f"正在加载模型: {model_name}...")
model = AutoModelForImageTextToText.from_pretrained(
    model_name, 
    dtype="bfloat16",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)


def think_and_answer_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    r"""
    Improved reward function with strict non-greedy matching.
    It ensures we match the *first* closing tag and forbids nested/duplicate tags.
    """
    # 核心改进点：(?:(?!STR).)*
    # 意思是：匹配任意字符，但前提是这个字符后面紧跟的不是 "STR"
    # 1. <think> 开头
    # 2. (?:(?!</think>).)*?  --> 匹配思考内容，绝对不允许包含 </think>
    # 3. </think> 结束
    # 4. \s* 中间只允许空白
    # 5. <answer> 开始
    # 6. (?:(?!</answer>).)*? --> 匹配答案内容，绝对不允许包含 </answer>
    # 7. </answer> 结束
    # 8. \s*$ 必须紧接着字符串结尾
    pattern = r"^<think>(?:(?!</think>).)*?</think>\s*<answer>(?:(?!</answer>).)*?</answer>\s*$"
    
    completion_contents = [completion[0]["content"] for completion in completions]
    
    # 必须使用 DOTALL 让 . 匹配换行符
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    
    return [0.1 if match else 0.0 for match in matches]


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[float | None]:
    r"""
    Correctness reward function that specifically extracts content from <answer>...</answer> tags.
    Since <answer> acts as the box, we allow extraction without explicit \boxed{} commands.
    """
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution, strict=True):
        # 提取 <answer> 内容
        # 使用 re.DOTALL 确保能匹配多行内容
        answer_match = re.search(r"<answer>(.*?)</answer>", content, flags=re.DOTALL)
        
        if answer_match:
            extracted_content = answer_match.group(1).strip()
            
            if grade_answer(extracted_content, sol):
                reward = 0.9
            else:
                reward = 0.0
        else:
            # 找不到 <answer> 标签，直接判错
            reward = 0.0
            
        rewards.append(reward)

    return rewards



output_dir = ""

training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-6,                  # 全量微调建议使用较低学习率
    lr_scheduler_type="constant",
    warmup_steps=0,
    num_train_epochs=1,
    per_device_train_batch_size=1,       # 显存敏感，设为1
    gradient_accumulation_steps=8,       # 梯度累积，弥补小 batch size
    max_completion_length=16384,          # 生成的最大长度
    num_generations=8,                   # GRPO 采样数量 (group size)
    num_generations_eval=8,                # 验证时采样数量
    temperature=1.0,
    top_p=1.0,
    mask_truncated_completions=True,

    bf16=True,
    logging_steps=1,
    report_to="swanlab",                 # 实验记录
    log_completions=True,
    num_completions_to_print=1,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=1,
    eval_strategy="steps",
    eval_steps=10,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# 冻结视觉塔
for name, param in model.visual.named_parameters():
    param.requires_grad = False

print("已冻结视觉塔 (Vision Tower) 参数")

model.gradient_checkpointing_enable()
model.enable_input_require_grads()


trainer = GRPOTrainer(
    model=model,
    reward_funcs=[think_and_answer_format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


print("开始训练 (Full Fine-Tuning)...")
trainer.train()

print(f"正在保存模型至 {output_dir} ...")
trainer.save_model(output_dir)

print("训练任务完成！")