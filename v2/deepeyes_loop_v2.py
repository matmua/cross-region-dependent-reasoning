# deepeyes_loop.py
# -*- coding: utf-8 -*-

import os
import json
from typing import List, Tuple, Optional

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from prompt import PROMPT
from deepeyes_tools_v2 import VisualToolBoxV5
from qwen_vl_utils import process_vision_info


# ==========================
# 1. 基本配置：改这里
# ==========================

MODEL_PATH = "/root/local-nvme/models/Lixiang__ChenShawn-DeepEyes-7B/Lixiang/ChenShawn-DeepEyes-7B"
IMAGE_PATH = "/root/local-nvme/projects/deepeyes/comparation.png"
DEVICE = "cuda"          # 和你原来的代码一样
MAX_TURNS = 6
MAX_NEW_TOKENS = 512

# qas_cache 根目录（你已经建好）
QAS_ROOT = os.path.join(os.path.dirname(__file__), "qas_cache")


# ==========================
# 2. 全局模型 / processor
# ==========================

_model = None
_processor = None

def get_model_and_processor():
    global _model, _processor
    if _model is None or _processor is None:
        print(f"[LOAD] Loading model from {MODEL_PATH} ...")
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print("[LOAD] Model & processor loaded.")
    return _model, _processor


# ==========================
# 3. 单轮推理（复用你原来的写法）
# ==========================

def call_vlm(messages: List[dict]) -> str:
    """
    输入：
        messages: Qwen 的多模态 chat 消息列表（包含 system/user/assistant，以及 image）
    输出：
        模型新生成的一段文本（里面可能包含 <think> / <tool_call> / <answer>）
    """
    model, processor = get_model_and_processor()

    # 1) 把 messages 变成 ChatML 文本
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 2) 从 messages 里抽取图片 / 视频信息（和你原来代码一样）
    image_inputs, video_inputs = process_vision_info(messages)

    # 3) 用 processor 打包成张量
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    # 4) 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    # 5) 截掉 prompt 部分，只留新生成内容
    gen_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


# ==========================
# 4. 调试目录 & 保存函数
# ==========================

def prepare_debug_dir(image_path: str) -> str:
    """
    在 qas_cache/ 下为当前图片创建一个子目录：
    home.png -> qas_cache/home 或 home_1, home_2...
    """
    os.makedirs(QAS_ROOT, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    subdir = os.path.join(QAS_ROOT, base)

    if os.path.exists(subdir):
        idx = 1
        while os.path.exists(f"{subdir}_{idx}"):
            idx += 1
        subdir = f"{subdir}_{idx}"

    os.makedirs(subdir, exist_ok=True)
    return subdir


def messages_to_debug_serializable(messages: List[dict]) -> List[dict]:
    """
    把 messages 里不能 JSON 序列化的 PIL.Image 替换成占位符，方便写入 .json。
    """
    debug_msgs: List[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    # 用描述字符串代替 PIL.Image 对象
                    new_content.append(
                        {"type": "image", "image": "<PIL.Image (omitted in debug)>"}
                    )
                else:
                    new_content.append(item)
            debug_msgs.append({"role": m.get("role"), "content": new_content})
        else:
            debug_msgs.append(m)
    return debug_msgs


def save_turn_debug(
    debug_dir: str,
    turn_idx: int,
    messages: List[dict],
    out_text: str,
):
    """
    把当前轮次的 messages + 模型输出保存到 debug 目录。
    """
    # messages（去掉 PIL.Image）
    debug_msgs = messages_to_debug_serializable(messages)
    msg_path = os.path.join(debug_dir, f"messages_turn{turn_idx}.json")
    with open(msg_path, "w", encoding="utf-8") as f:
        json.dump(debug_msgs, f, ensure_ascii=False, indent=2)

    # 模型原始输出
    out_path = os.path.join(debug_dir, f"output_turn{turn_idx}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)


# ==========================
# 5. 提取 <answer>（方便结束）
# ==========================

def extract_final_answer(text: str) -> Optional[str]:
    import re
    ans = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return ans[-1].strip() if ans else None


# ==========================
# 6. DeepEyes 主循环
# ==========================

def run_deepeyes(image_path: str, question: str) -> Tuple[str, List[str]]:
    """
    返回：
        final_answer: 最终答案（去掉 <answer> 标签）
        outputs:      每一轮模型原始输出文本，方便 debug
    """
    # 1) 准备调试目录
    debug_dir = prepare_debug_dir(image_path)
    print(f"[DEBUG] Debug dir: {debug_dir}")

    # 2) 加载原始图像
    img = Image.open(image_path).convert("RGB")

    # 3) 初始化工具环境（把 debug_dir 传进去）
    toolbox = VisualToolBoxV5()
    toolbox.reset(
        raw_prompt=None,
        multi_modal_data={"image": [img]},
        origin_multi_modal_data={"image": [img]},
        debug_dir=debug_dir,
    )

    # 4) 构造首轮 messages
    extra_rules = """
                    [当前任务的特别规则，必须严格遵守]
                    必须在一次think里调用3次image_zoom_in_tool，否则结果不被接收，不允许输出answer!
                    """
    text_qas = question + "\n" + PROMPT.USER_PROMPT_V5 #+ "\n" + extra_rules 
    messages: List[dict] = [
        {
            "role": "system",
            "content": PROMPT.SYSTEM_PROMPT_V5,   # DeepEyes 的 system prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {
                    "type": "text",
                    "text": text_qas,
                },
            ],
        },
    ]

    outputs: List[str] = []
    final_answer: Optional[str] = None

    for t in range(MAX_TURNS):
        turn_idx = t + 1
        print(f"[DeepEyes] Turn {turn_idx}/{MAX_TURNS}")

        # 5) 调用模型推理
        out_text = call_vlm(messages)
        outputs.append(out_text)

        # 5.1 保存调试信息：messages + out_text
        save_turn_debug(debug_dir, turn_idx, messages, out_text)

        # 5.2 把这次 assistant 的完整输出放进 messages（作为一条纯文本）
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": out_text},
                ],
            }
        )

        # ==========================
        # ✅ 仅修 <tool_call> 检测逻辑：
        #    不再靠字符串 "<tool_call>"，而是用 toolbox.extract_action
        #    这样能识别“坏格式/裸 JSON”的 tool call
        # ==========================
        actions = toolbox.extract_action(out_text)
        ans = extract_final_answer(out_text)

        # 6) 只有当：没有任何 tool 调用 且 有 answer，才结束
        if (not actions) and (ans is not None):
            final_answer = ans
            break

        # 6.1 没有 tool 调用 且 没有 answer：按你原逻辑也结束（原来是直接 break）
        if not actions:
            final_answer = out_text.strip()
            break

        # 7) 有 tool_call：交给工具环境，执行 zoom-in
        obs, reward, done, info = toolbox.execute(out_text, turn=turn_idx)

        if done:
            # 工具认为 episode 结束（通常是已经解析出 final_answer）
            if "final_answer" in info:
                final_answer = info["final_answer"].strip()
            elif final_answer is None:
                final_answer = out_text.strip()
            break

        # 8) 工具返回 zoom 后图像，构造下一轮 user 消息
        zoom_imgs = obs["multi_modal_data"]["image"]  # List[PIL.Image]
        if not zoom_imgs:
            # 没有真的裁剪出图像，那就直接回到循环，让模型继续说
            continue

        user_content = []
        for im in zoom_imgs:
            user_content.append({"type": "image", "image": im})
        # 文本部分：就是 DeepEyes 的 TURN_PROMPT_V5 / USER_PROMPT_V5
        user_content.append({"type": "text", "text": PROMPT.TURN_PROMPT_V5})

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

    if final_answer is None:
        final_answer = outputs[-1].strip()

    # 9) 写一个 summary.json 方便快速回顾
    summary_path = os.path.join(debug_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_path": image_path,
                "question": question,
                "final_answer": final_answer,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return final_answer, outputs


# ==========================
# 7. 简单测试入口
# ==========================

if __name__ == "__main__":
    q = "The first image shows an object made of geometric shapes, which together form an object. Does this SAME object appear in the second image? For example, if a component shape were to be rotated or translated, it would be a different object. Respond with {yes} or {no} (inside the curly brackets). There may be extra shapes in Image 2 that are not part of the original object; as long as the object from Image 1 is present, the answer is yes even if there are other shapes present."
    ans, outs = run_deepeyes(IMAGE_PATH, q)
    print("\n==== FINAL ANSWER ====\n", ans)
