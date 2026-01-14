# eval_obj_reid.py
# -*- coding: utf-8 -*-
"""
用 DeepEyes zoom-in 流程评估 obj-reid-pixelperfect 数据集 (T3/fs1_nd0)。
- 对每个 example_x 目录：
  - 读取 img1.png + img2.png
  - 横向拼接，中间画竖线
  - 用 meta['prompt'] (+ extra_rules) 作为问题
  - 用 meta['truth'] ('yes' / 'no') 作为标签
- 其它逻辑与之前的 deepeyes_loop + qas_cache 调试一致。
"""

import os
import json
import re
from typing import List, Tuple, Optional

import torch
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from prompt import PROMPT           # 你已有的 prompt.py
from deepeyes_tools import VisualToolBoxV5  # 前面写好的工具环境
from qwen_vl_utils import process_vision_info


# ==========================
# 1. 基本配置：按需修改
# ==========================

MODEL_PATH = "/root/local-nvme/models/Lixiang__ChenShawn-DeepEyes-7B/Lixiang/ChenShawn-DeepEyes-7B"

DATA_ROOT = "/root/local-nvme/resource/vlmtunnel-main/datasets/obj-reid-pixelperfect/T3/fs1_nd0"

DEVICE = "cuda"
MAX_TURNS = 6
MAX_NEW_TOKENS = 512

QAS_ROOT = os.path.join(os.path.dirname(__file__), "qas_cache")

# 可选额外规则；留空就什么都不加
EXTRA_RULES = "评判标准：只有图1中的图形组合完全在图二中，才能认为一致，包括相对位置、相对比例等，zoom的对象应该是有利于你思考的区域，对不确定的问题，必须使用工具进行下一轮验证"  # 例如： "在输出 <answer> 之前至少调用两次 image_zoom_in_tool。\n"


# ==========================
# 2. 模型 & processor（全局缓存）
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
# 3. 拼接两张图，中间画竖线
# ==========================

def concat_images_with_separator(
    img1: Image.Image,
    img2: Image.Image,
    sep_width: int = 10,
    sep_color=(0, 0, 0),
) -> Image.Image:
    """
    横向拼接 img1 和 img2，中间用指定颜色的竖线隔开。
    高度取两者最大值，较矮的居中对齐。
    """
    w1, h1 = img1.size
    w2, h2 = img2.size
    H = max(h1, h2)
    W = w1 + sep_width + w2

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    # 放第一张（左对齐，垂直居中）
    y1 = (H - h1) // 2
    canvas.paste(img1, (0, y1))

    # 画竖线
    draw = ImageDraw.Draw(canvas)
    x_sep_start = w1
    draw.rectangle(
        [x_sep_start, 0, x_sep_start + sep_width - 1, H - 1],
        fill=sep_color,
    )

    # 放第二张
    y2 = (H - h2) // 2
    canvas.paste(img2, (w1 + sep_width, y2))

    return canvas


# ==========================
# 4. 单轮推理（沿用你之前的写法）
# ==========================

def call_vlm(messages: List[dict]) -> str:
    """
    输入：
        messages: Qwen 的多模态 chat 消息列表（包含 system/user/assistant，以及 image）
    输出：
        模型新生成的一段文本（里面可能包含 <think> / <tool_call> / <answer>）
    """
    model, processor = get_model_and_processor()

    # 1) messages -> ChatML
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 2) 抽取 image/video
    image_inputs, video_inputs = process_vision_info(messages)

    # 3) 打包张量
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

    # 5) 只取新生成 token
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
# 5. 调试目录 & 保存函数
# ==========================

def prepare_debug_dir(example_name: str) -> str:
    """
    在 qas_cache/ 下为当前 example 创建一个子目录：
    example_0 -> qas_cache/example_0 或 example_0_1, example_0_2...
    """
    os.makedirs(QAS_ROOT, exist_ok=True)
    base = example_name
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
                    new_content.append(
                        {"type": "image", "image": "<PIL.Image (omitted)>"}
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
    debug_msgs = messages_to_debug_serializable(messages)
    msg_path = os.path.join(debug_dir, f"messages_turn{turn_idx}.json")
    with open(msg_path, "w", encoding="utf-8") as f:
        json.dump(debug_msgs, f, ensure_ascii=False, indent=2)

    out_path = os.path.join(debug_dir, f"output_turn{turn_idx}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)


# ==========================
# 6. 辅助函数：提取 <answer>、{yes}/{no}
# ==========================

def extract_final_answer(text: str) -> Optional[str]:
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m[-1].strip() if m else None


def extract_yes_no(answer_text: str) -> Optional[str]:
    """
    从最终答案中抽取 {yes}/{no}，返回 'yes'/'no' 小写；失败返回 None。
    """
    m = re.search(r"\{(yes|no)\}", answer_text, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # 兜底：如果只出现一次 yes/no，也勉强用一下
    text_lower = answer_text.lower()
    has_yes = "yes" in text_lower
    has_no = "no" in text_lower
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return None


# ==========================
# 7. 跑单个 example
# ==========================

def run_single_example(example_dir: str, extra_rules: str = "") -> Tuple[str, str]:
    """
    Args:
        example_dir: 例如 ".../fs1_nd0/example_0"
        extra_rules: 额外规则字符串，可为空

    Returns:
        (pred_label, truth_label)
    """
    example_name = os.path.basename(example_dir.rstrip("/"))
    # 每个 example 对应 qas_cache 里的一个子目录
    debug_dir = prepare_debug_dir(example_name)
    print(f"\n[EXAMPLE] {example_name}, debug dir: {debug_dir}")

    # 1) 读两张图
    img1_path = os.path.join(example_dir, "img1.png")
    img2_path = os.path.join(example_dir, "img2.png")
    meta_path = os.path.join(example_dir, "meta.json")

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    concat_img = concat_images_with_separator(img1, img2)

    # ✅ 把拼接后的图像保存到对应 cache 目录里
    concat_save_path = os.path.join(debug_dir, "concat.png")
    concat_img.save(concat_save_path)

    # 2) 读 meta：prompt + truth
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    question = meta["prompt"]
    truth_label = meta["truth"].lower()  # "yes" / "no"

    # optional extra_rules：可以为空，不影响逻辑
    if extra_rules and extra_rules.strip():
        question_full = question + "\n" + extra_rules.strip() + "\n"
    else:
        question_full = question

    # 3) 初始化工具环境（用拼接后的图像）
    toolbox = VisualToolBoxV5()
    toolbox.reset(
        raw_prompt=None,
        multi_modal_data={"image": [concat_img]},
        origin_multi_modal_data={"image": [concat_img]},
        debug_dir=debug_dir,
    )

    # 4) 构造首轮 messages
    messages: List[dict] = [
        {
            "role": "system",
            "content": PROMPT.SYSTEM_PROMPT_V5,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": concat_img},
                {
                    "type": "text",
                    "text": question_full + PROMPT.USER_PROMPT_V5,
                },
            ],
        },
    ]

    outputs: List[str] = []
    final_answer: Optional[str] = None
    zoom_count = 0

    for t in range(MAX_TURNS):
        turn_idx = t + 1
        print(f"[DeepEyes] Turn {turn_idx}/{MAX_TURNS}")

        out_text = call_vlm(messages)
        outputs.append(out_text)

        # 当前轮 messages + 输出存到 cache 目录
        save_turn_debug(debug_dir, turn_idx, messages, out_text)

        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": out_text}],
            }
        )

        if "<tool_call>" not in out_text:
            ans = extract_final_answer(out_text)
            final_answer = ans if ans is not None else out_text.strip()
            break

        obs, reward, done, info = toolbox.execute(out_text, turn=turn_idx)

        if done:
            if "final_answer" in info:
                final_answer = info["final_answer"].strip()
            elif final_answer is None:
                final_answer = out_text.strip()
            break

        zoom_imgs = obs["multi_modal_data"]["image"]
        zoom_count += len(zoom_imgs)

        if not zoom_imgs:
            continue

        user_content = []
        for im in zoom_imgs:
            user_content.append({"type": "image", "image": im})
        user_content.append({"type": "text", "text": PROMPT.TURN_PROMPT_V5})

        messages.append({"role": "user", "content": user_content})

    if final_answer is None:
        final_answer = outputs[-1].strip()

    pred_label = extract_yes_no(final_answer)
    if pred_label is None:
        pred_label = "unknown"

    # 每个 example 一个 summary.json
    summary_path = os.path.join(debug_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "example_dir": example_dir,
                "truth": truth_label,
                "pred": pred_label,
                "final_answer": final_answer,
                "zoom_count": zoom_count,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[RESULT] truth={truth_label}, pred={pred_label}, zooms={zoom_count}")
    return pred_label, truth_label



# ==========================
# 8. 数据集循环 & 统计准确率
# ==========================

def eval_dataset(data_root: str, extra_rules: str = ""):
    """
    按 example_0, example_1, example_2 的数字顺序遍历数据集，
    而不是按字符串顺序 example_0, example_1, example_10, example_100...
    """
    example_entries = []
    for d in os.listdir(data_root):
        full = os.path.join(data_root, d)
        if not os.path.isdir(full):
            continue
        if not d.startswith("example_"):
            continue

        # 从 "example_123" 里提取数字 123，用于排序
        m = re.search(r"example_(\d+)", d)
        if m:
            idx = int(m.group(1))
        else:
            # 万一格式不对，扔到后面去
            idx = 10**9
        example_entries.append((idx, full))

    # 按数字 idx 排序
    example_entries.sort(key=lambda x: x[0])
    example_dirs = [path for idx, path in example_entries]

    total = 0
    correct = 0
    unknown = 0

    for ex_dir in example_dirs:
        pred, truth = run_single_example(ex_dir, extra_rules=extra_rules)
        total += 1
        if pred == "unknown":
            unknown += 1
        if pred == truth:
            correct += 1

    print("\n========== SUMMARY ==========")
    print(f"Total examples : {total}")
    print(f"Correct        : {correct}")
    print(f"Unknown preds  : {unknown}")
    if total > 0:
        acc = correct / total
        print(f"Accuracy       : {acc:.4f}")
    print("=============================")



# ==========================
# 9. main
# ==========================

if __name__ == "__main__":
    eval_dataset(DATA_ROOT, extra_rules=EXTRA_RULES)
