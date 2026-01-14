# eval_obj_reid_robust.py
# -*- coding: utf-8 -*-
"""
验证版：忽视 <tool_call> 漏洞（裸 JSON tool call 也能执行）
- 兼容模型输出：第2个 tool_call 缺少 "<tool_call>" 开标签，但仍有 {"name":..., "arguments":...}
- 执行前把“扫描出来的 tool call JSON”重新包装成 <tool_call>...</tool_call> 交给 toolbox.execute()

基于你现有 eval_obj_reid.py 的最小改动点：
- 原来用 `if "<tool_call>" not in out_text: ... break` 会漏掉坏格式工具调用 :contentReference[oaicite:1]{index=1}
"""

import os
import json
import re
from typing import List, Tuple, Optional, Dict, Any

import torch
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from prompt import PROMPT
from deepeyes_tools_v2 import VisualToolBoxV5
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

EXTRA_RULES = "评判标准：只有图1中的图形组合完全在图二中，才能认为一致，包括相对位置、相对比例等，zoom的对象应该是有利于你思考的区域，对不确定的问题，必须使用工具进行下一轮验证"


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
    w1, h1 = img1.size
    w2, h2 = img2.size
    H = max(h1, h2)
    W = w1 + sep_width + w2

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    y1 = (H - h1) // 2
    canvas.paste(img1, (0, y1))

    draw = ImageDraw.Draw(canvas)
    x_sep_start = w1
    draw.rectangle(
        [x_sep_start, 0, x_sep_start + sep_width - 1, H - 1],
        fill=sep_color,
    )

    y2 = (H - h2) // 2
    canvas.paste(img2, (w1 + sep_width, y2))
    return canvas


# ==========================
# 4. 单轮推理
# ==========================

def call_vlm(messages: List[dict]) -> str:
    model, processor = get_model_and_processor()

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )

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
    os.makedirs(QAS_ROOT, exist_ok=True)
    subdir = os.path.join(QAS_ROOT, example_name)
    if os.path.exists(subdir):
        idx = 1
        while os.path.exists(f"{subdir}_{idx}"):
            idx += 1
        subdir = f"{subdir}_{idx}"
    os.makedirs(subdir, exist_ok=True)
    return subdir


def messages_to_debug_serializable(messages: List[dict]) -> List[dict]:
    debug_msgs: List[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    new_content.append({"type": "image", "image": "<PIL.Image (omitted)>"})
                else:
                    new_content.append(item)
            debug_msgs.append({"role": m.get("role"), "content": new_content})
        else:
            debug_msgs.append(m)
    return debug_msgs


def save_turn_debug(debug_dir: str, turn_idx: int, messages: List[dict], out_text: str):
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
    m = re.search(r"\{(yes|no)\}", answer_text, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    text_lower = answer_text.lower()
    has_yes = "yes" in text_lower
    has_no = "no" in text_lower
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return None


# ==========================
# 6.5 关键：扫描全文裸 JSON tool call，并包装成 <tool_call>...</tool_call>
# ==========================

_ALLOWED_TOOLS = {"image_zoom_in_tool", "image_rotate_tool"}

def scan_json_toolcalls(text: str) -> List[Dict[str, Any]]:
    """
    从全文扫描所有可 json.loads 的对象（json.JSONDecoder.raw_decode），
    过滤出 {"name": ..., "arguments": ...} 结构的 tool call。
    """
    decoder = json.JSONDecoder()
    out: List[Dict[str, Any]] = []
    seen = set()

    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except Exception:
            i += 1
            continue

        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            name = obj.get("name")
            if isinstance(name, str) and name in _ALLOWED_TOOLS:
                key = json.dumps(obj, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    out.append(obj)

        i = end
    return out


def wrap_toolcalls_as_xml(toolcalls: List[Dict[str, Any]]) -> str:
    """
    把 tool call dict 列表包装成严格的 <tool_call>...</tool_call> 块，供旧 execute 正确解析。
    """
    blocks = []
    for obj in toolcalls:
        blocks.append("<tool_call>\n" + json.dumps(obj, ensure_ascii=False) + "\n</tool_call>")
    return "\n".join(blocks)


# ==========================
# 7. 跑单个 example
# ==========================

def run_single_example(example_dir: str, extra_rules: str = "") -> Tuple[str, str]:
    example_name = os.path.basename(example_dir.rstrip("/"))
    debug_dir = prepare_debug_dir(example_name)
    print(f"\n[EXAMPLE] {example_name}, debug dir: {debug_dir}")

    img1_path = os.path.join(example_dir, "img1.png")
    img2_path = os.path.join(example_dir, "img2.png")
    meta_path = os.path.join(example_dir, "meta.json")

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    concat_img = concat_images_with_separator(img1, img2)

    concat_save_path = os.path.join(debug_dir, "concat.png")
    concat_img.save(concat_save_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    question = meta["prompt"]
    truth_label = meta["truth"].lower()

    if extra_rules and extra_rules.strip():
        question_full = question + "\n" + extra_rules.strip() + "\n"
    else:
        question_full = question

    toolbox = VisualToolBoxV5()
    toolbox.reset(
        raw_prompt=None,
        multi_modal_data={"image": [concat_img]},
        origin_multi_modal_data={"image": [concat_img]},
        debug_dir=debug_dir,
    )

    messages: List[dict] = [
        {"role": "system", "content": PROMPT.SYSTEM_PROMPT_V5},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": concat_img},
                {"type": "text", "text": question_full + PROMPT.USER_PROMPT_V5},
            ],
        },
    ]

    outputs: List[str] = []
    final_answer: Optional[str] = None
    zoom_count = 0

    # 额外统计：验证“漏洞恢复”是否发生
    malformed_turns = 0
    recovered_calls_total = 0

    for t in range(MAX_TURNS):
        turn_idx = t + 1
        print(f"[DeepEyes] Turn {turn_idx}/{MAX_TURNS}")

        out_text = call_vlm(messages)
        outputs.append(out_text)

        save_turn_debug(debug_dir, turn_idx, messages, out_text)

        messages.append({"role": "assistant", "content": [{"type": "text", "text": out_text}]})

        # ====== 关键改动：不靠 "<tool_call>" 字符串，而是抽取 & 扫描 ======
        # 1) 标准块（若你的 tools.py 已经容错，这里也会包含裸 JSON）
        std_actions = toolbox.extract_action(out_text)
        # 2) 扫描全文裸 JSON tool call（即使缺 <tool_call> 开标签也能抓到）
        scanned_toolcalls = scan_json_toolcalls(out_text)

        # 如果出现“扫描到 toolcall，但标准抽取不到/抽取更少”，说明命中了你说的漏洞
        if scanned_toolcalls and len(std_actions) < len(scanned_toolcalls):
            malformed_turns += 1

        # 执行用的 action_string：优先用“包装后的严格格式”，确保 execute 能吃到全部工具调用
        # （这一步就是“忽视漏洞”的核心）
        if scanned_toolcalls:
            recovered_calls_total += len(scanned_toolcalls)
            exec_action_string = wrap_toolcalls_as_xml(scanned_toolcalls)
        else:
            exec_action_string = out_text  # 没工具就原样

        # 结束条件：只有“没有任何工具调用”且“有 answer”才结束
        # （不会因为 out_text 里没有 "<tool_call>" 就提前结束）
        if not scanned_toolcalls:
            ans = extract_final_answer(out_text)
            if ans is not None:
                final_answer = ans
                break
            # 没工具也没 answer：跟你原本脚本一致，结束
            final_answer = out_text.strip()
            break

        # 有工具：执行（即使 out_text 原始格式坏，exec_action_string 也能让它正确执行）
        obs, reward, done, info = toolbox.execute(exec_action_string, turn=turn_idx)

        if done:
            if isinstance(info, dict) and "final_answer" in info:
                final_answer = str(info["final_answer"]).strip()
            elif final_answer is None:
                final_answer = out_text.strip()
            break

        # 取 zoom 图
        zoom_imgs = []
        if isinstance(obs, dict):
            mm = obs.get("multi_modal_data") or {}
            zoom_imgs = mm.get("image") or []
        zoom_count += len(zoom_imgs)

        if not zoom_imgs:
            continue

        user_content = [{"type": "image", "image": im} for im in zoom_imgs]
        user_content.append({"type": "text", "text": PROMPT.TURN_PROMPT_V5})
        messages.append({"role": "user", "content": user_content})

    if final_answer is None:
        final_answer = outputs[-1].strip()

    pred_label = extract_yes_no(final_answer) or "unknown"

    summary_path = os.path.join(debug_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "example_dir": example_dir,
                "truth": truth_label,
                "pred": pred_label,
                "final_answer": final_answer,
                "zoom_count": zoom_count,
                "malformed_turns_detected": malformed_turns,
                "recovered_toolcalls_total": recovered_calls_total,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[RESULT] truth={truth_label}, pred={pred_label}, zooms={zoom_count}, malformed_turns={malformed_turns}")
    return pred_label, truth_label


# ==========================
# 8. 数据集循环 & 统计准确率
# ==========================

def eval_dataset(data_root: str, extra_rules: str = ""):
    example_entries = []
    for d in os.listdir(data_root):
        full = os.path.join(data_root, d)
        if not os.path.isdir(full):
            continue
        if not d.startswith("example_"):
            continue
        m = re.search(r"example_(\d+)", d)
        idx = int(m.group(1)) if m else 10**9
        example_entries.append((idx, full))

    example_entries.sort(key=lambda x: x[0])
    example_dirs = [path for _, path in example_entries]

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
        print(f"Accuracy       : {correct / total:.4f}")
    print("=============================")


if __name__ == "__main__":
    eval_dataset(DATA_ROOT, extra_rules=EXTRA_RULES)
