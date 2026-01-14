# deepeyes_tools.py
# -*- coding: utf-8 -*-

import os
import re
import json
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Set

from PIL import Image
from prompt import PROMPT


# ===== 极简 ToolBase：代替 verl.workers.agent.tool_envs.ToolBase =====

class ToolBase:
    def __init__(self, name: str):
        self.name = name

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        raise NotImplementedError

    def execute(self, action_string: str, **kwargs):
        raise NotImplementedError


# ===== DeepEyes 的 v5 视觉工具箱 =====

class VisualToolBoxV5(ToolBase):
    """
    精简自 visual_toolbox_v5.py，只保留 zoom-in / rotate 逻辑和 prompt 组装，
    额外支持把 zoom patch 保存到 debug 目录。

    ✅ 修复点：
    1) extract_action() 现在能容错解析“坏格式”的 tool call（缺 <tool_call>、裸 JSON、标签不匹配等）
    2) execute() 不会因为发现 <answer> 就提前结束：只有在“无 tool call 且有 answer”时才结束
    """
    name = "visual_toolbox_v5"
    user_prompt = PROMPT.TURN_PROMPT_V5
    max_action_per_turn = 3

    # 只允许这些工具名，避免 JSON 扫描误伤
    _ALLOWED_TOOL_NAMES = {"image_zoom_in_tool", "image_rotate_tool"}

    def __init__(self):
        super().__init__(name=self.name)
        self.chatml_history = None
        self.multi_modal_data: Dict[str, List[Image.Image]] = {}
        self.height = 0
        self.width = 0
        self.debug_dir: str | None = None

    # ---------- 从模型输出里抽取 <answer> 和 tool calls ----------

    def extract_answer(self, action_string: str) -> Optional[str]:
        answer_list = re.findall(r"<answer>(.*?)</answer>", action_string, re.DOTALL)
        return answer_list[-1].strip() if answer_list else None

    def _normalize_tool_obj(self, obj: Dict[str, Any]) -> Optional[str]:
        """
        将 {"name":..., "arguments":...} 规范化成稳定的 JSON 字符串，便于去重。
        """
        if not isinstance(obj, dict):
            return None
        name = obj.get("name")
        args = obj.get("arguments")
        if name not in self._ALLOWED_TOOL_NAMES:
            return None
        if not isinstance(args, dict):
            return None
        # 规范化输出（排序 key，避免同一对象不同字符串导致去重失败）
        return json.dumps({"name": name, "arguments": args}, ensure_ascii=False, sort_keys=True)

    def _try_parse_json_object(self, s: str) -> Optional[Dict[str, Any]]:
        """
        尝试从字符串中解析一个 JSON object：
        - 直接 json.loads
        - 如果失败，尝试截取第一个 {...} 子串再 loads
        """
        s = s.strip()
        if not s:
            return None

        # 1) 直接 loads
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # 2) 尝试提取第一个 { ... } 子串
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

        return None

    def _scan_json_tool_calls(self, text: str) -> List[str]:
        """
        在整段文本里扫描所有 JSON object，找到形如：
        {"name": "...", "arguments": {...}}
        的对象，容错处理“缺 <tool_call> 标签”的情况。
        """
        calls: List[str] = []
        decoder = json.JSONDecoder()
        i = 0
        n = len(text)

        while i < n:
            j = text.find("{", i)
            if j == -1:
                break
            try:
                obj, end = decoder.raw_decode(text[j:])
            except json.JSONDecodeError:
                i = j + 1
                continue

            norm = self._normalize_tool_obj(obj) if isinstance(obj, dict) else None
            if norm:
                calls.append(norm)
                i = j + end
            else:
                i = j + 1

        return calls

    def extract_action(self, action_string: str) -> List[Dict[str, str]]:
        """
        返回 [{"tool_call": "<json-string>"}...]

        ✅ 兼容：
        - 标准：<tool_call> {json} </tool_call>
        - 漏洞：裸 JSON 后面跟着 </tool_call>（缺开头标签）
        - 漏洞：<tool_call> 里不是纯 JSON（能提取其中 {...}）
        - 漏洞：文本里出现多个 tool call，但标签缺失/混乱
        """
        collected: List[str] = []
        seen: Set[str] = set()

        # A) 先按标签抓（最“干净”）
        tagged_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", action_string, re.DOTALL)
        for blk in tagged_blocks:
            obj = self._try_parse_json_object(blk)
            if not obj:
                continue
            norm = self._normalize_tool_obj(obj)
            if norm and norm not in seen:
                seen.add(norm)
                collected.append(norm)

        # B) 再全局 JSON 扫描（兜底：抓漏掉 <tool_call> 的）
        for norm in self._scan_json_tool_calls(action_string):
            if norm not in seen:
                seen.add(norm)
                collected.append(norm)

        return [{"tool_call": s} for s in collected]

    # ---------- 主入口：执行一轮工具 ----------

    def execute(self, action_string: str, **kwargs):
        """
        输入：本轮模型完整输出（包含 <think> / <tool_call> / <answer>）
        输出：
            obs:   {"prompt": 下一轮 user ChatML, "multi_modal_data": {"image": [PIL...]}}
            reward: 0.0（推理阶段不用）
            done:   是否 episode 结束
            info:   额外信息（含 final_answer / draft_answer / error 等）
        """
        turn_idx = kwargs.get("turn", "x")

        # ✅ 先解析 actions，再解析 answer
        #    这样就能实现：有 answer 但也有 tool_calls -> 不提前结束
        actions = self.extract_action(action_string)
        answer = self.extract_answer(action_string)

        # 1) 只有当“无 tool 调用 且有 answer”时才结束
        if (not actions) and answer:
            return "", 0.0, True, {"final_answer": answer}

        # 2) 无 tool、无 answer：结束（你也可以改成 done=False 让模型继续输出）
        if not actions:
            return "", 0.0, True, {}

        # 3) 有 tool_calls：继续执行（即使有 answer，也不结束）
        if len(actions) > self.max_action_per_turn:
            actions = actions[: self.max_action_per_turn]

        current_images: List[Image.Image] = []
        last_tool_name = None
        zoom_idx = 0

        # 可选：记录每个 zoom 的元信息，便于 loop 回灌
        zoom_meta: List[Dict[str, Any]] = []

        try:
            for act in actions:
                tool_call_str = act["tool_call"]
                tool_call = json.loads(tool_call_str)
                tool_name = tool_call["name"]
                args = tool_call["arguments"]
                last_tool_name = tool_name

                if tool_name == "image_zoom_in_tool":
                    # ✅ 兼容 bbox_2d 或 bbox
                    bbox = args.get("bbox_2d", None)
                    if bbox is None:
                        bbox = args.get("bbox", None)

                    if (not isinstance(bbox, list)) or len(bbox) != 4:
                        raise ValueError("ZOOM IN ARGUMENTS ARE INVALID: bbox_2d/bbox must be [l,t,r,b]")

                    bbox = self.maybe_resize_bbox(*bbox)
                    if not bbox:
                        raise ValueError("ZOOM IN ARGUMENTS ARE INVALID after resize/validate")

                    img = self.multi_modal_data["image"][0]
                    cropped = img.crop(bbox)
                    current_images.append(cropped)
                    zoom_idx += 1

                    zoom_meta.append({
                        "bbox": bbox,
                        "label": args.get("label", ""),
                    })

                    # DEBUG：保存 zoom patch
                    if self.debug_dir:
                        os.makedirs(self.debug_dir, exist_ok=True)
                        filename = f"zoom_turn{turn_idx}_idx{zoom_idx}.png"
                        save_path = os.path.join(self.debug_dir, filename)
                        cropped.save(save_path)

                        log_path = os.path.join(self.debug_dir, "zoom_log.txt")
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(
                                f"turn={turn_idx}, idx={zoom_idx}, "
                                f"bbox={bbox}, label={args.get('label', '')}\n"
                            )

                elif tool_name == "image_rotate_tool":
                    angle = args.get("angle", None)
                    if angle is None:
                        raise ValueError("ROTATE ARGUMENTS ARE INVALID: missing angle")
                    img = self.multi_modal_data["image"][0]
                    rotated = img.rotate(angle)
                    current_images.append(rotated)

                else:
                    raise ValueError(f"Unknown tool name: {tool_name}")

            # 4) 构造下一轮 user prompt（保持你原逻辑）
            tool_response = "<tool_response>\n<image>\n</tool_response>\n" * len(current_images)
            prompt = (
                "<|im_end|>\n<|im_start|>user\n"
                + tool_response
                + self.user_prompt
                + "<|im_end|>\n<|im_start|>assistant\n"
            )

            obs = {
                "prompt": prompt,
                "multi_modal_data": {"image": current_images},
            }
            reward = 0.0
            done = False

            info: Dict[str, Any] = {
                "status": "success",
                "tool_used": last_tool_name,
                "zoom_meta": zoom_meta,
            }

            # ✅ 如果同轮有 answer，把它当草稿保存（不结束）
            if answer:
                info["draft_answer"] = answer

            print(f"[DEBUG] SUCCESS ACTION {action_string=}")
            return obs, reward, done, info

        except Exception as e:
            print(f"[DEBUG] Execute WRONG - {str(e)} {action_string=}")
            err_prompt = (
                "<|im_end|>\n<|im_start|>user\n"
                + f"Error: {str(e)}"
                + "<|im_end|>\n<|im_start|>assistant\n"
            )
            obs = {"prompt": err_prompt, "multi_modal_data": None}
            reward = 0.0
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info

    # ---------- reset：一条对话开始时调用一次 ----------

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        """
        origin_multi_modal_data: {"image": [PIL.Image,...]}，第一张就是原图
        支持 debug_dir，用于保存 zoom patch 等调试信息。
        """
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert "image" in self.multi_modal_data
        assert len(self.multi_modal_data["image"]) > 0

        img0 = self.multi_modal_data["image"][0]
        self.height = img0.height
        self.width = img0.width

        self.debug_dir = kwargs.get("debug_dir", None)
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

    # ---------- bbox 合法性检查 & 最小尺寸扩展 ----------

    def validate_bbox(self, left, top, right, bottom) -> bool:
        try:
            assert left < right and bottom > top
            h = bottom - top
            w = right - left
            assert max(h, w) / min(h, w) <= 100
            return True
        except Exception as err:
            print(f"[validate_bbox ERROR] {err}")
            return False

    def maybe_resize_bbox(self, left, top, right, bottom):
        left = max(0, int(left))
        top = max(0, int(top))
        right = min(self.width, int(right))
        bottom = min(self.height, int(bottom))

        if not self.validate_bbox(left, top, right, bottom):
            return None

        h = bottom - top
        w = right - left

        # DeepEyes 逻辑：太小就扩展到至少 28px
        if h < 28 or w < 28:
            cx = (left + right) / 2.0
            cy = (top + bottom) / 2.0
            ratio = 28 / min(h, w)
            new_half_h = ceil(h * ratio * 0.5)
            new_half_w = ceil(w * ratio * 0.5)
            new_left = int(cx - new_half_w)
            new_right = int(cx + new_half_w)
            new_top = int(cy - new_half_h)
            new_bottom = int(cy + new_half_h)

            new_left = max(0, new_left)
            new_top = max(0, new_top)
            new_right = min(self.width, new_right)
            new_bottom = min(self.height, new_bottom)

            if not self.validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            return [new_left, new_top, new_right, new_bottom]

        return [left, top, right, bottom]
