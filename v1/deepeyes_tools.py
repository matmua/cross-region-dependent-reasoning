# deepeyes_tools.py
# -*- coding: utf-8 -*-

import os
import re
import json
from math import ceil, floor
from typing import Any, Dict, List

from PIL import Image
from prompt import PROMPT   # 就是你上传的 prompt.py


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
    现在额外支持把 zoom patch 保存到 debug 目录。
    """
    name = "visual_toolbox_v5"
    user_prompt = PROMPT.TURN_PROMPT_V5
    max_action_per_turn = 3

    def __init__(self):
        super().__init__(name=self.name)
        self.chatml_history = None
        self.multi_modal_data: Dict[str, List[Image.Image]] = {}
        self.height = 0
        self.width = 0
        self.debug_dir: str | None = None  # 每次 run_deepeyes 会传进来

    # ---------- 从模型输出里抽取 <answer> 和 <tool_call> ----------

    def extract_answer(self, action_string: str):
        answer_list = re.findall(r"<answer>(.*?)</answer>", action_string, re.DOTALL)
        return answer_list[-1] if answer_list else None

    def extract_action(self, action_string: str):
        tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", action_string, re.DOTALL)
        if not tool_calls:
            return []
        return [{"tool_call": call.strip()} for call in tool_calls]

    # ---------- 主入口：执行一轮工具 ----------

    def execute(self, action_string: str, **kwargs):
        """
        输入：本轮模型完整输出（包含 <think> / <tool_call> / <answer>）
        输出：
            obs:   {"prompt": 下一轮 user ChatML, "multi_modal_data": {"image": [PIL...]}}
            reward: 0.0（推理阶段不用）
            done:   是否 episode 结束
            info:   额外信息（含 final_answer / error 等）
        """
        # 当前是第几轮，方便命名 debug 文件
        turn_idx = kwargs.get("turn", "x")

        # 1）如果已经有 <answer>，直接结束
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {"final_answer": answer}

        # 2）否则解析所有 <tool_call>
        actions = self.extract_action(action_string)
        if not actions:
            # 没有 answer 也没有 tool_call，当 episode 结束处理
            return "", 0.0, True, {}

        if len(actions) > self.max_action_per_turn:
            actions = actions[: self.max_action_per_turn]

        current_images: List[Image.Image] = []
        last_tool_name = None
        zoom_idx = 0  # 本轮第几个 zoom patch

        try:
            for act in actions:
                tool_call_str = act["tool_call"]
                tool_call = json.loads(tool_call_str)
                tool_name = tool_call["name"]
                args = tool_call["arguments"]
                last_tool_name = tool_name

                # ---- zoom-in 工具：裁剪 bbox_2d（像素坐标） ----
                if tool_name == "image_zoom_in_tool":
                    bbox = args["bbox_2d"]  # [left, top, right, bottom]
                    bbox = self.maybe_resize_bbox(*bbox)
                    if not bbox:
                        raise ValueError("ZOOM IN ARGUMENTS ARE INVALID")

                    img = self.multi_modal_data["image"][0]
                    cropped = img.crop(bbox)
                    current_images.append(cropped)
                    zoom_idx += 1

                    # === DEBUG：保存 zoom patch 到 qas_cache 子目录 ===
                    if self.debug_dir:
                        os.makedirs(self.debug_dir, exist_ok=True)
                        filename = f"zoom_turn{turn_idx}_idx{zoom_idx}.png"
                        save_path = os.path.join(self.debug_dir, filename)
                        cropped.save(save_path)
                        # 记录 bbox 信息
                        log_path = os.path.join(self.debug_dir, "zoom_log.txt")
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(
                                f"turn={turn_idx}, idx={zoom_idx}, "
                                f"bbox={bbox}, label={args.get('label', '')}\n"
                            )

                # ---- rotate 工具（几乎不用，你以后想删掉也可以） ----
                elif tool_name == "image_rotate_tool":
                    angle = args["angle"]
                    img = self.multi_modal_data["image"][0]
                    rotated = img.rotate(angle)
                    current_images.append(rotated)

                else:
                    raise ValueError(f"Unknown tool name: {tool_name}")

            # 3）构造下一轮 user prompt（和源码一致：一张图配一个 <tool_response><image>）
            tool_response = "<tool_response>\n<image>\n</tool_response>\n" * len(
                current_images
            )
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
            info = {"status": "success", "tool_used": last_tool_name}
            print(f"[DEBUG] SUCCESS ACTION {action_string=}")
            return obs, reward, done, info

        except Exception as e:
            # 出错时，构造一个带错误信息的 user turn
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
        额外支持 debug_dir，用于保存 zoom patch 等调试信息。
        """
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert "image" in self.multi_modal_data
        assert len(self.multi_modal_data["image"]) > 0

        img0 = self.multi_modal_data["image"][0]
        self.height = img0.height
        self.width = img0.width

        # debug 目录（每次 run_deepeyes 新建一个子目录传进来）
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
        # 裁剪到图像范围
        left = max(0, left)
        top = max(0, top)
        right = min(self.width, right)
        bottom = min(self.height, bottom)
        if not self.validate_bbox(left, top, right, bottom):
            return None

        h = bottom - top
        w = right - left

        # DeepEyes 里的逻辑：太小就扩展到至少 28px
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
            if not self.validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            return [new_left, new_top, new_right, new_bottom]

        return [left, top, right, bottom]
        