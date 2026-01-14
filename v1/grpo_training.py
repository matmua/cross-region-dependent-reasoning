#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO-style LoRA fine-tuning for DeepEyes-style multi-turn tool loop.
Key memory fixes vs naive implementation:
- Generate rollouts under no_grad, store turn_records.
- Compute advantages from rewards first.
- Re-compute logprob WITH grad one episode at a time, and backward per-episode (no storing multiple graphs).
- Optional: only score the LAST turn's generated tokens (treat whole loop as one "completion" proxy).
- Resize all images/patches before processor to reduce vision token count.
- Enable gradient checkpointing correctly (requires enable_input_require_grads).
"""

import os
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

from prompt import PROMPT
from deepeyes_tools import VisualToolBoxV5
from qwen_vl_utils import process_vision_info


# -------------------------
# 0) Config
# -------------------------

@dataclass
class TrainCfg:
    model_path: str = "/root/local-nvme/models/Lixiang/ChenShawn-DeepEyes-7B"
    train_jsonl: str = "./train.json"      # each line: {"image": "...", "prompt": "...", "answer": "...", "id": "..."}
    output_dir: str = "./train_output_dir"
    debug_root: str = "./qas_cache"

    seed: int = 42
    epochs: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0

    num_generations: int = 2            # GRPO group size; reduce if OOM
    max_turns: int = 6
    max_new_tokens: int = 256           # reduce to save memory
    temperature: float = 0.7
    top_p: float = 0.9

    # Memory controls
    max_input_tokens: int = 2048        # truncate chat template tokens
    max_image_side: int = 672           # resize images/patches to <= this side
    last_turn_only: bool = True         # compute logprob only on last turn tokens

    grad_accum_steps: int = 1
    log_every: int = 1


CFG = TrainCfg()


# -------------------------
# 1) Dataset
# -------------------------

class JsonlQADataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        # normalize keys
        return {
            "id": r.get("id", str(idx)),
            "image": r["image"],
            "prompt": r["prompt"],
            "answer": r["answer"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # one sample per step is simplest for RL fine-tuning (and avoids long padding)
    assert len(batch) == 1, "Use per_device_batch_size=1 for this script."
    return batch[0]


# -------------------------
# 2) Helpers
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_debug_dir(debug_root: str, sample_id: str) -> str:
    os.makedirs(debug_root, exist_ok=True)
    subdir = os.path.join(debug_root, sample_id)
    if os.path.exists(subdir):
        i = 1
        while os.path.exists(f"{subdir}_{i}"):
            i += 1
        subdir = f"{subdir}_{i}"
    os.makedirs(subdir, exist_ok=True)
    return subdir


def resize_pil(im: Image.Image, max_side: int) -> Image.Image:
    # make a copy to avoid in-place modification surprises
    im = im.copy()
    w, h = im.size
    if max(w, h) <= max_side:
        return im
    im.thumbnail((max_side, max_side), Image.BICUBIC)
    return im


def extract_final_answer(text: str) -> Optional[str]:
    # simplest: parse <answer> ... </answer>
    import re
    m = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not m:
        return None
    return m[-1].strip()


# -------------------------
# 3) Generation (no grad)
# -------------------------

def build_inputs(processor, messages: List[dict], device: torch.device, max_input_tokens: int, max_image_side: int):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    if image_inputs:
        image_inputs = [resize_pil(im, max_image_side) for im in image_inputs]

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt",
    )
    # only move tensors
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    return inputs


@torch.no_grad()
def generate_one_turn(
    gen_model,
    processor,
    messages: List[dict],
    device: torch.device,
    cfg: TrainCfg,
) -> Tuple[str, List[int]]:
    inputs = build_inputs(processor, messages, device, cfg.max_input_tokens, cfg.max_image_side)

    out = gen_model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, prompt_len:].tolist()

    out_text = processor.batch_decode(
        [gen_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return out_text, gen_ids


def rollout_episode(
    gen_model,
    processor,
    image_path: str,
    question: str,
    device: torch.device,
    cfg: TrainCfg,
    debug_dir: Optional[str] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    img = Image.open(image_path).convert("RGB")
    img = resize_pil(img, cfg.max_image_side)

    toolbox = VisualToolBoxV5()
    toolbox.reset(
        raw_prompt=None,
        multi_modal_data={"image": [img]},
        origin_multi_modal_data={"image": [img]},
        debug_dir=debug_dir,
    )

    text_qas = question + PROMPT.USER_PROMPT_V5
    messages: List[dict] = [
        {"role": "system", "content": PROMPT.SYSTEM_PROMPT_V5},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": text_qas},
            ],
        },
    ]

    turn_records: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None

    for t in range(cfg.max_turns):
        turn_idx = t + 1
        # NOTE: shallow copy is OK because we only read it later.
        messages_before = [m for m in messages]

        out_text, gen_ids = generate_one_turn(gen_model, processor, messages, device, cfg)
        turn_records.append(
            {
                "turn_idx": turn_idx,
                "messages_before": messages_before,
                "gen_ids": gen_ids,
                "out_text": out_text,
            }
        )

        messages.append({"role": "assistant", "content": [{"type": "text", "text": out_text}]})

        actions = toolbox.extract_action(out_text)
        ans = extract_final_answer(out_text)

        if (not actions) and (ans is not None):
            final_answer = ans
            break

        if not actions:
            final_answer = out_text.strip()
            break

        obs, _reward, done, info = toolbox.execute(out_text, turn=turn_idx)
        if done:
            final_answer = (info.get("final_answer") or out_text).strip()
            break

        zoom_imgs = obs["multi_modal_data"]["image"]
        if not zoom_imgs:
            continue

        # IMPORTANT: resize tool patches too
        zoom_imgs = [resize_pil(im, cfg.max_image_side) for im in zoom_imgs]

        user_content = [{"type": "image", "image": im} for im in zoom_imgs]
        user_content.append({"type": "text", "text": PROMPT.TURN_PROMPT_V5})
        messages.append({"role": "user", "content": user_content})

    if final_answer is None and turn_records:
        final_answer = extract_final_answer(turn_records[-1]["out_text"]) or turn_records[-1]["out_text"].strip()

    return final_answer, turn_records


# -------------------------
# 4) Reward (simple)
# -------------------------

def compute_reward(pred: Optional[str], gold: str, raw_text: Optional[str] = None) -> float:
    """Reward shaping to avoid all-zero rewards at the start.
    - If output contains <answer>...</answer>, give 0.1 (format reward)
    - If extracted answer matches gold exactly, add 0.9
    """
    r = 0.0
    if raw_text is not None and ("<answer>" in raw_text and "</answer>" in raw_text):
        r += 0.1
    if pred is None:
        return r
    p = pred.strip().lower()
    g = gold.strip().lower()
    if p == g:
        r += 0.9
    return r


# -------------------------
# 5) Logprob (WITH grad)
# -------------------------

def sum_logprob_for_episode(
    model,
    processor,
    turn_records: List[Dict[str, Any]],
    device: torch.device,
    cfg: TrainCfg,
) -> torch.Tensor:
    """
    Return sum logprob of generated tokens for this episode.
    If cfg.last_turn_only=True, only score the last turn's generated tokens.
    """
    total = torch.zeros([], device=device)

    if not turn_records:
        return total

    records = [turn_records[-1]] if cfg.last_turn_only else turn_records

    for rec in records:
        messages_before = rec["messages_before"]
        gen_ids_list = rec["gen_ids"]
        if not gen_ids_list:
            continue

        inputs = build_inputs(processor, messages_before, device, cfg.max_input_tokens, cfg.max_image_side)

        prompt_ids = inputs["input_ids"]
        prompt_mask = inputs["attention_mask"]
        prompt_len = prompt_ids.shape[1]

        gen_ids = torch.tensor([gen_ids_list], device=device, dtype=torch.long)
        gen_mask = torch.ones_like(gen_ids)

        full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
        full_mask = torch.cat([prompt_mask, gen_mask], dim=1)

        model_inputs = dict(inputs)
        model_inputs["input_ids"] = full_ids
        model_inputs["attention_mask"] = full_mask

        out = model(**model_inputs)
        logits = out.logits  # [B, T, V]

        # token logprob: for each token, look at previous position logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)      # [B, T-1, V]
        target = full_ids[:, 1:]                                      # [B, T-1]

        tok_lp = torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

        # only sum over generated tokens positions (exclude prompt)
        gen_tok_lp = tok_lp[:, (prompt_len - 1):]  # aligns with full_ids[:, prompt_len:]
        total = total + gen_tok_lp.sum()

    return total


# -------------------------
# 6) Main train
# -------------------------

def build_model_and_processor(cfg: TrainCfg):
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,   # let accelerate place
    )
    processor = AutoProcessor.from_pretrained(cfg.model_path)

    # ---- memory-friendly training ----
    base.config.use_cache = False
    # base.gradient_checkpointing_enable()  # 先关闭：避免 checkpoint 警告/梯度链路问题；若显存紧再打开

    # LoRA
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    return model, processor


def main():
    cfg = CFG
    set_seed(cfg.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum_steps, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    ds = JsonlQADataset(cfg.train_jsonl)
    if accelerator.is_main_process:
        print(f"[info] dataset_size={len(ds)}")

    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model, processor = build_model_and_processor(cfg)

    # only optimize trainable params (LoRA)
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    # For generation with DDP: unwrap
    gen_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    global_step = 0
    model.train()

    for epoch in range(cfg.epochs):
        for batch in dl:
            global_step += 1

            sample_id = str(batch["id"])
            image_path = batch["image"]
            question = batch["prompt"]
            gold = batch["answer"]

            debug_dir = prepare_debug_dir(cfg.debug_root, sample_id) if accelerator.is_main_process else None

            # ---- 1) rollout K episodes (NO grad) ----
            episodes: List[List[Dict[str, Any]]] = []
            rewards: List[float] = []

            for k in range(cfg.num_generations):
                final_ans, turn_records = rollout_episode(
                    gen_model=gen_model,
                    processor=processor,
                    image_path=image_path,
                    question=question,
                    device=device,
                    cfg=cfg,
                    debug_dir=debug_dir,
                )
                raw_text = turn_records[-1]["out_text"] if turn_records else ""
                r = compute_reward(final_ans, gold, raw_text=raw_text)
                episodes.append(turn_records)
                rewards.append(r)

            rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
            # 如果组内 reward 全相同（std≈0），用无 baseline 的 adv 进行 bootstrap，否则用组内均值 baseline
            if torch.isfinite(rewards_t).all() and rewards_t.std(unbiased=False) < 1e-6:
                adv = rewards_t
            else:
                adv = rewards_t - rewards_t.mean()

            # ---- 2) recompute logprob WITH grad one-by-one; backward each time ----
            optimizer.zero_grad(set_to_none=True)
            loss_scalar = torch.zeros([], device=device)

            for k in range(cfg.num_generations):
                lp = sum_logprob_for_episode(
                    model=model,
                    processor=processor,
                    turn_records=episodes[k],
                    device=device,
                    cfg=cfg,
                )
                # GRPO-style: maximize advantage-weighted logprob => minimize negative
                loss_k = (-adv[k] / cfg.num_generations) * lp
                accelerator.backward(loss_k)
                loss_scalar = loss_scalar + loss_k.detach()

            optimizer.step()

            if accelerator.is_main_process and (global_step % cfg.log_every == 0):
                print(f"[step {global_step}] loss={loss_scalar.item():.6f} rewards={rewards}")

            # free cache to reduce fragmentation
            torch.cuda.empty_cache()

    # ---- save adapter ----
    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        # save LoRA adapter
        unwrapped.save_pretrained(cfg.output_dir)
        processor.save_pretrained(cfg.output_dir)
        print(f"[done] saved to {cfg.output_dir}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
