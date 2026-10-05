"""LoRA 미세조정(blueprint 4.2). 손실은 응답 토큰에만 건다.

    python src/train_lora.py  ->  outputs/lora/ (어댑터), results/train_log.json
"""
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm import MODEL_ID, messages

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(r=16, alpha=32, dropout=0.05, targets=["q_proj", "k_proj", "v_proj", "o_proj"], lr=2e-4, epochs=2,
           grad_accum=8, max_len=512, seed=42, warmup_frac=0.05)


def encode(tok, prompt, response):
    """프롬프트 부분은 -100으로 가려 응답 토큰(끝 토큰 포함)만 손실에 들어가게 한다."""
    prefix = tok.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True)
    full = prefix + response + tok.eos_token
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"][:CFG["max_len"]]
    labels = [-100] * len(p_ids) + ids[len(p_ids):]
    return torch.tensor([ids]), torch.tensor([labels[:len(ids)]])


def main():
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    rows = [json.loads(l) for l in (ROOT / "data/rft_train.jsonl").read_text("utf-8").splitlines()]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=CFG["r"], lora_alpha=CFG["alpha"], lora_dropout=CFG["dropout"],
                                             target_modules=CFG["targets"], task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    data = [encode(tok, r["prompt"], r["response"]) for r in rows]
    steps = math.ceil(len(data) * CFG["epochs"] / CFG["grad_accum"])
    warm = max(1, int(steps * CFG["warmup_frac"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / (steps - warm)))
    model.train()
    log, t0, step, acc_loss, k = [], time.time(), 0, 0.0, 0
    for epoch in range(CFG["epochs"]):
        order = list(range(len(data)))
        random.shuffle(order)
        for i in order:
            ids, labels = data[i]
            loss = model(input_ids=ids, labels=labels).loss / CFG["grad_accum"]
            loss.backward()
            acc_loss += loss.item()
            k += 1
            if k % CFG["grad_accum"] == 0:
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                log.append({"step": step, "epoch": epoch, "loss": round(acc_loss, 4), "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0)})
                print(f"step {step}/{steps} loss {acc_loss:.4f} {time.time() - t0:.0f}s", flush=True)
                acc_loss = 0.0
    if k % CFG["grad_accum"]:
        opt.step(); opt.zero_grad()
    out = ROOT / "outputs/lora"
    model.save_pretrained(out)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/train_log.json").write_text(json.dumps({"config": CFG, "examples": len(data), "optimizer_steps": steps,
        "trainable_params": trainable, "total_params": total, "seconds": round(time.time() - t0), "log": log}, indent=2), "utf-8")
    print(f"학습 예시 {len(data)}개, 학습 파라미터 {trainable:,}/{total:,}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
