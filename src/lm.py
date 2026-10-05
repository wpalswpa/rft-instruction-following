"""모델 적재와 배치 생성(CPU). 학습·평가가 같은 채팅 형식을 쓰게 한 곳에 모은다."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"  # 2026-10-05 기준 허브 최신 커밋에 고정
# Qwen2.5 채팅 템플릿의 기본 시스템 문구. 모든 조건이 같은 문구로 시작하고, B1만 뒤에 한 문장을 덧붙인다.
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
STRICT_SUFFIX = " Follow every formatting instruction in the user's message exactly."


def load(adapter_dir=None):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, dtype=torch.float32)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


def messages(prompt, strict=False):
    return [{"role": "system", "content": DEFAULT_SYSTEM + (STRICT_SUFFIX if strict else "")},
            {"role": "user", "content": prompt}]


def prompt_text(tok, prompt, strict=False):
    return tok.apply_chat_template(messages(prompt, strict), tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(tok, model, prompts, strict=False, sample=False, n=1, max_new_tokens=256, batch_size=64, seed=0, on_batch=None):
    """prompts 순서대로 문항당 n개 응답을 돌려준다. sample=False면 탐욕 디코딩(n=1)."""
    texts = [prompt_text(tok, p, strict) for p in prompts for _ in range(n)]
    out = []
    torch.manual_seed(seed)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
        kwargs.update(dict(do_sample=True, temperature=0.8, top_p=1.0, top_k=0) if sample else dict(do_sample=False))
        gen = model.generate(**enc, **kwargs)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            out.append(tok.decode(row, skip_special_tokens=True))
        if on_batch:
            on_batch(min(i + batch_size, len(texts)), len(texts))
    return [out[j * n:(j + 1) * n] for j in range(len(prompts))]
