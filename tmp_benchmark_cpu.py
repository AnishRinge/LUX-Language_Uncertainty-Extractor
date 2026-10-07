import json
import os
import time
from pathlib import Path

import torch
import run_cp4_2_pilot as cp42


scenarios = cp42._load_scenarios()
# pick a representative prompt already present in production/resume state
scenario = next(s for s in scenarios if s.get("sample_id") == "squad_v2_5ad3e4e5604f3c001a3ff58f")


def measure(seq, threads, interop=1):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass

    tokenizer, model = cp42._load_model()
    prompt_text = cp42.render_prompt(
        tokenizer,
        scenario["prompt"]["context"],
        scenario["prompt"]["question"],
    )
    encoded = tokenizer(prompt_text, return_tensors="pt")
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    each = []
    for seed in seq:
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **encoded,
                do_sample=cp42.DO_SAMPLE,
                temperature=cp42.TEMPERATURE,
                top_p=cp42.TOP_P,
                top_k=cp42.TOP_K,
                max_new_tokens=cp42.MAX_NEW_TOKENS,
                pad_token_id=pad_token_id,
            )
        elapsed = time.perf_counter() - t0
        seq_out = out.sequences[0] if hasattr(out, "sequences") else out[0]
        prompt_len = int(encoded["input_ids"].shape[1])
        gen_ids = seq_out[prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        each.append({"seed": seed, "elapsed": elapsed, "tokens": int(gen_ids.numel()), "text": text})
    del model, tokenizer
    return each

print("cpu_count", os.cpu_count())
for threads in [1, 2, 4, 8]:
    print(f"BENCH threads={threads}")
    result = measure([1001], threads)
    for r in result:
        print(r)

# batch-2 sanity check with same prompt repeated twice.
print("BATCH-2 sanity")
try:
    tokenizer, model = cp42._load_model()
    prompt_text = cp42.render_prompt(
        tokenizer,
        scenario["prompt"]["context"],
        scenario["prompt"]["question"],
    )
    batch = tokenizer([prompt_text, prompt_text], return_tensors="pt", padding=True)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    torch.manual_seed(1001)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **batch,
            do_sample=cp42.DO_SAMPLE,
            temperature=cp42.TEMPERATURE,
            top_p=cp42.TOP_P,
            top_k=cp42.TOP_K,
            max_new_tokens=32,
            pad_token_id=pad_token_id,
        )
    elapsed = time.perf_counter() - t0
    seq = out.sequences if hasattr(out, "sequences") else out
    print("batch_elapsed", elapsed)
    print("batch_shape", seq.shape)
    for i in range(seq.shape[0]):
        print("batch_out", i, tokenizer.decode(seq[i], skip_special_tokens=True)[:180])
finally:
    del model, tokenizer
