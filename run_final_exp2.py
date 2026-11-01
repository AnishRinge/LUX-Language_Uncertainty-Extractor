import json, numpy as np
from pathlib import Path
sel = json.load(open("data/production/squad_v2_expanded_500/selection.json"))
canonical = json.load(open("data/prompts/squad_v2_canonical_scenarios.json"))
targets_all = []
t1 = json.load(open("data/production/cp4_7_0_gemini_3_5_flash_lite/targets.json"))
targets_all.extend(t1["targets"])
for bid in [3,4,5]:
    t = json.load(open("data/production/cp4_7_0_gemini_3_5_flash_lite/targets_batch_%02d.json" % bid))
    targets_all.extend(t["targets"])
target_by_pid = {r["prompt_id"]: r for r in targets_all}
import run_production as production
tokenizer = production.load_tokenizer()
def token_count(text): return len(tokenizer(text, add_special_tokens=False)["input_ids"])
exp_dir = Path("experiments/final_cohort_2026_10_24")
feat_rows = []
for selected in sel["selected_scenarios"]:
    pid = selected["prompt_id"]
    if pid not in target_by_pid or not target_by_pid[pid].get("target_eligible"): continue
    idx = selected["dataset_index"]
    scenario = canonical[idx]
    prompt = scenario["prompt"]
    question = prompt["question"]
    context = prompt["context"]
    feat_rows.append({"prompt_id":pid,"dataset_index":idx,"question_chars":len(question),"context_chars":len(context),"question_tokens":token_count(question),"context_tokens":token_count(context),"question_ends_with_question_mark":1 if question.strip().endswith("?") else 0,"target_risk":target_by_pid[pid]["empirical_hallucination_risk"]})
hidden_by_idx={}
for fr in feat_rows:
    idx=fr["dataset_index"]
    hs_path=Path("data/production/squad_v2_expanded_500/hidden_states")/("hs_%d.json"%idx)
    if hs_path.exists():
        hs=json.load(open(hs_path))
        if "hidden_states" in hs:
            last=hs["hidden_states"][-1]
            arr=np.array(last)
            hidden_by_idx[idx]=arr.mean(axis=0).tolist()
        elif "last_layer_mean_pooled" in hs:
            hidden_by_idx[idx]=hs["last_layer_mean_pooled"]
X_prompt=[]; X_hidden=[]; y=[]; pids_list=[]
for fr in feat_rows:
    if fr["dataset_index"] in hidden_by_idx:
        X_prompt.append([fr["question_chars"],fr["context_chars"],fr["question_tokens"],fr["context_tokens"],fr["question_ends_with_question_mark"]])
        X_hidden.append(hidden_by_idx[fr["dataset_index"]])
        y.append(fr["target_risk"]); pids_list.append(fr["prompt_id"])
X_prompt=np.array(X_prompt); X_hidden=np.array(X_hidden); y=np.array(y); pids_list=np.array(pids_list)
np.save(str(exp_dir/"X_prompt.npy"),X_prompt); np.save(str(exp_dir/"X_hidden.npy"),X_hidden); np.save(str(exp_dir/"y.npy"),y); np.save(str(exp_dir/"pids.npy"),pids_list)
json.dump({"n_prompts":len(pids_list)}, open(str(exp_dir/"dataset_info.json"),"w"))
print(len(pids_list))
