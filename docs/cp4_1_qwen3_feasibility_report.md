# CP4.1 — Qwen3 Feasibility & Target Model Setup Report

**Date:** September 7, 2026  
**Status:** **PASS**  
**Target Model:** `Qwen/Qwen3-1.7B`  
**Model Revision:** `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`  

---

## 1. Objective & Findings
Tested `Qwen/Qwen3-1.7B` locally for LUX pre-generation hidden-state extraction and generation workflow.

- **Load Time:** `10.21s`
- **Frozen Status:** **CONFIRMED** (`requires_grad=False`, eval mode active).
- **Hidden Layers:** `29` (Embedding + Transformer decoder layers/blocks)
- **Hidden Dimension:** `2048`
- **Tensor Dtype:** `torch.float16`
- **Device:** `cpu`
- **RAM Before Model Load:** `304.36 MB`
- **RAM After Model Load:** `4292.70 MB` (Model Weight Delta: `3988.34 MB`)
- **Peak Process RSS RAM:** `3710.76 MB`

## 2. Test Prompts & Performance
- **Prompt:** `What is the capital of France?`
  - Seq Len: 7 | Fwd Latency: 1.4884s | Gen Latency: 4.8829s | Process RSS RAM: 3707.64 MB
- **Prompt:** `Explain entropy briefly.`
  - Seq Len: 5 | Fwd Latency: 1.0267s | Gen Latency: 5.0259s | Process RSS RAM: 3710.76 MB

## 3. Conclusion
CP4.1 is **PASS**. Qwen3-1.7B is fully functional and feasible for LUX pre-generation hidden-state extraction within local hardware constraints.
