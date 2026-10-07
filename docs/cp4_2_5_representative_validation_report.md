# CP4.2.5 Representative Validation

## Exact protocol

- Model: `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; CPU-only, frozen/eval/no gradients.
- Thinking disabled through `qwen3_chat_template:enable_thinking=False`.
- Prompts: 19919, 61608, 108548, 31073, 69929, 108653. Seeds: 1001, 1002, 1003.
- A: temperature=0.7, top_p=0.8, top_k=20; B: temperature=1.3, top_p=1.0, top_k=20.

## Results

- Generation success: `36` / `36`; processor/logits cross-check at tolerance 0.0: `True`.
- Fidelity anchor (61608/A/1001): `True`.

| Protocol | Generations | EOS | Truncated | Unique outputs |
|---|---:|---:|---:|---:|
| A | 18 | 18 | 0 | 8 |
| B | 18 | 18 | 0 | 9 |

## Diversity and scope

The six-class conservative framework was applied per prompt, protocol, answerability class, and overall. Overall most severe class: **materially different wording**. Potential semantic diversity requiring human adjudication: **True**.

## Limitations and conservative conclusion

This is a fixed six-prompt representative sample with three seeds per protocol. It does not justify dataset-wide conclusions or a production decoding change. No numeric semantic-diversity score is claimed.
