# Commit Decision History

> 此文件是 `commits.jsonl` 的人类可读视图，可由工具重生成。
> Canonical store: `commits.jsonl` (JSONL, append-only)

| Date | Context-Id | Commit | Summary | Decisions | Bugs | Risk |
|------|-----------|--------|---------|-----------|------|------|
| 2026-04-15 | 4dfe5aaf | bc4f77a | refactor!: replace TSE with pDFNet2+ PSE | Sigmoid gate, causal conditioning, epoch-aware PSEMixer, cosine-warmup LR | — | — |
| 2026-04-15 | fe000599 | 5c3c761 | feat(scripts): TOML-driven CLI | TOML config, --override, auto-split | — | — |
| 2026-04-15 | 2405dd98 | 4a06197 | refactor(models): native ECAPA-TDNN | Separate ecapa_tdnn.py, wespeaker FBank helper, channels//4 SE, weights_only | — | FBank parity |
| 2026-04-15 | a62c4a32 | 3164aa9 | fix(models): SpeechBrain ECAPA format | Rewrote ecapa_tdnn.py for SpeechBrain arch, 3C fusion, Conv1d SE, global ASP | ECAPA ckpt load fail | — |
| 2026-04-15 | 959010f4 | b066574 | fix(data): enrollment/target leakage + gain/bandwidth augmentation | Reserve enrollment anchor during target sampling; sinc FIR lowpass; random gain | Enrollment shared target files | — |
| 2026-04-16 | f5b16f47 | f60095d | fix(loss): align SpectralLoss with pDFNet2 paper defaults | under_suppression_weight 2.0→1.0 (paper symmetric); gamma 0.6→1.0 (official default) | Conditioning collapse from asymmetric loss | — |
| 2026-04-16 | 854fa36c | (pending) | fix(training): align optimizer/LR with paper | Adam→AdamW; lr 1e-3→5e-4; wd 0→0.05; per-step LR; warmup 3ep from 1e-4; batch ramp 8→128 | — | — |
