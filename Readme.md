# Category-Aware LLM Routing on TACO

A controlled study of whether routing each problem category to its best-performing model beats always using the single best model, run across 13 open-weight code LLMs on the TACO competitive-programming benchmark.

Everything lives in one self-contained notebook: `TACO_Routing_Study_v4.ipynb`. Corpus construction → generation → execution grading → cross-validated routing → policy comparison → tables and figures. Every stage checkpoints to disk and resumes after a crash.

This is a companion to the HumanEval+ notebook, using the same protocol and the same statistics, so the two runs form a genuine replication pair.

---

## The question

Given a pool of similarly-sized code models, does knowing a problem's category (dynamic programming, graphs, strings, …) let you pick a better model than always sending everything to the strongest one? And if so, does that advantage survive cross-validation, budget matching, and a sanity check on whether the category axis means anything in the first place?

Three arms address it:

1. **Category routing** — a route map fit on 9 folds and scored on the 10th, against a cross-validated best-single baseline.
2. **Width vs. depth at matched budget** — is a gated cascade better than just resampling one model *k* times for the same number of generations?
3. **Repair vs. blind resampling** — TACO gives you the failing input, the expected output, and the actual output. Does feeding that back beat spending the same generations on fresh samples?

## Protocol

| | |
|---|---|
| Benchmark | TACO test split, pulled from the Hub |
| Categories | 9, built from TACO's own tags via a fixed priority order |
| Models | 13 (11 in the routing pool; 2 base models excluded) |
| pass@1 | Greedy decoding, T=0.0 |
| pass@k | Unbiased estimator (Chen et al., 2021) from 10 samples at T=0.8, top-p 0.95 |
| Grading | All provided test cases up to a cap of 10, pre-cap count recorded |
| Cascade gating | First 2 cases only; grading always uses the full set |
| Routing | 10-fold cross-validation |
| Significance | Exact McNemar, Wilson + bootstrap CIs, Benjamini–Hochberg and Holm correction |

### Models

Qwen2.5-Coder (3B / 7B / 14B), DeepSeek-Coder-6.7B, Mistral-7B, Llama-3.1-8B, Phi-3.5-Mini, CodeLlama (7B / 13B), CodeGemma-7B, Granite-4.1-8B, StarCoder2 (7B / 15B).

StarCoder2 variants are base models. They get completion-style prompts rather than a chat template, and they are excluded from the routing pool — a base model evaluated under a chat template produces a fake "bad at category X" result. Precision defaults to uniform bf16 so that per-category winners are a statement about models, not about quantization.

## Design decisions worth knowing about

The version history of this notebook is mostly a list of ways the earlier versions were quietly wrong. The fixes are the interesting part:

- **Routing is cross-validated.** Fitting a route map on all problems and then scoring it on the same problems measures memorization, not routing. Every routing number here is out-of-fold.
- **Cascades gate on cases they are not graded on.** Escalating based on the same tests that determine the verdict is the most common way to manufacture a good cascade.
- **pass@k is the unbiased estimator**, not "any of 5 samples passed," which is biased downward at small sample counts.
- **Token costs come from the real tokenizer**, logged at generation time, not `len(chars) / 4`.
- **The taxonomy is treated as a modelling choice.** The tag→category mapping is many-to-one and resolved by priority order. The multi-tag rate is reported, and Table 8 re-runs the entire routing analysis under 20 random permutations of that priority. If the conclusions move, the category axis is an artifact.
- **Two harness-soundness checks ship with the results.** TACO is unsaturated — the best model here sits around 9.6% — which is good for preserving per-category variance but shifts the burden onto you to show the low numbers are the benchmark, not a broken pipeline. Table 2b (difficulty gradient: easy items must score visibly higher) and Table 2c (failure composition: failures should be wrong answers, not syntax errors) do that work.
- **Both TACO problem shapes are handled.** stdin problems and call-based problems (`fn_name` in `input_output`) need different harnesses; conflating them silently scores every call-based problem as a failure.
- **Empty generations are never checkpointed.** An OOM that wrote empty strings to the JSONL used to mark a problem "done" forever, so the resume skipped it and the model silently scored zero on it.

## Running it

**Step 1 — smoke test.** Set `DRY_RUN = True` and run everything. About 10 minutes, no GPU. This exercises the corpus builder, both executor modes, all statistics, and all figures against mock generations.

**Step 2 — the sweep.** `DRY_RUN = False`, run §0 → §4. GPU required, fully resumable. Interrupt it, restart the kernel, re-run: it continues from the last completed problem.

**Step 3 — analysis.** §5 → §9, any time. CPU-only, roughly a minute, re-runnable as often as you like since it reads only the on-disk checkpoints.

Nothing to attach — TACO comes from the Hub and the taxonomy is built from its tags.

### Requirements

Python with `transformers>=4.45`, `torch`, `datasets`, `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`. The install cell handles these.

vLLM is used when available and is 10–20× faster. There is no Windows build, so on Windows the `transformers` path *is* the pipeline; the notebook auto-sizes batches against measured free VRAM and caps the fraction of the card it may claim, which matters because the Windows driver silently spills to shared system RAM instead of raising OOM.

Budget roughly 25 GB of Hub cache per 13B model — about 250 GB for the full set. Point `HF_HOME` at a drive with room *before* the first download; a truncated download fails later with a confusing "does not appear to have files named…" error.

### Credentials

Set `HF_TOKEN` in the environment (or as a Kaggle secret). Gated repos — Llama-3.1, CodeGemma — are skipped automatically if it is absent. No token is ever hard-coded.

## Outputs

```
taco_run/
  corpus.json            problems, categories, difficulty, test cases
  taxonomy_audit.json    multi-tag rate, case counts, difficulty and type mix
  gen/<model>.jsonl      raw generations, appended per problem
  grade/<model>.jsonl    execution verdicts, appended per problem
  repair/<model>.jsonl   repair-arm attempts
  tables/                tab1–tab9, CSV and LaTeX
  figures/               fig1–fig7
  headline_numbers.json  the numbers a results section needs
  results_paragraph.txt  auto-drafted results prose — edit and verify before use
  run_manifest.json      reproducibility record
```

Section 9.3 maps each artifact to the paper element it belongs in. Section 9.4 lists what the numbers do and do not license you to claim — in particular, anything from a `TRIAL`, `QUICK`, or `DRY` run is not a result, and routing numbers without the CV prefix are not routing numbers.

