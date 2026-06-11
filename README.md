# Evaluating Clinical AI Agents for ICU Transfer Prediction

A reproducible pipeline that tests whether a foundational LLM, operating under
stateless / zero-retention conditions, can recognise clinical deterioration
from a structured ward narrative and recommend ICU escalation **before** the
transfer happens.

Dataset: **MIMIC-III Clinical Database Demo v1.4** (100 patients, open access).

## Deliverables

| File | Purpose |
|------|---------|
| `extract.py` | Builds the cohort, computes the ICU-transfer timestamp, enforces the 12 h lookback window, and writes `extract.jsonl` (one patient per line, temporally ordered, no post-event data). |
| `benchmark.py` | Reads `extract.jsonl`, submits each narrative to the model under strictly stateless conditions, captures the A/B/C decision, writes `results.csv`. |
| `results.csv` | One row per patient: `patient_id, model, predicted_action, ground_truth_action, latency_s, parse_success, rationale`. |
| `READOUT.md` | 10-minute readout: accuracy, a correct example, a failure example, temporal-data observations, dataset nature & limitations. |

## Quick start

```bash
# 1. Environment
python3 -m venv .venv && . .venv/bin/activate
pip install pandas anthropic requests

# 2. Get the data (open access, no signup)
curl -L -o mimic-demo.zip \
  https://physionet.org/static/published-projects/mimiciii-demo/mimic-iii-clinical-database-demo-1.4.zip
unzip -q mimic-demo.zip

# 3. Build the cohort -> extract.jsonl
python extract.py --target-n 100

# 4. Benchmark a model -> results.csv
export ANTHROPIC_API_KEY=sk-ant-...
python benchmark.py --backend claude --claude-model claude-opus-4-8

# Bonus: an open-source model via any OpenAI-compatible endpoint
export OSS_BASE_URL=https://api.together.xyz/v1 OSS_API_KEY=...
python benchmark.py --backend oss --oss-model meta-llama/Llama-3.3-70B-Instruct-Turbo --out results_oss.csv

# No API key handy? Offline deterministic demo:
python benchmark.py --backend mock --out results_mock.csv
```

## Design decisions (the parts that matter)

**Target event.** A *genuine ward→ICU transfer*: the first ICU care-unit entry
(`curr_careunit` ∈ {MICU, SICU, CCU, TSICU, CSRU}) within a hospital admission
that is **preceded by a non-ICU/ward period**. Direct ICU admissions are
excluded — they never had a ward observation window to evaluate. The transfer
timestamp is the ICU `intime`.

**Observation window.** `[transfer_ts − 13h, transfer_ts − 1h]`: a 12-hour
window ending **1 hour before** transfer. Every event used satisfies
`charttime < transfer_ts`, so nothing from the moment of transfer or after can
leak into a prompt (temporal integrity).

**Stateless / zero-retention.** Each patient is a fresh, single-turn API call
with no conversation history. The request shape is stateless; true *zero data
retention* additionally requires the org-level ZDR setting on the provider.

**Ground truth.** Every cohort patient was transferred, so the correct action
is **C (Escalate)** for all of them. See `READOUT.md` for why this single-class
design only measures **sensitivity/recall**, not specificity.

## Author

**Viseth Sean** ([@visethchapman](https://github.com/visethchapman)) — case study
for Centific. Code licensed under MIT (see `LICENSE`); the MIMIC-III demo dataset
is provided by PhysioNet under ODbL v1.0.
