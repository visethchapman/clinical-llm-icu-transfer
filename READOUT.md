# 10-Minute Readout — Evaluating an LLM for ICU-Transfer Escalation

**Task.** Given a structured narrative of a ward patient's last 12 hours
(ending 1 h before an ICU transfer), can a foundational LLM, queried statelessly,
recommend escalation (action **C**) *before* the transfer occurs?

**Dataset.** MIMIC-III Clinical Database Demo v1.4.
**Model.** `claude-opus-4-8` (Anthropic SDK, single-turn, stateless). 59 patients,
one independent API call each.

---

## 1. Headline results

| Metric | Value |
|---|---|
| Cohort (ward→ICU transfer events) | **59** (47 unique patients) |
| **Recall** — escalated correctly (pred = C) | **28 / 59 = 47.5%** |
| Decision distribution | **C ×28, B ×21, A ×10** |
| Clean JSON parse rate | **59 / 59 = 100%** |
| Latency per call | mean 3.44 s, median 3.40 s (range 1.6–8.6 s) |

The single most important finding: **the model escalated on fewer than half of
patients who genuinely deteriorated.** It did *not* reflexively answer "C" — it
chose "continue monitoring" (A) 10 times and "routine labs" (B) 21 times on
patients who were all, in fact, transferred to ICU within the hour.

---

## 2. The result has to be read through a one-class design

Every cohort patient *was* transferred, so the ground-truth action is **C for
all 59**. Consequently:

- We can measure **sensitivity / recall** (did it catch the deterioration?) and
  **nothing else** — there are no "stayed-on-the-floor" controls, so we cannot
  compute specificity or a false-positive rate.
- A model that blindly answers **C** every time would score **100% recall** while
  being clinically useless. Claude's 47.5% is therefore *not* a model that's
  "wrong half the time" in a vacuum — it's a model that is **discriminating**
  (using the data to sometimes hold back), and the open question this design
  cannot answer is whether that discrimination is *correct*. **A real evaluation
  needs a matched control arm of non-transferred ward patients.** That is a
  property of the task as specified, not of the implementation.

---

## 3. One correct prediction, with its reasoning

**Patient `10126_160445`** — 44-year-old woman, admitting problem *liver failure*.
Lab-only window, unambiguous trajectory:

```
Lactate     1.5 → 7.4 mmol/L     Hemoglobin  11.4 → 7.9 g/dL
Platelets   ~51–54 K/uL          WBC          4.4 → 3.8 K/uL
```

**Model → C (Escalate)** in 3.3 s. Verbatim rationale:

> *"Liver failure with rising lactate 7.4, falling hemoglobin 7.9,
> thrombocytopenia — signs of shock/bleeding requiring ICU."*

This is exactly right, and notice it needed only the *lab trend* — no vital signs
— to make the call. The narrative's earliest→latest encoding (not just a snapshot)
is what surfaced the rising lactate.

---

## 4. One failure — and why it's the interesting kind

No format violations occurred (100% parse rate — see §5), so the failures here
are **clinical misses**, all 10 of them where the model answered A/B on a patient
who was escalated. The most instructive:

**Patient `42199_178513`** — 55-year-old man, admitting problem *shortness of breath*.

```
VITAL SIGNS: none charted on the ward during the window
LABS: WBC 6.4, Hb 8.6, Cr 0.7, BUN 12, Na 141, K 4.1, HCO3 25, glucose 110  (all ~normal)
```

**Model → A (Continue routine floor monitoring).** Verbatim rationale:

> *"Labs largely normal except mild anemia; no vitals charted, but no acute
> instability indicated."*

The model's reasoning is *locally correct* — the labs really are reassuring. But
the patient presented with **shortness of breath** and deteriorated **respiratorily**:
the discriminating signal would have been a rising respiratory rate or falling
SpO₂ — **exactly the vital signs MIMIC does not record on the ward** (§6). The
miss is driven by a **data-modality gap, not a reasoning error.** This is the
single most important practical lesson: on this data substrate, respiratory and
haemodynamic deteriorations are nearly invisible, so any lab-only escalation model
will systematically miss them.

> The 4 "zero ward data" patients (handed only age + diagnosis) are a related
> failure surface: the model leans to B/C and hedges, which is the safe behaviour,
> but it is guessing.

---

## 5. A note on parse success / format handling

`parse_success` is a first-class column because *the model chose B* and *the model
emitted unparseable text we salvaged* are different events. The parser is layered:
(1) strict JSON `{"decision","rationale"}`; (2) regex fallback to a standalone
A/B/C token (recovers a decision but flags `parse_success = False`). **Claude
produced clean JSON on 59/59 calls — zero format violations**, with free-text
prompting (no forced/structured output). A smaller open-source model is where
format violations actually appear; `benchmark.py --backend oss` runs the identical
harness against any local/OSS model for that comparison.

---

## 6. How the model handled the temporal data

- **The 12 h window is labs, not vitals.** Only **9/59 (15%)** of windows contained
  any charted vital sign; **54/59 (92%)** contained labs. MIMIC-III's `CHARTEVENTS`
  is populated almost entirely *during ICU stays* — ward vitals live in nursing
  flowsheets the research DB doesn't capture. `LABEVENTS` is hospital-wide. So the
  model reasons over a **lab trajectory**, and §4's failure is the direct
  consequence.
- **Trends beat snapshots.** Encoding earliest→latest (`Lactate 1.5 → 7.4`) is what
  let the model see deterioration; a single value would have been ambiguous.
- **Temporal integrity held.** Every event satisfies `charttime < transfer_ts`, and
  the 1 h gap guards against last-minute peri-transfer labs. No leakage.

---

## 7. Nature of the dataset & limitations

1. **The demo cannot supply 100 ward→ICU transfers.** "Cohort Size: 100 patients"
   describes the *demo database* (100 patients total). Only **47** of them ever had
   a ward→ICU transfer (**59** events incl. bouncebacks); most ICU stays are *direct*
   admissions with no ward window. Reaching n=100 requires full MIMIC-III (~46k
   admissions) — `extract.py` scales to it unchanged.
2. **ICU-biased charting** (§6) — ward vitals are structurally absent; lab-based
   signals are the reliable substrate. This persists on full MIMIC-III.
3. **One-class design** (§2) — recall only; no specificity, no calibration.
4. **Tiny N + patient overlap** — 59 events / 47 patients (5 patients contribute
   ≥2 admissions; 4 events are bounceback readmissions). CIs are wide; treat as
   directional.
5. **Date-shifting** — MIMIC shifts dates to the future and ages >89 to ~300 y
   (handled: reported as ">89-year-old"). Absolute times are fictional; *relative*
   intervals within an admission — all the 12 h window needs — are preserved.
6. **Label ≠ optimal care** — "was transferred" is objective and deterministic, but
   it encodes the *actual* clinical decision, not necessarily the *ideal* one; some
   transfers are precautionary, some late.
7. **Zero-retention is partly an account setting** — the request is stateless
   single-turn; full ZDR also requires the provider org-level flag. (Data here is
   the open-access, de-identified *demo*.)

---

### Reproduce

```bash
python extract.py --target-n 100                       # -> extract.jsonl (59 patients)
export ANTHROPIC_API_KEY=sk-ant-...
python benchmark.py --backend claude --claude-model claude-opus-4-8   # -> results.csv
# open-source comparison (bonus), fully offline:
python benchmark.py --backend oss --oss-model <local-model> --out results_oss.csv
```
