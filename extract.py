#!/usr/bin/env python3
"""
extract.py — Build the ICU-transfer prediction cohort from MIMIC-III.

Pipeline:
  1. Identify the target event: a patient on a GENERAL WARD who is subsequently
     transferred to an ICU (the first ICU care-unit entry that is preceded by a
     non-ICU/ward period within the same hospital admission).
  2. Compute the ICU transfer timestamp (the ICU intime).
  3. Enforce a 12-hour observation window ENDING 1 hour before that timestamp,
     so no data from the moment of transfer or after can leak into the prompt.
  4. Synthesize a clinician-style narrative from the vitals + labs in that window.
  5. Emit a .jsonl file, one patient per line, temporally ordered, with the
     ground-truth action (always "C" — every cohort patient was transferred).

Temporal integrity: every charted/lab event used satisfies
    window_start <= charttime <= window_end  AND  charttime < transfer_ts.

Usage:
    python extract.py --mimic-dir mimic-iii-clinical-database-demo-1.4 \
                      --out extract.jsonl --lookback-hours 12 --gap-hours 1 \
                      --target-n 100
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Clinical configuration
# ---------------------------------------------------------------------------
ICU_UNITS = {"MICU", "SICU", "CCU", "TSICU", "CSRU"}

# CHARTEVENTS itemid -> canonical vital name. Covers both the CareVue and
# MetaVision generations present in MIMIC-III. Temperatures in F are converted.
VITALS = {
    "Heart rate (bpm)":      [211, 220045],
    "Respiratory rate (/min)": [618, 220210, 615],
    "SpO2 (%)":              [646, 220277],
    "Systolic BP (mmHg)":    [51, 455, 220050, 220179],
    "Mean arterial BP (mmHg)": [52, 456, 220052, 220181],
    "GCS (total)":           [198],
}
TEMP_F = {678, 679, 223761}   # Fahrenheit itemids -> convert to C
TEMP_C = {676, 677, 223762}   # Celsius itemids

# LABEVENTS itemid -> canonical lab name + unit.
LABS = {
    "WBC (K/uL)":          51301,
    "Lactate (mmol/L)":    50813,
    "Creatinine (mg/dL)":  50912,
    "BUN (mg/dL)":         51006,
    "Hemoglobin (g/dL)":   51222,
    "Platelets (K/uL)":    51265,
    "Sodium (mEq/L)":      50983,
    "Potassium (mEq/L)":   50971,
    "Bicarbonate (mEq/L)": 50882,
    "Anion gap (mEq/L)":   50868,
    "Glucose (mg/dL)":     50931,
    "pH":                  50820,
    "Bands (%)":           51144,
}


def find_transfer_events(transfers: pd.DataFrame) -> pd.DataFrame:
    """Return one row per genuine ward->ICU transfer.

    Target event = any TRANSFERS row that ENTERS an ICU care unit
    (`curr_careunit` in ICU_UNITS) directly from a ward / non-ICU location
    (`prev_careunit` not in ICU_UNITS, i.e. NaN or a regular ward), and is not
    the hospital admission row itself (`eventtype != 'admit'`).

    This captures both the patient's initial ward->ICU transfer AND later
    ICU->ward->ICU "bounceback" readmissions — every one of which is a real
    deterioration-on-the-ward event with its own 12 h observation window.
    Excluded: direct ICU admissions (no prior ward) and ICU->ICU unit moves
    (prev_careunit already in ICU), neither of which has a ward window.
    """
    transfers = transfers.sort_values(["hadm_id", "intime"])
    events = []
    for hadm, g in transfers.groupby("hadm_id"):
        seq = 0
        for _, r in g.iterrows():
            enters_icu = r.curr_careunit in ICU_UNITS
            from_ward = r.prev_careunit not in ICU_UNITS   # NaN or non-ICU ward
            if enters_icu and from_ward and r.eventtype != "admit":
                seq += 1
                events.append({
                    "subject_id": int(r.subject_id),
                    "hadm_id": int(hadm),
                    "icustay_id": None if pd.isna(r.icustay_id) else int(r.icustay_id),
                    "icu_unit": r.curr_careunit,
                    "transfer_ts": r.intime,
                    "event_seq": seq,   # >1 marks a bounceback within the same admission
                })
    return pd.DataFrame(events)


def _trend(vals: list[float]) -> str:
    """Compact earliest->latest + range description for a vital series."""
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.0f}"
    return f"{vals[0]:.0f} → {vals[-1]:.0f} (range {min(vals):.0f}–{max(vals):.0f}, n={len(vals)})"


def build_narrative(demo: dict, vitals_df: pd.DataFrame, labs_df: pd.DataFrame,
                    lookback_h: int, gap_h: int) -> str:
    age = demo["age_str"]
    sex = demo["sex"]
    dx = demo["diagnosis"]
    lines = [
        f"A {age} {sex} was admitted to a general hospital ward with a presenting "
        f"problem of: {dx}.",
        f"The observations below were recorded on the ward during the {lookback_h}-hour "
        f"period ending {gap_h} hour before the current decision point. No later data is available.",
        "",
        "VITAL SIGNS (earliest → latest within the window):",
    ]
    if vitals_df.empty:
        lines.append("  - No vital signs were charted on the ward during this window.")
    else:
        for name in VITALS.keys() | {"Temperature (C)"}:
            s = vitals_df[vitals_df["vital"] == name].sort_values("charttime")
            if s.empty:
                continue
            vals = s["valuenum"].tolist()
            lines.append(f"  - {name}: {_trend(vals)}")

    lines += ["", "LABORATORY RESULTS (most recent value within the window):"]
    if labs_df.empty:
        lines.append("  - No laboratory results were resulted on the ward during this window.")
    else:
        for name in LABS.keys():
            s = labs_df[labs_df["lab"] == name].sort_values("charttime")
            if s.empty:
                continue
            last = s.iloc[-1]["valuenum"]
            extra = ""
            if len(s) > 1:
                first = s.iloc[0]["valuenum"]
                extra = f" (was {first:g} earlier in window)"
            lines.append(f"  - {name}: {last:g}{extra}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build MIMIC-III ICU-transfer cohort -> jsonl")
    ap.add_argument("--mimic-dir", default="mimic-iii-clinical-database-demo-1.4")
    ap.add_argument("--out", default="extract.jsonl")
    ap.add_argument("--lookback-hours", type=int, default=12)
    ap.add_argument("--gap-hours", type=int, default=1)
    ap.add_argument("--target-n", type=int, default=100,
                    help="Desired cohort size; capped at the number of true ward->ICU events.")
    args = ap.parse_args()

    base = Path(args.mimic_dir)
    if not base.exists():
        print(f"ERROR: MIMIC dir not found: {base}", file=sys.stderr)
        return 1

    print("Loading tables ...", file=sys.stderr)
    transfers = pd.read_csv(base / "TRANSFERS.csv", parse_dates=["intime", "outtime"])
    admissions = pd.read_csv(base / "ADMISSIONS.csv", parse_dates=["admittime", "dischtime"])
    patients = pd.read_csv(base / "PATIENTS.csv", parse_dates=["dob"])

    events = find_transfer_events(transfers)
    print(f"Found {len(events)} genuine ward->ICU transfer events "
          f"({events.subject_id.nunique()} unique patients).", file=sys.stderr)
    if len(events) < args.target_n:
        print(f"NOTE: requested target-n={args.target_n} but only {len(events)} ward->ICU "
              f"events exist in this dataset. Using all {len(events)}.", file=sys.stderr)

    events = events.merge(
        admissions[["hadm_id", "admittime", "diagnosis"]], on="hadm_id", how="left"
    ).merge(patients[["subject_id", "gender", "dob"]], on="subject_id", how="left")
    events = events.sort_values("transfer_ts").head(args.target_n).reset_index(drop=True)

    wanted_hadms = set(events.hadm_id)

    # Stream the big event tables, keeping only rows for cohort admissions.
    print("Scanning CHARTEVENTS (vitals) ...", file=sys.stderr)
    vital_ids = {i for ids in VITALS.values() for i in ids} | TEMP_F | TEMP_C
    chart_keep = []
    for chunk in pd.read_csv(base / "CHARTEVENTS.csv",
                             usecols=["hadm_id", "itemid", "charttime", "valuenum", "error"],
                             parse_dates=["charttime"], chunksize=100_000):
        c = chunk[chunk.hadm_id.isin(wanted_hadms) & chunk.itemid.isin(vital_ids)]
        c = c[(c.error != 1) & c.valuenum.notna()]
        if len(c):
            chart_keep.append(c)
    chart = pd.concat(chart_keep) if chart_keep else pd.DataFrame(
        columns=["hadm_id", "itemid", "charttime", "valuenum"])

    print("Scanning LABEVENTS ...", file=sys.stderr)
    lab_ids = set(LABS.values())
    labs_all = pd.read_csv(base / "LABEVENTS.csv",
                           usecols=["hadm_id", "itemid", "charttime", "valuenum"],
                           parse_dates=["charttime"])
    labs_all = labs_all[labs_all.hadm_id.isin(wanted_hadms) & labs_all.itemid.isin(lab_ids)
                        & labs_all.valuenum.notna()]

    id2vital = {}
    for name, ids in VITALS.items():
        for i in ids:
            id2vital[i] = name
    id2lab = {v: k for k, v in LABS.items()}

    lookback, gap = args.lookback_hours, args.gap_hours
    out_path = Path(args.out)
    n_written = 0
    n_empty_window = 0
    with out_path.open("w") as fh:
        for _, ev in events.iterrows():
            ts = ev.transfer_ts
            w_end = ts - pd.Timedelta(hours=gap)
            w_start = w_end - pd.Timedelta(hours=lookback)

            # --- vitals in window (strictly before transfer) ---
            cv = chart[(chart.hadm_id == ev.hadm_id)
                       & (chart.charttime >= w_start) & (chart.charttime <= w_end)
                       & (chart.charttime < ts)].copy()
            if len(cv):
                def _vital_name(r):
                    if r.itemid in TEMP_F:
                        return "Temperature (C)"
                    if r.itemid in TEMP_C:
                        return "Temperature (C)"
                    return id2vital.get(r.itemid)
                cv["vital"] = cv.apply(_vital_name, axis=1)
                cv.loc[cv.itemid.isin(TEMP_F), "valuenum"] = \
                    (cv.loc[cv.itemid.isin(TEMP_F), "valuenum"] - 32) * 5.0 / 9.0
                cv = cv.dropna(subset=["vital"])
            vitals_df = cv if len(cv) else pd.DataFrame(columns=["vital", "valuenum", "charttime"])

            # --- labs in window ---
            lv = labs_all[(labs_all.hadm_id == ev.hadm_id)
                          & (labs_all.charttime >= w_start) & (labs_all.charttime <= w_end)
                          & (labs_all.charttime < ts)].copy()
            if len(lv):
                lv["lab"] = lv.itemid.map(id2lab)
            labs_df = lv if len(lv) else pd.DataFrame(columns=["lab", "valuenum", "charttime"])

            # --- demographics ---
            if pd.isna(ev.dob) or pd.isna(ev.admittime):
                age_str = "patient of unknown age"
            else:
                # Use native datetime to avoid Timedelta overflow: MIMIC shifts
                # the DOB of patients >89 so the apparent age is ~300 years.
                age_yrs = (ev.admittime.to_pydatetime() - ev.dob.to_pydatetime()).days / 365.25
                age_str = ">89-year-old" if age_yrs > 89 else f"{int(round(age_yrs))}-year-old"
            sex = {"M": "man", "F": "woman"}.get(ev.gender, "patient")
            dx = str(ev.diagnosis) if not pd.isna(ev.diagnosis) else "not documented"
            demo = {"age_str": age_str, "sex": sex, "diagnosis": dx}

            narrative = build_narrative(demo, vitals_df, labs_df, lookback, gap)
            n_chart, n_lab = len(vitals_df), len(labs_df)
            if n_chart == 0 and n_lab == 0:
                n_empty_window += 1

            # Unique id; append the event sequence only for bouncebacks (seq>1)
            # so single-event admissions keep the clean subject_hadm form.
            pid = f"{ev.subject_id}_{ev.hadm_id}"
            if int(ev.event_seq) > 1:
                pid += f"_{int(ev.event_seq)}"
            rec = {
                "patient_id": pid,
                "subject_id": int(ev.subject_id),
                "hadm_id": int(ev.hadm_id),
                "icustay_id": ev.icustay_id,
                "icu_unit": ev.icu_unit,
                "transfer_ts": ts.isoformat(),
                "window_start": w_start.isoformat(),
                "window_end": w_end.isoformat(),
                "ground_truth_action": "C",          # every cohort patient WAS transferred
                "n_vital_measurements": int(n_chart),
                "n_lab_measurements": int(n_lab),
                "narrative": narrative,
            }
            fh.write(json.dumps(rec) + "\n")
            n_written += 1

    print(f"\nWrote {n_written} patients -> {out_path}", file=sys.stderr)
    print(f"  windows with NO ward data at all: {n_empty_window} "
          f"({100*n_empty_window/max(n_written,1):.0f}%)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
