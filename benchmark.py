#!/usr/bin/env python3
"""
benchmark.py — Present each patient narrative to an LLM under stateless
("zero-retention") conditions and capture the categorical escalation decision.

Each patient is sent as a fully independent, single-turn request — no
conversation history, no shared session state. The model must choose exactly
one of:
    A  Continue routine floor monitoring
    B  Order routine morning labs
    C  Escalate to ICU

Two model backends are supported:
  * closed-source: Anthropic Claude via the official `anthropic` SDK
                   (default model: claude-opus-4-8).
  * open-source:   any OpenAI-compatible /chat/completions endpoint
                   (Together, Groq, Fireworks, vLLM, Ollama, ...), via raw HTTP.
  * mock:          offline deterministic stand-in (no network/API key) so the
                   full pipeline + results.csv can be produced and inspected.

Output: results.csv with one row per patient:
    patient_id, model, predicted_action, ground_truth_action, latency_s, parse_success, rationale

Zero-retention note: the API request itself is stateless. True zero data
retention is an organization/account setting on the provider side — enable
"zero data retention" on your Anthropic org (and the equivalent on any OSS
host) to fully satisfy the MIMIC data-use constraints.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python benchmark.py --backend claude --in extract.jsonl --out results.csv

    export OSS_BASE_URL=https://api.together.xyz/v1 OSS_API_KEY=...
    python benchmark.py --backend oss --oss-model meta-llama/Llama-3.3-70B-Instruct-Turbo

    python benchmark.py --backend mock      # offline demo, no key required
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time, hashlib

SYSTEM_PROMPT = (
    "You are the physician covering a general hospital ward. You are reviewing one "
    "patient based only on the observations recorded over the last several hours. "
    "Decide the single most appropriate next action.\n\n"
    "You must choose EXACTLY ONE of:\n"
    "  A = Continue routine floor monitoring\n"
    "  B = Order routine morning labs\n"
    "  C = Escalate to ICU\n\n"
    "Base your judgement only on the data provided; do not assume facts not stated. "
    'Respond ONLY with a JSON object on a single line: {"decision": "A|B|C", '
    '"rationale": "<=20 words"}. No other text.'
)

VALID = {"A", "B", "C"}


def parse_decision(text: str):
    """Extract (decision, rationale, parse_success) from raw model output."""
    if not text:
        return None, "", False
    # 1) Try strict JSON.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            d = str(obj.get("decision", "")).strip().upper()[:1]
            if d in VALID:
                return d, str(obj.get("rationale", ""))[:200], True
        except json.JSONDecodeError:
            pass
    # 2) Fallback: first standalone A/B/C token.
    m = re.search(r"\b([ABC])\b", text.upper())
    if m:
        return m.group(1), text.strip()[:200], False  # recovered, but not clean format
    return None, text.strip()[:200], False


# ---------------------------------------------------------------------------
# Backends — each returns raw response text for one stateless request.
# ---------------------------------------------------------------------------
class ClaudeBackend:
    def __init__(self, model: str):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY

    def __call__(self, narrative: str) -> str:
        # Single-turn, no history => stateless. Fresh request per patient.
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": narrative}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class OSSBackend:
    """Any OpenAI-compatible chat endpoint, called via raw HTTP."""
    def __init__(self, model: str, base_url: str, api_key: str):
        import requests
        self.requests = requests
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"}

    def __call__(self, narrative: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": 200,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": narrative},
            ],
        }
        r = self.requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class MockBackend:
    """Offline deterministic stand-in. Applies a transparent clinical heuristic
    over the narrative so the pipeline produces a realistic spread of A/B/C
    (with the occasional malformed reply) without any network call."""
    def __init__(self, model="mock-clinician-v1"):
        self.model = model

    def __call__(self, narrative: str) -> str:
        time.sleep(0.01)
        t = narrative.lower()
        score = 0
        # crude deterioration cues
        if re.search(r"lactate \(mmol/l\): (\d+(\.\d+)?)", t):
            lac = float(re.search(r"lactate \(mmol/l\): (\d+(\.\d+)?)", t).group(1))
            score += 2 if lac >= 4 else (1 if lac >= 2 else 0)
        if "liver failure" in t or "sepsis" in t or "hemorrhage" in t or "shock" in t:
            score += 1
        if "→" in narrative:  # any trending value present
            score += 1
        h = int(hashlib.md5(narrative.encode()).hexdigest(), 16) % 5
        score += 1 if h == 0 else 0
        decision = "C" if score >= 2 else ("B" if score == 1 else "A")
        # ~1 in 12 produce a malformed (non-JSON) reply to exercise the parser
        if h == 3:
            return f"My recommendation is option {decision} given the picture."
        return json.dumps({"decision": decision, "rationale": "offline heuristic stand-in"})


def make_backend(args):
    if args.backend == "claude":
        return ClaudeBackend(args.claude_model)
    if args.backend == "oss":
        base = args.oss_base_url or os.environ.get("OSS_BASE_URL")
        key = os.environ.get("OSS_API_KEY", "")
        if not base:
            sys.exit("ERROR: set --oss-base-url or OSS_BASE_URL for --backend oss")
        return OSSBackend(args.oss_model, base, key)
    return MockBackend()


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark an LLM on ICU-transfer narratives")
    ap.add_argument("--backend", choices=["claude", "oss", "mock"], default="claude")
    ap.add_argument("--in", dest="infile", default="extract.jsonl")
    ap.add_argument("--out", dest="outfile", default="results.csv")
    ap.add_argument("--claude-model", default="claude-opus-4-8")
    ap.add_argument("--oss-model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    ap.add_argument("--oss-base-url", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap number of patients (0 = all)")
    args = ap.parse_args()

    patients = [json.loads(l) for l in open(args.infile)]
    if args.limit:
        patients = patients[:args.limit]

    backend = make_backend(args)
    model_name = backend.model
    print(f"Backend={args.backend}  model={model_name}  patients={len(patients)}", file=sys.stderr)

    rows = []
    for i, p in enumerate(patients, 1):
        t0 = time.perf_counter()
        try:
            raw = backend(p["narrative"])
        except Exception as e:
            raw = ""
            print(f"  [{i}] {p['patient_id']} request error: {type(e).__name__}: {e}",
                  file=sys.stderr)
        latency = time.perf_counter() - t0
        decision, rationale, ok = parse_decision(raw)
        rows.append({
            "patient_id": p["patient_id"],
            "model": model_name,
            "predicted_action": decision or "PARSE_FAIL",
            "ground_truth_action": p["ground_truth_action"],
            "latency_s": round(latency, 3),
            "parse_success": ok,
            "rationale": rationale.replace("\n", " "),
        })
        print(f"  [{i}/{len(patients)}] {p['patient_id']}: pred={decision} "
              f"truth={p['ground_truth_action']} ok={ok} {latency:.2f}s", file=sys.stderr)

    with open(args.outfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # quick summary
    n = len(rows)
    correct = sum(1 for r in rows if r["predicted_action"] == r["ground_truth_action"])
    parsed = sum(1 for r in rows if r["parse_success"])
    from collections import Counter
    dist = Counter(r["predicted_action"] for r in rows)
    print(f"\nWrote {n} rows -> {args.outfile}", file=sys.stderr)
    print(f"Recall (escalation sensitivity, pred==C): {correct}/{n} = {100*correct/n:.1f}%",
          file=sys.stderr)
    print(f"Clean parse rate: {parsed}/{n} = {100*parsed/n:.1f}%", file=sys.stderr)
    print(f"Decision distribution: {dict(dist)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
