"""18 API/security checks against a running backend (python tests/api_checks.py)."""
from __future__ import annotations

import json
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
results: dict[str, dict] = {}


def chk(name: str, cond: bool, detail: str = "") -> None:
    results[name] = {"pass": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + (f"  — {detail}" if detail else ""))


def main() -> int:
    chk("health contract", "status" in requests.get(f"{BASE}/health", timeout=10).json())
    caps = requests.get(f"{BASE}/api/v1/protection/capabilities", timeout=10).json()
    chk("capabilities.reasoning_review", "reasoning_review" in caps)
    chk("capabilities.contextual_ai", "contextual_ai" in caps, caps.get("contextual_ai", {}).get("provider", ""))
    lst = requests.get(f"{BASE}/api/v1/protection/analyses", timeout=20).json()
    items = lst.get("analyses", lst)
    chk("history list", isinstance(items, list) and len(items) > 0, f"{len(items)} records")
    one = items[0]["id"]
    d = requests.get(f"{BASE}/api/v1/protection/analyses/{one}", timeout=20).json()
    chk("history detail persists quality", "audio_quality" in d)
    chk("history detail keeps evidence quotes", "evidence_quotes" in (d.get("content") or {}))
    llm = (d.get("ai") or {}).get("llm") or {}
    chk("llm fallback fields present", "fallback" in llm and "configured" in llm, str(llm.get("fallback")))
    rr = llm.get("reasoner") or {}
    chk("reasoner keeps stage one", "stage_one_classification" in rr and "final_risk_category" in rr)
    r = requests.get(f"{BASE}/api/v1/protection/analyses/{one}/report", timeout=20)
    chk("JSON report export", r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"))
    c = requests.get(f"{BASE}/api/v1/analyses/export.csv", timeout=20)
    chk("CSV export", c.status_code == 200 and "csv" in c.headers.get("content-type", ""), f"{len(c.content)}B")
    chk("evaluation endpoint", requests.get(f"{BASE}/api/v1/evaluation", timeout=40).status_code in (200, 202))
    chk("family voices list", requests.get(f"{BASE}/api/v1/protection/family-voices", timeout=20).status_code == 200)
    chk("unknown analysis id -> 404",
        requests.get(f"{BASE}/api/v1/protection/analyses/doesnotexist", timeout=10).status_code in (404, 422))
    chk("consent required", requests.post(f"{BASE}/api/v1/protection/analyze", timeout=20).status_code in (400, 422))
    dev = requests.options(f"{BASE}/api/v1/protection/analyses",
                           headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"}, timeout=10)
    prod = requests.options(f"{BASE}/api/v1/protection/analyses",
                            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"}, timeout=10)
    chk("CORS allows localhost:5173", dev.headers.get("access-control-allow-origin") in ("http://localhost:5173", "*"))
    chk("CORS rejects unknown origin", prod.headers.get("access-control-allow-origin") is None)
    h = requests.get(f"{BASE}/health", timeout=10).json()
    chk("health reports model readiness", "model_ready" in h and "speaker_model_ready" in h)
    chk("no secrets echoed in capabilities", "gsk_" not in json.dumps(caps) and "api_key" not in json.dumps(caps).lower())
    passed = sum(v["pass"] for v in results.values())
    print(f"\n{passed}/{len(results)} API/security checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
