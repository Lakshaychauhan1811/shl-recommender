"""
Evaluation suite v2 — replays the 10 real gold-standard conversation traces
(C1-C10) turn-by-turn against a running instance of the API (local or deployed
on Render) and scores retrieval quality, recommendation relevance,
groundedness, and end-of-conversation accuracy.

Usage:
    # against local dev server
    uvicorn main:app --reload &
    python eval.py

    # against a deployed instance
    API_BASE_URL=https://shl-recommender-2u14.onrender.com python eval.py

Falls back to catalog-integrity-only checks if the API is unreachable.
"""
import os
import sys
import json
import time
import urllib.request
import urllib.error

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from main import CATALOG  # noqa: E402  (also validates main.py imports cleanly)

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
TRACES_PATH = os.path.join(HERE, "real_traces.json")

with open(TRACES_PATH) as f:
    REAL_TRACES = json.load(f)

CATALOG_URLS = {c["link"].rstrip("/") + "/" for c in CATALOG}
CATALOG_NAMES = {c["name"].lower().strip() for c in CATALOG}


# ── HTTP helper (stdlib only, no extra deps) ─────────────────────────────────
def post_chat(messages, timeout=90):
    body = json.dumps({"messages": messages}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get_health(timeout=150):
    req = urllib.request.Request(f"{BASE_URL}/health", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def norm_url(u):
    return (u or "").rstrip("/") + "/"


# ── Reporting ─────────────────────────────────────────────────────────────────
passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"    ✅ {name}")
        passed += 1
    else:
        print(f"    ❌ {name}: {detail}")
        failed += 1


print("=" * 70)
print("SHL Recommender — Real Trace Evaluation (C1-C10, live API)")
print(f"Target: {BASE_URL}")
print("=" * 70)

# ── 0. Catalog integrity (static, no network needed) ─────────────────────────
print("\n[0] Catalog integrity")
check("377 products loaded", len(CATALOG) == 377, f"got {len(CATALOG)}")
check("All have name/link/keys", all("name" in i and "link" in i and "keys" in i for i in CATALOG))
urls = [i["link"] for i in CATALOG]
check("No duplicate URLs", len(urls) == len(set(urls)))

# ── 1. Warm-up / health check ────────────────────────────────────────────────
print("\n[1] API reachability")
api_up = False
for attempt in range(2):
    try:
        t0 = time.time()
        h = get_health()
        dt = time.time() - t0
        check("GET /health returns ok", h.get("status") == "ok", str(h))
        print(f"    time cold-start/response time: {dt:.1f}s")
        api_up = True
        break
    except Exception as e:
        if attempt == 0:
            print(f"    (first attempt timed out - Render free tier may be cold-starting, retrying once...)")
            continue
        check("GET /health reachable", False, str(e))
        print("\nWARNING: API unreachable - skipping live trace replay.")
        print("   Set API_BASE_URL env var to your deployed URL, or run")
        print("   `uvicorn main:app --reload` locally, then re-run eval.py.")

# ── 2. Replay real traces (only if API is up) ────────────────────────────────
retrieval_hits, retrieval_total = 0, 0     # recall: expected items actually returned
groundedness_hits, groundedness_total = 0, 0  # every returned rec must exist in catalog
clarify_correct, clarify_total = 0, 0      # empty-rec turns correctly left empty
end_correct, end_total = 0, 0              # end_of_conversation flag matches expected

if api_up:
    for trace_name, turns in REAL_TRACES.items():
        print(f"\n[Trace {trace_name}] {len(turns)} turns")
        history = []
        for i, turn in enumerate(turns, 1):
            history.append({"role": "user", "content": turn["user"]})
            time.sleep(2.5)  # stay under Groq free-tier 30 RPM cap across the full replay
            try:
                resp = post_chat(history)
            except Exception as e:
                check(f"{trace_name} turn {i}: API call", False, str(e))
                continue

            actual_recs = resp.get("recommendations") or []
            actual_urls = {norm_url(r.get("url")) for r in actual_recs}
            expected_recs = turn["expect_recs"]
            expected_urls = {norm_url(r["url"]) for r in expected_recs}

            # Groundedness: every returned rec must be a real catalog item
            groundedness_total += len(actual_recs)
            grounded_ok = sum(1 for u in actual_urls if u in CATALOG_URLS)
            groundedness_hits += grounded_ok
            if actual_recs:
                check(
                    f"{trace_name} turn {i}: all recs grounded in catalog",
                    grounded_ok == len(actual_urls),
                    f"{len(actual_urls) - grounded_ok} of {len(actual_urls)} not found in catalog",
                )

            if expected_urls:
                # Recommend/refine turn: check recall of expected items
                retrieval_total += len(expected_urls)
                hits = len(expected_urls & actual_urls)
                retrieval_hits += hits
                check(
                    f"{trace_name} turn {i}: recommends expected shortlist",
                    hits == len(expected_urls),
                    f"{hits}/{len(expected_urls)} expected items present "
                    f"(missing: {expected_urls - actual_urls})",
                )
            else:
                # Clarify/compare/decline turn: recs should be empty
                clarify_total += 1
                is_empty = len(actual_recs) == 0
                clarify_correct += int(is_empty)
                check(
                    f"{trace_name} turn {i}: correctly withholds recs (clarify/compare/decline)",
                    is_empty,
                    f"got {len(actual_recs)} recs, expected 0",
                )

            # end_of_conversation
            end_total += 1
            actual_end = bool(resp.get("end_of_conversation", False))
            end_correct += int(actual_end == turn["expect_end"])
            check(
                f"{trace_name} turn {i}: end_of_conversation flag correct",
                actual_end == turn["expect_end"],
                f"expected {turn['expect_end']}, got {actual_end}",
            )

# ── 3. Aggregate metrics ──────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("Aggregate metrics")
print("=" * 70)
if api_up:
    def pct(a, b):
        return f"{(100 * a / b):.1f}%" if b else "n/a"

    print(f"Retrieval recall (expected items actually returned): {pct(retrieval_hits, retrieval_total)} "
          f"({retrieval_hits}/{retrieval_total})")
    print(f"Groundedness (returned recs that exist in catalog):  {pct(groundedness_hits, groundedness_total)} "
          f"({groundedness_hits}/{groundedness_total})")
    print(f"Clarify/compare/decline correctness (recs=[]):        {pct(clarify_correct, clarify_total)} "
          f"({clarify_correct}/{clarify_total})")
    print(f"end_of_conversation accuracy:                         {pct(end_correct, end_total)} "
          f"({end_correct}/{end_total})")
else:
    print("(skipped - API was unreachable)")

print(f"\n{'=' * 70}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL CHECKS PASSED")
else:
    print("WARNING: Some checks failed - see above")
print("=" * 70)

sys.exit(1 if failed else 0)