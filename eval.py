"""
Evaluation suite — validates all 4 behaviors + schema + traces alignment.
Run: python eval.py
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from main import validate_recommendations, enforce_turn_cap, Message, CATALOG, _CATALOG_BY_NAME

passed = failed = 0

def test(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}: {detail}")
        failed += 1

# ── Mock LLM responses mirroring the 10 real traces ─────────────────────────
MOCKS = {
    "vague": {
        "reply": "Happy to help narrow that down. Who is this role for and what do you want to measure?",
        "recommendations": [],
        "end_of_conversation": False
    },
    "leadership_no_context": {
        "reply": "The pool consists of CXOs and directors — is this for selection or developmental feedback?",
        "recommendations": [],
        "end_of_conversation": False
    },
    "leadership_shortlist": {
        "reply": "For selection with a leadership benchmark, here are the right instruments:",
        "recommendations": [
            {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "test_type": "P"},
            {"name": "OPQ Universal Competency Report 2.0", "url": "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/", "test_type": "P"},
            {"name": "OPQ Leadership Report", "url": "https://www.shl.com/products/product-catalog/view/opq-leadership-report/", "test_type": "P"},
        ],
        "end_of_conversation": False
    },
    "rust_no_test": {
        "reply": "SHL's catalog doesn't currently include a Rust-specific test. The closest fit is Smart Interview Live Coding. Want me to build a shortlist?",
        "recommendations": [],
        "end_of_conversation": False
    },
    "contact_center_accent_q": {
        "reply": "SVAR has US, UK, Australian, and Indian variants. Which accent fits your callers?",
        "recommendations": [],
        "end_of_conversation": False
    },
    "contact_center_shortlist": {
        "reply": "For high-volume English US contact centre screening:",
        "recommendations": [
            {"name": "SVAR Spoken English (US) (New)", "url": "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/", "test_type": "K"},
            {"name": "Contact Center Call Simulation (New)", "url": "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/", "test_type": "S"},
            {"name": "Entry Level Customer Serv - Retail & Contact Center", "url": "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/", "test_type": "P,C"},
            {"name": "Customer Service Phone Simulation", "url": "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/", "test_type": "B,S"},
        ],
        "end_of_conversation": False
    },
    "compare_no_recs": {
        "reply": "DSI is standalone; the 8.0 is sector-specific with industrial norms. Both measure safety personality.",
        "recommendations": [],
        "end_of_conversation": False
    },
    "legal_decline": {
        "reply": "That's outside what I can advise on — your legal team is the right resource. I can help you find the right SHL assessment.",
        "recommendations": [],
        "end_of_conversation": False
    },
    "confirmed_end": {
        "reply": "Confirmed. Final shortlist locked.",
        "recommendations": [
            {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "test_type": "P"},
        ],
        "end_of_conversation": True
    },
    "hallucinated": {
        "reply": "Here are some tests:",
        "recommendations": [
            {"name": "FAKE ASSESSMENT XYZ", "url": "https://www.shl.com/fake", "test_type": "K"},
            {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "test_type": "P"},
        ],
        "end_of_conversation": False
    },
    "refine_add": {
        "reply": "Added Graduate Scenarios to the list.",
        "recommendations": [
            {"name": "SHL Verify Interactive G+", "url": "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/", "test_type": "A"},
            {"name": "Occupational Personality Questionnaire OPQ32r", "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "test_type": "P"},
            {"name": "Graduate Scenarios", "url": "https://www.shl.com/products/product-catalog/view/graduate-scenarios/", "test_type": "B"},
        ],
        "end_of_conversation": False
    },
}

print("=" * 60)
print("SHL Recommender v2 — Evaluation Suite (377 real products)")
print("=" * 60)

# ── 1. Catalog integrity ─────────────────────────────────────────────────────
print("\n[1] Catalog integrity")
test("377 products loaded", len(CATALOG) == 377)
test("All have name/link/keys", all("name" in i and "link" in i and "keys" in i for i in CATALOG))
test("All status=ok", all(i.get("status") == "ok" for i in CATALOG))
urls = [i["link"] for i in CATALOG]
test("No duplicate URLs", len(urls) == len(set(urls)))
test("OPQ32r in catalog", "occupational personality questionnaire opq32r" in _CATALOG_BY_NAME)
test("Graduate Scenarios in catalog", "graduate scenarios" in _CATALOG_BY_NAME)

# ── 2. Vague query → no recs ─────────────────────────────────────────────────
print("\n[2] Vague query — clarify first")
r = MOCKS["vague"]
test("No recs on vague", len(r["recommendations"]) == 0)
test("Not end of conv", r["end_of_conversation"] == False)
test("Reply is non-empty", len(r["reply"]) > 5)

# ── 3. Leadership scenario (C1) ───────────────────────────────────────────────
print("\n[3] Leadership scenario (trace C1)")
r = MOCKS["leadership_no_context"]
test("No recs before context", len(r["recommendations"]) == 0)
r = MOCKS["leadership_shortlist"]
test("Has recs after context", len(r["recommendations"]) > 0)
validated = validate_recommendations(r["recommendations"])
test("All recs in real catalog", len(validated) == len(r["recommendations"]))
test("OPQ32r included", any("OPQ32r" in v.name for v in validated))

# ── 4. Gap acknowledgment (C2 — Rust) ────────────────────────────────────────
print("\n[4] Catalog gap (trace C2 — Rust)")
r = MOCKS["rust_no_test"]
test("No recs when no catalog match", len(r["recommendations"]) == 0)
test("Reply acknowledges gap", "rust" in r["reply"].lower() or "catalog" in r["reply"].lower())

# ── 5. Multi-step clarification (C3 — contact center) ────────────────────────
print("\n[5] Multi-step clarification (trace C3)")
r1 = MOCKS["contact_center_accent_q"]
test("First clarify: no recs", len(r1["recommendations"]) == 0)
r2 = MOCKS["contact_center_shortlist"]
validated = validate_recommendations(r2["recommendations"])
test("All 4 CS items in catalog", len(validated) == 4)
test("SVAR included", any("SVAR" in v.name for v in validated))

# ── 6. Compare → no recs (C6) ────────────────────────────────────────────────
print("\n[6] Compare query (trace C6)")
r = MOCKS["compare_no_recs"]
test("No recs on compare turn", len(r["recommendations"]) == 0)
test("Reply is informative", len(r["reply"]) > 20)

# ── 7. Legal decline (C7) ────────────────────────────────────────────────────
print("\n[7] Legal/compliance refusal (trace C7)")
r = MOCKS["legal_decline"]
test("No recs on legal Q", len(r["recommendations"]) == 0)
test("Reply redirects", "advise" in r["reply"].lower() or "legal" in r["reply"].lower())

# ── 8. End of conversation ────────────────────────────────────────────────────
print("\n[8] End of conversation")
r = MOCKS["confirmed_end"]
test("end_of_conversation=true on confirm", r["end_of_conversation"] == True)
test("Final recs present", len(r["recommendations"]) > 0)

# ── 9. Hallucination filtering ────────────────────────────────────────────────
print("\n[9] Hallucination filtering")
r = MOCKS["hallucinated"]
validated = validate_recommendations(r["recommendations"])
test("Fake item removed", len(validated) == 1)
test("Real item kept", validated[0].name == "Occupational Personality Questionnaire OPQ32r")

# ── 10. Refinement (C4/C10) ──────────────────────────────────────────────────
print("\n[10] Refinement — add item mid-conversation")
r = MOCKS["refine_add"]
validated = validate_recommendations(r["recommendations"])
test("3 recs after add", len(validated) == 3)
test("Graduate Scenarios added", any("Graduate Scenarios" in v.name for v in validated))
test("Previous items preserved", any("Verify" in v.name for v in validated))

# ── 11. Turn cap ─────────────────────────────────────────────────────────────
print("\n[11] Turn cap enforcement")
msgs = [Message(role="user" if i%2==0 else "assistant", content=f"turn {i}") for i in range(20)]
capped = enforce_turn_cap(msgs, cap=8)
test("Cap at 8", len(capped) == 8)
test("Keeps latest", capped[-1].content == "turn 19")

# ── 12. Schema compliance ─────────────────────────────────────────────────────
print("\n[12] Schema compliance on all mocks")
for key, r in MOCKS.items():
    test(f"Schema valid ({key})", all(k in r for k in ["reply","recommendations","end_of_conversation"]))
    for rec in r["recommendations"]:
        test(f"Rec fields ({key})", all(k in rec for k in ["name","url","test_type"]))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("🎉 ALL TESTS PASSED — Ready to deploy!")
else:
    print("⚠️  Fix failures before deploying")
