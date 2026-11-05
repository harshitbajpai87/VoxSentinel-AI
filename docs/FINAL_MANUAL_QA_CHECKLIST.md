# Final Manual QA Checklist

Automated results live in `FINAL_VALIDATION_REPORT.md`. This is what a human should still click through.

## Landing and navigation
- [ ] Landing page renders with gradient headline and hero cards; **no raw code** visible as text.
- [ ] Sidebar shows exactly the intended items (Overview, Analyze Audio, Live Recording, Speaker
      Verification, Analysis History, Family Protection).
- [ ] No dark-theme surfaces; badges and tables use the light tokens.

## Analyze Audio
- [ ] Clean clip → transcript, category, risk and evidence quotes shown.
- [ ] Consent unchecked → request blocked with a readable message.
- [ ] 0.5 s clip → INCONCLUSIVE quality line, not a scary verdict.
- [ ] Clipped clip → QUALIFIED voice verdict with a plain-English reason.
- [ ] Provider line shows Groq + model; on quota it says rules-only with the reason.

## Family Protection
- [ ] Enrol a reference voice, compare another speaker → similarity and outcome shown.
- [ ] Impostor clip → "not consistent with reference", no certainty claim.

## History and exports
- [ ] History lists runs; detail shows voice, quality, evidence, review status.
- [ ] JSON export downloads and parses.
- [ ] CSV export has a header row and one row per analysis.

## Reasoning review
- [ ] Softening shows quoted evidence from the transcript verbatim.
- [ ] Break the provider key → original stage-one verdict still displayed unchanged.

## Language spot-check (with a Hindi speaker)
- [ ] Hindi ransom script → Devanagari transcript, CRITICAL.
- [ ] Hinglish family call → no scam warning.
- [ ] Hinglish ransom script → expect the known ASR weakness; record it, do not file it as a classifier bug.

## Release hygiene
- [ ] No `.env`, keys, `node_modules` or caches in the ZIP.
- [ ] ZIP opens; `backend/ frontend/ docs/ training/ e2e/` all present and readable.
