# Compliance and claims review (Phase 125)

Generated 2026-09-12T14:10:16+00:00 by `python scripts/50_compliance_review.py`.

**Result: PASS**

| Check | T125 | Findings | Scope |
|---|---|---|---|
| language | T125.1 | 0 | 401 files, 108411 lines, 7 rules; each rule carries its own allowances, matched over a three-line window so a denial split by a line wrap still counts; code files additionally allow the corpus's own diagnosis field and track names |
| disclaimer | T125.2 | 0 | 8 of 8 declared surfaces carry it (literally or by reference to the canonical constant), 15 of 15 built dashboard pages, and 2 canonical constants verified word for word |
| perfection | T125.3 | 0 | 20 result tables scanned at threshold 0.98; 0 unexplained, 1 explained in src/reporting/compliance.py |
| objectives | T125.4 | 0 | 6 of 6 objectives exported, compared character for character |
| counts | T125.5 | 0 | 7 published counts and 3 required discrepancy notes checked against the audit |
| generated | T125.6 | 0 | 35 of 37 published tables re-derived cell by cell from their own CSV (8626 numeric cells); every sidecar checked for a command and live sources; 2 narrative document(s) checked for a stated source instead |

## language — T125.1 -- screening wording only, no diagnostic language

No findings.

## disclaimer — T125.2 -- the disclaimer reaches every surface

No findings.

## perfection — T125.3 -- no unexplained near-perfect metric

No findings.

## objectives — T125.4 -- the six locked objectives, verbatim

No findings.

## counts — T125.5 -- every documented count matches audited reality

No findings.

## generated — T125.6 -- no result was hand-entered

No findings.
