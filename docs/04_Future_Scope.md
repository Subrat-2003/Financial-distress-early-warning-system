# FDEWS — Future Scope

**Version:** 1.0
**Status:** Final
**Last Updated:** July 2026
**Author:** Subrat Kumar Jena

## 1. Document Scope

This document describes potential future improvements to FDEWS, distinguished from current functionality. It does not describe what the system does today — that is covered in the Functional Specification. Every item here is grounded in a limitation already disclosed in the repository or the Functional Specification; no new capability is proposed without a traceable basis.

## 2. Referenced Documents

| Document | Relationship |
|---|---|
| `01_Business_Problem.md` | Establishes the business rationale future enhancements would extend |
| `02_PRD.md` | Defines current product requirements this scope builds beyond |
| `03_Functional_Specification.md` | Source of current implementation status and disclosed limitations |

## 3. Current Known Limitations (Repository-Verified)

*(Basis for all proposed enhancements below — not new claims)*

| Limitation | Status |
|---|---|
| Sentiment signal extracted for only a subset of companies; remainder default to neutral | ✅ Implemented (disclosed) |
| LSTM model shows high recall but low precision; not viable for production | 🧪 Experimental |
| Sentiment extraction quality degrades on boilerplate/templated MD&A language | ✅ Implemented (disclosed) |
| Walk-forward validation assumes stationarity of distress patterns; unverified beyond 2020–2024 | ✅ Implemented (disclosed) |
| `persistent_distress_flag` uses a fixed 2-quarter window; longer windows untested | ✅ Implemented (disclosed) |
| General pipeline error handling (malformed filings, corrupted downloads) not implemented | 🔲 Future Scope |

## 4. Proposed Future Enhancements

### 4.1 Data & Sentiment Coverage

| Enhancement | Rationale | Priority | Status |
|---|---|---|---|
| Extend sentiment extraction to the full company set | Currently only a subset has an extracted signal; the remainder default to neutral, understating risk for companies with unprocessed but parseable filings | High | 🔲 Future Scope |
| Improve sentiment extraction robustness on boilerplate/templated disclosure language | Disclosed quality degradation on companies using generic MD&A language | Medium | 🔲 Future Scope |

### 4.2 Model & Evaluation

| Enhancement | Rationale | Priority | Status |
|---|---|---|---|
| Extended temporal validation beyond the 2020–2024 window | Current walk-forward split has not been verified for stationarity in more recent market conditions; directly affects confidence in the production model's ongoing validity | High | 🔲 Future Scope |
| Continued temporal model experimentation (building on existing LSTM benchmarking) | LSTM's current precision is insufficient for production use; further architectural or training changes could be evaluated as an experimental track without affecting the production model | Medium | 🔲 Future Scope |
| Evaluation of longer persistence windows for `persistent_distress_flag` | Only a 2-quarter window has been tested; longer windows are unverified | Low | 🔲 Future Scope |

### 4.3 Pipeline Reliability

| Enhancement | Rationale | Priority | Status |
|---|---|---|---|
| Formal error handling for malformed filings and ingestion failures | Currently unimplemented per the Functional Specification's Error Handling section; a reliability gap in the current pipeline | High | 🔲 Future Scope |
| Extended schema-validation coverage further upstream (closer to raw ingestion) | The Feature Order Lock currently validates only at the inference stage | Medium | 🔲 Future Scope |

## 5. Out of Scope for the Current Version

*(Aligned with PRD Out-of-Scope — not a permanent product decision, but not planned within the current version)*

- Real-time or streaming filing ingestion (current pipeline is batch-oriented)
- Non-SEC or international disclosure regimes (consistent with PRD Out-of-Scope)
- Automated investment decision-making (consistent with PRD Out-of-Scope Clarifications)

## 6. Assumptions

This document reflects reasonable, repository-grounded extensions of currently disclosed limitations. It does not represent a committed roadmap, budget, or timeline — it is intended to demonstrate forward-looking product thinking for a portfolio project. No new functionality is claimed as planned or in progress beyond what is stated here.
