# FDEWS — Product Requirements Document (PRD)

**Version:** 1.0
**Status:** Final
**Last Updated:** July 2026
**Author:** Subrat Kumar Jena

---

## 1. Document Overview

This PRD translates the Business Problem into product-level requirements. It defines what the platform must do, for whom, and within what boundaries — without prescribing implementation detail. Technical performance targets (e.g., prediction quality, latency, explainability depth) are intentionally deferred to the Functional Specification.

## 2. Product Goals

- Enable systematic, scalable monitoring of financial distress signals across publicly listed companies
- Combine structured financial data and disclosure narrative into a single risk assessment
- Present risk findings in a way that supports — rather than replaces — analyst judgment
- Provide explainable risk insights so users understand *why* a company was flagged, not just *that* it was flagged

## 3. Target Users

*(Consistent with Business Problem — Section 6)*

| Stakeholder | Interest in the Product |
|---|---|
| Credit / Risk Analysts | Need scalable early-warning signals across a monitored company list |
| Portfolio / Investment Teams | Need to identify deteriorating holdings before losses materialize |
| Lenders | Need forward-looking risk indicators to inform lending and covenant decisions |
| Regulators / Auditors | Have an interest in early, systemic identification of financial distress patterns |

## 4. Scope

*(Consistent with Business Problem — Section 9)*

**In Scope**
- Analysis of publicly available SEC financial disclosures (10-K, 10-Q)
- Structured financial ratio evaluation
- Qualitative disclosure text analysis (MD&A sentiment)
- Distress risk scoring at the individual company level
- Explainable risk insights to support analyst review and decision-making

**Out of Scope**
- Private (non-SEC-filed) company analysis
- Real-time market/trading data integration
- Investment or trading decision automation
- Regulatory/compliance filing obligations
- Coverage of non-US regulatory disclosure regimes

## 5. Functional Requirements

*Requirements describe product capabilities available to the user — not internal processing steps. Each requirement is implementation-independent: it would remain true regardless of which underlying model or technique powers it.*

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | The product shall provide a distress risk assessment for each covered company, based on its public financial disclosures | Must Have |
| FR-02 | The product shall present the key factors contributing to a company's risk assessment in an understandable format | Must Have |
| FR-03 | The product shall incorporate both quantitative financial data and qualitative disclosure narrative into the risk assessment shown to the user | Must Have |
| FR-04 | The product shall allow analysts to search for and select a company to view its risk assessment | Must Have |
| FR-05 | The product shall allow analysts to trace a risk assessment back to the underlying source disclosure | Should Have |
| FR-06 | The product shall present risk assessments through an accessible, analyst-facing interface | Must Have |

*Note: Multi-company comparison was considered but is not part of the current delivered dashboard. It is documented under Future Scope rather than included here, to keep this PRD accurate to what exists today.*

## 6. Success Criteria

*(Business-level success — technical model performance is addressed separately in the Functional Specification)*

- Analysts can obtain a company's risk assessment without requiring data science expertise to interpret it
- Risk assessments include explainable contributing factors that support analyst confidence in the output
- The product reduces reliance on fully manual review, supporting monitoring across a larger set of companies than manual review alone would allow
- The workflow integrates structured and qualitative disclosure signals into a single, coherent assessment rather than requiring analysts to reconcile separate tools

## 7. Non-Functional Requirements (Business-Level)

*(Qualitative product expectations — measurable targets reserved for Functional Specification)*

| Requirement | Description |
|---|---|
| Explainability | Every risk assessment must be accompanied by understandable reasoning, not a bare score |
| Usability | The interface must be usable by an analyst without requiring data science expertise |
| Scalability | The system must be capable of processing disclosures across a large number of companies, not a single case at a time |
| Data Integrity | Risk assessments must be traceable back to the source disclosure data used to generate them |
| Reliability | The system should behave predictably and consistently across repeated runs on the same input |

## 8. Assumptions & Constraints

- The system relies on the availability and consistency of public SEC disclosure data
- Disclosure quality and completeness vary by company, which may affect assessment depth
- The product is designed for analyst-assisted decision-making, not autonomous decision-making

## 9. Out-of-Scope Clarifications

- The product does not issue investment recommendations or buy/sell signals
- The product provides risk assessments to support human decision-making and should be used alongside professional financial analysis
- The product does not process non-public or confidential financial information
