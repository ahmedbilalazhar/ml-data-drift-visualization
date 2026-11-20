I reviewed the critique document you attached and combined it with the Consensus-verified literature. The critique changed some important decisions:

- The topic is still strong, but the **original claim "visualization is underdeveloped" is too broad** because prior systems already exist (DriftVis, ConceptExplorer, DriftLens).
- The stronger research gap is now:

  > Existing work has detectors, localization methods, visualization prototypes, and monitoring systems separately, but fewer open/reproducible studies combine detection + quantitative feature localization + model degradation analysis + one visual monitoring layer.

- Strictly getting **8–10 Q1 journal papers from only 2025/2026 is risky**. The critique estimates only around 5–7 strict papers and recommends asking the instructor whether surveys, conferences, or 2024 papers are acceptable.

Therefore, my final decision is:

# Final Project Direction

## Title:

**Explainable Visualization and Monitoring of Machine Learning Data Drift in Dynamic Environments**

Do NOT use only "ML Data Drift Visualization". This title gives you more research depth:

- Drift detection
- Drift localization
- Explainability
- ML monitoring

---

# Ranked Final Paper List (15 Papers)

Ranking criteria:

1. Fit with your exact project
2. Research quality
3. Accessibility
4. Usefulness for Phase 1 table
5. AI Engineering relevance

---

# Tier 1 — Must Include (Core Papers)

## Rank 1 ⭐⭐⭐⭐⭐

## 1. Greco et al. (2025)

**"DriftLens: ..."**
Venue: IEEE Transactions on Knowledge and Data Engineering (TKDE)

Why ranked #1:

- Highest relevance.
- Directly combines drift monitoring + visualization + explanation.
- Open code available.
- Strong AI engineering connection.

Verified:

- IEEE TKDE 37(10):6232–6245
- DOI available
- arXiv version available
- Code repository available.

Use for:

- Main related work
- Visualization comparison
- Research gap

---

## Rank 2 ⭐⭐⭐⭐⭐

## 2. Hinder, Vaquet & Hammer (2024)

**"One or two things we know about concept drift — Part B: locating and explaining concept drift"**

Venue:
Frontiers in Artificial Intelligence

Why:

- Best paper for the "explainable drift" part.

Contribution:

- Drift localization
- Drift explanation
- Human interpretation

Consensus also verified that the paper focuses on locating and explaining drift to enable human operators to understand changes.

Use for:

- Literature foundation
- Research gap

---

## Rank 3 ⭐⭐⭐⭐⭐

## 3. Lukats et al. (2024)

**"A benchmark and survey of fully unsupervised concept drift detectors on real-world data streams"**

Venue:
International Journal of Data Science and Analytics

Why:

- Strong experimental reference.
- Real datasets.
- Evaluation metrics.

Contribution:

- Compares 10 unsupervised detectors.
- Tests 7 algorithms on 11 real-world streams.

Use for:

- Methodology
- Metrics

---

## Rank 4 ⭐⭐⭐⭐⭐

## 4. Szűcs & Németh (2025)

Venue:
Knowledge and Information Systems (KAIS)

Why:

- Recent experimental paper.
- Verified metadata.
- Q1 in Information Systems category.

Important:
Do not simply say "Q1"; specify category.

---

## Rank 5 ⭐⭐⭐⭐⭐

## 5. Guerrero Cano, Aguiar & Cano (2026)

Venue:
Machine Learning

Why:

- Very recent.
- Drift adaptation focus.
- Useful for model degradation connection.

Verified metadata.

---

# Tier 2 — Strong Supporting Papers

## Rank 6 ⭐⭐⭐⭐☆

## 6. Tran, Le-Khac & Bertolotto (2025)

**"Concept drift detection in image data stream: a survey on current literature, limitations and future directions"**

Venue:
Artificial Intelligence Review

Why:

- Excellent survey.
- Helps identify future research gaps.

Limitation:
Survey, not experimental.

Use:
Background only.

Verified:
Survey of 14 image drift methods.

---

## Rank 7 ⭐⭐⭐⭐☆

## 7. Hinder et al. (2023)

**"Model Based Explanations of Concept Drift"**

Venue:
Neurocomputing

Why:

- Explains drift rather than only detecting it.

Use:
Explainability section.

Correction:
Author list includes Brinkrolf.

---

## Rank 8 ⭐⭐⭐⭐☆

## 8. Bayram, Ahmed & Kassler (2022)

**"From Concept Drift to Model Degradation: An Overview on Performance-Aware Drift Detectors"**

Venue:
Knowledge-Based Systems

Why:

- Very highly cited.
- Connects drift → model failure.

Use:
Motivation.

---

## Rank 9 ⭐⭐⭐⭐☆

## 9. Gâlmeanu & Andonie (2025)

Venue:
Information Visualization

Why:

- Closest visualization-focused paper.

Important:
Not Q1.

SCImago classification:
Q3.

Still valuable because it directly matches visualization.

---

## Rank 10 ⭐⭐⭐⭐☆

## 10. Greco-related DriftLens extension / author materials

Use only if needed:

- More technical details
- Visualization implementation

---

# Tier 3 — Backup Papers (Use if access problems happen)

## Rank 11

## 11. Assis & de Souza (2025)

**"ADWIN-U: adaptive windowing for unsupervised drift detection on data streams"**

Venue:
KAIS

Why:

- Recent.
- Experimental.
- Good detector comparison.

Status:
Needs final verification.

---

## Rank 12

## 12. da Silva et al. (2025)

**"An unsupervised noise-resistant method for detection of incremental drifts"**

Venue:
KAIS

Why:

- Recent.
- Relevant drift detection method.

Needs verification.

---

## Rank 13

## 13. Yang et al. (2020)

**"Diagnosing Concept Drift with Visual Analytics" (DriftVis)**

Venue:
IEEE VAST

Why:

- Foundational visualization work.

Important:
Do not ignore old papers because visualization prior art exists.

---

## Rank 14

## 14. Wang et al. (2020)

**"ConceptExplorer"**

Venue:
IEEE VAST

Why:

- Visual analysis of drift in multi-source time series.

Use:
Visualization comparison.

---

## Rank 15

## 15. Souza et al. (2020)

Venue:
Data Mining and Knowledge Discovery

Why:

- Important methodological paper.

Use:
Supports argument that natural datasets often lack ground-truth drift points.

---

# Final 3 Papers I Would Build The Project Around

If I were making the decision for you:

# 🥇 Final Choice 1: Greco et al. 2025 — DriftLens

Reason:

- Closest to your final system.
- Modern.
- Visualization + explanation.
- Strong AI engineering relevance.

Role:
Main comparison paper.

---

# 🥈 Final Choice 2: Hinder et al. 2024 — Locating and Explaining Concept Drift

Reason:

- Gives theoretical foundation.
- Helps justify your research gap.

Role:
Research background.

---

# 🥉 Final Choice 3: Lukats et al. 2024 — Benchmark of Unsupervised Drift Detectors

Reason:

- Gives experimental methodology.
- Gives evaluation ideas.

Role:
Your methodology design.

---

# Extra 3 Backup Papers (because access problems happen)

Keep these ready:

1. Szűcs & Németh 2025 (KAIS)
2. Guerrero Cano et al. 2026 (Machine Learning)
3. Tran et al. 2025 (Artificial Intelligence Review)

---

# My Final Literature Strategy for Phase 1

I would NOT try to force 10 "Q1 2025/2026 only" papers.

I would build:

| Category                   | Papers |
| -------------------------- | ------ |
| Recent experimental papers | 5–6    |
| Recent surveys             | 2      |
| Visualization foundations  | 2      |
| Methodology references     | 2      |

This will produce a much stronger literature review.

The final project direction I recommend:

**"Interactive Visual Analytics for Detecting, Localizing, and Explaining Machine Learning Data Drift in Dynamic Environments"**

This is the strongest balance between:

- getting marks in DAV,
- satisfying Phase 1,
- building an AI Engineering portfolio,
- and actually being implementable.
