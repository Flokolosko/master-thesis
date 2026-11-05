# Literature inventory

Search conducted 2026-11-05 (web search: Google Scholar/Semantic Scholar/arXiv/publisher pages via
search engine; PubMed/PMC for clinical items). Detailed per-method entries — definition, requirements,
computation, complexity, strengths, limitations, applicability, timing, validation evidence,
recommendation and category — are in:

| File | Scope | Methods | Verified refs |
|---|---|---|---|
| [inventory/inventory_markov_suitability.md](inventory/inventory_markov_suitability.md) | A suitability & temporal abstraction, B Markov/state sufficiency | 19 (M1–M19) | ~40 |
| [inventory/inventory_overlap_coverage_ope.md](inventory/inventory_overlap_coverage_ope.md) | C overlap/positivity, D coverage, G OPE reliability | 24 (C1–C8, D1–D8, G1–G8) | 37 |
| [inventory/inventory_temporal_confounding.md](inventory/inventory_temporal_confounding.md) | E temporal data quality, F confounding/identifiability | 23 (E1–E11, F1–F12) | ~40 |

**Verification status.** "Verified" means existence (title, authors, venue, year; DOI/arXiv where given)
was confirmed by web search. Most entries were read at abstract level only; each inventory states the
reading depth per item. Items under "Unverified leads" in those files must be checked before citing.
Known items to re-check: Weiskopf & Weng 2013 DOI; Austin 2009 volume/pages; the ISPOR-SMDM issue number;
full author lists of Allen et al. 2024 and Eleveld et al. 2018.

## Category overview (counts across the three inventories)
1 = established, directly applicable · 2 = established, needs adaptation · 3 = research candidate · 4 = unsuitable as pre-training metric.
See `metric_comparison.md` for which were selected and why.

## Key references used by the implementation

| Topic | Reference | Identifier | Used for |
|---|---|---|---|
| Healthcare RL guidelines | Gottesman et al. (2019) *Nat Med* 25:16–18 | doi:10.1038/s41591-018-0310-5 | qualitative baseline |
| OPE in observational health data | Gottesman et al. (2018) | arXiv:1805.12298 | ESS / non-zero weights in sepsis |
| Formulation sensitivity | Luo et al. (2024) ICML, PMLR 235 | arXiv:2405.18556 | motivation |
| Markov test | Shi et al. (2020) ICML, PMLR 119:8807–8817 | arXiv:2002.01751 | open item (FBL test) |
| CI test | Shah & Peters (2020) *Ann Stat* 48(3) | doi:10.1214/19-AOS1857 | GCM test; limits of CI testing |
| CMI estimation | Kraskov et al. (2004) *PRE* 69:066138; Runge (2018) AISTATS; Mukherjee et al. (2019) UAI | doi:10.1103/PhysRevE.69.066138; arXiv:1709.01447; arXiv:1906.01824 | alternatives to Gaussian CMI |
| Approximate information state | Subramanian et al. (2022) *JMLR* 23(12) | arXiv:2010.08843 | interpretation of history gain |
| State representations in sepsis | Killian et al. (2020) ML4H, PMLR 136 | arXiv:2011.11235 | history-aware representations |
| Multi-state Markov tests | Titman & Putter (2022) *Biostatistics* 23(2) | doi:10.1093/biostatistics/kxaa030 | candidate extension |
| Time discretisation | Tallec et al. (2019) ICML; Fatemi et al. (2022) CHIL; Sun & Tang (2025, preprint) | arXiv:1901.09732; arXiv:2203.09365; arXiv:2511.20913 | Δt sensitivity, semi-MDP |
| Positivity | Petersen et al. (2012) *SMMR* 21(1) | doi:10.1177/0962280210386207 | support diagnostics |
| High-dimensional overlap | D'Amour et al. (2021) *J Econometrics* 221(2) | doi:10.1016/j.jeconom.2019.10.014 | limitation of overlap in rich states |
| Behaviour-policy calibration | Raghu et al. (2018) ICML CausalML workshop | arXiv:1807.01066 | calibration gate |
| Importance sampling | Kish (1965); Vehtari et al. (2024) *JMLR* 25(72) | arXiv:1507.02646 | ESS, Pareto-k̂ |
| Concentrability | Munos & Szepesvári (2008) *JMLR* 9; Chen & Jiang (2019) ICML; Rashidinejad et al. (2021) | arXiv:1905.00360; arXiv:2103.12021 | coverage theory |
| OPE benchmarks | Voloshin et al. (2021); Tang & Wiens (2021) MLHC; Hargrave et al. (2024) EpiCare | arXiv:1911.06854; arXiv:2107.11003 | validation context |
| EHR data quality | Kahn et al. (2016) *eGEMs* 4(1):1244 | doi:10.13063/2327-9214.1244 | temporal indicator frame |
| Informative observation | Agniel et al. (2018) *BMJ* 361:k1479; Lipton et al. (2016); Che et al. (2018) | doi:10.1136/bmj.k1479; arXiv:1606.04130; doi:10.1038/s41598-018-24271-9 | informative missingness |
| Identification of DTRs | Robins (1986); Murphy (2003) *JRSS-B*; Hernán & Robins (2020) | doi:10.1111/1467-9868.00389 | untestable assumptions |
| Confounding-robust OPE | Namkoong et al. (2020) NeurIPS; Kallus & Zhou (2020) NeurIPS | arXiv:2003.05623; arXiv:2002.04518 | recommended sensitivity analyses |
| Sensitivity analysis | VanderWeele & Ding (2017) *Ann Intern Med*; Tan (2006) *JASA* | doi:10.7326/M16-2607; doi:10.1198/016214506000000023 | recommended next steps |
| Negative controls | Lipsitch et al. (2010) *Epidemiology* 21(3) | doi:10.1097/EDE.0b013e3181d61eeb | recommended next steps |
| Simulation methodology | Morris et al. (2019) *Stat Med* 38(11) | doi:10.1002/sim.8086 | ADEMP |
| PK/PD models | Schnider et al. (1998) *Anesthesiology* 88(5); Eleveld et al. (2018) *BJA* 120(5); Aubouin-Pairault et al. (2023) *JOSS* 8(88) | doi:10.1097/00000542-199805000-00006; doi:10.1016/j.bja.2018.01.018; doi:10.21105/joss.05480 | synthetic anaesthesia environment |

## Notes on items from the earlier idea notes
- **FCAF-RL**: not found in any source; not used.
- **BOReL / "MDP ambiguity"** (Dorfman et al. 2021): concerns task inference in offline *meta*-RL; cite only as an analogy.
- **Bellman–Wasserstein distance** (arXiv:2510.12638): exists, measures behaviour-policy quality vs random, needs a trained critic; excluded.
- **"Evidence Ladder"** (arXiv:2609.23374): recent preprint; related work only.
