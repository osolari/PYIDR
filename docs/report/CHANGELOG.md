# CHANGELOG - Phase Three Technical Development Pass

## Summary

- Total changes applied by category: A technical/error corrections: 20; B theoretical development: 4; C rigor/completeness: 10; D clarity/organization: 5; E planned/placeholder clarification: 10; F new planned theory/method/experiment/handoff additions: 13; G substantive positioning/citation-verification additions: 2; H LaTeX/build/minor fixes: 16.
- Technical-error fixes applied: claim-strength qualifications; corrected finite/infinite identifiability scope; corrected PY truncation-tail claim; corrected Bayes-mFDR threshold and denominator handling; separated Harris from geometric ergodicity; qualified variational consistency; harmonized S5/K settings; corrected atom/support counts and correlation parameterization.
- Theoretical-development changes applied: sharpened assumptions, finite-sieve identifiability theorem, conditional contraction theorem, theory-scope limitation for heavy-tail atoms, and theory-to-experiment bridge.
- Methodology-strengthening changes applied: clarified rank/marginal pipeline, atom parameter domains, PLOD-compatible correlation parameterization, PY predictive reassignment probabilities, valid correlation updates, diagnostics, VI truncation, and operational Bayes-mFDR computation.
- Experiment-plan changes applied: added ablations, stress tests, Monte Carlo uncertainty reporting, preprocessing/QC requirements, reproducibility artifacts, and S5-sparse stress-test distinction.
- Planned experiments preserved or clarified: S1--S5 simulations, real-data applications, runtime analysis, contraction diagnostics, PY-versus-DP behavior, and sensitivity analyses were preserved and locally clarified as planned.
- Expected or projected results preserved or clarified: all expected-behavior language was retained but revised to avoid implying observed results.
- Placeholder figures and tables preserved, clarified, or added: all placeholder figures and tables were preserved; figure titles/captions and table captions now identify planned or illustrative status; S5-sparse planned row added to the simulation-design table.
- Implementation-plan material preserved or clarified: MCMC, VI, figure generation, data preprocessing, reproducibility, and coding-agent tasks were expanded.
- Coding-agent-relevant tasks added or clarified: model implementation, sampler, VI, simulations, real-data preprocessing, figure/table generation, diagnostics, citation verification, and build tasks were documented in `CODING_AGENT_HANDOFF.md`.
- Proposed changes not applied: none from the approved Phase Two plan were intentionally rejected; some citation/software items remain as handoff verification tasks rather than unsupported manuscript claims.
- New package, macro, auxiliary file, figure file, table file, bibliography entry, placeholder, proposed experiment, theoretical result, methodology addition, or implementation task added: no new LaTeX package was added; a table column macro `Y` was added; no new bibliography entry was added, but three bibliography entries were updated; no new auxiliary source file was added beyond this changelog and `CODING_AGENT_HANDOFF.md`; placeholder figures were regenerated; S5-sparse was added as a planned stress-test row; finite-sieve and conditional theorem refinements were added; multiple implementation and experiment tasks were added in the handoff.

## Applied changes in document order

### Project/build/deliverables

- Build-H1 | `math_commands.tex` | H | Replaced warning-prone definition marker with `\coloneqq`. | Removes avoidable math font warnings while preserving meaning.
- Build-H2 | `main.tex`, table files | H | Added a wrapped table column macro and reformatted wide tables using `tabularx`/`resizebox`. | Prevents clipped table content and overfull boxes.
- Build-H3 | Section 3 and Appendix displays | H | Broke long theoretical and KL displays into multi-line forms. | Improves build quality and PDF readability.
- Build-H4 | `README.md` | H | Updated root-file, BibTeX, figure-generation, and manuscript-status notes. | Improves Overleaf and coding-agent readiness.
- Deliverable-H1 | `CHANGELOG.md` | H | Replaced the prior changelog with this identifier-keyed changelog. | Documents the current approved pass.
- Deliverable-F1 | `CODING_AGENT_HANDOFF.md` | F | Added a standalone coding-agent handoff file. | Supports the next implementation and experiment iteration.

### Abstract

- Abs-A1 | Abstract | A | Qualified identifiability, contraction, mFDR, VI, and geometric-ergodicity claims. | Prevents overclaiming relative to the current proof status.
- Abs-E1 | Abstract | E | Clarified that figures and result tables are illustrative/placeholders. | Preserves planned material without implying completed experiments.
- Abs-D1 | Abstract | D | Reordered and tightened model, inference, theory, and protocol summary. | Improves readability and publication-development readiness.

### Introduction

- 1-A1 | Section 1.1 | A | Replaced unsupported universal pairwise-FDR wording with calibrated-risk language. | Avoids unsupported empirical claims.
- 1-C1 | Section 1.2 | C | Recast DP/PY cluster comparison as order-based rather than exact. | Improves stochastic-process precision.
- 1-D1 | Section 1.3 | D | Converted dense contribution prose into structured contribution bullets. | Improves reviewability.
- 1-E1 | Section 1.3 | E | Clarified empirical evaluations as planned. | Avoids fabricated-completion implications.
- 1-F1 | Section 1.3 | F | Added design map connecting IDR limitations to PY-IDR components. | Strengthens methodological motivation.
- 1-G1 | Section 1.3 | G | Replaced unsupported novelty absolutes with verification-aware positioning. | Reduces citation/novelty risk.

### Related work

- 2-H1 | Table 1 | H | Reformatted the related-work table to fit within text width. | Improves layout and compilation quality.
- 2-A1 | Related work | A | Retained nestedIDR but moved implementation availability to verification status. | Avoids unsupported software-availability claims.
- 2-A2 | Related work and bibliography | A | Updated eCV citation metadata to package/source-verification status. | Improves citation accuracy without inventing article metadata.
- 2-A3 | Related work and bibliography | A | Updated BNP Archimedean citation to the 2025 version of record. | Improves bibliographic accuracy.
- 2-D1 | Section 2 | D | Clarified direct baselines, methodological neighbors, and complementary methods. | Improves organization.
- 2-G1 | Handoff | G | Added recent/software-dependent citation checks as handoff tasks. | Preserves citation integrity.

### Method: model and notation

- 3-C1 | Table 2 | C | Expanded notation for active clusters, PY reassignment, basis indicators, and thresholds. | Reduces notation ambiguity.
- 3-A1 | Section 3.2.1 | A | Clarified empirical ranks as initialization/diagnostics or approximation, not replacement for modeled marginals. | Resolves rank/marginal inconsistency.
- 3-A2 | Section 3.2.2 | A | Added PLOD-compatible constraints for structured correlation matrices. | Aligns model parameterization with assumptions.
- 3-C2 | Section 3.2.2 | C | Added atom parameter domains for Gaussian, t, Clayton, and Gumbel atoms. | Improves implementability.
- 3-C3 | Section 3.2.2 | C | Defined PY stick-breaking variables explicitly. | Makes model self-contained.
- 3-A3 | Section 3.2.5 | A | Corrected structural-class and atomic-type support counts. | Fixes consistency error.

### Method: inference

- 3-A4 | Section 3.3.1 | A | Reframed sampler as PY Polya-urn auxiliary-atom MCMC rather than inconsistent slice/Algorithm-8 hybrid. | Makes sampler description coherent.
- 3-M1 | Section 3.3.1 | F | Added occupied/new-cluster reassignment probabilities under the PY predictive rule. | Makes algorithm implementation-ready.
- 3-A5 | Algorithm 1 | A | Replaced invalid Cholesky-correlation wording with covariance-to-correlation normalization and PLOD rejection. | Prevents invalid correlation matrices.
- 3-C4 | Section 3.3.1 | C | Clarified Bernstein basis-membership likelihood contribution. | Connects marginal model to sampler.
- 3-M2 | Section 3.3.1 | F | Added initialization, ESS/R-hat, label-invariant, NUTS, tie, and cluster diagnostics. | Improves robust implementation.
- 3-H1 | Sections 3.3.1 and 4.9 | H | Harmonized analytic MCMC complexity with runtime discussion. | Removes cross-section inconsistency.
- 3-C5 | Section 3.3.2 | C | Clarified finite truncation, variational factors, minibatch scaling, and approximation status. | Improves VI rigor.

### Method: idr and Bayes-mFDR

- 3-A6 | Section 3.3.3 | A | Added largest-threshold/maximum-rejection formulation and zero-denominator convention. | Aligns with Sun-Cai-style step-up logic.
- 3-C6 | Section 3.3.3 | C | Added operational sorted-running-average rejection rule. | Improves method-to-code traceability.

### Theory

- 3-T1 | Section 3.4 assumptions | B | Split assumptions into smooth truth, mass/separation, KL support, sieve/tail, threshold regularity, and sampler regularity. | Makes theorem dependencies explicit.
- 3-A7 | Theorem 3.1 | A | Replaced unsupported diagonal recovery with finite-sieve linear-independence identifiability. | Fixes proof gap.
- 3-T2 | Theorem 3.1 | B | Separated finite-sieve identifiability from unrestricted infinite-PY extensions. | Prevents overextension of finite-mixture theory.
- 3-A8 | Theorem 3.2 | A | Replaced invalid positive-discount exponential tail implication with explicit sieve-tail assumption/proof obligation. | Corrects central contraction issue.
- 3-A9 | Theorem 3.2 | A | Corrected entropy reference to the sieve-covering lemma. | Improves cross-reference accuracy.
- 3-T3 | Theorem 3.2 | B | Added bounded-smooth theory-scope remark for heavy-tail atoms. | Aligns theory with methodology.
- 3-A10 | Theorem 3.2 and Appendix Lemma A.2 | A | Revised KL decomposition to state required transform/support conditions. | Repairs proof dependency.
- 3-A11 | Theorem 3.3 | A | Added threshold regularity and denominator conditions; qualified rate. | Avoids overstating mFDR transfer.
- 3-A12 | Theorem 3.4 | A | Split Harris ergodicity from conditional geometric ergodicity. | Fixes logical inconsistency.
- 3-A13 | Theorem 3.5 | A | Recast VI consistency as finite-truncation conditional statement. | Avoids overstating variational theory.
- 3-B1 | End of Section 3.4 | B | Added theory-to-experiment bridge. | Connects theory to planned evaluation.

### Experiments

- 4-E1 | Section 4 opening | E | Preserved protocol-first framing and propagated local status language. | Maintains planned-material clarity.
- 4-A1 | Simulation tables | A | Changed placeholder table setting from unsupported K=4 to planned K=5. | Fixes design inconsistency.
- 4-A2 | S5 design/ROC text | A | Added S5-sparse planned stress-test variant for pi*=0.10. | Harmonizes S5 descriptions.
- 4-C1 | Simulation protocol | C | Added seed ledger, truth-object storage, rank/tie handling, and stored outputs. | Improves reproducibility.
- 4-F1 | Section 4.6 | F | Added planned ablations for PY vs DP, heavy-tail atoms, learned marginals, structure classes, inference mode, and aggregation. | Strengthens experiment design.
- 4-F2 | Section 4.3 | F | Added Monte Carlo SEs, uncertainty bands, calibration diagnostics, posterior predictive checks, and failure summaries. | Improves statistical rigor.
- 4-E2 | Figures | E | Revised captions to label figures as illustrative synthetic placeholders. | Prevents false empirical interpretation.
- 4-E3 | Result tables | E | Revised table captions to planned-output shells. | Preserves placeholders clearly.
- 4-F3 | Section 4.6 | F | Added stress tests for weak dependence, ties, missing replicates, singularity, high K, and marginal misspecification. | Improves robustness plan.
- 4-C2 | Section 4.7 | C | Added data manifests, feature-universe, replicate matching, score dictionaries, QC, tie, and missingness requirements. | Makes real-data plan implementable.
- 4-A3 | Pan-UKBB text and bibliography | A | Updated Pan-UKBB citation metadata and real-data wording. | Improves source accuracy and status handling.
- 4-E4 | Real-data tables | E | Preserved placeholder tables and clarified planned status. | Avoids false observed results.
- 4-C3 | Runtime section | C | Separated analytic complexity from synthetic plotted values and future measured runtime. | Avoids implied completed timing experiments.
- 4-F4 | Runtime/reproducibility | F | Added reproducibility requirements for seeds, environment, manifests, configs, output schemas, and containers/workflows. | Supports companion implementation.

### Conclusion

- 5-A1 | Conclusion opening | A | Aligned conclusion with qualified theorem status. | Ensures consistency across manuscript.
- 5-E1 | Practical recommendations | E | Marked recommendations as provisional implementation guidance. | Avoids empirical overclaiming.
- 5-C1 | Limitations | C | Added identifiability, tail, bounded-copula, mFDR, geometric-ergodicity, and VI limitations. | Improves scientific rigor.
- 5-D1 | Acknowledgments/code/data | D | Preserved placeholders and future code/data status. | Maintains publication-development readiness.

### Appendix

- App-A-A1 | Appendix opening | A | Harmonized appendix status as supporting lemmas and proof obligations. | Removes conflict with full-proof claim.
- App-A-A2 | Lemma A.2 | A | Revised KL lemma to state transform/support conditions. | Improves correctness.
- App-A-A3 | Lemma A.3 | A | Removed invalid exponential conclusion for positive-discount PY residual. | Fixes mathematical error.
- App-A-H1 | Lemma A.4 | H | Corrected support-count wording. | Fixes consistency.
- App-A-F1 | Appendix remark | F | Added theory-status roadmap. | Documents established vs future theoretical work.
- App-A-F2 | Appendix remark and handoff | F | Added theory-to-code diagnostics. | Connects assumptions to implementation checks.

### Figures and script

- Fig-H1 | `figures/generate_figures.py` | H | Changed hard-coded absolute output paths to project-local paths. | Makes figure regeneration portable.
- Fig-E1 | Figure PDFs/script | E | Removed internal hard-coded figure numbers from generated titles and regenerated PDFs. | Avoids LaTeX numbering conflicts.
- Fig-E2 | Figure captions/script | E | Preserved synthetic/illustrative labels in captions and internal figure text. | Prevents placeholder values from being treated as observed.

### Bibliography

- Bib-A1 | `refs.bib` | A | Updated eCV, BNP Archimedean copula, and Pan-UKBB metadata where verified; left remaining items for handoff verification. | Improves citation accuracy.
- Bib-F1 | Handoff | F | Added citation-verification tasks rather than unsupported new references. | Preserves citation integrity.
- Bib-H1 | `refs.bib` | H | Applied safe formatting/metadata normalization only where supported. | Improves bibliography polish.

### Coding-agent handoff

- CA-F1 | `CODING_AGENT_HANDOFF.md` | F | Added implementation-ready task plan for model, sampler, VI, simulations, real-data, figures, diagnostics, and reproducibility. | Enables next agent iteration.
- CA-F2 | `CODING_AGENT_HANDOFF.md` | F | Added projected/expected-results constraints and replacement protocol. | Prevents accidental fabrication.
- CA-H1 | `CHANGELOG.md`, `CODING_AGENT_HANDOFF.md` | H | Added build status, changed-file summary, and known caveats. | Supports final delivery and future handoff.
