# Changelog: Hierarchical Nonparametric Bayesian IDR via Pitman-Yor Copula Mixtures

This changelog documents the substantive corrections applied during proofreading and adaptation to the SAIM journal template. Corrections fall into three categories: critical mathematical errors identified by literature verification, citation upgrades for stronger or more accurate references, and structural changes for the SAIM template adaptation.

## Critical mathematical corrections

### 1. Identifiability strategy: replaced Allman-Matias-Rhodes with Yakowitz-Spragins

The original draft invoked Allman, Matias, and Rhodes (2009, *Annals of Statistics*) for identifiability of the copula mixture component. This was a structural mismatch: the AMR theorem is Kruskal-based and requires conditional independence of the observed variables given the latent class, which Gaussian copula components by definition violate (the variates within each component are correlated through the correlation matrix). The corrected Theorem 3.1 invokes Yakowitz and Spragins (1968, *Annals of Mathematical Statistics*) instead, which only requires linear independence of the joint component densities. Gaussian copula densities with distinct correlation matrices are real-analytic on the open cube, satisfy linear independence, and are therefore identifiable up to label permutation. A new Remark 3.2 documents the reasoning explicitly.

### 2. Posterior contraction: added Sklar regularity caveat and corrected primary citation

The original draft attributed the rate `n^{-beta/(2*beta+K)}(log n)^t` to Wu and Ghosal (2010), which provides only L^1-consistency rather than a rate. The corrected Theorem 3.2 cites Shen, Tokdar, and Ghosal (2013, *Biometrika*) as the canonical multivariate Dirichlet-mixture contraction result, supplemented by Scricciolo (2014, *Bayesian Analysis*) for the Pitman-Yor adaptation specifically. A new supporting lemma (Lemma A.1, Sklar regularity for copula contraction) is added in the appendix to handle the transformation from densities on R^K to copulas on [0,1]^K, which is not automatic and requires regularity assumptions on the marginals.

### 3. Bernstein-Dirichlet contraction: corrected attribution from Ghosal (2001) to Rousseau (2010)

The original draft attributed the adaptive Bernstein contraction rate `n^{-beta/(2*beta+1)}(log n)^t` to Ghosal (2001, *Annals of Statistics*), which provides only the suboptimal non-adaptive rate `n^{-1/3}(log n)^{5/6}`. The corrected Section 3.5.2 cites Rousseau (2010, *Annals of Statistics*) for the adaptive rate, with Petrone (1999) and Petrone-Wasserman (2002) as background and Kruijer-Rousseau-vdV (2010) for general location-scale extension.

### 4. Geometric ergodicity: softened to Harris ergodicity with explicit conjecture

The original draft asserted V-uniform geometric ergodicity of the slice-sampled Polya-urn sampler under Meyn-Tweedie Theorem 16.0.1. This is unproven for slice-sampled Pitman-Yor under Neal Algorithm 8: existing references (Walker 2007, Kalli-Griffin-Walker 2011, Papaspiliopoulos-Roberts 2008) establish correctness of the invariant distribution but not geometric rates. The corrected Theorem 3.4 establishes Harris ergodicity unconditionally via Meyn-Tweedie Theorem 13.0.1, and states geometric ergodicity as a conjecture supported by recent quantitative-mixing analyses for Dirichlet-process samplers (Ascolani et al. 2024, arXiv:2304.01731). A new Remark 3.6 documents this separation.

### 5. mFDR control: added Heller-Rosset and Castillo-Roquain rate bridge

The original draft suggested that posterior contraction at rate epsilon_n implies asymptotic mFDR control at the same rate via Sun-Cai (2007, JASA). Sun-Cai establishes asymptotic mFDR optimality of the oracle compound-decision rule and asymptotic mFDR control of consistent plug-ins as the number of hypotheses tends to infinity, but does not provide a result that converts a posterior contraction rate into an mFDR rate. The corrected Theorem 3.3 invokes Heller and Rosset (2021, JRSSB) and Castillo and Roquain (2020, Annals of Statistics) for the explicit rate-to-mFDR bridge.

### 6. Variational consistency: qualified the scope of Wang-Blei (2019)

The original draft cited Wang and Blei (2019, JASA) as the basis for variational consistency in the nonparametric setting. Their Theorem 5 is a parametric variational Bernstein-von Mises result requiring finite-dimensional theta in a compact parameter space. The corrected Theorem 3.5 invokes Wang-Blei for finite truncation only, and cites Alquier and Ridgway (2020, Annals of Statistics) and Yang-Pati-Bhattacharya (2020, Annals of Statistics) for the nonparametric extension.

## Citation upgrades and additions

### Most relevant new references identified

The closest published competitor is **Pan, Nieto-Barajas, and Craiu (2024, arXiv:2412.09539)**, which develops a Bayesian nonparametric mixture of Archimedean copulas with Pitman-Yor mixing. This is now cited prominently in the introduction and related work, and added to the comparator list in Table 2. The most direct hierarchical-IDR competitor is **Ranalli, Lyu, Koch, and Li (2026, *Statistics in Medicine*)**, the nestedIDR framework, which is now treated as the principal hierarchical baseline.

The expanded literature review now covers the modern (2020-2026) landscape across four themes: IDR variants and reproducibility methods (eCV, nestedIDR, MaRR, IDR2D, ChIP-R, GMCM-AD, NPMLE replicability), Bayesian nonparametric copula models (Pan-Nieto-Barajas-Craiu, Burda-Prokhorov, Zhuang-Diao-Yi, Liu-Li, Feldman-Kowal, Dalla Valle et al., Smith-Loaiza-Maya-Nott, Murray et al., Smith-Khaled), Pitman-Yor mixture theory (Pitman-Yor 1997, Ishwaran-James, Favaro-Teh, Walker, Kalli-Griffin-Walker, Papaspiliopoulos-Roberts, Ascolani et al., Bystrova et al., Shen-Tokdar-Ghosal, Canale-De Blasi, Scricciolo, Rousseau, Kruijer-Rousseau-vdV), and modern multiple-testing methodology (Sun-Cai, Cai-Sun-Wang, Heller-Rosset, Castillo-Roquain, Bogomolov-Heller, Wang-Owen, Wang-Ramdas, Lei-Fithian, Ignatiadis-Huber, Heller-Yaacoby).

### Real-data citations

ENCODE 4 (Moore et al. 2020, *Nature*; Hitz et al. 2023 pipeline preprint), DREAM5 (Marbach et al. 2012, *Nature Methods*), Tabula Muris (2018 *Nature*; Senis 2020 *Nature*), and Pan-UKBB (Bycroft et al. 2018 *Nature*; Karczewski et al. 2024 medRxiv preprint) are added with full bibliographic detail.

## Structural changes for SAIM template adaptation

### Document structure

The paper has been restructured from the original article-class layout into the SAIM `saim.cls` template with custom title box. Section structure follows the SAIM convention: abstract embedded in the title macro; introduction; related work; method (combining model, inference, and theory); experiments; conclusion; appendix. All inline algorithm definitions use the `algorithm2e` package as required by the template.

### Figures and tables

All six figures (idr-FDR calibration, ROC/PR, posterior contraction, PY-vs-DP cluster behaviour, real-data evaluation, runtime scaling) are rendered as illustrative PDF figures with synthetic numerical content and are clearly labeled as illustrative on each figure caption. Tables include three notation/comparison tables (notation, related-work comparison, simulation design, datasets) and four placeholder-cell results tables (simulation FDR, simulation power, K-effect, real-data discoveries, real-data clusters, real-data runtime).

### Bibliography

The bibliography uses `plainnat` style with the SAIM `natbib` setup. All preprints are flagged as `arXiv preprint` in the journal field; final journal information will be substituted at submission time for entries currently appearing as preprints.

### Build

Build instructions: `pdflatex main; bibtex main; pdflatex main; pdflatex main`. The final compiled PDF is 31 pages with full bibliography and appendix.

## Files in the package

- `main.tex` -- main entry point with title, author, abstract, and section inputs
- `saim.cls` -- SAIM journal class file with title-box macro
- `math_commands.tex` -- math macros and theorem environments
- `refs.bib` -- bibliography with all verified references
- `latexmkrc` -- latexmk configuration
- `assets/saim_logo.png` -- SAIM logo for title box
- `sections/0-abstract.tex` through `sections/6.appendix.tex` -- modular section files
- `tables/table_notation.tex`, `table_related_work.tex`, `table_sim_design.tex`, `table_sim_results.tex`, `table_datasets.tex`, `table_real_results.tex` -- standalone table snippets
- `figures/generate_figures.py` -- Python script that generates all six illustrative figures
- `figures/fig_*.pdf` -- six rendered figures
- `CHANGELOG.md` -- this file
