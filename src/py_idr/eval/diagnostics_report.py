"""§3.3.1 diagnostics report (plan 12).

Builds a :class:`DiagnosticsReport` from an arviz ``InferenceData`` (or our
:class:`MultiChainResult`). The report mirrors the explicit diagnostic list of
§3.3.1 of the revised report:

- Split-$\\widehat R$ on every reported scalar must be $\\le 1.01$.
- Effective sample size on every reported scalar must be $\\ge 400$.
- NUTS divergent-transition count must be $< 0.5\\%$ of post-warmup samples.
- At least 4 overdispersed chains; flagged if fewer.

The verdict is one of ``"pass"`` / ``"warn"`` / ``"fail"``:

- ``"pass"``  — every threshold met.
- ``"warn"``  — a threshold is missed but the chain is usable (e.g., ESS just
  below 400, or 3 chains instead of 4).
- ``"fail"``  — split-$\\widehat R$ blows past 1.10 OR there are divergent
  transitions in the post-warmup samples OR less than 2 chains.

Per the durable user feedback, the diagnostic values come from
:mod:`arviz` — we do NOT reimplement split-$\\widehat R$, ESS, etc. The
:class:`DiagnosticsReport` is just the bridge between arviz output and the
§3.3.1 contract.

See plans/12_diagnostics.md.
"""

from __future__ import annotations

from typing import Any, Literal

import arviz as az
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# §3.3.1 thresholds.
RHAT_PASS_THRESHOLD: float = 1.01
RHAT_FAIL_THRESHOLD: float = 1.10
ESS_PASS_THRESHOLD: float = 400.0
DIVERGENCE_FRACTION_THRESHOLD: float = 0.005
MIN_CHAINS_PASS: int = 4
MIN_CHAINS_FAIL: int = 2


class DiagnosticsReport(BaseModel):
    r"""Bundled chain-quality verdict, serialisable to ``diagnostics.json``.

    Built from an :class:`arviz.InferenceData` via
    :func:`build_diagnostics_report`. The verdict logic is:

    - **pass**: every per-variable $\hat R \le$ :data:`RHAT_PASS_THRESHOLD`,
      every per-variable bulk + tail ESS $\ge$ :data:`ESS_PASS_THRESHOLD`,
      divergence fraction $\le$ :data:`DIVERGENCE_FRACTION_THRESHOLD`,
      number of chains $\ge$ :data:`MIN_CHAINS_PASS`.
    - **warn**: thresholds not all met but no catastrophic failure
      (e.g., only 2-3 chains, some borderline ESS).
    - **fail**: at least one of $\hat R > 1.05$, ESS $< 100$, or
      divergence fraction $> 0.05$. In this case the posterior should
      not be trusted without re-running with more chains / longer warmup.

    Why pydantic?
    -------------
    The report is serialised to JSON next to the run manifest as part of
    plan 13's reproducibility ledger. ``extra = "forbid"`` ensures
    schema drift is caught at load time.

    See the **diagnostics guide** in the docs site for the operational
    interpretation of each field.
    """

    model_config = ConfigDict(extra="forbid")

    rhat: dict[str, float] = Field(
        default_factory=dict,
        description="Per-variable split-Rhat. Empty when the chain has <2 chains.",
    )
    ess_bulk: dict[str, float] = Field(
        default_factory=dict,
        description="Per-variable bulk ESS. Drawn from arviz.summary.",
    )
    ess_tail: dict[str, float] = Field(
        default_factory=dict,
        description="Per-variable tail ESS.",
    )
    num_chains: int = Field(..., ge=1)
    num_draws_per_chain: int = Field(..., ge=1)
    num_divergent_transitions: int = Field(default=0, ge=0)
    divergence_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    label_invariant_summary: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Label-switching-invariant per-cluster summaries (W10.12). Populated "
            "by py_idr.eval.label_switching.build_label_invariant_summary when "
            "the build_diagnostics_report caller supplies a cluster_assignments "
            "array. Empty dict on chains that have no multi-cluster trace."
        ),
    )
    verdict: Literal["pass", "warn", "fail"] = Field(...)
    warnings: list[str] = Field(default_factory=list)


def _extract_scalars_from_summary(
    summary_df: Any,  # pandas.DataFrame; typed as Any to avoid a top-level pandas import
    column: str,
    variables: list[str],
) -> dict[str, float]:
    """Pull a column from arviz.summary into a {var_name: value} dict.

    arviz.summary returns object-dtype columns when some entries are NaN;
    we coerce defensively and skip non-finite values.
    """
    out: dict[str, float] = {}
    for var in variables:
        if var in summary_df.index and column in summary_df.columns:
            raw = summary_df.loc[var, column]
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                out[var] = value
    return out


def build_diagnostics_report(
    idata: az.InferenceData,
    *,
    scalar_vars: list[str] | None = None,
    num_divergent_transitions: int = 0,
    cluster_assignments: np.ndarray | None = None,
    T_max: int | None = None,
) -> DiagnosticsReport:
    """Build a :class:`DiagnosticsReport` from an arviz InferenceData.

    Parameters
    ----------
    idata
        Posterior trace as an arviz InferenceData (or DataTree). Shape on each
        variable is ``(num_chains, num_draws_per_chain)``.
    scalar_vars
        Names of the scalar variables to summarise. Defaults to the keys of
        ``idata.posterior``.
    num_divergent_transitions
        Total NUTS divergent transitions across all chains, from
        :attr:`py_idr.inference.mcmc.run_chain.ChainResult.num_divergent_transitions`
        and its multi-cluster / multi-chain variants (W8.11 plumbing).
    cluster_assignments
        Optional per-iteration cluster-label trace, shape ``(num_draws, n)``,
        from the multi-cluster chain
        (:attr:`ChainResultMulti.cluster_assignments`). When supplied (along
        with ``T_max``), the report's ``label_invariant_summary`` field is
        populated via :func:`py_idr.eval.label_switching.build_label_invariant_summary`.
        Pass ``None`` (the default) on the $T=1$ path or when label-invariant
        diagnostics are not needed.
    T_max
        Cluster truncation passed to the chain driver. Required iff
        ``cluster_assignments`` is supplied.

    Returns
    -------
    A :class:`DiagnosticsReport` with the verdict applied per the §3.3.1
    thresholds.
    """
    posterior = idata.posterior
    # Detect chain / draw counts from the first variable.
    first_var = next(iter(posterior.data_vars))
    num_chains, num_draws = posterior[first_var].shape[:2]

    if scalar_vars is None:
        scalar_vars = list(posterior.data_vars)

    summary_df = az.summary(idata, var_names=scalar_vars)

    rhat_col = "r_hat" if "r_hat" in summary_df.columns else None
    rhat = (
        _extract_scalars_from_summary(summary_df, rhat_col, scalar_vars)
        if (rhat_col is not None and num_chains >= 2)
        else {}
    )
    ess_bulk = _extract_scalars_from_summary(summary_df, "ess_bulk", scalar_vars)
    ess_tail = _extract_scalars_from_summary(summary_df, "ess_tail", scalar_vars)

    total_post_warmup_samples = num_chains * num_draws
    divergence_fraction = (
        num_divergent_transitions / total_post_warmup_samples
        if total_post_warmup_samples > 0
        else 0.0
    )

    warnings: list[str] = []
    verdict: Literal["pass", "warn", "fail"] = "pass"

    if num_chains < MIN_CHAINS_FAIL:
        verdict = "fail"
        warnings.append(
            f"only {num_chains} chain(s); need >= {MIN_CHAINS_FAIL} for any split-Rhat."
        )
    elif num_chains < MIN_CHAINS_PASS:
        verdict = "warn"
        warnings.append(f"only {num_chains} chains; §3.3.1 recommends >= {MIN_CHAINS_PASS}.")

    for var, r_hat in rhat.items():
        if r_hat > RHAT_FAIL_THRESHOLD:
            verdict = "fail"
            warnings.append(f"split-Rhat({var}) = {r_hat:.3f} > {RHAT_FAIL_THRESHOLD} (fail).")
        elif r_hat > RHAT_PASS_THRESHOLD:
            if verdict == "pass":
                verdict = "warn"
            warnings.append(f"split-Rhat({var}) = {r_hat:.3f} > {RHAT_PASS_THRESHOLD} (warn).")

    for var, ess in ess_bulk.items():
        if ess < ESS_PASS_THRESHOLD:
            if verdict == "pass":
                verdict = "warn"
            warnings.append(f"ess_bulk({var}) = {ess:.0f} < {ESS_PASS_THRESHOLD:.0f} (warn).")

    if divergence_fraction > DIVERGENCE_FRACTION_THRESHOLD:
        verdict = "fail"
        warnings.append(
            f"divergent transitions = {divergence_fraction:.4f} of post-warmup "
            f"> {DIVERGENCE_FRACTION_THRESHOLD:.4f} (fail)."
        )
    elif num_divergent_transitions > 0:
        if verdict == "pass":
            verdict = "warn"
        warnings.append(f"{num_divergent_transitions} divergent transitions in post-warmup (warn).")

    label_invariant_summary: dict[str, float] = {}
    if cluster_assignments is not None:
        if T_max is None:
            raise ValueError(
                "cluster_assignments was supplied; T_max is required to size "
                "the sorted-cluster-size buffer."
            )
        # Local import to keep eval.diagnostics_report independent of
        # scipy at the top level (scipy is already a transitive runtime
        # dep but importing eagerly slows the module's first load).
        from py_idr.eval.label_switching import build_label_invariant_summary

        label_invariant_summary = build_label_invariant_summary(
            cluster_assignments, T_max=int(T_max)
        )

    return DiagnosticsReport(
        rhat=rhat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        num_chains=num_chains,
        num_draws_per_chain=num_draws,
        num_divergent_transitions=num_divergent_transitions,
        divergence_fraction=divergence_fraction,
        label_invariant_summary=label_invariant_summary,
        verdict=verdict,
        warnings=warnings,
    )
