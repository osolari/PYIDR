"""CLI entrypoint: ``py-idr <command>``.

Installed via the ``[project.scripts]`` block in ``pyproject.toml``.
Subcommands:

- ``doctor`` — environment sanity probes (JAX backend, devices, version).
- ``fit`` — fit PY-IDR to a dataset (stub today; W2/W4 in plans/).
- ``sim`` — run a simulation sweep and emit a long-format CSV.
- ``figure`` — reproduce one of the report figures from a result CSV.
- ``table`` — reproduce one of the report tables from a result CSV.
- ``evaluate`` — load a finished trace and run the decision pipeline
  (Sun-Cai + Bayes-mFDR) at a user-specified $\\alpha$.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from py_idr.doctor import doctor as run_doctor

app = typer.Typer(help="PY-IDR — Pitman-Yor IDR command line.")


@app.command()
def doctor() -> None:
    """Run environment sanity probes.

    Prints a verdict (pass / warn / fail) covering:

    - JAX backend selection (CPU, CUDA, or Metal) and devices visible.
    - PY-IDR package version + dependency versions.
    - Working directory and the data directory the run scripts would use.

    Exit code is 0 on pass, 1 on warn, 2 on fail. Suitable for use in
    CI sanity-check steps.
    """
    code = run_doctor()
    raise typer.Exit(code=code)


@app.command()
def fit(
    config: str = typer.Option(..., "--config", help="Hydra config path."),
    inference: str = typer.Option("mcmc", "--inference", help="mcmc | vi"),
    out: str = typer.Option("runs", "--out", help="Output directory for the trace."),
) -> None:
    """Fit PY-IDR to a dataset.

    **Stub today.** When implemented (plans/04 + plans/05), this command
    will resolve the Hydra config at ``--config``, load the dataset
    manifest it points to, run the chain (MCMC or VI per
    ``--inference``), write the trace as Zarr under ``--out/<run_id>/``,
    and emit a :class:`py_idr.data.schema.RunManifest` JSON alongside.

    The full S5 / real-data sweeps go through this command; for ad-hoc
    runs, the Python entrypoints in :mod:`py_idr.simulation.replicates`
    or :func:`py_idr.inference.mcmc.run_chain.run_multi_chain_simple` are
    more ergonomic.
    """
    raise NotImplementedError(
        f"fit({config=}, {inference=}, {out=}) — implemented by W2 (MCMC) and W4 (VI). "
        f"See plans/04_inference_mcmc.md and plans/05_inference_vi.md."
    )


@app.command()
def sim(
    regime: str = typer.Option(..., "--regime", help="S1 | S2 | S3 | S4 | S5 | S5-sparse"),
    K: int = typer.Option(4, "--K"),  # noqa: N803
    n: int = typer.Option(10_000, "--n"),
    reps: int = typer.Option(100, "--reps"),
    base_seed: int = typer.Option(0, "--base-seed"),
    alphas: str = typer.Option(
        "0.01,0.05,0.10,0.20",
        "--alphas",
        help="Comma-separated Sun-Cai operating levels.",
    ),
    num_chains: int = typer.Option(2, "--num-chains"),
    n_warmup: int = typer.Option(50, "--n-warmup"),
    n_samples: int = typer.Option(100, "--n-samples"),
    nuts_warmup: int = typer.Option(20, "--nuts-warmup"),
    M: int = typer.Option(5, "--M"),  # noqa: N803
    chain_type: str = typer.Option(
        "auto",
        "--chain-type",
        help="auto | T=1 | T>1. 'auto' picks T>1 for S5/S5-sparse else T=1.",
    ),
    T_max: int = typer.Option(10, "--T-max", help="Cluster truncation (T>1 path)."),  # noqa: N803
    H: int = typer.Option(5, "--H", help="Aux-atom count per Pólya-urn step (T>1)."),  # noqa: N803
    py_hyperparam_warmup: int = typer.Option(
        50, "--py-hyperparam-warmup", help="(alpha, sigma) NUTS warmup (T>1)."
    ),
    save_idr: bool = typer.Option(
        False,
        "--save-idr/--no-save-idr",
        help=(
            "Also write a per-replicate idr / true_Z NPZ sidecar "
            "alongside results.csv. Required input for F4 (ROC/PR)."
        ),
    ),
    em_init: bool = typer.Option(
        False,
        "--em-init/--no-em-init",
        help=(
            "Seed each PY-IDR chain via the pairwise Vanilla-IDR EM "
            "warm start (W8.5). Recommended for sweeps with short "
            "warmup budgets — gives the chain an informed (pi, rho, Z) "
            "start so a low n_warmup is no longer disastrous."
        ),
    ),
    method: str = typer.Option(
        "py-idr",
        "--method",
        help=(
            "Fit method: 'py-idr' (default, PY-IDR MCMC), 'vanilla-idr' "
            "(Li-Brown-Huang-Bickel 2011 pairwise EM), or 'maxrank' "
            "(rank-based baseline, Philtron et al. 2018 style)."
        ),
    ),
    out: str = typer.Option(
        "runs/sim",
        "--out",
        help="Output directory; a per-run subdir is created under it.",
    ),
) -> None:
    """Run a simulation sweep and emit a long-format CSV.

    For each regime ``--regime`` runs ``--reps`` independent replicates
    (seed schedule: ``base_seed + i``), fits the chain once per
    replicate, and evaluates Sun-Cai at every ``--alpha``. The output
    is a long-format CSV plus a sibling :class:`RunManifest` JSON.

    Chain routing follows ``--chain-type``:

    - ``auto`` (default) — multi-cluster path for S5 / S5-sparse,
      T=1 fast path for S1-S4.
    - ``T=1`` — force the single-Gaussian fast path everywhere
      (useful for ablation sweeps that pit T=1 against T>1 on S5).
    - ``T>1`` — force the multi-cluster path everywhere.

    Output layout:

    ::

        <out>/<run_id>/
            results.csv      # one row per (replicate, alpha)
            run_manifest.json

    Example (ablation: T=1 vs T>1 on S5):

    .. code-block:: bash

        py-idr sim --regime S5 --K 2 --n 200 --reps 20 \\
                   --chain-type T=1 --out runs/s5_t1
        py-idr sim --regime S5 --K 2 --n 200 --reps 20 \\
                   --chain-type T>1 --out runs/s5_tgt1
    """
    from py_idr.simulation.sweep import (  # type: ignore[attr-defined]
        run_sweep,
        run_sweep_with_traces,
        write_idr_sidecar,
        write_sweep_csv,
    )
    from py_idr.utils.run import (  # type: ignore[attr-defined]
        build_run_manifest,
        detect_git_sha,
        new_run_id,
        now_utc,
    )

    alphas_tuple = tuple(float(a) for a in alphas.split(","))

    run_id = new_run_id()
    out_dir = Path(out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    method_key = method.lower()
    if method_key not in ("py-idr", "vanilla-idr", "maxrank"):
        raise typer.BadParameter(
            f"--method must be 'py-idr', 'vanilla-idr', or 'maxrank'; got {method!r}"
        )

    typer.echo(
        f"Running sweep: method={method_key} regime={regime} K={K} n={n} reps={reps} "
        f"alphas={alphas_tuple} → {out_dir}"
        + (f" chain_type={chain_type} save_idr={save_idr}" if method_key == "py-idr" else "")
    )

    if method_key == "vanilla-idr":
        from py_idr.simulation.sweep import (  # type: ignore[attr-defined]
            run_sweep_vanilla_idr,
        )

        if save_idr:
            typer.echo(
                "WARNING: --save-idr is a no-op for --method vanilla-idr "
                "(no NPZ sidecar produced)."
            )
        rows = run_sweep_vanilla_idr(
            regime,
            K=K,
            n=n,
            num_replicates=reps,
            base_seed=base_seed,
            alphas=alphas_tuple,
        )
    elif method_key == "maxrank":
        from py_idr.simulation.sweep import (  # type: ignore[attr-defined]
            run_sweep_maxrank,
        )

        if save_idr:
            typer.echo(
                "WARNING: --save-idr is a no-op for --method maxrank " "(no NPZ sidecar produced)."
            )
        rows = run_sweep_maxrank(
            regime,
            K=K,
            n=n,
            num_replicates=reps,
            base_seed=base_seed,
            alphas=alphas_tuple,
        )
    elif save_idr:
        sweep_result = run_sweep_with_traces(
            regime,
            K=K,
            n=n,
            num_replicates=reps,
            base_seed=base_seed,
            alphas=alphas_tuple,
            num_chains=num_chains,
            n_warmup=n_warmup,
            n_samples=n_samples,
            nuts_warmup=nuts_warmup,
            M=M,
            chain_type=chain_type,
            T_max=T_max,
            H=H,
            py_hyperparam_warmup=py_hyperparam_warmup,
            em_init=em_init,
        )
        rows = sweep_result.rows
        npz = write_idr_sidecar(sweep_result.idr_traces, out_dir / "idr_traces.npz")
        typer.echo(f"Wrote idr sidecar to {npz}")
    else:
        rows = run_sweep(
            regime,
            K=K,
            n=n,
            num_replicates=reps,
            base_seed=base_seed,
            alphas=alphas_tuple,
            num_chains=num_chains,
            n_warmup=n_warmup,
            n_samples=n_samples,
            nuts_warmup=nuts_warmup,
            M=M,
            chain_type=chain_type,
            T_max=T_max,
            H=H,
            py_hyperparam_warmup=py_hyperparam_warmup,
            em_init=em_init,
        )
    csv = write_sweep_csv(rows, out_dir / "results.csv")

    # Manifest sibling: records run_id + git_sha + config so the run is
    # reproducible from this one directory. The full resolved config is
    # persisted to ``run_config.json`` so ``scripts/replay_run.sh`` can
    # reconstruct the original ``py-idr sim`` command.
    config_dict = {
        "regime": regime,
        "K": K,
        "n": n,
        "reps": reps,
        "base_seed": base_seed,
        "alphas": list(alphas_tuple),
        "num_chains": num_chains,
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "nuts_warmup": nuts_warmup,
        "M": M,
        "chain_type": chain_type,
        "T_max": T_max,
        "H": H,
        "py_hyperparam_warmup": py_hyperparam_warmup,
        "save_idr": save_idr,
        "em_init": em_init,
        "method": method_key,
    }
    config_blob = json.dumps(config_dict, sort_keys=True)
    config_hash = __import__("hashlib").sha256(config_blob.encode()).hexdigest()

    config_path = out_dir / "run_config.json"
    config_path.write_text(json.dumps(config_dict, indent=2, sort_keys=True))

    manifest = build_run_manifest(
        run_id=run_id,
        git_sha=detect_git_sha() or "0" * 40,
        config_hash=config_hash,
        seed=base_seed,
        started_at=now_utc(),
        inference_method="mcmc",
    )
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    typer.echo(f"Wrote {len(rows)} rows to {csv}")
    typer.echo(f"Wrote manifest to {manifest_path}")
    typer.echo(f"Wrote config to {config_path}")


@app.command()
def figure(
    fig: str = typer.Option(..., "--fig", help="f1 | f2 | f3 | f4 | f5"),
    results: str = typer.Option(..., "--results", help="Path to source CSV."),
    out: str = typer.Option(..., "--out", help="Output PDF path."),
    regimes: str = typer.Option(
        "S1,S2,S4,S5",
        "--regimes",
        help="Comma-separated regime list.",
    ),
    idr_npz: str = typer.Option(
        "",
        "--idr-npz",
        help=(
            "Path to the idr_traces.npz sidecar produced by "
            "`py-idr sim --save-idr`. Required for f4."
        ),
    ),
) -> None:
    """Reproduce one of the report figures from a results CSV.

    Implemented today: ``f1`` (realised FDR vs nominal $\\alpha$
    calibration across regimes). Other labels (``f2``..``f6``) raise
    :class:`NotImplementedError` with a pointer to the relevant plan.
    """
    label = fig.lower()
    regimes_tuple = tuple(r.strip() for r in regimes.split(","))
    if label == "f1":
        from py_idr.figures.f1_calibration import (
            make_calibration_figure,  # type: ignore[attr-defined]
        )

        out_path = make_calibration_figure(results, out, regimes=regimes_tuple)
        typer.echo(f"Wrote {out_path}")
        return

    if label == "f2":
        from py_idr.figures.f2_cluster_count import (
            make_cluster_count_figure,  # type: ignore[attr-defined]
        )

        # For F2 the default regimes are the multi-cluster ones if the
        # caller left the F1 default of S1/S2/S4/S5 in place.
        if regimes == "S1,S2,S4,S5":
            regimes_tuple = ("S5", "S5-sparse")
        out_path = make_cluster_count_figure(results, out, regimes=regimes_tuple)
        typer.echo(f"Wrote {out_path}")
        return

    if label == "f3":
        from py_idr.figures.f3_runtime import (
            make_runtime_figure,  # type: ignore[attr-defined]
        )

        # For F3 we usually want every (regime, method) cell, not just
        # the F1 default. Pass None to keep all.
        regimes_arg = regimes_tuple if regimes != "S1,S2,S4,S5" else None
        out_path = make_runtime_figure(results, out, regimes=regimes_arg)
        typer.echo(f"Wrote {out_path}")
        return

    if label == "f4":
        from py_idr.figures.f4_roc_pr import (
            make_roc_pr_figure,  # type: ignore[attr-defined]
        )
        from py_idr.simulation.sweep import (
            load_idr_sidecar,  # type: ignore[attr-defined]
        )

        if not idr_npz:
            raise typer.BadParameter(
                "f4 requires --idr-npz pointing at the idr_traces.npz "
                "sidecar produced by `py-idr sim --save-idr`."
            )
        idr_z_by_regime = load_idr_sidecar(idr_npz)
        # Match the F2 default fall-through: if the user kept the F1
        # default --regimes string, the natural F4 default is the
        # multi-cluster regimes only.
        if regimes == "S1,S2,S4,S5":
            present = ("S5", "S5-sparse")
            regimes_arg_f4: tuple[str, ...] | None = (
                tuple(r for r in present if r in idr_z_by_regime) or None
            )
        else:
            regimes_arg_f4 = regimes_tuple
        out_path = make_roc_pr_figure(idr_z_by_regime, out, regimes=regimes_arg_f4)
        typer.echo(f"Wrote {out_path}")
        return

    if label == "f5":
        from py_idr.figures.f5_contraction import (
            make_contraction_figure,  # type: ignore[attr-defined]
        )

        # F5 is meaningful only for regimes with finite rho_true; the
        # default narrows to S1 unless the user overrode --regimes.
        regimes_arg_f5 = ("S1",) if regimes == "S1,S2,S4,S5" else regimes_tuple
        out_path = make_contraction_figure(results, out, regimes=regimes_arg_f5)
        typer.echo(f"Wrote {out_path}")
        return

    raise NotImplementedError(
        f"figure --fig={fig} not implemented yet. Today 'f1'..'f5' "
        f"are wired up. See plans/09_figures_and_tables.md for F6."
    )


@app.command()
def table(
    table: str = typer.Option(
        ..., "--table", help="t4-fdr | t4-power | t7 (today) | t1..t6 (planned)"
    ),
    results: str = typer.Option(..., "--results", help="Path to source CSV."),
    out: str = typer.Option(..., "--out"),
    alpha: float = typer.Option(0.05, "--alpha"),
) -> None:
    """Reproduce one of the report tables from a results CSV.

    Implemented today:

    - ``t4-fdr`` — realised FDR at nominal $\\alpha$ (default 0.05) across
      regimes; matches ``tab:sim-fdr``.
    - ``t4-power`` — same shape but with the ``power`` column;
      matches ``tab:sim-power``.

    Other labels raise :class:`NotImplementedError`.
    """
    from py_idr.tables.t4_sim_results import (
        make_simulation_results_table,  # type: ignore[attr-defined]
    )

    label = table.lower()
    if label == "t4-fdr":
        out_path = make_simulation_results_table(
            results, out, metric="realized_fdr", alpha=alpha, label="tab:sim-fdr"
        )
    elif label == "t4-power":
        out_path = make_simulation_results_table(
            results, out, metric="power", alpha=alpha, label="tab:sim-power"
        )
    elif label == "t7":
        from py_idr.tables.t7_K_effect import (
            make_K_effect_table,  # type: ignore[attr-defined]
        )

        out_path = make_K_effect_table(results, out, regime="S5", alpha=alpha, label="tab:sim-K")
    else:
        raise NotImplementedError(
            f"table --table={table} not implemented yet. Today only "
            "'t4-fdr', 't4-power', and 't7' are wired up. See plans/09_figures_and_tables.md."
        )
    typer.echo(f"Wrote {out_path}")


@app.command()
def evaluate(
    idr_npz: str = typer.Option(
        ...,
        "--idr-npz",
        help="Path to an idr_traces.npz sidecar produced by `py-idr sim --save-idr`.",
    ),
    alpha: float = typer.Option(0.05, "--alpha", help="Sun-Cai operating level."),
    regimes: str = typer.Option(
        "",
        "--regimes",
        help="Comma-separated regimes to include (default: all present).",
    ),
    out: str = typer.Option(
        "",
        "--out",
        help="Optional output CSV path; if omitted, prints to stdout.",
    ),
) -> None:
    """Re-evaluate an idr sidecar at a user-specified Sun-Cai $\\alpha$.

    Reads the per-replicate ``(idr, true_Z)`` arrays from the NPZ
    sidecar, applies Sun-Cai at ``--alpha`` to each replicate, and
    prints per-regime mean realised FDR / power / $k_\\alpha$.

    The sweep CSV already carries these metrics for the alphas the
    sweep itself was run at. This command is for **after-the-fact
    re-evaluation at a different $\\alpha$**, without re-running the
    chain.

    Example:

    .. code-block:: bash

        py-idr sim --regime S5 --reps 20 --save-idr --out runs/sim
        py-idr evaluate --idr-npz runs/sim/<run_id>/idr_traces.npz \\
                        --alpha 0.10
    """
    import jax.numpy as jnp
    import pandas as pd

    from py_idr.simulation.evaluation import evaluate_recovery  # type: ignore[attr-defined]
    from py_idr.simulation.sweep import load_idr_sidecar  # type: ignore[attr-defined]

    if not (0.0 < alpha < 1.0):
        raise typer.BadParameter(f"--alpha must be in (0, 1); got {alpha}")

    traces_by_regime = load_idr_sidecar(idr_npz)
    if regimes:
        allow = {r.strip() for r in regimes.split(",")}
        traces_by_regime = {k: v for k, v in traces_by_regime.items() if k in allow}
    if not traces_by_regime:
        raise typer.BadParameter(f"No matching regimes in {idr_npz}; try --regimes ''.")

    rows: list[dict[str, object]] = []
    for regime, replicates in traces_by_regime.items():
        for seed_i, (idr_arr, Z_arr) in enumerate(replicates):
            rec = evaluate_recovery(jnp.asarray(idr_arr), jnp.asarray(Z_arr), alpha=alpha)
            rows.append(
                {
                    "regime": regime,
                    "alpha": alpha,
                    "replicate_idx": seed_i,
                    "k_alpha": rec.k_alpha,
                    "realized_fdr": rec.realized_fdr,
                    "power": rec.power,
                }
            )

    df = pd.DataFrame(rows)
    summary = df.groupby("regime")[["realized_fdr", "power", "k_alpha"]].agg(["mean", "std"])
    typer.echo(f"Sun-Cai @ alpha = {alpha} — per-regime mean (std) across {len(rows)} rows:")
    typer.echo(summary.to_string())

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        typer.echo(f"Wrote {out_path}")


@app.command()
def verdict(
    results: str = typer.Option(..., "--results", help="Path to a sweep CSV."),
    out: str = typer.Option(..., "--out", help="Output verdict JSON path."),
    fdr_slack: float = typer.Option(
        0.02,
        "--fdr-slack",
        help="Tolerance on the H1 FDR-control check (absolute, default 0.02).",
    ),
) -> None:
    """Build a verdict ledger from a sweep CSV.

    Runs the three pre-registered canonical hypotheses (H1 FDR
    control, H2 PY-IDR beats Vanilla IDR on S5, H3 PY-IDR beats
    MaxRank on S1) against the sweep DataFrame and writes a
    :class:`py_idr.repro.verdict.VerdictLedger` JSON to ``--out``.

    Per §4.10 of the report, every companion-paper analysis ships
    with such a ledger so a reader can audit which acceptance
    criteria the run met. Each verdict records the pre-registered
    statement, the {pass / warn / fail / insufficient_data} outcome,
    the underlying evidence dict, and a one-line rationale.

    Insufficient-data outcomes are returned (not failures) when the
    sweep CSV lacks rows for the methods the hypothesis compares —
    e.g., H2 needs both PY-IDR (T>1) and Vanilla IDR rows.
    """
    import pandas as pd

    from py_idr.repro.verdict import build_verdict_ledger  # type: ignore[attr-defined]

    df = pd.read_csv(results)
    ledger = build_verdict_ledger(df, sweep_csv=results, fdr_slack=fdr_slack)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ledger.model_dump_json(indent=2))

    typer.echo(f"Wrote {out_path}")
    typer.echo(f"Verdict summary: {ledger.summary()}")
    for rec in ledger.records:
        typer.echo(f"  [{rec.hypothesis_id}] {rec.outcome:>18} — {rec.rationale}")


if __name__ == "__main__":
    app()
