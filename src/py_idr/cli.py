"""CLI entrypoint: ``py-idr <command>``.

Installed via the ``[project.scripts]`` block in ``pyproject.toml``.
Subcommands are:

- ``doctor`` — environment sanity probes (JAX backend, devices, version).
- ``fit`` — fit PY-IDR to a dataset (stub today; W2/W4 in plans/).
- ``sim`` — run a simulation cell (stub today; W3).
- ``figure`` — reproduce one of the report figures from a result CSV.
- ``table`` — reproduce one of the report tables from a result CSV.
- ``evaluate`` — load a finished trace and run the decision pipeline
  (Sun-Cai + Bayes-mFDR) at a user-specified $\\alpha$.

The stubs raise :class:`NotImplementedError` with a pointer to the plan
that will implement them; this keeps the surface stable while the
back-end fills in.
"""

from __future__ import annotations

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
    regime: str = typer.Option(..., "--regime", help="S1 | S2 | S3 | S4 | S5"),
    K: int = typer.Option(4, "--K"),  # noqa: N803
    n: int = typer.Option(10_000, "--n"),
    reps: int = typer.Option(100, "--reps"),
) -> None:
    """Run a simulation cell — one regime, many seeds.

    **Stub today.** When implemented (plans/07), this command will:

    1. Generate ``reps`` replicates of regime ``regime`` at
       ``(K, n)``-shape, using a deterministic seed schedule rooted at
       the base seed in the run manifest.
    2. Fit each replicate end-to-end via
       :func:`py_idr.simulation.replicates.run_replicate`.
    3. Concatenate the per-replicate ``ReplicateRow`` outputs into a
       long-format CSV under ``runs/sim/<run_id>/results.csv``.

    Today the Python equivalent is to call ``run_replicate_S<N>`` in a
    loop and ``pd.DataFrame([r.to_dict() for r in rows]).to_csv(...)``.
    """
    raise NotImplementedError(f"sim({regime=}, {K=}, {n=}, {reps=}) — implemented by W3.")


@app.command()
def figure(
    fig: str = typer.Option(..., "--fig", help="f1 | f2 | f3 | f4 | f5 | f6"),
    results: str = typer.Option(..., "--results", help="Path to source CSV."),
    out: str = typer.Option(..., "--out", help="Output PDF path."),
) -> None:
    """Reproduce one of the report figures from a results CSV.

    **Stub today.** When implemented (plans/09), this routes to the
    matching reproducer in :mod:`py_idr.figures`. The recognised figure
    labels match the ones in ``docs/report/figures/``: ``f1`` =
    realised-vs-nominal FDR calibration, ``f2`` = PY-vs-DP cluster count,
    ``f3`` = posterior contraction, ``f4`` = idr calibration, ``f5`` =
    real-data ROC/PR, ``f6`` = runtime scaling.

    The figure reproducers are pure plotting code — they consume the
    sweep CSV (column layout matches :class:`ReplicateRow`) and emit a
    paginated PDF.
    """
    raise NotImplementedError(
        f"figure({fig=}, {results=}, {out=}) — implemented per plans/09_figures_and_tables.md."
    )


@app.command()
def table(
    table: str = typer.Option(..., "--table", help="t1 | t2 | t3 | t4 | t5 | t6 | t7"),
    out: str = typer.Option(..., "--out"),
) -> None:
    """Reproduce one of the report tables from a results CSV.

    **Stub today.** When implemented (plans/09), this emits a LaTeX
    tabular file at ``--out`` matching one of the tables under
    ``docs/report/tables/``: ``t1`` = notation, ``t2`` = related work,
    ``t3`` = simulation design, ``t4`` = simulation results, ``t5`` =
    datasets, ``t6`` = real-data results, ``t7`` = ablations.
    """
    raise NotImplementedError(
        f"table({table=}, {out=}) — implemented per plans/09_figures_and_tables.md."
    )


@app.command()
def evaluate(
    trace: str = typer.Option(..., "--trace", help="Path to a Zarr trace."),
    alpha: float = typer.Option(0.05, "--alpha"),
) -> None:
    """Evaluate a finished run: local idr, Sun-Cai threshold, Bayes-mFDR.

    **Stub today.** When implemented (plans/06), this will:

    1. Load the Zarr trace at ``--trace`` (an ``arviz.InferenceData``
       plus the Z-trace).
    2. Compute the posterior local idr via
       :func:`py_idr.decision.local_idr.from_mcmc`.
    3. Apply the Sun-Cai step-up rule at level ``--alpha`` and report
       $k_\\alpha$, $\\gamma_\\alpha$, realised mFDR.
    4. Emit the Bayes-mFDR curve as a side artifact.

    For interactive / scripted use, call
    :func:`py_idr.decision.sun_cai.step_up_threshold` directly.
    """
    raise NotImplementedError(
        f"evaluate({trace=}, {alpha=}) — implemented per plans/06_decision_theory.md."
    )


if __name__ == "__main__":
    app()
