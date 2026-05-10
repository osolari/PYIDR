"""CLI entrypoint: `py-idr <command>`.

Subcommands are stubs at the skeleton stage and route into the Python modules.
"""

from __future__ import annotations

import typer

from py_idr.doctor import doctor as run_doctor

app = typer.Typer(help="PY-IDR — Pitman-Yor IDR command line.")


@app.command()
def doctor() -> None:
    """Run environment sanity probes."""
    code = run_doctor()
    raise typer.Exit(code=code)


@app.command()
def fit(
    config: str = typer.Option(..., "--config", help="Hydra config path."),
    inference: str = typer.Option("mcmc", "--inference", help="mcmc | vi"),
    out: str = typer.Option("runs", "--out", help="Output directory for the trace."),
) -> None:
    """Fit PY-IDR to a dataset (stub; implemented by W2/W4)."""
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
    """Run a simulation cell (stub; implemented by W3, plans/07_simulation_studies.md)."""
    raise NotImplementedError(f"sim({regime=}, {K=}, {n=}, {reps=}) — implemented by W3.")


@app.command()
def figure(
    fig: str = typer.Option(..., "--fig", help="f1 | f2 | f3 | f4 | f5 | f6"),
    results: str = typer.Option(..., "--results", help="Path to source CSV."),
    out: str = typer.Option(..., "--out", help="Output PDF path."),
) -> None:
    """Reproduce one of the paper figures (stub; implemented by W3/W4/W5)."""
    raise NotImplementedError(
        f"figure({fig=}, {results=}, {out=}) — implemented per plans/09_figures_and_tables.md."
    )


@app.command()
def table(
    table: str = typer.Option(..., "--table", help="t1 | t2 | t3 | t4 | t5 | t6 | t7"),
    out: str = typer.Option(..., "--out"),
) -> None:
    """Reproduce one of the paper tables (stub)."""
    raise NotImplementedError(
        f"table({table=}, {out=}) — implemented per plans/09_figures_and_tables.md."
    )


@app.command()
def evaluate(
    trace: str = typer.Option(..., "--trace", help="Path to a Zarr trace."),
    alpha: float = typer.Option(0.05, "--alpha"),
) -> None:
    """Evaluate a finished run: local idr, Sun-Cai threshold, Bayes-mFDR (stub)."""
    raise NotImplementedError(
        f"evaluate({trace=}, {alpha=}) — implemented per plans/06_decision_theory.md."
    )


if __name__ == "__main__":
    app()
