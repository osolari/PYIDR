# API reference

The PY-IDR API reference is auto-generated from docstrings via
[mkdocstrings](https://mkdocstrings.github.io/). Click into any module
below to see the full signature list, parameter types (with
`jaxtyping` shape annotations), and per-function documentation.

The module layout mirrors the [architecture map](../developer/architecture.md).

## Simulation

::: py_idr.simulation.scenarios
    options:
      show_submodules: false
      members:
        - SimulationResult
        - simulate_S1
        - simulate_S2
        - simulate_S3
        - simulate_S4

::: py_idr.simulation.replicates
    options:
      members:
        - ReplicateRow
        - run_replicate
        - run_replicate_S1
        - run_replicate_S2
        - run_replicate_S3
        - run_replicate_S4

::: py_idr.simulation.evaluation

## Inference

::: py_idr.inference.mcmc.run_chain
    options:
      members:
        - ChainResult
        - MultiChainResult
        - run_chain_simple
        - run_multi_chain_simple
        - summarise

::: py_idr.inference.mcmc.sweep_simple

::: py_idr.inference.mcmc.sweep_multi

::: py_idr.inference.mcmc.class_assign

::: py_idr.inference.mcmc.pi_update

::: py_idr.inference.mcmc.nuts_atoms

::: py_idr.inference.mcmc.atom_dispatch

::: py_idr.inference.mcmc.type_update

::: py_idr.inference.mcmc.py_hyperparam_update

## Decision

::: py_idr.decision.local_idr

::: py_idr.decision.sun_cai

::: py_idr.decision.bayes_mfdr

## Marginals

::: py_idr.marginals.transform

::: py_idr.marginals.bernstein

::: py_idr.marginals.dirichlet_weights

::: py_idr.marginals.pilot

::: py_idr.marginals.degree_prior

## Copulas

::: py_idr.copulas.base

::: py_idr.copulas.gaussian

::: py_idr.copulas.clayton

::: py_idr.copulas.gumbel

::: py_idr.copulas.student_t

::: py_idr.copulas.independence

::: py_idr.copulas.registry

## Pitman-Yor

::: py_idr.pym.stickbreak

::: py_idr.pym.polya_urn

::: py_idr.pym.eppf

## Algebra

::: py_idr.algebra.plod

::: py_idr.algebra.correlation

## State containers

::: py_idr.model.state

::: py_idr.model.multi_state

## Diagnostics & schema

::: py_idr.eval.diagnostics_report

::: py_idr.data.schema

::: py_idr.utils.run
