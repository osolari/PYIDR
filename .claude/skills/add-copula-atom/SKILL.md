---
name: add-copula-atom
description: Add a new copula family to the PY-IDR base measure. Use when the user asks to extend the atomic-type registry beyond {Gauss, t5, Clayton, Gumbel} (e.g. "add a Frank copula", "support a t_3 atom", "register a vine copula").
---

# Skill: Add a Copula Atom

Adding a new copula family to PY-IDR touches the algebraic primitives, the proposal
distribution used in Algorithm 1 step 4, the PLOD probe, and the simulation harness.

## When to invoke

Triggered by any of:

- "add a <name> copula"
- "support a new atomic type"
- "extend the base measure with <family>"
- "register an additional copula family"

Do NOT use this skill if the user is asking to *parametrise* an existing family
differently (that's a model-config edit, not a registry edit).

## Prerequisites

- `.venv` activated; `py_idr.doctor()` is green.
- `git status` is clean OR all in-progress changes are on a feature branch.
- The new family's PLOD status is documented; non-PLOD families cannot enter the
  reproducible-component base measure (would break Theorem 3.1).

## Procedure

1. **Verify PLOD status.** Compute $C(u, ..., u) - u^K$ analytically or numerically on
   a $u$-grid with $K \in \{2, 5, 10\}$. If the family is not PLOD, stop and explain
   to the user why; the family can still be a comparator but not an atomic type.
2. **Add the density module.** Create `src/py_idr/copulas/<name>.py` implementing
   the abstract `Copula` interface from `copulas/base.py`:
   - `log_density(u: jnp.ndarray) -> jnp.ndarray`
   - `cdf_diagonal(u: jnp.ndarray) -> jnp.ndarray`
   - constructors that match the base measure's parameter constraints.
3. **Register.** Append the name to `ATOMIC_TYPES` in `copulas/registry.py`. Add the
   family-specific log-prior over its parameters under the appropriate branch.
4. **Wire the proposal.** In `copulas/proposal.py`, ensure the independence Metropolis
   sampler over `ATOMIC_TYPES` has a uniform mass on the new family.
5. **Add a unit test.** `tests/unit/test_copula_logpdfs.py` gets a new parametrised
   case verifying the log-density against a closed-form or scipy reference.
6. **Add a property test.** `tests/property/test_plod_invariance.py` should now also
   sample from the new family and confirm PLOD on a $u$-grid.
7. **Add a simulation hook.** If the user wants S5-like regimes that mix in the new
   family, edit `simulation/scenarios.py` to include it in the equally-weighted
   tail-dependent mixture.
8. **Run the test gate:**
   ```bash
   pytest -m "unit or property" tests/unit/test_copula_logpdfs.py \
                                tests/property/test_plod_invariance.py
   ```
9. **Document.** Add a short paragraph to `docs/math/copula_family_<name>.md` linking
   to the canonical reference, the PLOD status, and the parameter support.

## Expected outputs

- `src/py_idr/copulas/<name>.py`
- `tests/unit/test_copula_logpdfs.py` updated
- `tests/property/test_plod_invariance.py` updated
- `docs/math/copula_family_<name>.md` (new)
- A new entry in `ATOMIC_TYPES` and `proposal.py`.

## Failure modes

- **PLOD violation.** Refuse to add the family as an atomic type; explain that it
  would break Theorem 3.1's identifiability hinge. Suggest registering it under
  `comparators/` instead.
- **Numerical instability at boundary.** If the log-density blows up near $u_j \in \{0, 1\}$,
  clamp $u$ to `[1e-6, 1 - 1e-6]` consistently with the rest of the package.
- **Non-vectorizable density.** All density evaluations must be `jax.jit`-able and
  `vmap`-friendly. If the family requires a numerical inversion (e.g., a vine
  conditional), use `jax.lax.while_loop` with a fixed iteration cap.
