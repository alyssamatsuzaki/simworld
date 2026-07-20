# RegWorld — working notes for Claude Code

## What this is
A policy world model of regulatory propagation (firms × consumers × regulator × associations),
built as the maximal sixteen-tool stack from `PLAN.md`. Read `PLAN.md` §2 for the tool map, §7 for
the model equations, §10 for the phases and stages. `PROGRESS.md` says where we are.

## Non-negotiables
- Mesa >= 3.0 AgentSet API. No RandomActivation / self.schedule.
- Gymnasium >= 1.0 five-tuple. `truncated` at horizon; `terminated` only on systemic collapse.
- Nothing outside `src/regworld/evaluation/` may import `regworld.dgp` or read `artifacts/oracle/`. Ever.
- No `print()` in src/ (ruff T20). No bare `np.random.*` — seeded Generators, passed explicitly.
- No `torch-scatter` / `torch-sparse`. No `hydra-ray-launcher`.
- Every JAX stage (calibration) runs in a subprocess with JAX_PLATFORMS=cpu.
- `make lint && make typecheck && make test` before every commit. `make smoke` stays under 6 minutes.
- Never stub a stage to pass a gate. Mark BLOCKED/DEGRADED honestly in PROGRESS.md.

## Commands
make setup | lint | typecheck | test | smoke | all | sweep | slurm | dashboard | docker-build

## Where things live
src/regworld/dgp/          the answer key (import-restricted); rules.py holds the shared pure equations
src/regworld/{data,graphs,abm,calibration,causal,models,training,environments,agents,
              evaluation,ensemble,sensitivity,visualization}
configs/                   Hydra groups (profile, compute, data, dgp, population, network, behavior,
                           abm, objective, calibration, causal, emulator, env, policy, rl, ensemble,
                           sensitivity, tracking, eval)
scripts/                   Hydra entry points, one per stage; run_pipeline.py runs all of them.

## If a library API differs from PLAN.md
Follow the library. Log it in DEVIATIONS.md with one line of rationale. Do not pin backwards.
If a gate fails twice, stop and report what failed, what you tried, and what you need.
