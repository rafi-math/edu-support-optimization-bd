# EquiReach: Optimizing Resources. Advancing Fairness.

EquiReach is a reusable decision tool. A planner submits a scenario. The engine returns who gets served, what each fairness rule costs, and whether the answer is proven optimal.

This folder is Mode A only: client-selection with equity floors, a command-line tool, and tests. The FastAPI service, Docker image, and Mode B siting are not in this build.

## What is validated and what is new

- Validated: selection with equity floors (exact MILP, price of fairness, interaction of floors), in the earlier study on MICS 2019 survey data from rural Chattogram. Repository: https://github.com/rafi-math/edu-support-optimization-bd
- New and tested only on synthetic data: center opening, assignment, travel limit, capacity, cost, and the heuristic fallback.
- Real siting needs real coordinates, distances, capacities and costs from the planner. Straight-line distance is not road or river distance.

The sign of an interaction (subadditive, additive, superadditive) depends on the submitted data. This engine does not force a sign.

## Run steps

Use the Python environment already in this folder (`Python 3.13.4` with numpy, scipy, pandas, pytest). Do not use the repository-root `.venv` or system Python 3.14.

From `apps/equireach`:

```text
.venv\Scripts\python.exe -m pytest tests
.venv\Scripts\python.exe -m equireach select --clients path\to\clients.csv --k 10 --rule "age_band:6-12:33.3"
```

CSV template: `templates/clients_template.csv`. Required columns: `client_id`, `priority_score`. Optional: `weight`, `need_flag`, `label`, `lat`, `lng`. Every other column is an attribute the rule builder can read.

`--rule "attribute:level:percent"` sets a share floor. `--rule-count "attribute:level:N"` sets a count floor and is valid with `exactly_K` only.

The CLI prints JSON. The objective is rounded to 6 decimals. Private survey files stay on the author's machine and must not be copied into this folder.

## Mode A MILP

Variables: `x_i` in {0, 1}.

- `v_i = priority_i` if unweighted, else `priority_i * weight_i`.
- `a_i = 1` if unweighted, else `weight_i`.
- Objective: maximise sum `v_i x_i`.
- Coverage: sum `x_i` = K (`exactly_K`) or <= K (`at_most_K`).
- Each rule with group `G` and minimum fraction `f`: sum_{i in G} `a_i x_i` − `f` × sum_i `a_i x_i` >= 0. An optional cap uses the same expression with an upper bound.
- With `exactly_K` and unweighted this is the count floor `ceil(f * K - 1e-9)` used in the validated study.
- Solver: `scipy.optimize.milp` (HiGHS), relative MIP gap 0, time limit `time_limit_s`.
- Exact mode only when the client count is at most 20,000. Otherwise the heuristic runs and a warning is returned.

`proven_optimal` is true only when HiGHS reports status optimal. A time-limited feasible solve returns that solution with `proven_optimal` false. The heuristic always returns solver `greedy_swap` and `proven_optimal` false. `need_flag` is never used in the objective or in a constraint.

## Heuristic (fallback only)

1. Rank clients by `v_i` descending, `client_id` ascending. Take the top K.
2. Floor repair, per rule and then jointly: if a group's share is below its minimum, swap out the lowest-v served client whose removal does not break another met floor, and swap in the highest-v unserved client from the deficient group.
3. If a floor is still unmet: status `partial`, name the rule, give the achieved share.

## Tests

Mode A tests cover: exact K (or clamp), brute-force match on tiny instances, floor effect or `partial`, count-floor equivalence, price-of-fairness monotonicity, interaction arithmetic, baseline comparison, honesty of `proven_optimal`, `need_flag` shuffle invariance, validation errors, empty/infeasible status, CLI match against `allocate()`, and that the engine does not write user files.

## Limits of this build

- Mode B (centers, travel, capacity, cost) is not implemented.
- There is no HTTP API, Docker image, login, or database in this folder.
- Coordinates, if present, are unused in Mode A. Straight-line distance is not road or river distance.
- Results come only from the submitted scenario. Nothing is invented.
