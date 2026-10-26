# EquiReach engine spec (file 1 of 3)

**Give this file to:** Cursor, Agent mode, in an empty folder. Save it as `docs/01_engine_spec.md` next to `docs/02_api_contract.md`.
**Do not give it to:** Lovable, Bolt or v0. They cannot run this engine.
**Job:** build the Python engine, its tests, a command-line tool, the FastAPI service and a Dockerfile. No user interface.

## Rules for you, the coding agent

- Create the files in the repository. Do not paste whole files into the chat.
- Build Mode A first, with its tests. Run `pytest`, fix failures, then stop and report. Start Mode B only after that.
- Code comments are short and say why a constraint exists, not what the line does.
- README and comments use plain, direct English. No marketing language.
- Never invent results. Every number the API returns comes from the submitted scenario.

## What EquiReach is

EquiReach is a reusable decision tool. A planner submits a scenario. The engine returns who gets served, what each fairness rule costs, and whether the answer is proven optimal.

- Name: EquiReach. Tagline: Optimizing Resources. Advancing Fairness.
- Show both in the README title, in `/api/config`, and in the header of every export.
- It is not a report about one place. Do not embed, require or depend on any dataset, survey file or earlier result.
- Method: allocate fixed capacity to maximise the total priority of served clients. Impose lower-bound equity floors on groups among the served. Report the price of fairness (priority lost as a floor tightens). Report whether two floors are subadditive or superadditive in cost. Claim optimality only when the solver proves it.

## Two modes

- **Mode A, selection (default).** Clients only, no centers. Choose exactly K clients to maximise total priority, with equity floors. This is the method validated in the author's earlier study.
- **Mode B, siting (new).** Adds centers, a travel limit, capacity and cost. Run it only when centers are supplied and `params.mode` is `siting`.
- Mode B must never change Mode A results. If no centers are supplied, run Mode A.

## What is validated and what is new

Put this paragraph in the README and keep it true:

- Validated: selection with equity floors (exact MILP, price of fairness, interaction of floors), in the earlier study on MICS 2019 survey data from rural Chattogram. Repository: https://github.com/rafi-math/edu-support-optimization-bd
- New and tested only on synthetic data: center opening, assignment, travel limit, capacity, cost, and the heuristic fallback.
- Real siting needs real coordinates, distances, capacities and costs from the planner. Straight-line distance is not road or river distance.

## Stack

Python 3.11 or newer, FastAPI, uvicorn, numpy, scipy (1.9 or newer, for `scipy.optimize.milp`), pandas, pydantic v2, pytest. No database. No paid services. Do not write user data to disk or to logs.

Dockerfile: python slim image, listens on port 7860. CORS origins come from the environment variable `ALLOWED_ORIGINS` (comma-separated). Allow `*` only when `DEV=1`.

## Data model

A scenario is `params` + `clients[]` + `centers[]` + `rules[]`. Field names are fixed in `02_api_contract.md`.

**Clients** (one row per demand point)
- `client_id`: unique string.
- `label`: optional.
- `lat`, `lng`: required in Mode B only. Optional in Mode A.
- `priority_score` in [0, 1]: the objective weight. It may be a model probability, but then it must be out-of-fold, or `need_capture_pct` is inflated.
- `weight` > 0, default 1.
- `attributes`: any names and levels the planner defines (age_band, gender, disability, wealth_quintile, region, and so on). The rule builder reads whatever exists.
- `need_flag`: optional 0 or 1. Used only for evaluation metrics. Never in the objective or in a constraint.

**Centers** (Mode B)
- `center_id` (unique), `label`, `lat`, `lng`, `capacity` (positive integer), `build_cost` >= 0, `operate_cost_per_client` >= 0 (default 0), `tags`.

**Params**
- `mode`: `selection` or `siting`.
- `coverage_mode`: `exactly_K` or `at_most_K`. Default `exactly_K`, as in the validated runs. Clamp K to the eligible count with a warning. `at_most_K` lets a floor shrink coverage instead of changing who is served, so return a warning whenever it is used. Price-of-fairness and interaction sweeps always run with `exactly_K`.
- `coverage_target`: a count K or a percent of clients. Convert internally.
- `objective_weighting`: `unweighted` or `weighted`.
- `exact`: default true. `time_limit_s`: default 20.
- Mode B only: `max_travel_km` > 0; `center_mode` (`open_exactly_N` or `open_within_budget`); `n_centers` (needed for `open_exactly_N`); `max_centers` (optional cap for `open_within_budget`); `budget` >= 0 (null or 0 means unbounded; say so in a warning).
- `source`: `synthetic` or `imported`. `seed`: demo only.

**Rules** (zero or more)
- `rule_id`, `attribute`, `level`, `min_share_pct` in [0, 100], optional `max_share_pct`.
- Meaning: among the served, the share with attribute equal to level is at least the minimum. Shares use weight if weighted, else count.

## Allocation

Notation: v_i = priority_i × (weight_i if weighted, else 1). a_i = weight_i if weighted, else 1.

### Mode A (selection)

Variables: x_i in {0, 1}.

- Objective: maximise the sum of v_i x_i.
- Coverage: sum of x_i = K (`exactly_K`) or <= K (`at_most_K`).
- Each rule r with group G_r and minimum fraction f_r: sum over G_r of a_i x_i − f_r × (sum over all i of a_i x_i) >= 0. Optional cap c_r: the same expression with c_r, <= 0.
- With `exactly_K` and unweighted, this equals "at least ceil(f_r × K) served from the group". That is the count-floor form used in the validated study. Compute the ceiling with a small tolerance, ceil(f_r × K − 1e-9), so a value like 243.00000000000003 does not become 244. Write a test for this equivalence.
- Solver: `scipy.optimize.milp` (HiGHS), relative MIP gap 0, time limit `time_limit_s`.
- Size guard: exact mode only when clients <= 20,000 (make it a setting). Otherwise use the heuristic and say so in `warnings`.

### Mode B (siting)

Distance: haversine kilometers. A pair (client, center) is eligible if distance <= `max_travel_km`. A client with no eligible center is ineligible (report the count).

Variables: y_j in {0, 1} (center open); x_i in {0, 1} (client served); z_ij in {0, 1} (client i assigned to center j; create only for eligible pairs).

- Objective: maximise the sum of v_i x_i.
- Coverage: as in Mode A.
- Sum over j of z_ij = x_i. A served client has exactly one center. An unserved client has none.
- z_ij <= y_j. Only open centers serve.
- Sum over i of z_ij <= capacity_j × y_j.
- `open_exactly_N`: sum of y_j = `n_centers`. `open_within_budget`: sum over j of (build_cost_j y_j + operate_cost_j × sum over i of z_ij) <= budget when budget > 0, and sum of y_j <= `max_centers` when it is set.
- Rules: as in Mode A.
- Size guard: exact mode only when clients × centers <= 250,000 and K <= 2,000 and centers <= 30. Otherwise use the heuristic.

### Honesty rules

- `proven_optimal` is true only when HiGHS reports status optimal. Then `solver` is `scipy_highs_milp`.
- If the time limit stops the solve with a feasible solution, return that solution, `mip_gap` from the result, and `proven_optimal` false.
- The heuristic always returns `solver` `greedy_swap` and `proven_optimal` false.
- Never set `proven_optimal` to true without a proof. When nothing is feasible, return status `infeasible` and a warning. Never return a made-up number.

### Heuristic (fallback only)

Mode A:
1. Rank clients by v_i descending (tie-break `client_id` ascending). Take the top K.
2. Floor repair, per rule and then jointly. If a group's share is below its minimum, swap out the lowest-v served client whose removal does not break another met floor, and swap in the highest-v unserved client from the deficient group. Repeat until met or no valid swap. Make a second pass if a later rule broke an earlier one.
3. If a floor is still unmet: status `partial`, name the rule, give the achieved share.

Mode B:
1. Build the eligible sets.
2. Open centers. `open_exactly_N`: open N centers greedily, each time the one with the largest marginal served priority (per added cost if a budget is set). `open_within_budget`: keep opening the best marginal center while total cost stays within budget and coverage is not yet met.
3. Go through clients by v_i descending (tie-break `client_id` ascending). Assign each to the nearest open eligible center that still has capacity. Stop when K is reached.
4. The served set is the set assigned in step 3.
5. Floor repair as in Mode A, respecting capacity.

## Sweeps (through `POST /api/sweep`, never inside `/api/allocate`)

All sweeps run with `exactly_K`.

- **Price of fairness for one rule.** 12 steps of its minimum share, from 0 up to the largest feasible value (stop at the first infeasible step). Other rules are off unless `others` is `current`. For each step return `min_pct`, `objective`, `priority_loss_vs_zero` (objective at 0 minus objective at the step), `need_capture_pct` (if `need_flag` exists), `served_count`, `achieved_share_pct`, `status`, `proven_optimal`.
- **Interaction (needs 2 or more rules).** `cost_both` = loss when all rules are at their current minimums. `cost_each` = loss when only that rule is on at its minimum. `interaction` = `cost_both` minus the sum of the `cost_each` values. `epsilon` = 0.5% of `cost_both` (never below 1e-9). Label SUBADDITIVE if `interaction` < −epsilon, SUPERADDITIVE if > epsilon, else ADDITIVE. If `cost_both` <= 1e-9 the label is NO_COST. Also return the interaction as a percent of `cost_both`.
- Stop a sweep when `sweep_time_limit_s` (default 55) is reached. Return the steps finished and a `SWEEP_PARTIAL` warning.
- **Baseline comparison** (in `/api/allocate` when `flags.compare_baseline` is true): run again with all rules removed. Return both summaries and the per-metric difference. That difference is the realised price of fairness.

## Outputs

Field names are fixed in `02_api_contract.md`.

- `summary`: served_count, coverage_target, coverage_met, shortfall, eligible_count, ineligible_count, coverage_pct, total_priority_captured, mean_priority_served, need_capture_pct (only if `need_flag` exists; evaluation only), equity_compliance[{attribute, level, min_pct, achieved_pct, met}], cost_total, budget_used_pct (if a budget is set), centers_opened, center_ids, avg_travel_km, max_travel_km_used, p90_travel_km, solver, proven_optimal, mip_gap, status (`ok`, `partial` or `infeasible`). In Mode A the center and travel fields are null.
- `centers[]`: center_id, label, lat, lng, load, capacity, utilization, build_cost, operate_cost, tag. Empty in Mode A.
- `assignments[]`: every served client, in full.
- `unserved[]`: every other client, with state `eligible_unserved` or `ineligible`.
- `preview[]`: the top 50 served by priority.
- `warnings[]`: travel limit excludes X clients; budget binds before the target; rule unmet (achieved a% against b%); exact mode disabled because the instance is too large; `at_most_K` in use; time limit reached.

## Validation and edge cases

On import and on every run:

- lat in [−90, 90] and lng in [−180, 180] (Mode B only); priority_score in [0, 1]; weight > 0; capacity a positive integer; costs >= 0; unique ids.
- coverage_target >= 1. If it is above the eligible count, clamp and warn. `max_travel_km` > 0.
- A rule's attribute and level must exist on at least one client, otherwise return an error that names it. `min_share_pct` in [0, 100]. If the minimums of mutually exclusive levels of one attribute add up to more than 100%, flag it as impossible by construction.
- Handle zero eligible clients, zero centers, a budget too small to open any center, and infeasible floors by returning a status and warnings. Never crash. Never fake a number.
- Drop empty rows and report how many. A non-numeric value in a numeric column rejects that row with a reason.

## Synthetic demo (seeded, labelled synthetic)

`GET /api/demo?kind=selection|siting&seed=20261018` returns a scenario with `source` set to `synthetic`.

- About 1,200 clients and, for siting, about 25 centers inside a bounding box the caller can set. Include coordinates in both kinds so the map is not empty.
- Attributes: age_band, gender, disability, wealth_quintile.
- priority_score rises with age_band and with lower wealth, plus noise. need_flag is correlated with priority_score. Weights are about 1 with some variation. Centers have varied capacity and cost.
- Make the method easy to see: the raw top-K should lean toward older, higher-priority clients.
- Do not tune the generator to force any interaction sign. Return whatever the engine computes. Say in the README that the sign depends on the data.

## Command-line tool (for parity checks)

```
python -m equireach select --clients clients.csv --k 730 --rule "age_band:6-12:33.3" --rule "gender:female:40"
```

- The CSV needs `client_id` and `priority_score`. Optional: `weight`, `need_flag`. Every other column is an attribute.
- `--rule "attribute:level:percent"` sets a share floor. `--rule-count "attribute:level:N"` sets an exact count floor (valid with `exactly_K` only). Use the count form for parity checks, so percent rounding cannot change the result.
- Use exact Mode A. Print JSON with the objective to 6 decimals, served_count, achieved shares, status and proven_optimal.
- Purpose: the author reproduces published results on private data, on their own machine. That file never enters the repository.

## Tests (acceptance for the engine and API)

1. Mode A `exactly_K` returns exactly K served, or clamps with a warning. On instances with n <= 20 the exact result equals brute-force enumeration.
2. Adding a floor raises that group's share, or returns `partial` with the achieved share and a warning. It is never silently ignored.
3. The count-floor equivalence in Mode A holds (`exactly_K`, unweighted).
4. Sweep: loss at 0% is 0, and loss never decreases as a floor tightens (`exactly_K`).
5. Interaction is computed from real runs. The epsilon rule is applied and the percent of `cost_both` is returned.
6. The baseline comparison equals the realised difference.
7. `proven_optimal` is true only when HiGHS reports optimal. The heuristic is always false. A time-limited solve returns the best found with its gap.
8. Shuffling `need_flag` leaves the selection unchanged.
9. Bad lat, lng, score, cost or id values are rejected with row reasons. Empty, zero and infeasible inputs return a status, not a crash.
10. Mode B: a smaller `max_travel_km` lowers the eligible count and warns. Capacity is respected. Every served client has exactly one center. Open centers equal `n_centers` when `open_exactly_N`.
11. Imported data alone drives the results (test with a tiny scenario).
12. No user data is stored: no files written, no rows logged.
13. CLI output matches the API for the same input.
14. `pip install -r requirements.txt && pytest` passes, `uvicorn` starts, `GET /health` returns ok, and the Docker image builds.

## Deliverables

Create these in the repository:

- `equireach/`: models, validation, exact engine, heuristic engine, sweeps, synthetic generator, cli.
- `api/main.py`: FastAPI, with the endpoints in `02_api_contract.md`.
- `tests/`, `requirements.txt`, `Dockerfile`.
- `README.md` with: purpose, the validated-versus-new paragraph, the synthetic warning, run steps, API summary, the CSV template and column rules, both MILP formulations, the heuristic steps, the tests list, and the limits of straight-line distance.
- `docs/DEPLOY_HF_SPACES.md`: steps to deploy the Docker image to a Hugging Face Space, with the README front matter that sets the Docker SDK and port 7860.
- `docs/AI_USE_LOG.md`: a header only. The author fills it in.
- `templates/clients_template.csv` and `templates/centers_template.csv`: header row only.

## Do not

Return only a plan. Hardcode any result, metric or chart series. Depend on any external dataset or on an earlier project's numbers. Claim optimality without a proof. Use `need_flag` as an input to optimization. Add login, payment or a database. Use paid APIs. Build any user interface.
