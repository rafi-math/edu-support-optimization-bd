# EquiReach Mode A parity validation

Date: 2026-10-27
Engine: EquiReach Mode A (client selection), exact MILP via scipy.optimize.milp (HiGHS)
Environment: Python 3.13.4, scipy 1.18.1, numpy 2.5.4, pandas 3.1.0
Input: the project's private Phase 3 optimization input (4861 modeling-ready rows). This file is restricted MICS-derived data and is not redistributed in this repository.
Selection count K: 730
Objective: maximize the sum of out-of-fold risk scores of the selected children (unweighted).
Fairness floors: exact count floors under exactly_K coverage.

Result: all five scenarios reproduced the published Phase 3 objectives to 6 decimal places, with served_count = 730, status = optimal, and proven_optimal = True.

| Scenario | Young floor | Girls floor | Objective | Served | Status | Proven optimal |
|---|---:|---:|---:|---:|---|---|
| raw risk, no floors | 0 | 0 | 325.663745 | 730 | optimal | True |
| girls 40% only | 0 | 292 | 322.039208 | 730 | optimal | True |
| young 33% only | 243 | 0 | 280.045189 | 730 | optimal | True |
| young 33% + girls 40% | 243 | 292 | 276.474216 | 730 | optimal | True |
| young 33% + girls parity | 243 | 367 | 270.226040 | 730 | optimal | True |

Scope of this validation:
- Validated: the Mode A selection engine, checked against the published Phase 3 fairness grid on the restricted Chattogram rural input.
- Not validated here: Mode B siting, the FastAPI backend, and the public dashboard. Those are exercised on synthetic demo data only.

Reproduction note:
- The validation was run locally from a private parity extract derived from the restricted input. The extract and the runner scripts are kept outside this repository and are not committed.
- The aggregate targets above match docs/data/phase3_fairness_summary.csv in this repository.
