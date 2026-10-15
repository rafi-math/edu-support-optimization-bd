# Educational Support Optimization in Rural Bangladesh

This repository contains code and documentation for a two-phase analysis of educational vulnerability and support allocation in rural Bangladesh.

Phase 1 uses Bangladesh MICS 2019 children's data to estimate out-of-school risk scores. Phase 2 uses a budget-constrained allocation model to select children for a hypothetical educational support package.

## Scope

- Data source: Bangladesh MICS 2019, children's file `fs.sav`
- Analysis area: rural Chattogram Division
- Age range: 6–17 years
- Target variable: out-of-school status in the current school year
- Weight variable: `FSWEIGHT`

## Important data limitations

The MICS 2019 children's file does not contain household coordinates, school identifiers, upazila identifiers, travel distances, center construction costs, or center capacities. Therefore, this project does not claim to optimize physical support-center locations using observed MICS infrastructure data.

Any distance, cost, capacity, or budget value used in the optimization model is explicitly treated as a scenario parameter chosen by the analyst or policy maker, not as an observed MICS variable.

## Repository structure

```text
.
├── docs/
│   └── ai_use_log.md
├── src/
├── notebooks/
├── reports/
├── .gitignore
└── README.md
