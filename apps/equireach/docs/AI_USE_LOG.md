# AI use log

The author fills in how AI tools were used on this project.

- 2026-10-27: Cursor Agent generated Mode A engine, CLI, and tests from the spec files. I checked that files stayed inside apps/equireach, verified ignore rules, ran pytest, and committed the checkpoint.

- 2026-10-27: Ran the five Mode A parity checks locally against the restricted Phase 3 input using a private extract kept outside the repo. All five objectives matched the published Phase 3 summary to 6 decimals with proven_optimal = True. I recorded the aggregate result in docs/PARITY_VALIDATION.md and verified that no private rows or IDs entered the repo.
