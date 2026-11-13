# Portable CPU analysis

- `reporting.py`: 39-variant behavioral tables and abstention discrimination.
- `interventions.py`: probability/log-odds effects for all saved causal controls.
- `figures.py`: publication-quality figures from those measurements.
- `paths.py`: account-independent workspace selection via `SHAME_WORKSPACE`.

Use [reproduction/run.py](../reproduction/run.py) to obtain and verify all required
inputs. These scripts are adapted from the original paper-analysis snapshot only
for portable paths and a stable output directory. The underlying numerical
calculations are preserved. Frozen original source and broader all-layer analyses
remain in the mechanism evidence archive.

Run `python reproduction/run.py all` from the repository root. Results are written
to `.artifacts/workspace/outputs/reporting_v1/`, never into the checked-in snapshot.
