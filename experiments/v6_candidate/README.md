# v6 archived source snapshot

This is the exact code and protocol used for the completed validation-only study. See [the measured report](../../docs/V6_VALIDATION.md) before interpreting results.

The `flycircuit_v6` package and `native_source` files retain their measured bytes; the portable verifier checks them against the original absolute-path plan. Original native training requires the original directory layout and processed FlyWire inputs, which are not redistributed. This archive is not a turnkey training release. Do not invoke archived launchers to resume v5.

The source-only unit checks run with `python -m unittest discover -s experiments/v6_candidate/tests` after adding this directory to PYTHONPATH. They do not establish performance or input-data availability.
