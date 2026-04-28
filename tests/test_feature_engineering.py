"""
tests/test_feature_engineering.py
-----------------------------------
Unit tests for core/feature_engineering.py.

Test cases to implement:
  - compute() returns a DataFrame with the expected column set
  - compute() produces no NaN values past the warm-up period
  - compute_latest() returns a single-row Series matching the last row of
    compute() on the same input
  - validate_no_lookahead() passes on correctly shifted features
  - validate_no_lookahead() fails (raises) when an unshifted close is included
  - Rolling stats (vol, z-score) use .shift(1) so bar T uses only bars < T
  - RSI value is within [0, 100]
  - ATR is positive for all non-NaN rows
  - Output is deterministic (same input → same output)
  - compute() handles missing bars (NaN rows in input) without crashing
"""
