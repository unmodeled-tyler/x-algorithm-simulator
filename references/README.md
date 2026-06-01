# Architecture References

This directory contains copied reference implementations and architectural inspiration from X-style feed systems:

- `candidate-pipeline/`
- `grox/`
- `home-mixer/`
- `phoenix/`
- `thunder/`

They are not part of the runnable `xsim` simulator. The active project lives in:

- `xsim/`
- `simulator/`
- `tests/`

The references are intentionally excluded from the default lint and test loop because they have separate dependencies, language runtimes, and historical assumptions.
