# Contributing

## Before submitting a change
- Keep measured, derived and inferred quantities distinct.
- Document units, sign conventions, assumptions, calibration requirements and uncertainty limits.
- Add behavioral regression tests for reported defects; input-rejection tests alone are not sufficient.
- Use synthetic truth-known tests where possible, and label real-data integration tests separately.
- Do not describe catalog entries as implemented end-to-end workflows unless a runnable workflow and validation exist.

## Local validation
```bash
python -m pip install -e '.[test]'
python -m compileall -q app.py tis tests
ruff check --select E9,F63,F7,F82 app.py tis tests
pytest -q --cov=tis --cov-report=term-missing
```

Network-dependent MAST integration tests run on demand or on their scheduled cadence.
