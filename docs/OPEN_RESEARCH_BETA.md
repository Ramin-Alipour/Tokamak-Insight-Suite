# Open Research Beta Policy

Tokamak Insight Suite (TIS) is an open research toolkit intended to let tokamak researchers and analysts try, inspect, extend, and contribute to analysis workflows while the suite is being developed.

## Development principle

**Independent scientific validation is not a prerequisite for use, contribution, or release of an analysis module.** The project will work toward implementation coverage across the 80 diagnostic entries and associated analysis workflows, then improve them through reproducible community use, issue reports, shared test cases, and optional comparison with published or independently processed results.

## What remains required

Removing scientific validation as a gate does not mean removing software engineering safeguards. TIS continues to use automated tests, static checks, packaging checks, and CI to catch software defects. These tests do not establish that a model is physically appropriate for every device or discharge.

Each module should describe, as far as currently known:
- what it implements and what inputs it expects;
- units, assumptions, calibration needs, and instrument-response requirements;
- known limitations and unsupported use cases;
- enough provenance to reproduce a run when feasible.

## Community contributions

A reference result, publication, or raw-data benchmark is welcome but optional. Contributions should include reproducible parameters and provenance where possible. Do not upload confidential or restricted data. Community submissions are not automatically endorsed by the project.

## Interpretation

TIS is research software, not a tokamak control or safety system. Users are responsible for assessing whether a result is suitable for their scientific purpose. An implementation label describes software scope, not scientific approval or device-specific certification.
