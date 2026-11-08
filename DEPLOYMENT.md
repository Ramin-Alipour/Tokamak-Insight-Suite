# Deployment

## Streamlit

This repository is structured for Streamlit Community Cloud or another Streamlit-capable host:

1. Select this repository and the `main` branch.
2. Set the main app file to `app.py`.
3. Install packages from `requirements.txt`.
4. Deploy; the theme is configured in `.streamlit/config.toml`.

The deployment provider account and public app URL are controlled by the repository owner; no hosted application is claimed until a deployment is actually created and verified.

## Docker

Build and run the included image:

```bash
docker build -t tokamak-insight-suite .
docker run --rm -p 8501:8501 tokamak-insight-suite
```

Open `http://localhost:8501`. The container includes a Streamlit health check.

## Validation checklist

- [ ] Run `pytest -q`
- [ ] Start the app with `streamlit run app.py`
- [ ] Test file upload using a copy of the included sample dataset
- [ ] Compare derived values against an independent analytical/reference dataset
- [ ] Confirm instrument-specific units, calibration, geometry, and uncertainty budgets
- [ ] Validate each intended workflow on the actual diagnostic and device
