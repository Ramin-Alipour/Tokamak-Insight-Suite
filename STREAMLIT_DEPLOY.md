# Deploy Tokamak Insight Suite on Streamlit Community Cloud

The repository already contains the Streamlit entrypoint `app.py` and declares its Python dependencies in `pyproject.toml`. `requirements.txt` uses `-e .`, so Community Cloud installs the project and its declared runtime dependencies.

## Deploy

1. Sign in at https://share.streamlit.io/ with the GitHub account that has admin access to this repository.
2. Select **Create app** → **Yup, I have an app**.
3. Repository: `Ramin-Alipour/Tokamak-Insight-Suite`
4. Branch: `main` after this deployment-preparation pull request has been merged (or `deploy/streamlit-community-cloud` to test this branch).
5. Main file path: `app.py`
6. In Advanced settings, choose Python 3.11 or 3.12.
7. Click **Deploy** and wait for the build logs to report success.

Streamlit Community Cloud assigns the public app URL after the first deployment. That URL cannot be truthfully stated before the app is created in the Streamlit workspace.

## First-launch checks

- Confirm the Home page renders and the version string is shown.
- Open Diagnostic Catalog with no uploaded file.
- Open Public Data Access and test a known public MAST shot only if network access is available.
- Upload a small CSV and confirm Data Import & QC, Time Series & MHD, and Reports & Export work.
- Check the app logs if dependency installation or runtime startup fails.

## Scientific validity

A successful cloud launch proves that the UI starts; it does not validate every diagnostic calculation. Use the project's tests and validation notes for algorithm-specific evidence. Public data access depends on external archive availability and network policy. Never upload private experimental data to a public deployment unless you intend it to be accessible under the app's privacy settings.

Official deployment instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
Official dependency guidance: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
