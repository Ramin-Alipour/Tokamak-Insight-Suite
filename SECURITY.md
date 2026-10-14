# Security policy

## Supported versions
Security fixes target the current default branch. There is no guarantee for old ZIPs or untagged snapshots.

## Reporting a vulnerability
Please do not publish exploit details in a public issue. Contact the repository maintainer privately through GitHub's security advisory mechanism. Include the affected file/function, reproduction steps, impact and any safe proof of concept.

## Current safeguards and limits
- Public URL imports reject non-global destinations, pin validated DNS addresses to sockets, validate redirects and disable environment proxies.
- Tabular import is byte-limited; XLSX archive expansion and Parquet row/row-group counts are bounded.
- Streamlit binds to localhost by default. Deployments exposed to a network must add authentication, TLS and an appropriately restrictive reverse proxy.
- TIS is a research analysis tool, not a control, interlock or safety system.
