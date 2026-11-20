# Render deployment

## Architecture

Deploy two private-owner web services and one managed PostgreSQL database:

- `atlas-api`: FastAPI Docker service
- `atlas-ui`: React/Nginx Docker service
- `atlas-postgres`: managed PostgreSQL

Redis is not required. pgvector is not used by the current application.
The application and migration remain paper-only.

## Prepare GitHub

1. Create or identify the GitHub repository remote.
2. Confirm `.env`, `atlas.db`, virtual environments, and `node_modules` are excluded.
3. Commit and push only after explicit user authorization.
4. In Render, create a Blueprint from this repository and select `render.yaml`.

## Required Render environment

Set `ATLAS_ALLOWED_ORIGINS` as a JSON array containing every exact HTTPS UI origin,
for example:

```json
["https://atlas.example.com"]
```

Set `ATLAS_BACKEND_URL` to the exact HTTPS API origin, for example:

```text
https://atlas-api.example.com
```

Configure `ATLAS_MARKET_DATA_SOURCE` and `ATLAS_MARKET_DATA_IS_SYNTHETIC`
only for an authorized local or mounted history source. Do not use synthetic
fixtures as production historical performance evidence.

## Custom domain

Add the UI custom domain to `atlas-ui` and the API custom domain to `atlas-api`.
The DNS provider must configure the records Render displays for each service,
normally CNAME records. Keep the same registrable domain for UI and API where
possible, for example `atlas.example.com` and `api.atlas.example.com`, so secure
cookies remain same-site. HTTPS is issued only after DNS verification.

Do not report a domain as connected until Render marks the certificate and
domain healthy.

## Verification

After deployment, verify:

- `https://<api-domain>/health/live`
- `https://<api-domain>/health/ready`
- UI login and owner bootstrap
- Anonymous `/api/dashboard` returns `401`
- Logout revokes the session
- Paper fill and close routes require approval and preserve paper-only status
