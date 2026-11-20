# Private API guide

The API is served by `atlas.web.app:create_app`. All endpoints under `/api` require the owner session cookie. State-changing endpoints also require the CSRF cookie value in `X-CSRF-Token` and an exact allow-listed `Origin`. Login/bootstrap set the session and CSRF cookies and return the CSRF value to the same-origin UI.

## Public endpoints

- `GET /health/live`: process liveness only.
- `GET /health/ready`: database connectivity readiness.
- `GET /auth/bootstrap-status`: reports only whether owner setup is required.
- `POST /auth/bootstrap`: one-time bootstrap token, username and password; creates owner and session.
- `POST /auth/login`, `GET /auth/session`, `POST /auth/logout`.

## Owner endpoints

- `GET /api/dashboard`, `GET /api/market-data/status`.
- `GET/POST /api/watchlists`, `DELETE /api/watchlists/{id}`, `POST /api/watchlists/{id}/items`.
- `GET/POST /api/instruments`.
- `GET /api/strategies/builtins`, `GET/POST /api/strategies`.
- `POST/GET /api/backtests`.
- `POST/GET /api/portfolios`, `GET /api/portfolio/{id}`, `GET /api/trades`.
- `GET/PUT /api/risk/policy`.
- `POST /api/proposals`, `GET /api/orders`, `POST /api/orders/{id}/approve`, `POST /api/orders/{id}/reject`.
- `POST /api/orders/{id}/fill`, `POST /api/positions/{id}/close`.
- `POST/GET /api/agents/runs`, `GET /api/agents/runs/{id}/observations`.
- `GET /api/audit?limit=100`.

Backtests require a configured `SYMBOL.csv` and explicit source/synthetic environment values. The request specifies strategy, symbol, underlying, UTC start/end, timeframe, initial capital, quantity, unit multiplier, last-bar completeness, spread, slippage, and per-execution fee. Zero transaction costs are explicit inputs.

Proposal requests use a typed derivative contract, timezone-aware quote and signal timestamps, data source, synthetic flag, direction, strategy identity, entry/stop/target references, quantity, point value, and horizon. Configure a risk policy first. The API does not accept a client-supplied risk decision. Approval is a second request and re-evaluates server-side; current risk accounting supports only portfolios without existing positions.

Fill and close requests require a two-sided quote, timezone-aware timestamp, data source, synthetic/real provenance, and fee. Approved orders can be filled; filled positions can be partially or fully closed. FastAPI-generated OpenAPI docs are disabled by default.
