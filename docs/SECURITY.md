# Security model and limits

## Enforced controls

- ATLAS is paper-only: no broker integration, credential fields, live order API, or live mode switch.
- One owner is created through a secret-protected bootstrap endpoint. The owner primary key is fixed to `1`; database uniqueness/checks prevent a second account. Public registration is absent.
- Password hashes use `pwdlib`'s recommended Argon2 implementation. Login errors are generic, and login attempts are throttled.
- Browser sessions are random opaque tokens. The database stores only a SHA-256 digest; sessions have an expiry and can be revoked.
- The session cookie is HttpOnly and SameSite=Strict. CSRF cookies are paired with a matching request header and exact allow-listed Origin for state changes. Production refuses to start if `ATLAS_COOKIE_SECURE` is false.
- Dashboard, market-data status, watchlists, instruments, strategies, backtests, portfolios, risk settings, proposals, orders, agents, and audit APIs require the owner session. Only liveness/readiness and the minimal bootstrap-status response are unauthenticated. API docs/OpenAPI are off by default and never enabled in production.
- Owner approval is separate from risk clearance. The approval endpoint rebuilds typed inputs, checks quote freshness, calls the existing sizing engine and risk evaluator, then persists approval only when both pass. AI tasks have no approval identity or order permissions.
- Portfolio/order/trade database constraints fix trading mode and order/trade type to `paper`. Nginx adds browser security headers; Compose binds the UI to loopback and keeps API/database ports internal.

FastAPI's security tutorial recommends Argon2 through `pwdlib`; the implementation follows that library guidance ([FastAPI documentation](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/)). CSRF checks use a double-submit token plus exact Origin validation, consistent with OWASP guidance to reject untrusted origins ([OWASP CSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html)).

## Operational limits

- Login throttling is in-process. Use one API worker as configured; do not scale workers or expose this app remotely until throttling is shared and abuse controls are reviewed.
- The bootstrap secret is supplied through local environment configuration. Keep it private and remove it from the shell after first setup. Owner creation makes the token unusable again.
- Local HTTP is supported only for loopback development. Use HTTPS and `ATLAS_COOKIE_SECURE=true` for any deployment that is not local HTTP.
- `is_synthetic` and quote/data source strings are caller assertions. ATLAS cannot verify a local file or manual quote's real-world origin or license.
- The API currently blocks risk approval when the portfolio contains positions because it cannot produce reconciled market value, realized/unrealized P&L, and underlying exposures.
- Web-approved paper orders require separate explicit fill/close quote requests and persist paper fills/trades/positions. Restart recovery, account equity reconciliation, transaction-cost configuration, and mark-to-market remain open work; don't interpret `approved` as a completed trade.
- Agent tasks are stored but not executed. No model credentials or autonomous AI execution path is present.
- Do not store secrets in audit details, structured logs, strategy JSON, agent inputs, or source control.
