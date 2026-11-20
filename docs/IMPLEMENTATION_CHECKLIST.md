# Remaining implementation checklist

This checklist separates completed local research/paper components from work
that needs a separate design and authorized inputs.

## Completed in this milestone

- [x] `src` package discovery and install metadata.
- [x] Validated local OHLCV ingestion and a CSV history provider.
- [x] Explicit synthetic-data provenance for local and mock providers.
- [x] EMA, RSI, ATR, two research strategies, and ATR stop references.
- [x] Derivative contracts/quotes, futures analytics, Black-Scholes Greeks,
  implied volatility, option-chain and position exposure analytics.
- [x] Deterministic event-driven backtesting, explicit spread/slippage/fees,
  and fixed-strategy walk-forward windows.
- [x] Decimal risk checks and sizing, user-approval requirement, and the
  in-memory paper order/position/trade ledger.
- [x] Local paper-ledger state transition logging and typed validation errors.
- [x] Retired the credential-based Groww authentication utility.
- [x] Added FastAPI application factory, owner-only bootstrap, Argon2 password
  hashes, opaque revocable sessions, HttpOnly cookies, CSRF double-submit and
  allow-listed Origin checks, login throttling, audit events, and health routes.
- [x] Added SQLAlchemy 2.x models and Alembic initial migration for web,
  research, paper and agent records; PostgreSQL Compose and local SQLite setup.
- [x] Added authenticated watchlist/instrument/strategy/backtest/risk/proposal/
  portfolio/order/agent/audit API routes and a React/TypeScript responsive UI.
- [x] Connected web backtests to the existing validated local CSV provider and
  event-driven `BacktestEngine`, with explicit data provenance and costs.
- [x] Connected paper approval to existing position sizing and M8.1 risk checks;
  quotes are freshness checked and approval cannot be client asserted.
- [x] Added focused web authentication/CSRF/session/rate-limit tests.

## Remaining implementation checklist

- [ ] Add an authorized historical-data adapter and an authorized paper quote
  adapter. Define supported instruments, source identity, timestamp semantics,
  freshness limits, and provenance before implementation. No external data
  source is configured by this repository today.
- [ ] Persist/replay and recover the existing paper ledger through restart
  recovery and reconciled account equity. Explicit web fills/closes, positions,
  fees, and trade history are now persisted.
- [ ] Reconcile web account equity, daily P&L, drawdown, open positions, and
  exposures from persistent fills and fresh marks. Current proposal approval
  fails closed when positions exist.
- [ ] Define concurrency behavior for simultaneous order approvals and fills;
  web proposals currently serialize to one active order per portfolio, while
  the original in-memory engine is single-process and not thread-safe.
- [ ] Replace process-local login throttling with a shared, persistent limiter
  before multi-worker or remote deployment.
- [ ] Configure structured log handlers, retention, and operator-facing audit
  export. Current logs are JSON request records without retention/export.
- [ ] Add strategy configuration/version storage and explicit validation
  review before promoting research configurations into paper use.
- [ ] Implement a durable agent worker only after permissions, timeout/retry,
  budget accounting, structured outputs, and approval gates have dedicated
  review. Current tasks are queued and not executed.
- [ ] Add frontend/API contract tests, migration upgrade/downgrade tests, and
  browser E2E coverage for auth, persistence, and restart recovery.
- [ ] Generate and commit a frontend lockfile after running npm in a working
  Node environment.
- [ ] Define exchange calendars, corporate-action handling, contract roll
  handling, option exercise/assignment, expiry settlement, and margin before
  claiming exchange-faithful simulation.
- [ ] Run compilation, imports, packaging/installability, migrations, full
  unittest suite, frontend build, Compose startup, restart-persistence checks,
  and browser E2E tests in working Python/Node/Docker/Postgres environments.
