# Architecture

ATLAS is split into small Python packages. All market and execution behavior
is local and deterministic; no module in `src/atlas` calls a broker or sends
orders to an external service.

## Research and backtest flow

1. `atlas.data.ohlcv` parses local CSV bars and rejects malformed OHLCV rows.
2. `atlas.market_data` exposes a provider protocol. The CSV and mock providers
   supply history; `ValidatedMarketDataProvider` checks each response before
   strategy or backtest consumers receive it.
3. `atlas.strategies` calculates deterministic research signals from a
   validated history prefix. Signals are descriptions of conditions, not
   executable orders.
4. `atlas.backtesting` evaluates only bars through the current candle. It
   queues close signals for the next bar, models explicit costs, and never
   liquidates a final open position automatically.

The CSV provider loads each configured file at construction. It supports
historical OHLCV only and raises `UnsupportedRequestError` for derivatives
quotes. The caller supplies `data_source` and `is_synthetic`; the provider
retains the source label and the response carries the synthetic flag. Neither
label is independently verified.

## Risk and paper-ledger flow

`atlas.sizing` reuses the Decimal formulas in `atlas.risk` to propose a whole
contract quantity. Sizing does not approve a trade. `evaluate_pre_trade_risk`
requires explicit user approval and independently evaluates the configured
policy, account snapshot, signal, quote provenance, price references, exposure,
daily loss, drawdown, and position limits.

`atlas.paper_trading` stages opening proposals pending approval. Its approval
operation resizes the exact proposal and reruns the risk evaluator against a
fresh ledger snapshot before the order becomes eligible for simulated fills.
Fills require explicit quote events, validate quote time and provenance, and
apply caller-configured slippage and fee assumptions. The ledger tracks cash,
positions, fills, realized and unrealized P&L, and closed-trade history in
memory.

The approval flag is an application-level handoff. A future UI or API must
only set it after an actual user action; this library cannot authenticate the
actor that calls a Python method. No broker or live-order interface exists.

## Derivative analytics

`atlas.derivatives` validates index futures/options contracts and quote models.
`atlas.analytics` contains descriptive futures analytics and European
Black-Scholes option pricing, Greeks, implied-volatility inversion, option
chain aggregation, portfolio exposure, and model scenario repricing. Inputs
such as rates, dividends, volatility, expiry instants, position quantities,
and provenance are explicit. Model results are not executable prices or
forecasts; exchange-specific exercise, margin, assignment, and settlement
rules are outside this milestone.

## Packaging and dependencies

The project uses the `src/atlas` package layout. `pyproject.toml` declares
setuptools package discovery for installable imports. The quantitative engine
is deterministic and separate from the optional FastAPI application package.

## Private web application

The UI is React/TypeScript/Vite under `frontend/`. FastAPI routes in
`src/atlas/web` authenticate the single local owner before exposing sensitive
research, portfolio, proposal, order, agent-task, or audit data. There is no
registration, broker adapter, or external order endpoint. SQLAlchemy 2.x
models and Alembic migrations persist the owner/session, watchlists, instruments,
strategies, backtests/results, portfolios, positions, paper orders/trades, risk
decisions, agent task records, and audit events. Compose uses Postgres; local
development defaults to SQLite.

Initial owner setup is guarded by a one-use secret. Passwords use the
recommended Argon2 hasher from `pwdlib`; sessions are opaque random values with
only their digest stored server-side. Writes require a matching CSRF cookie and
header plus an exact allow-listed `Origin`. Cookie Secure is mandatory in
production settings. SQLAlchemy sessions are request-scoped; SQLAlchemy
documents this as session-per-thread/task [in its Session Basics guide](https://docs.sqlalchemy.org/en/20/orm/session_basics.html).

The web backtest route uses the existing validated CSV provider and
`BacktestEngine`; it requires explicit data source and synthetic status, rejects
future end times, and records the declared last-candle completeness. Results
and provenance are persisted. There is no external market-data feed.

Paper proposals are converted to existing typed risk inputs. A proposal must
fit the configured position-sizing limit, then the API reruns the deterministic
risk evaluator on the user's approval action. It rejects stale quotes and
blocks risk approval for portfolios with positions until mark-to-market state
is reconciled. Approved web orders can be filled with explicit two-sided quotes,
and filled positions can be partially or fully closed. Fill/close records are
persisted in the paper trade and position tables; account equity and restart
recovery are not yet reconciled to these rows.

Agent tasks persist as queued, bounded-budget, `propose_only` requests; there is
no AI worker. Login throttling is process-local and Compose runs one API worker.
Redis is not configured because current features do not use it. See
[`SECURITY.md`](SECURITY.md), [`API.md`](API.md), and the implementation
checklist for boundaries and remaining work.
