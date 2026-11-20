# ATLAS V1

ATLAS is a private local quantitative research and paper-trading workspace. It has no broker adapter, broker credentials, live order route, or live-trading switch. Strategies and agents cannot approve risk or submit an order. Human approval and the deterministic risk evaluator are required for paper proposals.

## Existing research engine

The `src/atlas` engine includes validated OHLCV ingestion and local CSV history, EMA/RSI/ATR indicators, EMA trend and RSI mean-reversion strategies, futures/options analytics, Black-Scholes prices and Greeks, implied volatility, event-driven backtesting, explicit spread/slippage/fees, Decimal risk checks, position sizing, and an in-memory paper ledger. Its data-source limitations and assumptions are in [Architecture](docs/ARCHITECTURE.md).

The private web application adds owner bootstrap/authentication, revocable server-side sessions, CSRF protection, database migrations, persisted research metadata, backtest results, watchlists, instruments, strategies, paper proposals, portfolio records, queued agent tasks, and audit events. The browser UI is in `frontend/`; local Postgres and reverse-proxy deployment are in `docker-compose.yml`.

## Local development

Python 3.12 is required. From PowerShell in this repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
$env:ATLAS_BOOTSTRAP_TOKEN = [Convert]::ToBase64String((1..48 | ForEach-Object { Get-Random -Maximum 256 }))
.\.venv\Scripts\alembic.exe upgrade head
.\.venv\Scripts\atlas-api.exe
```

The local default database is SQLite at `atlas.db`. Open another terminal and run the UI with Node.js 22.12+:

```powershell
Set-Location frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The first visit asks for the one-time bootstrap token and creates the only owner account. The API is at `http://127.0.0.1:8000`; interactive API docs are disabled by default. Vite’s current Node requirement is documented in its [official guide](https://vite.dev/guide/).

Run the repository suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Local Docker Compose

Copy `.env.example` to `.env`, replace `ATLAS_DB_PASSWORD` with a long URL-safe random value and `ATLAS_BOOTSTRAP_TOKEN` with a different random value of at least 32 characters, then run:

```powershell
docker compose up --build
```

The UI is available at `http://localhost:8080`, bound to loopback only. PostgreSQL data persists in the `atlas-postgres` volume. The Compose API waits for PostgreSQL and applies Alembic migrations before serving. Compose does not publish the database or API ports to the host. Set `ATLAS_COOKIE_SECURE=true` only when serving the UI through HTTPS.

## Historical data

No market-data service is configured. For local CSV backtests, set all three variables before starting the API: `ATLAS_MARKET_DATA_DIR`, `ATLAS_MARKET_DATA_SOURCE`, and `ATLAS_MARKET_DATA_IS_SYNTHETIC` (`true` or `false`). Each file must be named `SYMBOL.csv`. `false` is an operator assertion that the local data is authorized real history; ATLAS does not independently verify it. The bundled sample file is a small synthetic parser fixture, not a research dataset. Compose mounts `./data` read-only at `/data`.

## Current scope and limits

- The API stores and risk-reviews paper proposals and persists explicit paper fills/closes with source and synthetic provenance. The existing Python paper ledger remains the primary deterministic quote-driven state machine.
- Web account equity, daily P&L, drawdown, and reconciled mark-to-market are not connected to fills yet.
- Risk review uses the explicitly configured policy and owner-supplied derivative quote. It blocks evaluation when a portfolio already has positions because the web layer has no reconciled mark-to-market snapshot.
- The login rate limiter is process-local and intended for the single local API worker. A multi-worker or shared deployment needs a shared limiter before exposure beyond loopback.
- Agent tasks persist as queued and propose-only; there is no AI provider or autonomous agent execution.
- Derivative quotes are not obtained from a provider. The owner submits timestamp, source, and synthetic status; those labels are not externally verified.

Read [Security](docs/SECURITY.md), [API](docs/API.md), [Development](docs/DEVELOPMENT.md), and the [remaining checklist](docs/IMPLEMENTATION_CHECKLIST.md) before extending the service.
