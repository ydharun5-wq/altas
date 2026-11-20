# Development

## Environment

- Windows and Python 3.12 are the supported development setup.
- Create `.venv` with `py -3.12 -m venv .venv` when needed.
- Install the package and test client dependencies with
  `\.venv\Scripts\python.exe -m pip install -e ".[test]"`.
- The engine's web layer requires FastAPI, SQLAlchemy 2.x, Alembic, Psycopg,
  Pydantic Settings, Uvicorn, and `pwdlib[argon2]`. See `pyproject.toml` for
  supported ranges. `.env.example` contains local setup values; `.env` is
  ignored and must not be committed.
- Run `\.venv\Scripts\alembic.exe upgrade head` before starting the API.
- Run the local API with `\.venv\Scripts\atlas-api.exe`. It binds to loopback.
- The frontend uses Vite 8, React, TypeScript, and Tailwind 4. Install and run
  it from `frontend/` with `npm install` and `npm run dev`. Use Node 20.19+ or
  22.12+ as required by the [Vite guide](https://vite.dev/guide/).

## Validation

Run the focused repository suite from the root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

For a packaging smoke check, install into the active environment and verify
that `import atlas` works without adding `src` to `sys.path`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -c "import atlas; print(atlas.__file__)"
```

The backtest and strategy research engines must receive a
`ValidatedMarketDataProvider`. Add new provider adapters behind that boundary.
Every quote source must preserve timestamps, contract identity, source
identity, and synthetic-data provenance. Missing or invalid inputs must fail
closed.

All market fixtures used in tests must be clearly synthetic. Do not describe
fixtures as historical market performance. Backtest execution assumptions
must remain explicit; zero-cost mode must be selected by name.

The API tests use FastAPI's `TestClient` and a temporary SQLite database. Do not
run the development server with multiple workers until the process-local login
limiter is replaced with shared state. Run migrations to change schema; do not
use `Base.metadata.create_all()` in a running deployment.

## Paper-only rules

- Do not add broker clients, credential loading, live-order methods, or a live
  trading configuration switch.
- Do not let a strategy, AI component, or sizing result submit or approve a
  paper order. The host must obtain user approval, and M8.1 risk checks remain
  mandatory after approval. Web approval reruns sizing and risk; it does not
  create a fill.
- Use only local data the operator is authorized to use, and set provenance
  accurately. Do not infer real-data status from a filename or provider name.
- Keep option pricing assumptions explicit and document any approximation.
