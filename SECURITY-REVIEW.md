# ATLAS Security Review

| # | Severity | File | Lines | Vulnerability | Confidence |
|---|----------|------|-------|---------------|------------|
| 1 | 🟡 MEDIUM | src/atlas/web/security.py | 95-151 | Login failure counters used read-modify-write transactions, allowing lost increments across concurrent requests or instances. | 10/10 |
| 2 | 🟡 MEDIUM | src/atlas/web/auth.py | 95-104 | Login limits were isolated by submitted username and socket IP, allowing identifier variation to create separate counters. | 9/10 |

## Finding 1: Atomic login throttling

**Severity:** 🟡 MEDIUM

**File and line:** [security.py](src/atlas/web/security.py#L95)

**Fix:** Replaced read-modify-write counters with PostgreSQL and SQLite dialect-specific atomic upserts. The owner-account counter and source-IP counter both increment and return the resulting count in the database itself. Expired rows are deleted and a single window is applied.

**Regression test:** [test_web_app.py](tests/test_web_app.py) includes:
- `test_login_rate_limit_is_enforced`
- `test_database_limiter_records_concurrent_failures`

**Remaining risk:** PostgreSQL concurrency should receive a further integration test with multiple actual processes. The local regression uses concurrent threads and SQLite. Render deployment should use the managed PostgreSQL database and observe real 429 behavior.

## Finding 2: Identifier bypass

**Severity:** 🟡 MEDIUM

**File and line:** [auth.py](src/atlas/web/auth.py#L95)

**Fix:** Login limiting now uses a fixed owner-account key and a direct source-IP key. Submitted usernames no longer create separate counters, and spoofed `X-Forwarded-For` values are ignored because the application does not derive IP identity from that header.

**Regression test:** [test_web_app.py](tests/test_web_app.py) includes `test_login_limit_cannot_be_reset_by_username_or_forwarded_ip`.

**Remaining risk:** Direct source-IP throttling may converge on one proxy address on hosted infrastructure. The fixed owner-account counter still prevents username-variation bypass. A trusted-proxy configuration is required if source-IP granularity is needed behind a proxy.

## Other controls

- Sessions use opaque random values and server-side SHA-256 digests.
- Cookies are HttpOnly and SameSite=Strict; secure-cookie mode is mandatory in production.
- State-changing routes require CSRF and exact allow-listed Origin.
- Passwords use the recommended Argon2 implementation.
- There is no registration route and owner bootstrap is one-time.
- No broker, live-order route, credentials, or live execution switch exists.
- Render configuration has no persistent application volume requirement; PostgreSQL owns durable database data.
- pgvector and Redis are not used by the current implementation.
