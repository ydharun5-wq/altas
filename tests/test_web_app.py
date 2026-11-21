"""Private web boundary tests; all auth and database state is local."""

from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from sqlalchemy import select
from fastapi import HTTPException

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.web.app import create_app
from atlas.web.config import Settings
from atlas.web.database import Base
from atlas.web.models import LoginAttempt, PaperOrder, Position
from atlas.web.security import login_limit_keys, record_login_failure


class WebApplicationSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        database_url = f"sqlite:///{Path(self.temp.name) / 'web-test.db'}"
        self.settings = Settings(
            database_url=database_url,
            environment="test",
            bootstrap_token="test-bootstrap-token-that-is-long-enough-123456",
            allowed_origins=("http://testserver",),
            login_max_failures=2,
            login_window_seconds=900,
        )
        self.app = create_app(settings=self.settings)
        Base.metadata.create_all(self.app.state.engine)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def _bootstrap(self):
        return self.client.post(
            "/auth/bootstrap",
            headers={"Origin": "http://testserver"},
            json={
                "bootstrap_token": self.settings.bootstrap_token,
                "username": "owner",
                "password": "a-local-test-password-123",
            },
        )

    def test_private_api_rejects_anonymous_requests(self):
        self.assertEqual(self.client.get("/api/dashboard").status_code, 401)
        self.assertEqual(self.client.get("/api/orders").status_code, 401)
        self.assertEqual(self.client.get("/auth/bootstrap-status").json(), {"setup_required": True})

    def test_bootstrap_is_single_use_and_creates_owner_session(self):
        first = self._bootstrap()
        self.assertEqual(first.status_code, 201, first.text)
        self.assertIn("httponly", first.headers.get("set-cookie", "").lower())
        self.assertTrue(first.json()["csrf_token"])
        self.assertEqual(self._bootstrap().status_code, 409)
        self.assertEqual(self.client.get("/api/dashboard").status_code, 200)

    def test_owner_mutations_require_csrf_token(self):
        self.assertEqual(self._bootstrap().status_code, 201)
        payload = {"name": "research"}
        headers = {"Origin": "http://testserver"}
        self.assertEqual(self.client.post("/api/watchlists", headers=headers, json=payload).status_code, 403)
        csrf = self.client.get("/auth/session").json()["csrf_token"]
        headers["X-CSRF-Token"] = csrf
        self.assertEqual(self.client.post("/api/watchlists", headers=headers, json=payload).status_code, 201)

    def test_untrusted_origin_is_rejected_even_with_session(self):
        self.assertEqual(self._bootstrap().status_code, 201)
        response = self.client.post(
            "/api/watchlists",
            headers={"Origin": "http://attacker.invalid", "X-CSRF-Token": self.client.cookies["atlas_csrf"]},
            json={"name": "blocked"},
        )
        self.assertEqual(response.status_code, 403)

    def test_logout_revokes_server_side_session(self):
        self.assertEqual(self._bootstrap().status_code, 201)
        csrf = self.client.get("/auth/session").json()["csrf_token"]
        response = self.client.post("/auth/logout", headers={"Origin": "http://testserver", "X-CSRF-Token": csrf})
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.client.get("/api/dashboard").status_code, 401)

    def test_paper_fill_and_close_persist_position_and_trade(self):
        self.assertEqual(self._bootstrap().status_code, 201)
        csrf = self.client.get("/auth/session").json()["csrf_token"]
        headers = {"Origin": "http://testserver", "X-CSRF-Token": csrf}
        portfolio = self.client.post(
            "/api/portfolios",
            headers=headers,
            json={"name": "paper", "initial_cash": 100000},
        )
        self.assertEqual(portfolio.status_code, 201, portfolio.text)
        portfolio_id = portfolio.json()["id"]

        with self.app.state.session_factory() as db:
            order = PaperOrder(
                id="approved-paper-order",
                portfolio_id=portfolio_id,
                symbol="NIFTY-FUT-TEST",
                side="long",
                quantity=1,
                order_type="paper",
                status="approved",
                limit_price=Decimal("100"),
                proposal_json={
                    "portfolio_id": portfolio_id,
                    "contract": {
                        "exchange": "NSE",
                        "underlying": "NIFTY",
                        "instrument_type": "FUT",
                        "expiry": "2026-12-31",
                        "contract_id": "NIFTY-FUT-TEST",
                        "lot_size": 1,
                        "tick_size": "0.05",
                    },
                    "quote": {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "bid": "99",
                        "ask": "100",
                        "last_traded_price": "100",
                        "volume": 1,
                        "open_interest": 1,
                        "change_in_open_interest": 0,
                    },
                    "data_source": "synthetic-fixture",
                    "is_synthetic": True,
                    "direction": "long",
                    "strategy_name": "fixture",
                    "strategy_version": "1",
                    "observation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "evaluation_at": datetime.now(timezone.utc).isoformat(),
                    "timeframe_seconds": 300,
                    "entry_reference_price": "100",
                    "stop_reference_price": "90",
                    "target_reference_price": "110",
                    "quantity": 1,
                    "point_value": "1",
                    "horizon": "intraday",
                    "rationale": "synthetic fixture",
                },
            )
            db.add(order)
            db.commit()

        now = datetime.now(timezone.utc).isoformat()
        filled = self.client.post(
            "/api/orders/approved-paper-order/fill",
            headers=headers,
            json={
                "timestamp": now,
                "bid": "99",
                "ask": "100",
                "data_source": "synthetic-fixture",
                "is_synthetic": True,
                "fee": "1",
            },
        )
        self.assertEqual(filled.status_code, 201, filled.text)
        detail = self.client.get(f"/api/portfolio/{portfolio_id}", headers={"Origin": "http://testserver"})
        self.assertEqual(detail.status_code, 200)
        position_id = detail.json()["positions"][0]["symbol"]
        positions = detail.json()["positions"]
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["quantity"], 1)

        with self.app.state.session_factory() as db:
            position = db.scalar(select(Position).where(Position.portfolio_id == portfolio_id))
            position_id = position.id

        closed = self.client.post(
            f"/api/positions/{position_id}/close",
            headers=headers,
            json={
                "quantity": 1,
                "timestamp": now,
                "bid": "101",
                "ask": "102",
                "data_source": "synthetic-fixture",
                "is_synthetic": True,
                "fee": "1",
            },
        )
        self.assertEqual(closed.status_code, 201, closed.text)
        trades = self.client.get("/api/trades")
        self.assertEqual(trades.status_code, 200)
        self.assertEqual(len(trades.json()), 2)

    def test_login_rate_limit_is_enforced(self):
        self.assertEqual(self._bootstrap().status_code, 201)
        self.client.post("/auth/logout", headers={"Origin": "http://testserver", "X-CSRF-Token": self.client.cookies["atlas_csrf"]})
        body = {"username": "owner", "password": "incorrect-password"}
        for _ in range(2):
            response = self.client.post("/auth/login", headers={"Origin": "http://testserver"}, json=body)
            self.assertEqual(response.status_code, 401)
        blocked = self.client.post("/auth/login", headers={"Origin": "http://testserver"}, json=body)
        self.assertEqual(blocked.status_code, 429)

    def test_login_limit_cannot_be_reset_by_username_or_forwarded_ip(self):
        self.assertEqual(self._bootstrap().status_code, 201)
        self.client.post("/auth/logout", headers={"Origin": "http://testserver", "X-CSRF-Token": self.client.cookies["atlas_csrf"]})
        for username in ("owner", "another-name", "owner"):
            response = self.client.post(
                "/auth/login",
                headers={
                    "Origin": "http://testserver",
                    "X-Forwarded-For": f"192.0.2.{len(username)}",
                },
                json={"username": username, "password": "incorrect-password"},
            )
            if response.status_code == 429:
                break
            self.assertIn(response.status_code, (401, 429))
        self.assertEqual(response.status_code, 429)

    def test_database_limiter_records_concurrent_failures(self):
        self.assertEqual(self._bootstrap().status_code, 201)
        keys = login_limit_keys("test-source")

        def fail_once(_):
            with self.app.state.session_factory() as db:
                try:
                    record_login_failure(db, keys, self.settings)
                    return "recorded"
                except HTTPException as exc:
                    return str(exc.status_code)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(fail_once, range(12)))
        self.assertIn("429", results)
        with self.app.state.session_factory() as db:
            counts = [
                db.get(LoginAttempt, key).failure_count
                for key in keys
            ]
        self.assertTrue(all(count >= 12 for count in counts))


if __name__ == "__main__":
    unittest.main()
