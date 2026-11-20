from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import groww_auth_check


class RetiredGrowwIntegrationTests(unittest.TestCase):
    def test_entry_point_is_disabled_without_credentials_or_network(self):
        output = StringIO()
        with redirect_stdout(output):
            result = groww_auth_check.main()

        self.assertEqual(result, 2)
        self.assertIn("integration is disabled", output.getvalue())
        self.assertIn("does not use broker credentials", output.getvalue())

    def test_script_contains_no_authentication_or_transport_code(self):
        source = (PROJECT_ROOT / "scripts" / "groww_auth_check.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("GROWW_API_KEY", source)
        self.assertNotIn("urlopen", source)
        self.assertNotIn("ACCESS_TOKEN", source)


if __name__ == "__main__":
    unittest.main()
