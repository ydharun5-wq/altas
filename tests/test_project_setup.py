from pathlib import Path
import importlib
import sys
import tomllib
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atlas
from atlas.indicators import atr, ema, rsi


class ProjectSetupTests(unittest.TestCase):
    def test_atlas_package_can_be_imported(self):
        self.assertIsNotNone(atlas)
        self.assertTrue(all(callable(item) for item in (atr, ema, rsi)))

    def test_all_public_package_boundaries_import_together(self):
        modules = (
            "atlas.application",
            "atlas.analytics",
            "atlas.backtesting",
            "atlas.data",
            "atlas.derivatives",
            "atlas.indicators",
            "atlas.market_data",
            "atlas.paper_trading",
            "atlas.risk",
            "atlas.sizing",
            "atlas.strategies",
        )
        for module_name in modules:
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_pyproject_discovers_src_layout_packages(self):
        root = Path(__file__).resolve().parents[1]
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(config["build-system"]["build-backend"], "setuptools.build_meta")
        self.assertEqual(config["tool"]["setuptools"]["packages"]["find"]["where"], ["src"])
        self.assertIn("atlas*", config["tool"]["setuptools"]["packages"]["find"]["include"])


if __name__ == "__main__":
    unittest.main()
