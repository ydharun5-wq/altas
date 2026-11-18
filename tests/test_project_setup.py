from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import atlas


class ProjectSetupTests(unittest.TestCase):
    def test_atlas_package_can_be_imported(self):
        self.assertIsNotNone(atlas)


if __name__ == "__main__":
    unittest.main()
