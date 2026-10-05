"""Release metadata that is easy to forget and awkward to discover later."""

from pathlib import Path
import unittest

from landlab_grid_validation import __version__

ROOT = Path(__file__).resolve().parents[1]


class TestPackaging(unittest.TestCase):
    def test_the_version_is_in_the_changelog(self):
        self.assertIn(__version__, (ROOT / "CHANGELOG.md").read_text())

    def test_the_license_and_citation_exist(self):
        self.assertIn("MIT License", (ROOT / "LICENSE").read_text())
        self.assertIn(f"version: {__version__}", (ROOT / "CITATION.cff").read_text())

    def test_the_version_is_declared_once(self):
        pyproject = (ROOT / "pyproject.toml").read_text()
        self.assertIn('dynamic = ["version"]', pyproject)  # setuptools reads __version__
        self.assertNotIn('version = "', pyproject.split("[tool.setuptools.dynamic]")[0])


if __name__ == "__main__":
    unittest.main()
