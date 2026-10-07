"""Regression tests for ``render_report`` in the CP4.2.1 sampling diagnostic.

These guard against the documented format-string defect where ``_fmt_pct(...)``
(a string such as ``"97.73%"``) was passed to a ``{0:.1f}`` placeholder, raising
``ValueError: Unknown format code 'f' for object of type 'str'``.

They run against the on-disk JSON artifact (no model weights, no generation),
skipping gracefully if the artifact is absent.
"""

import json
import os
import unittest

import run_cp4_2_sampling_diagnostic as diag

ARTIFACT_PATH = "data/pilots/cp4_2_sampling_diagnostic.json"


def _load_artifact():
    with open(ARTIFACT_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@unittest.skipUnless(
    os.path.exists(ARTIFACT_PATH),
    "diagnostic artifact not present; regression test skipped",
)
class TestRenderReportRegression(unittest.TestCase):
    def setUp(self):
        self.artifact = _load_artifact()

    def test_render_report_returns_string_without_error(self):
        report = diag.render_report(self.artifact)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)

    def test_nucleus_one_line_uses_numeric_not_pct_string(self):
        report = diag.render_report(self.artifact)
        # Before the fix this call raised ValueError; the formatted fraction
        # 129/132 -> 97.7 must appear as a numeric "{0:.1f}" result, i.e.
        # "(97.7% here)", not a crash and not a raw "97.73%" string substitution.
        self.assertIn("(97.7% here)", report)

    def test_render_report_stable_across_two_calls(self):
        first = diag.render_report(self.artifact)
        second = diag.render_report(self.artifact)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
