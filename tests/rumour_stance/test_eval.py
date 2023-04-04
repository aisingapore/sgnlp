import unittest

from sgnlp.models.rumour_stance.modules.report import (
    RumourVerificationModelEvaluationReport,
    StanceClassificationModelEvaluationReport,
)


class TestStanceClassificationModelEvaluationReport(unittest.TestCase):
    def setUp(self) -> None:

        y_true = [["DENY"], ["SUPPORT"], ["QUERY"], ["COMMENT"], ["COMMENT"]]
        y_pred = [["DENY"], ["SUPPORT"], ["QUERY"], ["COMMENT"], ["QUERY"]]

        self.report = StanceClassificationModelEvaluationReport(
            y_true=y_true, y_pred=y_pred, metrics={"eval_accuracy": 0.5}
        )

    def test_get_score(self):
        self.assertEqual(self.report.get_score(), 0.8333333333333333)


class TestRumourVerificationModelEvaluationReport(unittest.TestCase):
    def setUp(self) -> None:

        y_true = [["DENY"], ["SUPPORT"], ["QUERY"], ["COMMENT"], ["COMMENT"]]
        y_pred = [["DENY"], ["SUPPORT"], ["QUERY"], ["COMMENT"], ["QUERY"]]

        self.report = RumourVerificationModelEvaluationReport(
            y_true=y_true, y_pred=y_pred, metrics={"eval_accuracy": 0.5}
        )

    def test_get_score(self):
        self.assertEqual(self.report.get_score(), 0.5)
