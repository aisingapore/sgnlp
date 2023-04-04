import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)


class BaseEvaluationReport:
    """This is used as base class for derived StanceClassificationModelEvaluationReport and RumourVerificationModelEvaluationReport.

    Args:
            y_true (List[List[str]]): Groud truth.
            y_pred (List[List[str]]): Predictions.
            metrics (Dict[str, Union[float, int]]): _description_
            eval_file (Path, optional): Path of model evaluation file. Defaults to None.
            digits (int, optional): Number of digits for formatting floating point values in classification report. Defaults to 4.
    """

    def __init__(
        self,
        y_true: List[List[str]],
        y_pred: List[List[str]],
        metrics: Dict[str, Union[float, int]],
        eval_file: Path = None,
        digits: int = 4,
    ) -> None:
        self.metrics = metrics

        truth = np.concatenate(y_true)
        preds = np.concatenate(y_pred)
        self._calculate_acc_p_r_f1(truth, preds)

        report = self._generate_report(y_true=truth, y_pred=preds, digits=digits)
        logger.info("\n%s", report)

        if eval_file is not None:
            eval_file.write_text(report)

    def get_score(self) -> float:
        """Return Macro F1-score for StanceClassificationModel and accuracy for RumourVerificationModel."""
        raise NotImplementedError("Subclass should implement this.")

    def _generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        digits: int,
    ) -> str:
        """Generate a model evaluation report.

        Args:
            y_true (np.ndarray): Groud truth.
            y_pred (np.ndarray): Predictions.
            digits (int): Number of digits for formatting floating point values in classification report.

        Returns:
            str: Model evaluation report.
        """
        raise NotImplementedError("Subclass should implement this.")

    def _calculate_acc_p_r_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Calculate accuracy and macro precision, recall and F1 scores.

        Args:
            y_true (np.ndarray): Groud truth.
            y_pred (np.ndarray): Predictions.
        """
        self.accuracy = float(np.sum(y_pred == y_true)) / len(y_true)

        self.p_macro, self.r_macro, self.f_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro"
        )


class StanceClassificationModelEvaluationReport(BaseEvaluationReport):
    """Generate model evaluation report and scores by StanceClassificationModel."""

    def _generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        digits: int = 4,
    ) -> str:
        """Generate a evaluation report for StanceClassificationModel."""
        name_width: int = 0
        flatten_y_true: List[str] = [item for item in y_true]
        flatten_y_pred: List[str] = [item for item in y_pred]
        true_label_and_idx, pred_label_and_idx = set(), set()
        true_label_to_idx, pred_label_to_idx = defaultdict(set), defaultdict(set)

        for idx in range(len(flatten_y_true)):
            label = flatten_y_true[idx]
            true_label_and_idx.add((label, idx))
            true_label_to_idx[label].add(idx)

            name_width = max(name_width, len(label))

            label = flatten_y_pred[idx]
            pred_label_and_idx.add((label, idx))
            pred_label_to_idx[label].add(idx)

        micro_p, micro_r, micro_f1 = self._calculate_p_r_f1(
            y_true=true_label_and_idx, y_pred=pred_label_and_idx
        )

        last_line_heading = "macro avg"
        width = max(name_width, len(last_line_heading), digits)

        headers = ["precision", "recall", "f1-score", "support"]
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"

        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"

        avg_f1s = 0.0

        ps, rs, f1s, s = [], [], [], []
        for label in true_label_to_idx:
            true_idx = true_label_to_idx[label]
            pred_idx = pred_label_to_idx[label]
            nb_true = len(true_idx)

            p, r, f1 = self._calculate_p_r_f1(y_true=true_idx, y_pred=pred_idx)

            avg_f1s += f1

            report += row_fmt.format(
                *[label, p, r, f1, nb_true], width=width, digits=digits
            )

            ps.append(p)
            rs.append(r)
            f1s.append(f1)
            s.append(nb_true)

        report += "\n"

        report += row_fmt.format(
            "micro avg",
            micro_p,
            micro_r,
            micro_f1,
            np.sum(s),
            width=width,
            digits=digits,
        )
        report += row_fmt.format(
            last_line_heading,
            np.average(ps, weights=s),
            np.average(rs, weights=s),
            np.average(f1s, weights=s),
            np.sum(s),
            width=width,
            digits=digits,
        )

        report += "\nmacro f1: " + str(avg_f1s / 4)

        report += f"\nAccuracy: {self.accuracy}"

        return report

    def _calculate_p_r_f1(
        self,
        y_true: Set,
        y_pred: Set,
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall and F1-score for StanceClassificationModel.

        Args:
            y_true (Set): Groud truth.
            y_pred (Set): Predictions.

        Returns:
            Tuple[float, float, float]: Precision, recall and F1-score.
        """
        nb_correct = len(y_true & y_pred)
        nb_pred = len(y_pred)
        nb_true = len(y_true)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        return p, r, f1

    def get_score(self) -> float:
        """Get macro F1-score for StanceClassificationModel."""
        return self.f_macro


class RumourVerificationModelEvaluationReport(BaseEvaluationReport):
    """Generate evaluation report and scores for RumourVerificationModel."""

    def _generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        digits: int,
    ) -> str:
        """Generate a evaluation report for RumourVerificationModel."""
        report = {
            "precision": self.p_macro,
            "recall": self.r_macro,
            "f_score": self.f_macro,
        }
        report.update(self.metrics)
        return str(report)

    def get_score(self) -> float:
        """Get accuracy score for RumourVerificationModel."""
        return self.metrics["eval_accuracy"]
