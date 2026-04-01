"""Statistical hypothesis tests for model comparison."""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from loguru import logger


class HypothesisTests:
    """Statistical tests for comparing model performance."""

    @staticmethod
    def mcnemar_test(
        y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
    ) -> dict:
        """
        McNemar's test: are two classifiers' error rates significantly different?

        Returns dict with statistic, p_value, and conclusion.
        """
        correct_a = y_pred_a == y_true
        correct_b = y_pred_b == y_true

        # b = A right, B wrong; c = A wrong, B right
        b = ((correct_a) & (~correct_b)).sum()
        c = ((~correct_a) & (correct_b)).sum()

        if b + c == 0:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "conclusion": "Models have identical error patterns",
            }

        if b + c < 25:
            p_value = float(stats.binom_test(b, b + c, 0.5))
            stat = None
        else:
            stat = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = float(1 - stats.chi2.cdf(stat, df=1))

        significant = p_value < 0.05
        better = "Model A" if b > c else "Model B"

        result = {
            "statistic": float(stat) if stat else None,
            "p_value": p_value,
            "b_count": int(b),
            "c_count": int(c),
            "significant": significant,
            "conclusion": (
                f"{better} is significantly better (p={p_value:.4f})"
                if significant
                else f"No significant difference (p={p_value:.4f})"
            ),
        }
        logger.info(f"McNemar's test: {result['conclusion']}")
        return result

    @staticmethod
    def delong_test(
        y_true: np.ndarray,
        y_prob_a: np.ndarray,
        y_prob_b: np.ndarray,
    ) -> dict:
        """
        DeLong test for comparing two AUC values.
        Tests H0: AUC_A = AUC_B vs H1: AUC_A != AUC_B
        """
        auc_a = roc_auc_score(y_true, y_prob_a)
        auc_b = roc_auc_score(y_true, y_prob_b)

        n1 = y_true.sum()
        n0 = len(y_true) - n1

        pos_scores_a = y_prob_a[y_true == 1]
        neg_scores_a = y_prob_a[y_true == 0]
        pos_scores_b = y_prob_b[y_true == 1]
        neg_scores_b = y_prob_b[y_true == 0]

        v_a10 = np.array([np.mean(pos_scores_a > x) for x in neg_scores_a])
        v_a01 = np.array([np.mean(x > neg_scores_a) for x in pos_scores_a])
        v_b10 = np.array([np.mean(pos_scores_b > x) for x in neg_scores_b])
        v_b01 = np.array([np.mean(x > neg_scores_b) for x in pos_scores_b])

        s10_a = np.var(v_a10) / n0
        s01_a = np.var(v_a01) / n1
        s10_b = np.var(v_b10) / n0
        s01_b = np.var(v_b01) / n1

        s10_ab = np.cov(v_a10, v_b10)[0, 1] / n0
        s01_ab = np.cov(v_a01, v_b01)[0, 1] / n1

        var_diff = s10_a + s01_a + s10_b + s01_b - 2 * s10_ab - 2 * s01_ab

        if var_diff <= 0:
            return {
                "auc_a": float(auc_a),
                "auc_b": float(auc_b),
                "p_value": 1.0,
                "significant": False,
                "conclusion": "Cannot compute variance",
            }

        z = (auc_a - auc_b) / np.sqrt(var_diff)
        p_value = 2 * stats.norm.sf(abs(z))

        result = {
            "auc_a": float(auc_a),
            "auc_b": float(auc_b),
            "auc_diff": float(auc_a - auc_b),
            "z_statistic": float(z),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }
        logger.info(
            f"DeLong test: AUC_A={auc_a:.4f}, AUC_B={auc_b:.4f}, p={p_value:.4f}"
        )
        return result

    @staticmethod
    def paired_brier_test(
        y_true: np.ndarray,
        y_prob_a: np.ndarray,
        y_prob_b: np.ndarray,
    ) -> dict:
        """
        Paired t-test on per-observation Brier Score differences.
        Tests whether model A has significantly different Brier Score from model B.
        """
        bs_a = (y_prob_a - y_true) ** 2
        bs_b = (y_prob_b - y_true) ** 2
        diff = bs_a - bs_b

        t_stat, p_value = stats.ttest_rel(bs_a, bs_b)

        return {
            "mean_brier_a": float(bs_a.mean()),
            "mean_brier_b": float(bs_b.mean()),
            "mean_diff": float(diff.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "better_model": "A" if diff.mean() < 0 else "B",
        }
