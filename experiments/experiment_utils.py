from functools import partial

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
)
from skpsl import ProbabilisticScoringList
from skpsl.metrics import precision_at_recall_k_score, expected_entropy_loss


def from_scorevec(scores):
    scores = np.array(scores)
    features = np.nonzero(scores)[0]
    return list(features), list(scores[features])


def to_scorevec(features, scores, len_=None):
    if len_ is None:
        len_ = max(features)
    scores_ = np.zeros(len_)
    for f, s in zip(features, scores):
        scores_[f] = s
    return tuple(scores_)


def evaluate_metrics(
        y_,
        y_pred,
        y_prob=None,
        cp_thresh=.9
):
    result = dict()
    wpos = y_["train"].mean()

    y_preds = dict(y_pred=y_pred)
    if y_prob is not None:
        y_preds["y_bal"] = {sample: y_prob[sample][:, 1] > wpos for sample in ["train", "test"]}

        cp = partial(precision_at_recall_k_score, recall_level=cp_thresh)
        cp.__name__ = "constrained_precision"

        ee = lambda _, y_prob_: expected_entropy_loss(y_prob_)
        ee.__name__ = "expected_entropy"

        result |= {
            f"{metric.__name__}_{sample}": metric(y_[sample], y_prob[sample][:, 1])
            for metric in [brier_score_loss, roc_auc_score, cp, ee]
            for sample in ["train", "test"]
        }

    prec = partial(precision_score, zero_division=np.nan)
    prec.__name__ = "precision_score"

    result |= {
        f"{metric.__name__}_{rounding_type}_{sample}": metric(y_[sample], discrete_preds[sample])
        for metric in [accuracy_score, f1_score, prec, recall_score, balanced_accuracy_score]
        for rounding_type, discrete_preds in y_preds.items()
        for sample in ["train", "test"]
    }
    return result


def get_loss(name, wpos=None, recall_level=.9):
    match (name):
        case "expected_entropy" | "EE":
            return None
        case "constrained_precision" | "CP":
            return lambda y_true, y_prob: -precision_at_recall_k_score(y_true, y_prob, recall_level=recall_level)
        case "balanced_accuracy" | "BACC":
            if wpos is None:
                raise ValueError("wpos must be set for balanced accuracy")
            return lambda y_true, y_prob: -balanced_accuracy_score(y_true, y_prob > wpos)
        case "accuracy" | "ACC":
            return lambda y_true, y_prob: -accuracy_score(y_true, y_prob > 0.5)


def build_psl_score_cascade(score_sets, X, y, lookahead, optimization_metric=None):
    cascade = []
    for score_set in score_sets:
        psl = ProbabilisticScoringList(
            score_set=score_set,
            lookahead=lookahead,
            stage_loss=optimization_metric,
        )
        psl.fit(X, y)
        cascade.append(psl.stage_clfs[-1])
    return cascade
