import logging
from ast import literal_eval
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from skpsl import ProbabilisticScoringList
from skslim import Slim
from tqdm import tqdm

from experiment_utils import evaluate_metrics, get_loss

N_SPLITS = 50
RECALL_CONSTRAINT = 0.9
CALIBRATION_METHOD = "isotonic"

seed = 1
dataset_names = [
    "thorax",
    "covid",
    "heart",
    "breast",
    "adult",
]

c_values = [1e-3, 1e-5, 1e-7, 1e-9]
disable_slim = {"adult", "heart", "breast"}
resample_size_adult = 1000  # None
lookaheads = [2]
score_sets = [[-3, -2, -1, 1, 2, 3]]

optimization_metrics = ["expected_entropy", "constrained_precision", "balanced_accuracy", "accuracy"]


def worker(params):
    fold, data_name, score_set = params
    # print(fold, data_name, score_set)

    dataset = pd.read_csv(f"../data/{data_name}.csv")
    if data_name == "adult" and resample_size_adult:
        dataset = dataset.sample(resample_size_adult, random_state=fold)

    X = dataset.iloc[:, 1:].to_numpy()
    y = dataset.iloc[:, 0].to_numpy()

    # create splits
    shuffle = ShuffleSplit(n_splits=1, test_size=0.33, random_state=fold)
    split = next(shuffle.split(X))

    X_ = {name: X[idx] for name, idx in zip(["train", "test"], split)}
    y_ = {name: y[idx] for name, idx in zip(["train", "test"], split)}

    key = dict(dataset=data_name,
               fold=fold,
               score_set=score_set)

    results = []

    # fit and evaluate slim and slim cascade
    for c_value in c_values:
        if data_name in disable_slim:
            break
        slim_key = key | dict(method="SLIM", c_value=c_value)

        try:
            slim = Slim(
                random_state=42,
                C=c_value,
                max_score=max(score_set),
                min_score=min(score_set),
                balance_class_weights=True,
            )
            slim.fit(X_["train"], y_["train"])
            y_pred = {sample: slim.predict(X_[sample]) for sample in ["train", "test"]}

            features = np.nonzero(slim.scores)[0]

            results.append(slim_key | dict(stage=np.count_nonzero(slim.scores),
                                           features=list(features),
                                           scores=list(slim.scores[features]),
                                           slim_threshold=slim.threshold,
                                           solution_code=slim.solution_status_code)
                           | evaluate_metrics(y_=y_,
                                              y_pred=y_pred,
                                              cp_thresh=RECALL_CONSTRAINT
                                              ))

        except Exception as e:
            results.append(slim_key | dict(exception=str(e)))

    # fit psl
    for variant in ["PSL", "SLIM_PSL"]:
        if variant == "SLIM_PSL":
            if data_name in disable_slim:
                continue
            slim = Slim(
                random_state=42,
                C=1e-9,
                max_score=max(score_set),
                min_score=min(score_set),
                balance_class_weights=True,
                timeout=60
            )
            slim.fit(X_["train"], y_["train"])
            features = np.nonzero(slim.scores)[0]
            predef_features = list(features)
            predef_scores = list(slim.scores[features])
        else:
            predef_features = None
            predef_scores = None

        for lookahead, optimization_name in product(lookaheads, optimization_metrics):
            opt_func = get_loss(optimization_name, wpos=y_["train"].mean(),
                                recall_level=RECALL_CONSTRAINT)

            stage_clf_params = dict(calibration_method=CALIBRATION_METHOD)
            psl = ProbabilisticScoringList(score_set=set(score_set),
                                           stage_loss=opt_func,
                                           stage_clf_params=stage_clf_params,
                                           lookahead=lookahead)
            psl.fit(X_["train"], y_["train"],
                    predef_features=predef_features, predef_scores=predef_scores, strict=False)

            for k, stage in enumerate(psl.stage_clfs):
                # evaluate psl
                y_pred = {sample: stage.predict(X_[sample]) for sample in ["train", "test"]}
                y_proba = {sample: stage.predict_proba(X_[sample]) for sample in ["train", "test"]}

                results.append(key | dict(method=variant,
                                          stage=k,
                                          features=stage.features,
                                          scores=stage.scores,
                                          lookahead=lookahead,
                                          psl_calibration_method=CALIBRATION_METHOD,
                                          psl_optimization_metric=optimization_name)
                               | evaluate_metrics(y_=y_,
                                                  y_pred=y_pred,
                                                  y_prob=y_proba,
                                                  cp_thresh=RECALL_CONSTRAINT
                                                  ))
    return results


def is_unprocessed(params):
    fold, data_name, score_set = params
    # check if fold was already processed
    try:
        df = pd.read_csv(f"../results/results_{data_name}_features.csv")
        if (fold, tuple(score_set)) in set(
                (row["fold"], tuple(literal_eval(row["score_set"]))) for _, row in
                df[["fold", "score_set"]].iterrows()):
            return False
    except FileNotFoundError:
        pass
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.info("Starting experiment")

    grid = list(filter(is_unprocessed, product(range(N_SPLITS), dataset_names, score_sets)))

    with Pool(8) as p:
        pbar = tqdm(p.imap_unordered(worker, grid), total=len(grid))
        for results in pbar:
            result = results[0]
            pbar.set_postfix_str(f"{result['fold']} {result['dataset']} {result['score_set']}")
            filename = f"../results/results_{result['dataset']}_features.csv"
            try:
                df = pd.read_csv(filename)
            except FileNotFoundError:
                df = pd.DataFrame()
            pd.concat((df, pd.DataFrame(results))).to_csv(filename, index=False)
