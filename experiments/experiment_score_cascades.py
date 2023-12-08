import logging
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ShuffleSplit
from skpsl.metrics import constrained_precision
from skslim import Slim, RiskSlim

from experiment_utils import build_psl_score_cascade

# Note: In this experiment "stage" refers to the cardinality of the score set

N_SPLITS = 10
RECALL_CONSTRAINED = 0.9

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    seed = 1

    logger = logging.getLogger(__name__)
    logger.info("Starting experiment")

    dataset_names = [
        # "thorax",
        "covid",
        "mammo",
        "adult",
        "mushroom",
    ]

    for dataset_name in dataset_names:
        dataset = pd.read_csv(f"../data/{dataset_name}.csv")

        if dataset_name in ["adult", "mushroom"]:
            dataset = dataset.sample(n=700)

        y = dataset.iloc[:, 0].to_numpy()
        X = dataset.iloc[:, 1:].to_numpy()


        optimization_metrics = [
            ("expected_entropy", None),
            (
                "constraint_precision",
                lambda y_true, y_prob: -constrained_precision(y_true, y_prob),
            ),
            (
                "balanced_accuracy",
                lambda y_true, y_prob: -balanced_accuracy_score(y_true, y_prob > 0.5),
            ),
        ]

        shuffle = ShuffleSplit(n_splits=N_SPLITS, test_size=0.33, random_state=seed)

        c_values = [1e-3, 1e-5, 1e-7, 1e-9]
        # c_values = [1e-3]
        lookaheads = [2]
        boundaries = [1, 3, 5, 7, 9]
        score_sets = [list(range(-b, b + 1)) for b in boundaries]
        for s in score_sets:
            s.remove(0)

        splits = list(shuffle.split(X))


        def worker(fold):
            experiment_data = []

            train_index, test_index = splits[fold]

            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            # fit and evaluate slim and slim cascade
            for c_value, score_set in product(c_values, score_sets):
                try:
                    slim = Slim(
                        random_state=42,
                        C=c_value,
                        max_score=max(score_set),
                        min_score=min(score_set),
                    )
                    slim.fit(X_train, y_train)
                    y_pred_slim_train = slim.predict(X_train)
                    y_pred_slim_test = slim.predict(X_test)

                    evaluate_and_log(
                        experiment_data,
                        constrained_precision,
                        y_train,
                        y_test,
                        y_pred_slim_train,
                        y_pred_slim_test,
                        None,
                        None,
                        None,
                        fold,
                        "SLIM",
                        score_set,
                        np.count_nonzero(slim.scores),
                        list(np.nonzero(slim.scores)),
                        slim.scores,
                        lookahead=None,
                        c_value=c_value,
                        slim_threshold=slim.threshold,
                        solution_code=slim.solution_status_code,
                    )

                except Exception as e:
                    log_exception(
                        experiment_data=experiment_data,
                        fold=fold,
                        method="SLIM",
                        score_set=score_set,
                        e=e,
                    )
            # fit and evaluate riskslim
            for c_value, score_set in product(c_values, score_sets):
                try:
                    rslim = RiskSlim(
                        random_state=42,
                        C=c_value,
                        max_score=max(score_set),
                        min_score=min(score_set),
                    )
                    rslim.fit(X_train, y_train)

                    y_pred_rslim_train = rslim.predict(X_train)
                    y_pred_rslim_test = rslim.predict(X_test)
                    d_prob_rslim_train = rslim.predict_proba(X_train)
                    d_prob_rslim_test = rslim.predict_proba(X_test)
                    evaluate_and_log(
                        experiment_data,
                        constrained_precision,
                        y_train,
                        y_test,
                        y_pred_rslim_train,
                        y_pred_rslim_test,
                        d_prob_rslim_train,
                        d_prob_rslim_test,
                        None,
                        fold,
                        "RiskSLIM",
                        score_set,
                        np.count_nonzero(rslim.scores),
                        np.nonzero(rslim.scores),
                        rslim.scores,
                        lookahead=None,
                        c_value=c_value,
                        slim_threshold=rslim.threshold,
                        solution_code=rslim.solution_status_code,
                    )
                except Exception as e:
                    log_exception(
                        experiment_data=experiment_data,
                        fold=fold,
                        method="RiskSLIM",
                        score_set=score_set,
                        e=e,
                    )

                # fit psl
                # try:
            for lookahead, optimization_metric in product(
                    lookaheads, optimization_metrics
            ):
                optimization_name, optimizatoin_func = optimization_metric

                psl_score_cascade = build_psl_score_cascade(
                    score_sets,
                    X_train,
                    y_train,
                    lookahead=lookahead,
                    optimization_metric=optimizatoin_func,
                )

                for stage, stage_clf in enumerate(psl_score_cascade):
                    try:
                        # evaluate psl
                        y_pred_psl_train = stage_clf.predict(X_train)
                        y_pred_psl_test = stage_clf.predict(X_test)
                        d_prob_psl_train = stage_clf.predict_proba(X_train)
                        d_prob_psl_test = stage_clf.predict_proba(X_test)

                        print(stage, score_sets[stage])
                        evaluate_and_log(
                            experiment_data,
                            constrained_precision,
                            y_train,
                            y_test,
                            y_pred_psl_train,
                            y_pred_psl_test,
                            d_prob_psl_train,
                            d_prob_psl_test,
                            stage_clf.score(X_test),
                            fold,
                            "PSL",
                            score_sets[stage],
                            len(score_sets[stage]),
                            stage_clf.features,
                            stage_clf.scores,
                            lookahead=lookahead,
                            psl_optimization_metric=optimization_name,
                        )
                    except Exception as e:
                        log_exception(
                            experiment_data=experiment_data,
                            fold=fold,
                            method="PSL",
                            score_set=score_sets[stage],
                            e=e,
                        )
            return experiment_data


        with Pool(7) as p:
            results_data = []
            for results in p.imap_unordered(worker, range(N_SPLITS)):
                results_data.extend(results)

            experiment_df = (pd.DataFrame(results_data)
                             .to_csv(f"../results_old/results_{dataset_name}_scores.csv"))
