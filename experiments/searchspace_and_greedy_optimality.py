from itertools import product
from multiprocessing import Pool

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from skpsl.estimators.probabilistic_scoring_list import _ClassifierAtK, ProbabilisticScoringList
from tqdm import tqdm

from experiments.experiment_utils import get_loss, from_scorevec

LOOKAHEAD = [1, 2]
scoreset = [0, 1]

if __name__ == "__main__":
    for dataset in ["thorax", "covid"]:
        df = pd.read_csv(f"../data/{dataset}.csv")
        X = df.iloc[:, 1:].to_numpy()
        y = df.iloc[:, 0].to_numpy()
        wpos = y.mean()

        opt, meas = "BACC", "BACC"
        loss = get_loss(opt, wpos)
        perf = lambda y, pred: -get_loss(meas, wpos)(y, pred)


        def fit_predict(scores_):
            features, scores = from_scorevec(scores_)
            clf = _ClassifierAtK(features, scores=list(scores),
                                 initial_feature_thresholds=np.full_like(features, .5),
                                 threshold_optimizer=None)
            clf.fit(X, y)
            y_prob = clf.predict_proba(X)[:, 1]

            return scores_, perf(y, y_prob)
            # return accuracy_score(y, y_prob>.5)


        G = nx.Graph()
        node = dict()

        with Pool(12) as p:
            for scores, metric in tqdm(p.imap(fit_predict, product(scoreset, repeat=X.shape[1])),
                                       total=len(scoreset) ** X.shape[1]):

                id_ = np.count_nonzero(scores), metric
                node[tuple(scores)] = id_

                for index in np.nonzero(scores)[0]:
                    # get previous nodes by removing one non-zero feature
                    from_ = np.array(scores)
                    from_[index] = 0

                    G.add_edge(node[tuple(from_)], id_)

        pos = {p: p for p in G.nodes}

        sns.set(font_scale=1.5, rc={'text.usetex': True})
        sns.set_style("whitegrid")
        plt.rc('font', **{'family': 'serif'})
        fig, ax = plt.subplots()
        fig.set_size_inches(13, 4)
        ax.set_ylabel("Balanced Accuracy in sample")
        ax.set_xlabel("Model Complexity")

        print("drawing nodes")
        nx.draw_networkx_nodes(G, pos, node_color="#000000", node_size=10, ax=ax, alpha=.1)
        print("drawing edges")
        nx.draw_networkx_edges(G, pos, edge_color="#000000", ax=ax, width=0.5, node_size=10, alpha=.1)

        for l, c in zip(LOOKAHEAD, ["#de8f05", "#0173b2"]):
            # highlight cascade
            print("fitting cascade")
            psl = ProbabilisticScoringList(score_set=set(scoreset) - {0},
                                           stage_loss=loss,
                                           lookahead=l).fit(X, y)
            cascade = [(i, perf(y, clf.predict_proba(X)[:, 1]))
                       for i, clf in enumerate(psl.stage_clfs)]

            G = nx.Graph()
            for u, v in zip(cascade, cascade[1:]):
                G.add_edge(u, v)
            print("drawing cascade")
            nx.draw_networkx_nodes(G, pos, node_color=c, node_size=50, ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color=c, ax=ax, width=1, node_size=10, label=f"Lookahead $l={l}$")

        # plt.box(False)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.set_xlim(-0.2, X.shape[1] + 0.2)

        print("generating file")
        code = f"local{opt}_globalSUM_metr{meas}"
        fig.legend(bbox_to_anchor=(.9, .35))
        fig.suptitle(dataset.title())
        fig.savefig(f"../fig/{dataset}_{code}.pdf", bbox_inches="tight")
        # fig.savefig(f"../../fig/{dataset}_{code}.png", dpi=300)
