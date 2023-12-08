import shelve
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skslim import Slim
from tqdm import tqdm


def get_complexity(c):
    try:
        clf = Slim(max_score=3, random_state=42, C=c).fit(X_train, y_train)
        return c, np.count_nonzero(clf.scores)
    except AssertionError:
        return c, None


if __name__ == '__main__':
    dname = "covid"
    df = pd.read_csv(f"../../data/{dname}.csv")
    X = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)
    with shelve.open(f"{dname} score3.db") as db:
        cs = np.logspace(-10, -1, 100)
        with Pool(10) as p:
            for c, complexity in tqdm(p.imap_unordered(get_complexity, cs), total=cs.size):
                db[str(len(db))] = (c, complexity)
                pd.DataFrame(dict(db)).T.to_csv(f"{dname} score3.csv")
