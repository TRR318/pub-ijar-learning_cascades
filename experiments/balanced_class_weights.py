import numpy as np
import pandas as pd
from skpsl import ProbabilisticScoringList

dataset = pd.read_csv(f"../data/thorax.csv")
X_ = dataset.iloc[:, 1:].to_numpy()
y_ = dataset.iloc[:, 0].to_numpy()

clf_ = ProbabilisticScoringList(
    {-2, -1, 1, 2}
)
clf_.fit(X_, y_)
_, counts = np.unique(clf_.predict(X_), return_counts=True)
print(clf_.inspect().to_string())
print(counts)
