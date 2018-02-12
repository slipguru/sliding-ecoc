"""How to use SlidingECOC as a simple neural net-style."""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from secoc.estimator import SlidingECOC

X, y = make_classification()


def neural_net_fit(layer=1):
    x_transform = SlidingECOC(base_estimator=LogisticRegression(), method='proba').fit(X,y).transform(X)
    x = x_transform.copy()
    scores = [LogisticRegression().fit(x, y).score(x, y)]
    for i in range(layer):
        x = SlidingECOC(base_estimator=LogisticRegression(), method='proba', window_size=x.shape[1], max_features=1.).fit(x, y).transform(x)
        scores.append(LogisticRegression().fit(x, y).score(x, y))
    return scores


print neural_net_fit(3)
