"""Tests for Sliding-ECOC estimator."""
import numpy as np

from numpy.testing import assert_array_equal
from sklearn.datasets import make_classification

from secoc.estimator import SlidingECOC


def test_single_seed_features():
    X, y = make_classification(
        n_classes=3, n_clusters_per_class=1, n_features=20, n_samples=100,
        n_informative=5)
    estimator = SlidingECOC(
        max_features=.5, circular_features=True, stride=2,
        n_estimators_window=1, window_size=5, single_seed_features=True,
        verbose=0, oob_score=True, n_estimators=None,
        bootstrap_features=True).fit(X, y)

    diff = np.mod(estimator.estimators_features_[1] -
                  estimator.estimators_features_[0], 20)
    for i in range(2, len(estimator.estimators_features_)):
        diff_i = np.mod(
            estimator.estimators_features_[i] -
            estimator.estimators_features_[i - 1], 20)

        assert_array_equal(diff_i, diff)
