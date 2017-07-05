"""This module implements the sliding-window error correcting code."""

import array
import numpy as np
import warnings
import scipy.sparse as sp
import itertools


from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.ensemble.base import BaseEnsemble
from sklearn.externals import joblib as jl
from sklearn.externals.six.moves import zip as izip
from sklearn.multiclass import _check_estimator, _fit_binary, _predict_binary
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.multiclass import (_check_partial_fit_first_call,
                                      check_classification_targets,
                                      _ovr_decision_function)
from sklearn.utils.metaestimators import _safe_split


def random_binarizer(y):
    y = np.array(y)
    unique = np.unique(y)
    np.random.shuffle(unique)
    xx = unique[:np.random.randint(1, unique.size - 1)]
    return reduce(np.logical_or, [np.array(y) == x for x in xx]).astype(int)


class SlidingECOC(BaseEnsemble, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_estimator=None,
                 n_estimators=50,
                 estimator_params=tuple(),  # ?
                 window_size=5, stride=1,
                 # code_size=,  # division on the number of classes
                 oob_score=False,
                 max_features=None,
                 random_state=None, n_jobs=1):

        super(SlidingECOC, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        # self.base_estimator = base_estimator
        # self.code_size = code_size
        self.window_size = window_size
        self.stride = stride
        self.oob_score = oob_score
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : numpy array of shape [n_samples]
            Multi-class targets.

        Returns
        -------
        self
        """
        # if self.code_size <= 0:
        #     raise ValueError("code_size should be greater than 0, got {1}"
        #                      "".format(self.code_size))
        if self.max_features is None:
            self.max_features = self.window_size
        elif not 1 < self.max_features <= self.window_size:
            raise ValueError(
                "max_features must be included in [2, window_size], got {0}"
                " instead.".format(self.max_features))
        _check_estimator(self.base_estimator)
        random_state = check_random_state(self.random_state)

        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        # code_size_ = int(n_classes * self.code_size)

        n_samples, n_features = X.shape

        # FIXME: there are more elaborate methods than generating the codebook
        # randomly.
        # self.code_book_ = random_state.random_sample((n_classes, code_size_))
        # self.code_book_[self.code_book_ > 0.5] = 1
        #
        # if hasattr(self.base_estimator, "decision_function"):
        #     self.code_book_[self.code_book_ != 1] = -1
        # else:
        #     self.code_book_[self.code_book_ != 1] = 0
        #
        # classes_index = dict((c, i) for i, c in enumerate(self.classes_))
        #
        # Y = np.array([self.code_book_[classes_index[y[i]]]
        #               for i in range(X.shape[0])], dtype=np.int)

        # How many windows I have?
        # f = lambda p,d,k,s: p*(np.ceil((d-k)/s) + 1)

        # sliding window
        self.estimator_features_ = []
        self.estimator_splits_ = []
        for start in range(0, n_features - self.window_size + 1 + (
                n_features - self.window_size) % self.stride, self.stride):
            features = []
            y_binary_splits = []
            for p in range(self.n_estimators):
                # prepare the features
                # 1. choose randomly the max features belonging to [0, k-1]
                # 1a. how many?
                n_features_window = np.random.randint(1, min(
                    self.max_features, n_features - start - 1))
                # 1b. which ones?
                features.append(np.random.randint(
                    start,
                    min(start + self.window_size, n_features),
                    n_features_window))

                # 2. split y and binarise it
                y_binary_splits.append(random_binarizer(y))

            self.estimators_.extend(jl.Parallel(n_jobs=self.n_jobs)(
                jl.delayed(_fit_binary)(
                    self.base_estimator, X[:, feats], y_binary)
                for feats, y_binary in zip(features, y_binary_splits)))
            self.estimator_features_.extend(features)
            self.estimator_splits_.extend(y_binary_splits)

        return self

    def encode(self, X):
        check_is_fitted(self, 'estimators_')

        return np.array([estimator.predict(X[:, feats]) for estimator, feats in zip(
            self.estimators_, self.estimator_features_)]).T




    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted multi-class targets.
        """
        check_is_fitted(self, 'estimators_')
        # Y = np.array([_predict_binary(e, X) for e in self.estimators_]).T
        # pred = euclidean_distances(Y, self.code_book_).argmin(axis=1)
        # return self.classes_[pred]
