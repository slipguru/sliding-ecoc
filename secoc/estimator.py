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
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.multiclass import (_check_partial_fit_first_call,
                                      check_classification_targets,
                                      _ovr_decision_function)
from sklearn.utils.metaestimators import _safe_split


class SlidingECOC(BaseEnsemble, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_estimator=None, n_estimators=50,
                 code_size=1.5,  # division on the number of classes
                 window_size=5,
                 estimator_params=tuple(),  # ?
                 random_state=None, n_jobs=1):

        super(SlidingECOC, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.base_estimator = base_estimator
        self.code_size = code_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.window_size = window_size

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
        if self.code_size <= 0:
            raise ValueError("code_size should be greater than 0, got {1}"
                             "".format(self.code_size))

        _check_estimator(self.base_estimator)
        random_state = check_random_state(self.random_state)

        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        code_size_ = int(n_classes * self.code_size)

        # FIXME: there are more elaborate methods than generating the codebook
        # randomly.
        self.code_book_ = random_state.random_sample((n_classes, code_size_))
        self.code_book_[self.code_book_ > 0.5] = 1

        if hasattr(self.base_estimator, "decision_function"):
            self.code_book_[self.code_book_ != 1] = -1
        else:
            self.code_book_[self.code_book_ != 1] = 0

        classes_index = dict((c, i) for i, c in enumerate(self.classes_))

        Y = np.array([self.code_book_[classes_index[y[i]]]
                      for i in range(X.shape[0])], dtype=np.int)

        return self, Y  # debug

        self.estimators_ = jl.Parallel(n_jobs=self.n_jobs)(
            jl.delayed(_fit_binary)(self.base_estimator, X, Y[:, i])
            for i in range(Y.shape[1]))

        return self

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
        Y = np.array([_predict_binary(e, X) for e in self.estimators_]).T
        pred = euclidean_distances(Y, self.code_book_).argmin(axis=1)
        return self.classes_[pred]
