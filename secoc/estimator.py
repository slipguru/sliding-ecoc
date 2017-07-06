"""Sliding-window Error Correcting Ouput Code meta-estimator."""

import array
import numpy as np
import scipy.sparse as sp
import warnings
import itertools

from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.ensemble.base import BaseEnsemble
from sklearn.externals import joblib as jl
from sklearn.externals.six.moves import zip as izip
from sklearn.multiclass import _check_estimator, _fit_binary, _predict_binary
from sklearn.neighbors import KNeighborsClassifier
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

def _fit_binary_oob(estimator, X, y):
    return _fit_binary(estimator, X, y)

class SlidingECOC(BaseEnsemble, ClassifierMixin, MetaEstimatorMixin):
    """A SlidingECOC classifier.

    A SlidingECOC classifier is an ensemble meta-estimator specifically
    designed for multiclass problems. This meta-estimator fits base
    classifiers each on a random subset of the orignal feature-set in a
    sliding-window fashion. Size, stride and, consequently, overlap of the
    sliding window are user-defined. Each base classifier is trained to
    perform binary classification on a random split of the classes, therefore
    it produces a 0/1 prediction. Hence, the output of the learning step
    for each training sample is a binary encoding (e.g., 01001101010).
    When the out-of-bag error estimation strategy is enabled, only the top
    performing base classifiers are retained and the encoding length can be
    user defined. Each encoding is considered as a string of bits received after
    a transmission on a noisy channel. Once a new encoding is estimated for a
    new sample, its class predicton is finally achieved by maximum likelihood
    comparing its encoding with the training ones.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset (if
        `oob_score` is `True`). If None, then the base estimator is a
        decision tree.

    n_estimators : int or None, optional (default=100)
        The total number of base estimators used. If None, then
        `n_estimators = n_estimators_window * int(ceil((n_features -
        window_size) / stride) + 1)`.

    n_estimators_window : int, optional (default=10)
        The number of base estimators for each window (used only if
        `n_estimators` is None).

    window_size : int, optional (default=5)
        The size of the sliding window.

    stride : int, optional (default=1)
        The sliding window stride.

    code_size : int, float, string or None, optional (default="auto")
        The number of top performing `base_estimator` that are retained to
        to define each sample encoding. This parameters controls the complexity
        of the encoding and it should be tuned for better performance.

        - If int, then consider `code_size` classifiers.
        - If float, then `code_size` is a percentage and
          `int(code_size * n_tot_estimators)` estimators are considered for each
          encoding.
        - If "auto", then `code_size=n_tot_estimators`.
        - If "sqrt", then `code_size=sqrt(n_tot_estimators)`.
        - If "log2", then `code_size=log2(n_tot_estimators)`.
        - If None, then `code_size=n_tot_estimators` (same as "auto").

    oob_score : bool
        Whether to use out-of-bag samples to estimate
        the generalization error.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from the sliding window to train each
        base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `int(max_features * window_size)` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

     Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by a boolean mask.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    classes_ : array of shape = [n_classes]
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.
    """

    def __init__(self, base_estimator=None,
                 n_estimators=50,
                 estimator_params=tuple(),  # ?
                 window_size=5, stride=1,
                 code_size='auto',
                 oob_score=False,
                 max_features=None, verbose=False,
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
        self.verbose = verbose

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

        # How many windows I have?
        n_windows_ = int(np.ceil((n_features - self.window_size) / self.stride) + 1)
        if self.verbose:
            print("You are about to generate {0} estimators for {1} windows, "
                  "for a total of {2} estimators.".format(
                      self.n_estimators, n_windows_,
                      self.n_estimators * n_windows_))

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

            results = jl.Parallel(n_jobs=self.n_jobs)(
                jl.delayed(_fit_binary_oob)(
                    self.base_estimator, X[:, feats], y_binary)
                for feats, y_binary in zip(features, y_binary_splits))
            self.estimators_.extend()
            self.estimator_features_.extend(features)
            self.estimator_splits_.extend(y_binary_splits)

        self.X_train_encoding_ = self.encode(X)
        self.y_train_ = y
        return self

    def encode(self, X):
        check_is_fitted(self, 'estimators_')

        return np.array([estimator.predict(X[:, feats]) for estimator, feats in zip(
            self.estimators_, self.estimator_features_)]).T

    def predict(self, X, classifier=None):
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
        X_encoding_ = self.encode(X)
        # pred = euclidean_distances(Y, self.code_book_).argmin(axis=1)
        if classifier is None:
            classifier = KNeighborsClassifier()
        knn = classifier.fit(
            self.X_train_encoding_, self.y_train_)
        return knn.predict(X_encoding_)
