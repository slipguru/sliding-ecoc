"""Sliding-window Error Correcting Ouput Code meta-estimator."""

import array
import numpy as np
import numbers
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
from sklearn.utils import check_random_state, check_X_y
from sklearn.utils.validation import check_is_fitted

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.multiclass import (_check_partial_fit_first_call,
                                      check_classification_targets,
                                      _ovr_decision_function)
from sklearn.utils.metaestimators import _safe_split
from sklearn.ensemble.forest import _generate_sample_indices
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble.bagging import _generate_indices
from sklearn.ensemble.bagging import BaseBagging
from sklearn.ensemble.base import _partition_estimators
from sklearn.utils import indices_to_mask
from sklearn.utils.fixes import bincount
from sklearn.utils.validation import has_fit_parameter, check_is_fitted

MAX_INT = np.iinfo(np.int32).max


def _generate_feature_window_start(n_estimators_window, n_features, window_size,
                                   stride):
    for start in range(0, n_features - window_size + 1 + (
            n_features - window_size) % stride, stride):
        for i in range(n_estimators_window):
            yield start


def _generate_bagging_indices(random_state, bootstrap_features,
                              bootstrap_samples, n_features, n_samples,
                              max_features, max_samples):
    """Randomly draw feature and sample indices."""
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    feature_indices = _generate_indices(random_state, bootstrap_features,
                                        n_features, max_features)
    sample_indices = _generate_indices(random_state, bootstrap_samples,
                                       n_samples, max_samples)

    return feature_indices, sample_indices


def _parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight,
                               seeds, total_n_estimators, verbose):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
                                              "sample_weight")
    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run (total %d)..." %
                  (i + 1, n_estimators, total_n_estimators))

        random_state = np.random.RandomState(seeds[i])
        estimator = ensemble._make_estimator(append=False,
                                             random_state=random_state)

        # Draw random feature, sample indices
        features, indices = _generate_bagging_indices(random_state,
                                                      bootstrap_features,
                                                      bootstrap, n_features,
                                                      n_samples, max_features,
                                                      max_samples)

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                sample_counts = bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)

        # Draw samples, using a mask, and then fit
        else:
            estimator.fit((X[indices])[:, features], y[indices])

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features


def random_binarizer(y):
    y = np.array(y)
    unique = np.unique(y)
    np.random.shuffle(unique)
    xx = unique[:np.random.randint(1, unique.size - 1)]
    return reduce(np.logical_or, [np.array(y) == x for x in xx]).astype(int)


def _fit_binary_oob(estimator, X, y):
    indices = _generate_sample_indices(estimator.random_state, n_samples)
    fitted_estimator = _fit_binary(estimator, X[indices], y[indices])
    return None


class SlidingECOC(BaseBagging, ClassifierMixin, MetaEstimatorMixin):
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
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,

                 window_size=5,
                 stride=1,
                 code_size='auto'):

        super(SlidingECOC, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        self.window_size = window_size
        self.stride = stride
        self.code_size = code_size

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        """Build an ensemble of estimators from the training
           set (X, y) using a sliding window.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        max_samples : int or float, optional (default=None)
            Argument to use instead of self.max_samples.

        max_depth : int, optional (default=None)
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)

        # Convert data
        X, y = check_X_y(X, y, ['csr', 'csc'])

        # Remap output
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, (numbers.Integral, np.integer)):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.window_size)

        # if not (0 < max_features <= self.n_features_):
        #     raise ValueError("max_features must be in (0, n_features]")

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        # if not self.bootstrap and self.oob_score:
        #     raise ValueError("Out of bag estimation only available"
        #                      " if bootstrap=True")

        # if self.warm_start and self.oob_score:
        #     raise ValueError("Out of bag estimate only available"
        #                      " if warm_start=False")

        # if hasattr(self, "oob_score_") and self.warm_start:
        #     del self.oob_score_

        # if not self.warm_start or len(self.estimators_) == 0:
        if len(self.estimators_) == 0:
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        # if n_more_estimators < 0:
        #     raise ValueError('n_estimators=%d must be larger or equal to '
        #                      'len(estimators_)=%d when warm_start==True'
        #                      % (self.n_estimators, len(self.estimators_)))
        #
        # elif n_more_estimators == 0:
        #     warn("Warm-start fitting without increasing n_estimators does not "
        #          "fit new trees.")
        #     return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators,
                                                             self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        # if self.warm_start and len(self.estimators_) > 0:
        #     random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        all_results = jl.Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            jl.delayed(_parallel_build_estimators)(
                n_estimators[i], self, X[start:self.window_size], y, sample_weight,
                seeds[starts[i]:starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose)
            for i, start in zip(
                range(n_jobs), range(0, self.window_size, self.stride)))

        # Reduce
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self




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
            samples, features = [], []
            y_binary_splits = []
            for p in range(self.n_estimators):
                random_state = check_random_state(self.random_state)

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

            estimators_ = jl.Parallel(n_jobs=self.n_jobs)(
                jl.delayed(_fit_binary)(
                    self.base_estimator, X[mask][:, feats], y_binary[mask])
                for mask, feats, y_binary in zip(samples, features, y_binary_splits))

            self.estimators_.extend(estimators_)
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

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]
        n_classes_ = self.n_classes_
        classes_ = self.classes_

        predictions = np.zeros((n_samples, n_classes_))

        for estimator, samples, features in zip(self.estimators_,
                                                self.estimators_samples_,
                                                self.estimators_features_):
            # Create mask for OOB samples
            mask = ~samples

            if hasattr(estimator, "predict_proba"):
                predictions[mask, :] += estimator.predict_proba(
                    (X[mask, :])[:, features])

            else:
                p = estimator.predict((X[mask, :])[:, features])
                j = 0

                for i in range(n_samples):
                    if mask[i]:
                        predictions[i, p[j]] += 1
                        j += 1

        if (predictions.sum(axis=1) == 0).any():
            warn("Some inputs do not have OOB scores. "
                 "This probably means too few estimators were used "
                 "to compute any reliable oob estimates.")

        oob_decision_function = (predictions /
                                 predictions.sum(axis=1)[:, np.newaxis])
        oob_score = accuracy_score(y, classes_.take(np.argmax(predictions,
                                                              axis=1)))

        self.oob_decision_function_ = oob_decision_function
        self.oob_score_ = oob_score
