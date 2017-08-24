"""Sliding-window Error Correcting Ouput Code meta-estimator."""
from __future__ import division, print_function

import itertools
import numbers
import numpy as np

from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.ensemble.bagging import _generate_indices, MAX_INT, BaseBagging
from sklearn.ensemble.base import _partition_estimators
from sklearn.externals import joblib as jl
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state, check_X_y, column_or_1d
from sklearn.utils import indices_to_mask, check_array
from numpy import bincount
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import has_fit_parameter


def _predict_single_estimator(estimator, features, X):
    return estimator.predict(X[:, features])


def _predict_proba_single_estimator(estimator, features, X):
    return estimator.predict_proba(X[:, features])[:, 1]


def _generate_bagging_indices(random_state_features, random_state_samples,
                              random_state_max_features,
                              bootstrap_features, bootstrap_samples,
                              n_features, n_samples,
                              max_features, max_samples):
    """Randomly draw feature and sample indices."""
    # Draw indices
    feature_indices = _generate_indices(
        check_random_state(random_state_features), bootstrap_features,
        n_features,
        check_random_state(random_state_max_features).randint(1, max_features + 1))
    sample_indices = _generate_indices(
        check_random_state(random_state_samples), bootstrap_samples,
        n_samples, max_samples)

    return feature_indices, sample_indices


def _parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight,
                               seeds_features, seeds_samples, seeds_max_features,
                               total_n_estimators, verbose,
                               start_index, circular_features=False):
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
    estimators_samples = []
    estimators_splits = []

    for i in range(n_estimators):
        if verbose > 2:
            print("Building estimator %d of %d for this parallel run (total %d)..." %
                  (i + 1, n_estimators, total_n_estimators))

        random_state_max_features = np.random.RandomState(seeds_max_features[i])
        random_state_features = np.random.RandomState(seeds_features[i])
        random_state = np.random.RandomState(seeds_samples[i])
        estimator = ensemble._make_estimator(append=False,
                                             random_state=random_state)

        # Draw random feature, sample indices
        if circular_features:
            n_features_window = ensemble.window_size
            max_features_window = max_features
        else:
            n_features_window = min(ensemble.window_size, n_features - start_index[i])
            max_features_window = min(max_features, n_features - start_index[i])

        features, indices = _generate_bagging_indices(
            random_state_features, random_state, random_state_max_features,
            bootstrap_features, bootstrap,
            n_features_window, n_samples,
            max_features_window, max_samples)

        features += start_index[i]

        # ensure not going outside range, take the first ones instead
        np.mod(features, n_features, out=features)

        # Draw samples, using sample weights, and then fit
        y_binary = random_binarizer(y)
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

            estimator.fit(X[:, features], y_binary, sample_weight=curr_sample_weight)

        # Draw samples, using a mask, and then fit
        else:
            estimator.fit((X[indices])[:, features], y_binary[indices])

        estimators.append(estimator)
        estimators_features.append(features)
        estimators_samples.append(indices)
        estimators_splits.append(y_binary)

    return estimators, estimators_features, estimators_samples, estimators_splits


def random_binarizer(y):
    # y = np.array(y)
    unique = np.unique(y)
    if unique.size < 3:
        return y
    np.random.shuffle(unique)
    xx = unique[:np.random.randint(1, unique.size - 1)]
    return reduce(np.logical_or, [y == x for x in xx]).astype(int)


class SlidingECOC(BaseBagging, TransformerMixin):
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

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

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

    method : 'standard' or 'proba'
        Method to generate final encoding. 'standard' means that the encoding
        has only 0 or 1. 'proba' means that it's the probability to be 1, so
        the encoding has float numbers in [0,1].

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
                 n_estimators=100,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,

                 window_size=5,
                 n_estimators_window=10,
                 circular_features=False,
                 stride=1,
                 single_seed_features=False,
                 single_seed_samples=False,
                 method='standard',
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
        self.circular_features = circular_features
        self.code_size = code_size
        self.n_estimators_window = n_estimators_window
        self.single_seed_features = single_seed_features
        self.single_seed_samples = single_seed_samples
        self.method = method

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

        if not 0 < self.window_size <= self.n_features_:
            raise ValueError("window_size not valid")

        if not 0 < self.stride <= self.window_size:
            raise ValueError("stride not valid")

        # Check parameters
        if not self.circular_features:
            self.n_windows_ = int(np.ceil((
                self.n_features_ - self.window_size) / self.stride) + 1)
        else:
            self.n_windows_ = len(range(0, self.n_features_, self.stride))

        if self.n_estimators is None:
            self.n_estimators = self.n_estimators_window * self.n_windows_
        else:
            self.n_estimators_window = int(self.n_estimators / self.n_windows_)
            self.n_estimators = self.n_estimators_window * self.n_windows_

        # _check_estimator(self.base_estimator)  # ?
        self._validate_estimator()

        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, (numbers.Integral, np.integer)):
            max_samples = int(max_samples * X.shape[0])

        if not 0 < max_samples <= X.shape[0]:
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.window_size)

        if (self.bootstrap_features and max_features <= 0) or \
                not 0 < max_features <= self.window_size:
            raise ValueError("max_features must be in (0, window_size] if not"
                             " bootstrap_features.")

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score and \
                self._max_samples >= n_samples:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True or max_samples < n_samples")

        # if self.warm_start and self.oob_score:
        #     raise ValueError("Out of bag estimate only available"
        #                      " if warm_start=False")

        # if hasattr(self, "oob_score_") and self.warm_start:
        #     del self.oob_score_

        # if not self.warm_start or len(self.estimators_) == 0:
        # if len(self.estimators_) == 0:  # TODO think about warm_start or adding estimators
        # Free allocated memory, if any
        self.estimators_ = []
        self.estimators_features_ = []
        self._estimators_samples = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 1:
            raise ValueError('n_estimators=%d must be larger or equal to 1'
                             % (self.n_estimators))

        # using oob_score, take only "code_size" best estimators
        if isinstance(self.code_size, (numbers.Integral, np.integer)):
            self.code_size_ = self.code_size
        elif isinstance(self.code_size, (numbers.Real, np.float)):  # float
            self.code_size_ = int(self.code_size * self.n_estimators)
        elif self.code_size is None or self.code_size == 'auto':
            self.code_size_ = self.n_estimators
        elif self.code_size == 'sqrt':
            self.code_size_ = int(np.sqrt(self.n_estimators))
        elif self.code_size == 'log2':
            self.code_size_ = int(np.log2(self.n_estimators))
        else:
            raise ValueError("Value for code_size '{}' unrecognized".format(
                self.code_size))

        if self.code_size <= 0:
            raise ValueError("code_size should be greater than 0, got {1}"
                             "".format(self.code_size))

        if self.verbose:
            print("You are about to generate {0} estimators for {1} windows, "
                  "for a total of {2} estimators.".format(
                      self.n_estimators_window, self.n_windows_,
                      self.n_estimators))

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators,
                                                             self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        # if self.warm_start and len(self.estimators_) > 0:
        #     random_state.randint(MAX_INT, size=len(self.estimators_))
        if self.single_seed_features:
            # different features inside a single windows, then shift
            seeds_features = np.tile(
                random_state.randint(MAX_INT, size=self.n_estimators_window),
                self.n_windows_)
            seeds_max_features = np.tile(
                random_state.randint(MAX_INT, size=self.n_estimators_window),
                self.n_windows_)
        else:
            seeds_features = random_state.randint(MAX_INT, size=n_more_estimators)
            seeds_max_features = random_state.randint(MAX_INT, size=n_more_estimators)

        self._seeds_features = seeds_features
        self._seeds_max_features = seeds_max_features
        if self.verbose > 1:
            print("Seeds features: %s" % seeds_features)

        if self.single_seed_samples:
            seeds = np.tile(random_state.randint(MAX_INT, size=1),
                            n_more_estimators)
        else:
            seeds = random_state.randint(MAX_INT, size=n_more_estimators)

        self._seeds = seeds
        if self.verbose > 1:
            print("Seeds samples: %s" % seeds)

        start_index = (iter(sorted(self.n_estimators_window * range(
            0, self.n_features_, self.stride))))

        self.estimators_splits_ = []
        all_results = jl.Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            jl.delayed(_parallel_build_estimators)(
                n_estimators[i], self,
                X,
                y, sample_weight,
                seeds_features[starts[i]:starts[i + 1]],
                seeds[starts[i]:starts[i + 1]],
                seeds_max_features[starts[i]:starts[i + 1]],
                total_n_estimators,
                start_index=list(itertools.islice(start_index, n_estimators[i])),
                verbose=self.verbose,
                circular_features=self.circular_features)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))
        self._estimators_samples += list(itertools.chain.from_iterable(
            t[2] for t in all_results))
        self.estimators_splits_ += list(itertools.chain.from_iterable(
            t[3] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        # sliding window
        # for start in range(0, n_features - self.window_size + 1 + (
        #         n_features - self.window_size) % self.stride, self.stride):
        #     samples, features = [], []
        #     y_binary_splits = []
        #     for p in range(self.n_estimators):
        #         random_state = check_random_state(self.random_state)
        #
        #         # prepare the features
        #         # 1. choose randomly the max features belonging to [0, k-1]
        #         # 1a. how many?
        #         n_features_window = np.random.randint(1, min(
        #             self.max_features, n_features - start - 1))
        #         # 1b. which ones?
        #         features.append(np.random.randint(
        #             start,
        #             min(start + self.window_size, n_features),
        #             n_features_window))
        #
        #         # 2. split y and binarise it
        #         y_binary_splits.append(random_binarizer(y))
        #
        #     estimators_ = jl.Parallel(n_jobs=self.n_jobs)(
        #         jl.delayed(_fit_binary)(
        #             self.base_estimator, X[mask][:, feats], y_binary[mask])
        #         for mask, feats, y_binary in zip(samples, features, y_binary_splits))
        #
        #     self.estimators_.extend(estimators_)
        #     self.estimator_features_.extend(features)
        #     self.estimator_splits_.extend(y_binary_splits)

        return self

    @property
    def estimators_samples_(self):
        """The subset of drawn samples for each base estimator."""
        return self._estimators_samples

    def transform(self, X, code_size=None):
        """Transform the data X according to the fitted base model.

        Parameters
        ----------
        X: {array-like}, shape (n_samples, n_features)
            Data matrix to be transformed by the model

        Returns
        -------
        embedding : array, shape (n_samples, code_size)
            Transformed data
        """
        check_is_fitted(self, 'estimators_')
        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        if self.oob_score:
            idx = np.argsort(self.oob_score_)[::-1]
            idx = idx[:code_size or self.code_size_]
        else:
            idx = Ellipsis  # get everyone

        if self.method == 'standard':
            transformer = _predict_single_estimator
        elif self.method == 'proba':
            transformer = _predict_proba_single_estimator
        else:
            raise ValueError("Method must be 'standard' or 'proba'. "
                             "Got %s instead." % self.method)

        encoding = jl.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            jl.delayed(transformer)(estimator, feats, X)
            for estimator, feats in zip(
                np.array(self.estimators_)[idx],
                np.array(self.estimators_features_)[idx]))

        encoding = np.array(encoding).T

        # # Parallel loop
        # n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
        #                                                      self.n_jobs)
        #
        # all_proba = jl.Parallel(n_jobs=n_jobs, verbose=self.verbose)(
        #     jl.delayed(_parallel_predict_proba)(
        #         self.estimators_[starts[i]:starts[i + 1]],
        #         self.estimators_features_[starts[i]:starts[i + 1]],
        #         X,
        #         self.n_classes_)
        #     for i in range(n_jobs))
        #
        # # Reduce
        # proba = sum(all_proba) / self.n_estimators

        return encoding

    def _set_oob_score(self, X, y):
        n_samples = y.shape[0]

        # predictions = np.zeros((n_samples, self.n_estimators)) - 1
        # oob_score = np.zeros(self.n_estimators)

        # for i, (estimator, samples, split, features) in enumerate(zip(
        #         self.estimators_, self.estimators_samples_,
        #         self.estimators_splits_, self.estimators_features_)):
        #     # Create mask for OOB samples
        #     samples = indices_to_mask(samples, n_samples)
        #     mask = ~samples
        #
        #     predictions[mask, i] = estimator.predict(X[mask][:, features])
        #
        #     oob_score[i] = accuracy_score(split[mask], predictions[mask, i])
        #
        #     if self.verbose > 1 and i % 20 == 0:
        #         print("Encoding. Done %d/%d" % (i + 1, self.n_estimators),
        #               end="\r", file=sys.stderr)
        predictions, oob_score = zip(
            *jl.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                jl.delayed(_predict_score_single_estimator)(
                    estimator, X[:, features], samples, split, n_samples)
                for estimator, samples, split, features in zip(
                    self.estimators_, self.estimators_samples_,
                    self.estimators_splits_, self.estimators_features_)))

        predictions = np.array(predictions, dtype=np.int8).T
        # self.oob_decision_function_ = oob_decision_function
        self.prediction_ = predictions
        self.oob_score_ = oob_score

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(SlidingECOC, self)._validate_estimator(
            default=DecisionTreeClassifier())

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return y


def _predict_score_single_estimator(estimator, X, samples, split,
                                    n_samples):
    # Create mask for OOB samples
    samples = indices_to_mask(samples, n_samples)
    mask = ~samples

    predictions = np.empty(n_samples, dtype=np.int8)
    predictions[mask] = estimator.predict(X[mask])

    oob_score = accuracy_score(split[mask], predictions[mask])
    return predictions, oob_score
