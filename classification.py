import warnings
import numpy as np
from sys import stdout
import time

class BinaryLogisticRegression:
  """
  Notes
  -----
  Logistic Regression model for binary classification. 
  Trains the model using gradient descent method.

  Parameters
  ----------
  X: :type:`np.ndarray` of shape `(n_samples, n_features)`
     Features to train the model on.

  y: :type:`np.ndarray` of shape `(n_samples,)` or `(n_samples, 1)`
     Labels coresponding to the training data `X`.
  
  pretrained_weights: :type:`np.ndarray` of shape `(n_samples+1, 1)` default = `None`
     Initializes the weights using pretrained_weights, if provided.
  
  Methods
  -------
  :func:`fit`: Fits the model to the provided dataset and trains the weights.
  
  :func:`predict`: Predicts labels on new datasets.
  
  :func:`get_params`: Returns the trained parameters of the model.

  Examples
  --------
  >>> from classification import BinaryLogisticRegression
  >>> from sklearn.datasets import make_classification
  >>> from sklearn.model_selection import train_test_split
  >>> X, y = make_classification()
  >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
  >>> model = BinaryLogisticRegression(X_train, y_train)
  >>> model.fit(epochs=1000, lr=0.01, metrics=['accuracy', 'precision', 'recall', 'f1'], loss='bce')
  Epoch 1000/1000: loss: 0.1771 metric scores: {'accuracy': 0.9467, 'precision': 0.9302, 'recall': 0.9756, 'f1': 0.9524} time taken: 2.2941
  >>> preds = model.predict(y_test)
  >>> trained_weights = model.get_params()
  
  """
  def __init__(self, X, y, pretrained_weights = None):
    # check if the dataset and labels are consistent with our convention.
    if X is None or y is None:
      raise ValueError(
          "Training datasets X and y can't be `None`."
      )
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
      raise ValueError(
          "Training datasets must be "
          f"of type `numpy.ndarray`."
      )
    
    if len(X.shape) != 2:
      raise ValueError(
          "Expected an 2 dimentional array. "
          f"found a {len(X.shape)} dimentional array."
      )

    # initialize the class variables using given datasets
    self.X = X
    self.y = y.reshape(-1,1)
    self.n_features = X.shape[1]
    self.n_samples = X.shape[0]
    self.metric_scores = dict()

    # check if the size of lables is consistent with number of samples.
    if y.size != self.n_samples:
      raise ValueError(
          "Size of the labels doesn't match "
          "with the number of samples.  "
          f"size of labels found: {y.size}. "
          f"size accepted: {self.n_samples}"
      )
    
    # check if the pretrained weights are consistent with our convention
    if pretrained_weights is not None:
      if not isinstance(pretrained_weights, np.ndarray):
        raise TypeError(
            "Pretrained weights must be of the type numpy.ndarray. "
            f"found {pretrained_weights.__class__}."
        )
      if pretrained_weights.shape[0] != self.n_features+1:
        warnings.warn(
            f"Weights must be of the shape ({self.n_features+1},1). "
            f"found {pretrained_weights.shape}. "
            f"reshaping the weights to shape ({self.n_features+1},1).",
            RuntimeWarning
        )
        try:
          pretrained_weights = pretrained_weights.reshape(self.n_features+1,1)
        except ValueError:
          raise ValueError(
              "Pretrained weights are not consistent with the shape of the dataset. "
              f"shape expected: ({self.n_features+1},1). "
              f"found: {pretrained_weights.shape}."
          )
    
    # if consistent, initialize the weights of the model
    self.w = pretrained_weights
    self.loss = None

    # link the class methods using a dictionary for oss and metrics.
    self.LOSS_TAGS = {"mse": (self._mse, self._mse_grad), "bce": (self._bce, self._bce_grad)}
    self.METRICS_TAGS = {"accuracy": self._accuracy, "recall": self._recall, "precision": self._pre, "f1": self._f1}
    
    self.is_fitted = False
  
  def _sigmoid(self, z):
    """
    evaluates the sigmoid function at given input.
    """
    return 1./(1. + np.exp(-z))
  
  def _sigmoid_prime(self, z):
    """
    evaluate the derivative of sigmoid function.
    """
    return self._sigmoid(z)*(1. - self._sigmoid(z))
  
  def _decision_function(self, X_test):
    """
    evaluates the dicision function.
    """
    return self._sigmoid(X_test @ self.w)
  
  def _mse(self, preds):
    """
    evaluates the mean squared error between predicitons and labels.
    """
    return (1./self.n_samples)*np.sum((self.y - preds)**2)
  
  def _mse_grad(self, preds):
    """
    evaluates the gradient of mse function for predictions.
    """
    return (2./self.n_samples)*(self.X.T @ ((preds - self.y) * self._sigmoid_prime(preds) ))
  
  def _bce(self, preds):
    """
    evaluates the Binary Cross Entropy function on given predictions.
    """
    return (-1./self.n_samples)*np.sum(np.nan_to_num( self.y*np.log(preds) + (1-self.y)*np.log(1-preds) ))
  
  def _bce_grad(self, preds):
    """
    evaluates the gradient of binary cross entropy function at given predictions.
    """
    return (1./self.n_samples)*(self.X.T @ (preds - self.y))
  
  def _calc_loss(self, preds):
    """
    calculates the pre-set loss function at given predictions.
    """
    return self.LOSS_TAGS[self.loss][0](preds)
  
  def _calc_loss_grad(self, preds):
    """
    calculates gradient of pre-set loss function at given predictions.
    """
    return self.LOSS_TAGS[self.loss][1](preds)
  
  def _accuracy(self, preds):
    """
    calculates the accuracy of the models at given predictions.
    """
    return (1./self.n_samples)*np.sum(self.y == (preds > 0.5))
  
  def _recall(self, preds):
    """
    calculates the recall of the model at given predictions.
    """
    tp = np.sum((self.y == 1) & ((preds>0.5) == 1))
    fn = np.sum((self.y != 0) & ((preds>0.5) == 0))
    return tp/(tp+fn)
  
  def _pre(self, preds):
    """
    calculates the precision of the model at given predictions.
    """
    tp = np.sum((self.y == 1) & ((preds>0.5) == 1))
    fp = np.sum((self.y != 1) & ((preds>0.5) == 1))
    return tp/(tp+fp)
  
  def _f1(self, preds):
    """
    calculates the F1 score of the model at given predictions.
    """
    rec = self._recall(preds)
    prec = self._pre(preds)
    return (2.*rec*prec/(rec + prec))
  
  def _calc_metric(self, metric, preds):
    """
    calaculates values of the provided metrics at the training time.
    """
    return self.METRICS_TAGS[metric](preds)
    
  def _update_params(self, preds, lr):
    """
    updates parameters according to the gradient descent formula.
    """
    self.w = self.w - lr*self.LOSS_TAGS[self.loss][1](preds)
  
  def fit(self, epochs=100, lr=0.01, metrics=['accuracy'], loss='mse'):
    """
    Fits the model to the provided dataset.

    Parameters
    ----------
    epochs: :type:`int` Number of epochs to train the model.

    lr: :type:`float` learning rate of the model.

    metrics: :type:`python list` A list of metrics to evaluate.
                    Can be ['accuracy', 'f1', 'recall', 'precision']
    
    loss: :type:`string` The loss function to use for training the model.
                    Can be either `'mse'` for mean squared error or `'bce'` for binary cross entropy.
    
    Returns
    -------
    `None`
    
    Examples
    --------
    >>> from classification import BinaryLogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification()
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = BinaryLogisticRegression(X_train, y_train)
    >>> model.fit(epochs=1000, lr=0.01, metrics=['accuracy', 'precision', 'recall', 'f1'], loss='bce')
    Epoch 1000/1000: loss: 0.1771 metric scores: {'accuracy': 0.9467, 'precision': 0.9302, 'recall': 0.9756, 'f1': 0.9524} time taken: 2.2941
    >>> preds = model.predict(y_test)
    >>> trained_weights = model.get_params()

    Notes
    -----
    You can implement your own loss function and add the
    function and its gradient to the dinctionary `self.LOSS_TAGS`
    as a tuple `(func, func_grad)`. Then you can use the key to use
    the custom loss to train your model.

    Similarly, you can add any custom metric to `self.METRICS_TAGS`
    """

    # update the loss to given loss function
    self.loss = loss

    # check if the first column of dataset is ones (for intercept).
    # if it is, we need not add a column of ones. Else, add a column of ones.
    if self.X.shape[1] == self.n_features and np.sum(self.X[:,0] != np.ones((self.n_samples,), dtype=np.float32))!=0:
        self.X = np.concatenate((np.ones((self.n_samples,1), dtype=np.float32), self.X), axis=1)
    
    # if weights have not been initialized, initialize them using gaussian initialization.
    if self.w is None:
      self.w = np.random.randn(self.n_features+1, 1)
    
    # check if the given loss exists or is present in this version.
    if self.loss not in self.LOSS_TAGS.keys():
      raise TypeError(
          "The loss function passed as argument is "
          "not supported in this version "
          "or doesn't exists. "
          f"found: {self.loss}. "
          f"supported losses: {list(self.LOSS_TAGS.keys())}"
      )
    
    # check if the given metrix exists or is present in this version.
    for metric in metrics:
      if metric not in self.METRICS_TAGS.keys():
        raise TypeError(
            "The metric passed as argument is "
            "not supported in this version "
            "or doesn't exists. "
            f"found: {metric}. "
            f"supported metrics: {list(self.METRICS_TAGS.keys())}"
        )
    
    # start training.
    start = time.time()
    self.is_fitted = True
    for i in range(epochs):
      preds = self._decision_function(self.X)
      loss = self._calc_loss(preds)
      for metric in metrics:
        self.metric_scores[metric] = np.round(self._calc_metric(metric, preds), 4)
      self._update_params(preds, lr)
      stdout.write(
          f"\rEpoch {i+1}/{epochs}: loss: {loss:.4f} metric scores: {self.metric_scores} time taken: {(time.time() - start):.4f}"
      )
    print("")
  
  def predict(self, X_test, probs=False):
    """
    Predicts lables on a unlabled dataset.

    Parameters
    ----------
    X_test: :type:`np.ndarray` of shape `(new_n_samples, n_features)`
           Unlabled dataset.

    probs: :type:`boolean` default = `False`.
           If `True`, returns the probabilities.
    
    Returns
    -------
    preds: :type:`np.ndarray` of shape `(new_n_samples, 1)` 
           Predictions

    Examples
    --------
    >>> from classification import BinaryLogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification()
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = BinaryLogisticRegression(X_train, y_train)
    >>> model.fit(epochs=1000, lr=0.01, metrics=['accuracy', 'precision', 'recall', 'f1'], loss='bce')
    Epoch 1000/1000: loss: 0.1771 metric scores: {'accuracy': 0.9467, 'precision': 0.9302, 'recall': 0.9756, 'f1': 0.9524} time taken: 2.2941
    >>> preds = model.predict(y_test)
    >>> trained_weights = model.get_params()
    """
    # check if the array is consistent with the convention
    if not isinstance(X_test, np.ndarray):
      raise TypeError(
          "The argument passed must be a numpy array"
          " of shape (n?, n_features)."
      )
    elif len(X_test.shape) != 2:
      raise ValueError(
          f"Dimentions of the dataset must be 2. found {len(X_test.shape)}"
      )
    if X_test is not self.X and X_test.shape[1] != self.n_features:
      raise ValueError(
          "Shape of the new dataset is not consistent"
          "with the shape of training dataset. "
          f"shape found: (n_samples, {X_test.shape[1]}). "
          f"shape expected: (n_samples, {self.X.shape[1]-1})"
      )
    if X_test.shape[1] == self.n_features and np.sum(X_test[:,0] != np.ones((X_test.shape[0],), dtype=np.float32))!=0:
      X_test = np.concatenate((np.ones((X_test.shape[0],1), dtype=np.float32), X_test), axis=1)
    # if probs=True, return the probabilty of labels being 1.
    if probs:
      return self._decision_function(X_test)
    # otherwise, return the labels.
    return 1*(self._decision_function(X_test) > 0.5)

  def get_params(self):
    """
    Returns a copy of parameters of the fitted model.

    Returns
    -------
    w: :type:`np.ndarray` of shape `(n_features+1, 1)`
       Vector of weights of the model with intercept.

    Examples
    --------
    >>> from classification import BinaryLogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification()
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> model = BinaryLogisticRegression(X_train, y_train)
    >>> model.fit(epochs=1000, lr=0.01, metrics=['accuracy', 'precision', 'recall', 'f1'], loss='bce')
    Epoch 1000/1000: loss: 0.1771 metric scores: {'accuracy': 0.9467, 'precision': 0.9302, 'recall': 0.9756, 'f1': 0.9524} time taken: 2.2941
    >>> preds = model.predict(y_test)
    >>> trained_weights = model.get_params()
    """
    return self.w.copy()
  
  def __repr__(self):
    return f"<BinaryLogisticRegressor at {id(self)}>"
  
  def __str__(self):
    if self.is_fitted:
      return f"Trained BinaryLogisticRegressor with metric scores {self.metric_scores} and loss {self._calc_loss(self.predict(self.X, probs=True))} at {id(self)}"
    return f"Untrained BinaryLogisticRegressor at {id(self)}"

  def __getstate__(self):
    state = self.__dict__.copy()
    return state
  
  def __setstate__(self, state):
    self.__dict__.update(state)
