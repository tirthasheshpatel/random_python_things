import warnings
import numpy as np
from sys import stdout
import time

class BinaryLogisticRegression:
  def __init__(self, X, y, pretrained_weights = None):
    if X is None or y is None:
      raise ValueError(
          "Training datasets X and y can't be `None`. \n"
      )
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
      raise ValueError(
          "Training datasets must be "
          f"of type `numpy.ndarray`. \n"
      )
    self.X = X
    self.y = y.reshape(-1,1)
    self.n_features = X.shape[1]
    self.n_samples = X.shape[0]
    self.metric_scores = dict()
    if y.size != self.n_samples:
      raise ValueError(
          "Size of the labels doesn't match "
          "with the number of samples. \n"
          f"size of labels found: {y.size} \n"
          f"size accepted: {self.n_samples} \n"
      )
    if pretrained_weights is not None:
      if not isinstance(pretrained_weights, np.ndarray):
        raise TypeError(
            "Pretrained weights must be of the type numpy.ndarray. \n"
            f"found {pretrained_weights.__class__}. \n"
        )
      if pretrained_weights.shape[0] != self.n_features+1:
        warnings.warn(
            f"Weights must be of the shape ({self.n_features+1},1). \n"
            f"found {pretrained_weights.shape}. \n"
            f"reshaping the weights to shape ({self.n_features+1},1). \n",
            RuntimeWarning
        )
        try:
          pretrained_weights = pretrained_weights.reshape(self.n_features+1,1)
        except ValueError:
          raise ValueError(
              "Pretrained weights don't match with the "
              f"shape ({self.n_features+1},1). \n"
              f"Found {pretrained_weights.shape}. \n"
          )
    self.w = pretrained_weights
    self.loss = None
    self.LOSS_TAGS = {"mse": (self._mse, self._mse_grad), "bce": (self._bce, self._bce_grad)}
    self.METRICS_TAGS = {"accuracy": self._accuracy, "recall": self._recall, "precision": self._pre, "f1": self._f1}
    self.is_fitted = False
  
  def _sigmoid(self, z):
    return 1./(1. + np.exp(-z))
  
  def _sigmoid_prime(self, z):
    return self._sigmoid(z)*(1. - self._sigmoid(z))
  
  def _decision_function(self):
    return self._sigmoid(self.X @ self.w)
  
  def _mse(self, preds):
    return (1./self.n_samples)*np.sum((self.y - preds)**2)
  
  def _mse_grad(self, preds):
    return (2./self.n_samples)*(self.X.T @ ((preds - self.y) * self._sigmoid_prime(preds) ))
  
  def _bce(self, preds):
    return (-1./self.n_samples)*np.sum(np.nan_to_num( self.y*np.log(preds) + (1-self.y)*np.log(1-preds) ))
  
  def _bce_grad(self, preds):
    return (1./self.n_samples)*(self.X.T @ (preds - self.y))
  
  def _calc_loss(self, preds):
    return self.LOSS_TAGS[self.loss][0](preds)
  
  def _calc_loss_grad(self, preds):
    return self.LOSS_TAGS[self.loss][1](preds)
  
  def _accuracy(self, preds):
    return (1./self.n_samples)*np.sum(self.y == (preds > 0.5))
  
  def _recall(self, preds):
    tp = np.sum((self.y == 1) & ((preds>0.5) == 1))
    fn = np.sum((self.y != 0) & ((preds>0.5) == 0))
    return 100.*(1./self.n_samples)*tp/(tp+fn)
  
  def _pre(self, preds):
    tp = np.sum((self.y == 1) & ((preds>0.5) == 1))
    fp = np.sum((self.y != 1) & ((preds>0.5) == 1))
    return 100.*(1./self.n_samples)*tp/(tp+fp)
  
  def _f1(self, preds):
    rec = self._recall(preds)
    prec = self._pre(preds)
    return 100.*(1./self.n_samples)*(2.*rec*prec/(rec + prec))
  
  def _calc_metric(self, metric, preds):
    return self.METRICS_TAGS[metric](preds)
    
  def _update_params(self, preds, lr):
    self.w = self.w - lr*self.LOSS_TAGS[self.loss][1](preds)
  
  def fit(self, epochs=100, lr=0.01, metrics=['accuracy'], loss='mse'):
    self.loss = loss
    self.X = np.concatenate((np.ones((self.n_samples,1), dtype=np.float32), X), axis=1)
    
    if self.w is None:
      self.w = np.random.randn(self.n_features+1, 1)
      
    if self.loss not in self.LOSS_TAGS.keys():
      raise TypeError(
          "The loss function passed as argument is "
          "not supported in this version "
          "or doesn't exists. \n"
          f"found: {self.loss} \n"
          f"supported losses: {list(self.LOSS_TAGS.keys())} \n"
      )
      
    for metric in metrics:
      if metric not in self.METRICS_TAGS.keys():
        raise TypeError(
            "The metric passed as argument is "
            "not supported in this version "
            "or doesn't exists. \n"
            f"found: {metric}\n"
            f"supported metrics: {list(self.METRICS_TAGS.keys())}\n"
        )
        
    start = time.time()
    self.is_fitted = True
    for i in range(epochs):
      preds = self._decision_function()
      loss = self._calc_loss(preds)
      for metric in metrics:
        self.metric_scores[metric] = np.round(self._calc_metric(metric, preds), 4)
      self._update_params(preds, lr)
      stdout.write(
          f"\rEpoch {i+1}/{epochs}: loss: {loss:.4f} metric scores: {self.metric_scores} time taken: {(time.time() - start):.4f}"
      )
    print("\n")
  
  def predict(self, probs=False):
    if probs:
      return self._decision_function()
    return 1*(self._decision_function() > 0.5)
    
  def get_params(self):
    return self.w.copy()
  
  def __repr__(self):
    return f"<BinaryLogisticRegressor at {id(self)}>"
  
  def __str__(self):
    if self.is_fitted:
      return f"Trained BinaryLogisticRegressor with metric scores {self.metric_scores} and loss {self._calc_loss(self.predict(probs=True))} at {id(self)}"
    return f"Untrained BinaryLogisticRegressor at {id(self)}"

  def __getstate__(self):
    state = self.__dict__.copy()
    return state
  
  def __setstate(self, **state):
    self.__dict__.update(state)
