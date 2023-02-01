import numpy as np


class Softmax(object):

  def __init__(self, dims=[10, 3073]):
    self.init_weights(dims=dims)

  def init_weights(self, dims):
    """
    Initializes the weight matrix of the Softmax classifier.  
    Note that it has shape (C, D) where C is the number of 
    classes and D is the feature size.
    """
    self.W = np.random.normal(size=dims) * 0.0001

  def loss(self, X, y):
    """
    Calculates the softmax loss.
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    Inputs:
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
  
    Returns a tuple of:
    - loss as single float
    """

    # Initialize the loss to zero.
    loss = 0.0

    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the normalized softmax loss.  Store it as the variable loss.
    #   (That is, calculate the sum of the losses of all the training 
    #   set margins, and then normalize the loss by the number of 
    #   training examples.)
    # ================================================================ #
    for i in range(X.shape[0]):
      x_sample, y_sample = X[i], y[i]
      s = np.log(np.sum(np.exp(x_sample.dot(self.W.T) ))) - x_sample.dot(self.W[y_sample].T)
      loss+=s
    loss = loss / X.shape[0]
    pass
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss

  def loss_and_grad(self, X, y):
    """
    Same as self.loss(X, y), except that it also returns the gradient.

    Output: grad -- a matrix of the same dimensions as W containing 
      the gradient of the loss with respect to W.
    """

    # Initialize the loss and gradient to zero.
    loss = 0.0
    grad = np.zeros_like(self.W)
    num_sample = X.shape[0]
    num_class = self.W.shape[0]
    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the softmax loss and the gradient. Store the gradient
    #   as the variable grad.
    # ================================================================ #
    for i in range(num_sample):
      # compute loss
      x_sample, y_true_label = X[i], y[i]
      # s = np.log(np.sum(np.exp(x_sample.dot(self.W.T) ))) - x_sample.dot(self.W[y_true_label].T)
      # loss+=s
      score = x_sample.dot(self.W.T)
      score = score- score.max()
      true_score = np.exp(score[y_true_label])
      all_sum_score = np.sum(np.exp(score))
      loss -= np.log( true_score / all_sum_score)

      # compute grad
      #get the prob of every class for the i-th sample:
      score = x_sample.dot(self.W.T)
      score_exp = np.exp(score)
      exp_sum = np.sum(score_exp)
      for j in range(num_class):
        if j != y_true_label:
          grad[j] =  grad[j] + score_exp[j]/exp_sum  * x_sample
        else:
          grad[j] = grad[j] + (score_exp[j] / exp_sum - 1) * x_sample

    loss = loss / num_sample
    grad = grad / num_sample
    pass
 
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grad

  def grad_check_sparse(self, X, y, your_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in these dimensions.
    """
  
    for i in np.arange(num_checks):
      ix = tuple([np.random.randint(m) for m in self.W.shape])
  
      oldval = self.W[ix]
      self.W[ix] = oldval + h # increment by h
      fxph = self.loss(X, y)
      self.W[ix] = oldval - h # decrement by h
      fxmh = self.loss(X,y) # evaluate f(x - h)
      self.W[ix] = oldval # reset
  
      grad_numerical = (fxph - fxmh) / (2 * h)
      grad_analytic = your_grad[ix]
      rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
      print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))

  def fast_loss_and_grad(self, X, y):
    """
    A vectorized implementation of loss_and_grad. It shares the same
    inputs and ouptuts as loss_and_grad.
    """
    loss = 0.0
    grad = np.zeros(self.W.shape) # initialize the gradient as zero
    num_sample = X.shape[0]
    num_class = self.W.shape[0]
    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the softmax loss and gradient WITHOUT any for loops.
    # ================================================================ #
    score = X.dot(self.W.T)
    # X: (N,D), W: (C,D), score: (N,C)
    true_label_mat = np.zeros((num_sample,num_class))
    # every row of true_label_mat would only have one 1, where is the true label
    true_label_mat[np.arange(num_sample), y] = 1
    # calculate true score
    true_score = np.sum(np.multiply(score, true_label_mat), axis=1)
    false_score = np.sum(np.exp(score - np.multiply(score, true_label_mat)),axis = 1)
    loss += np.sum(np.log(false_score) - true_score)
    
    score -= np.max(score,axis=1,keepdims=True)
    prob = np.exp(score) / np.sum(np.exp(score),axis=1,keepdims=True)
    grad = X.T.dot(prob - true_label_mat)

    loss = loss / num_sample
    grad = grad.T / num_sample
    pass
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    return loss, grad

  def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train, dim = X.shape
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes

    self.init_weights(dims=[np.max(y) + 1, X.shape[1]])	# initializes the weights of self.W

    # Run stochastic gradient descent to optimize W
    loss_history = []

    for it in np.arange(num_iters):
      X_batch = None
      y_batch = None
      # ================================================================ #
      # YOUR CODE HERE:
      #   Sample batch_size elements from the training data for use in 
      #     gradient descent.  After sampling,
      #     - X_batch should have shape: (batch_size, dim)
      #     - y_batch should have shape: (batch_size,)
      #   The indices should be randomly generated to reduce correlations
      #   in the dataset.  Use np.random.choice.  It's okay to sample with
      #   replacement.
      # ================================================================ #
      idx = np.random.choice(num_train, batch_size)
      X_batch = X[idx]
      y_batch = y[idx]
      pass
      # ================================================================ #
      # END YOUR CODE HERE
      # ================================================================ #

      # evaluate loss and gradient
      loss, grad = self.fast_loss_and_grad(X_batch, y_batch)
      loss_history.append(loss)
      
      self.W += (-learning_rate * grad)
      # ================================================================ #
      # YOUR CODE HERE:
      #   Update the parameters, self.W, with a gradient step 
      # ================================================================ #
      pass

      # ================================================================ #
      # END YOUR CODE HERE
      # ================================================================ #

      if verbose and it % 100 == 0:
        print('iteration {} / {}: loss {}'.format(it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Inputs:
    - X: N x D array of training data. Each row is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[0])
    # ================================================================ #
    # YOUR CODE HERE:
    #   Predict the labels given the training data.
    # ================================================================ #
    score = X.dot(self.W.T)
    # score: (N,C)
    y_pred = np.argmax(score, axis=1)
    pass
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return y_pred

