import numpy as np
import math

def sigmoid(z):
  """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """
  
  g = 1 / 1 + np.exp(-z)

  return g

def compute_logreg_cost(X, y, w, b, lambda_=0.1):
  """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost : (scalar) cost 
    """
  
  m = X.shape[0]
  n = len(w)

  cost = 0.

  for i in range(m):
    z_wb_i = np.dot(w, X[i]) + b
    f_wb_i = sigmoid(z_wb_i)
    cost += -y[i] * np.log(f_wb_i) - (1-y[i]) * np.log(1-f_wb_i)
  
  cost = cost / m

  reg_cost = 0
  for j in range(n):
    reg_cost += (w[j] ** 2)

  reg_cost = reg_cost * (lambda_/(2*m))

  total_cost = cost + reg_cost

  return total_cost

def compute_gradient_descent_logregression(X, y, w, b, lambda_=0.1):
  """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
  
  m, n = X.shape
  dj_dw = np.zeros((n,))
  dj_db = 0

  for i in range(m):
    z_fw_i = np.dot(w, X[i]) + b
    f_fw_i = sigmoid(z_fw_i)
    error_i = f_fw_i - y[i]
    for j in range(n):
      dj_dw[j] += error_i * X[j]
    
    dj_dw = dj_dw / m
    dj_db = (dj_db + error_i)/m

  for j in range(n):
    dj_dw[j] = dj_dw + (lambda_/m) * w[j]

  return dj_dw, dj_db


def gradient_descent_logregression(X, y, w_in, b_in, alpha, lambda_=0.1, num_iters=1000, verbose=False): 
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      lambda_ (scalar)   : Controls amount of regularization
      num_iters (scalar) : number of iterations to run gradient descent
      verbose (logical)  : Print cost every at intervals 10 times or as many iterations if < 10
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
      # Calculate logistic regression parameters (i.e., derivatives)
      dj_dw, dj_db = compute_gradient_descent_logregression(X, y, w, b, lambda_)

      w = w - alpha * dj_dw
      b = b - alpha * dj_db

      # save regularized cost
      if i < 1000000: # prevent resource exhaustion 
            J_history.append(compute_logreg_cost(X, y, w, b, lambda_))

      # Print cost every at intervals 10 times or as many iterations if < 10
      if verbose == True:
        if i % math.ceil(num_iters / 10) == 0:
          print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history         #return final w,b and J history for graphing