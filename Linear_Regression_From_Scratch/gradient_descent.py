import numpy as np
import matplotlib.pyplot as plt

def initialize(dim):
  w1 = np.random.rand(dim)
  w0 = np.random.rand()
  return w1, w0

def compute_cost(X,Y, y_hat):
    m = len(Y)
    cost = (1/(2*m)) * np.sum(np.square(y_hat - Y))
    return cost

def predict_y(X,w1,w0):
  if len(w1)==1:
    w1 = w1[0]
    return X*w1+w0
  return np.dot(X,w1)+w0

def update_parameters(X,Y,y_hat,cost,w0,w1,learning_rate):
  m = len(Y)
  db=(np.sum(y_hat-Y))/m
  dw=np.dot(y_hat-Y,X)/m

  w0_new=w0-learning_rate*db
  w1_new=w1-learning_rate*dw
  return w0_new,w1_new

def run_gradient_descent(X,Y,alpha,max_iterations,stopping_threshold = 1e-6):
  dims = 1
  if len(X.shape)>1:
    dims = X.shape[1]
  w1,w0=initialize(dims)
  previous_cost = None
  cost_history = np.zeros(max_iterations)
  for itr in range(max_iterations):
    y_hat=predict_y(X,w1,w0)
    cost=compute_cost(X,Y,y_hat)
    # early stopping criteria
    if previous_cost and abs(previous_cost-cost)<=stopping_threshold:
      break
    cost_history[itr]=cost
    previous_cost = cost
    old_w1=w1
    old_w0=w0
    w0,w1=update_parameters(X,Y,y_hat,cost,old_w0,old_w1,alpha)

  return w0,w1,cost_history

# Set hyperparameters
learning_rate = 0.0001
iterations = 1000

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
m = 200

# generate random features X with two dimensions
X = 10 * np.random.rand(m, 2)  # Scaling by 10 for variety

# generate target variable y with some constant weights and bias
true_weights = np.array([3, 4]) 
bias = 2  
Y = X.dot(true_weights) + bias + np.random.randn(m)  # adding some random noise

w0,w1,cost_history = run_gradient_descent(X,Y,learning_rate,iterations)
print(w0,w1)

# Plot the cost history
plt.plot(range(1, iterations + 1), cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')
plt.show()