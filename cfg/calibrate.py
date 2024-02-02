import numpy as np
import torch
import torch.optim as optim

# Provided data
K_rate = np.array([19.34, 20.18, 21.27, 21.12, 20.52, 19.78, 19.3, 18.86, 18.59, 18.56, 18.32, 18.25, 18.31, 19.06, 18.3])

L = np.array([
255716.834,
241354.751,
241509.917,
245159.167,
249807.417,
253330.833,
258156,
263674,
267261,
270827.3,
275696,
278646.8,
262006.8,
274317.9,
283184.6642]) *1e6
# L = np.array([258675.833, 255716.834, 241354.751, 241509.917, 245159.167, 249807.417, 253330.833, 258156, 263674, 267261, 270827.3, 275696, 278646.8, 262006.8, 274317.9])
GDP = np.array([14474.228, 14769.862, 14478.067, 15048.97, 15599.731, 16253.97, 16843.196, 17550.687, 18206.023, 18695.106, 19477.337, 20533.058, 21380.976, 21060.474, 23315.081])*1e9
K = K_rate * GDP/100
# Convert numpy arrays to torch tensors
K = torch.tensor(K, dtype=torch.float32)
L = torch.tensor(L, dtype=torch.float32)
GDP = torch.tensor(GDP, dtype=torch.float32)

# Log-transforming K, L, and GDP
K_log = torch.log(K)
L_log = torch.log(L)
GDP_log = torch.log(GDP)

# Initialize alpha closer to 0 to avoid large initial steps
alpha = torch.tensor([0.3], requires_grad=True)
# A = torch.tensor([100.], requires_grad=True)
A = torch.tensor([10.])

# Further decrease learning rate for more precise updates
learning_rate = 0.001

# Gradient clipping value
clip_value = 0.1
# Number of iterations for the gradient descent
iterations = 10000000

# Loss function
def loss_function(GDP_pred, GDP_real):
    return torch.mean((GDP_pred - GDP_real) ** 2)

# Implementing the gradient descent with further adjustments
for i in range(iterations):
    # Predicted log-GDP using the log-transformed Cobb-Douglas production function
    GDP_pred_log = torch.log(A) + alpha * K_log + (1 - alpha) * L_log
    # Calculate loss using the log-transformed values
    loss = loss_function(GDP_pred_log, GDP_log)
    # Zero the gradients before running the backward pass
    alpha.grad = None
    # Perform backpropagation
    loss.backward()
    
    # Gradient clipping
    with torch.no_grad():
        alpha.grad = torch.clamp(alpha.grad, -clip_value, clip_value)
    
    # Update alpha
    with torch.no_grad():
        alpha -= learning_rate * alpha.grad
        # A -= learning_rate * A.grad
    
    # Print every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i + 1}/{iterations}, Loss: {loss.item()}, Alpha: {alpha.item()}, A: {A.item()}")

# Iteration 181001/10000000, Loss: 0.0031980823259800673, Alpha: 0.8590314984321594
