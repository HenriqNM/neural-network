import numpy as np

L = 3
n = [2, 3, 3, 1]

np.random.seed(42)
W1 = np.random.randn(n[1], n[0]) * np.sqrt(2.0 / n[0])
W2 = np.random.randn(n[2], n[1]) * np.sqrt(2.0 / n[1])
W3 = np.random.randn(n[3], n[2]) * np.sqrt(2.0 / n[2])
b1 = np.zeros((n[1], 1))
b2 = np.zeros((n[2], 1))
b3 = np.zeros((n[3], 1))

# Adam momentum
m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
m_W3, v_W3 = np.zeros_like(W3), np.zeros_like(W3)

m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
m_b3, v_b3 = np.zeros_like(b3), np.zeros_like(b3)

# Parametros de Adam
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

def prepare_data():
    X = np.array([
        [150, 70],
        [254, 73],
        [312, 68],
        [120, 60],
        [154, 61],
        [212, 65],
        [216, 67],
        [145, 67],
        [184, 64],
        [130, 69] 
    ])
    y = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    m = 10
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    A0 = X.T
    Y = y.reshape(n[L], m)

    return A0, Y, m

def sigmoid(arr):
    return 1 / (1 + np.exp(-1 * arr))

def relu(arr):
    return np.maximum(0, arr)

def relu_derivative(Z):
    return np.where(Z > 0, 1.0, 0.0)

def feed_foward(A0, W1, W2, W3, b1, b2, b3):
    Z1 = W1 @ A0 + b1
    A1 = relu(Z1)

    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)

    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    cache = {
        "A0": A0, "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3
    }
    return A3, cache

def backpropagation(cache, Y, W2, W3, m):
    A0 = cache["A0"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    Z1 = cache["Z1"]
    Z2 = cache["Z2"]

    # Layer 3 (Output)
    dZ3 = (1 / m) * (A3 - Y)
    dW3 = dZ3 @ A2.T
    db3 = np.sum(dZ3, axis=1, keepdims=True)

    # Layer 2
    dZ2 = (W3.T @ dZ3) * relu_derivative(Z2)
    dW2 = dZ2 @ A1.T
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    # Layer 1
    dZ1 = (W2.T @ dZ2) * relu_derivative(Z1)
    dW1 = dZ1 @ A0.T
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

A0, Y, m = prepare_data()
epochs = 5000
alpha = 0.1

for e in range(epochs):
    t = e + 1

    # 1. Feed Forward
    y_hat, cache = feed_foward(A0, W1, W2, W3, b1, b2, b3)

    # 2. Backpropagation
    dW1, db1, dW2, db2, dW3, db3 = backpropagation(cache, Y, W2, W3, m)

    # 3. Atualização do gradiente via Optimizador de Adam
    for W, dW, m_W, v_W in zip([W1, W2, W3], [dW1, dW2, dW3], [m_W1, m_W2, m_W3], [v_W1, v_W2, v_W3]):
        # Cálculo de momentos
        m_W[:] = beta1 * m_W + (1 - beta1) * dW
        v_W[:] = beta2 * v_W + (1 - beta2) * (dW ** 2)

        # Correção de Bias
        m_hat = m_W / (1 - (beta1 ** t))
        v_hat = v_W / (1 - (beta2 ** t))

        # Atualiza o peso
        W -= (alpha / (np.sqrt(v_hat) + epsilon)) * m_hat
    
    for b, db, m_b, v_b in zip([b1, b2, b3], [db1, db2, db3], [m_b1, m_b2, m_b3], [v_b1, v_b2, v_b3]):
        # Cálculo de momentos
        m_b[:] = beta1 * m_b + (1 - beta1) * db
        v_b[:] = beta2 * v_b + (1 - beta2) * (db ** 2)

        # Correção de Bias
        m_hat = m_b / (1 - (beta1 ** t))
        v_hat = v_b / (1 - (beta2 ** t))

        # Atualiza o bias
        b -= (alpha / (np.sqrt(v_hat) + epsilon)) * m_hat


    if e % 200 == 0:
        y_hat_clipped = np.clip(y_hat, 1e-15, 1 - 1e-15)
        cost = -np.mean(Y * np.log(y_hat_clipped) + (1 - Y) * np.log(1 - y_hat_clipped))
        print(f"Epoch {e} - Cost (BCE): {cost:.6f}")

print("Result:")
print(feed_foward(A0, W1, W2, W3, b1, b2, b3)[0])