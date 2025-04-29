import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
# y_true = np.sin(X) * 0.5 + 0.5 * X + np.random.normal(0, 0.1, X.shape)
# y_true = (X > 0).astype(float) + np.random.normal(0, 0.1, X.shape)
# y_true = X**2 + np.random.normal(0, 0.2, X.shape)
y_true = 0.1 * X**3 + np.random.normal(0, 0.2, X.shape)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size=1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1 # Distributes random values according to the gaussian distribution (Mean: 1, Std: 0.1). Then multiplies by 0.1 to scale down the values. This scaling down is sometimes called "Xavier" or "He" initialization.
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        self.history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2
        return self.a2
    
    def backward(self, X, y, learning_rate=0.1):
        m = X.shape[0]

        dz2 = (self.a2 - y) / m
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=10000, learning_rate=0.1, store_interval=50):
        for epoch in range(epochs):
            predictions = self.forward(X)
            self.backward(X, y, learning_rate)

            if epoch % store_interval == 0:

                weights_copy = {
                    'W1': self.W1.copy(),
                    'b1': self.b1.copy(),
                    'W2': self.W2.copy(),
                    'b2': self.b2.copy()
                }
                mse = np.mean((predictions - y) ** 2)
                self.history.append((weights_copy, mse))

                if epoch % (store_interval * 10) == 0:
                    print(f"Epoch {epoch}, MSE: {mse}")

        return self.history
    
nn = NeuralNetwork(input_size=1, hidden_size=10, output_size=1)

history = nn.train(X, y_true, epochs = 7000, learning_rate=1, store_interval=20)

def predict_with_weights(weights, X):
    W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']
    a1 = 1 / (1 + np.exp(-(np.dot(X, W1) + b1)))
    return np.dot(a1, W2) + b2


fig, ax = plt.subplots(figsize=(10, 6))
line_pred, = ax.plot([], [], 'r-', lw=2, label='Prediction')
line_data = ax.scatter(X, y_true, alpha=0.5, label='Training data')
ax.set_xlim(X.min(), X.max())
ax.set_ylim(min(y_true) - 0.5, max(y_true) + 0.5)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Neural Network Curve Fitting')
ax.legend()
txt = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    line_pred.set_data([], [])
    return line_pred,

def animate(i):
    weights, mse = history[i]
    X_sorted = np.sort(X, axis=0)
    y_pred = predict_with_weights(weights, X_sorted)
    line_pred.set_data(X_sorted, y_pred)
    txt.set_text(f'Epoch: {i * 20}, MSE: {mse:.4f}')
    return line_pred, txt

ani = FuncAnimation(fig, animate, frames=len(history), init_func=init, blit=True, interval=50)
plt.show()
                         