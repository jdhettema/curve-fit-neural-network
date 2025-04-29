# Neural Network Curve Fitting Visualizer

A lightweight implementation of a neural network built from scratch using NumPy that demonstrates the learning process of fitting curves to various data patterns. This project includes animated visualizations showing how a simple feed-forward neural network gradually improves its predictions as training progresses.

## Features

- Pure NumPy implementation without deep learning frameworks
- Animated visualization of the neural network training process
- Support for multiple data patterns (sine waves, step functions, quadratic, cubic)
- Real-time display of mean squared error during training
- Customizable network architecture and hyperparameters

## Demo

The visualizer animates how the neural network learns to fit a curve to scattered data points:
- Blue dots represent the training data (with added noise)
- The red line shows the neural network's current prediction
- The animation updates to show how the prediction improves with each training epoch

## Requirements

All dependencies are listed in `requirements.txt`:

```
contourpy==1.3.2
cycler==0.12.1
fonttools==4.57.0
kiwisolver==1.4.8
matplotlib==3.10.1
numpy==2.2.5
packaging==25.0
pillow==11.2.1
pyparsing==3.2.3
python-dateutil==2.9.0.post0
six==1.17.0
```

To install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with default settings:

```bash
python neural_network_visualizer.py
```

### Trying Different Data Patterns

The code includes several data generation options. To try different patterns, uncomment the desired pattern in the script:

```python
# Sine wave pattern
y_true = np.sin(X) * 0.5 + 0.5 * X + np.random.normal(0, 0.1, X.shape)

# Step function pattern 
# y_true = (X > 0).astype(float) + np.random.normal(0, 0.1, X.shape)

# Quadratic function pattern
# y_true = X**2 + np.random.normal(0, 0.2, X.shape)

# Cubic function pattern (more challenging)
# y_true = 0.1 * X**3 + np.random.normal(0, 0.2, X.shape)
```

### Adjusting Network Parameters

To modify the neural network architecture or training parameters:

```python
# Change network architecture (input dimensions, hidden layer size, output dimensions)
nn = NeuralNetwork(input_size=1, hidden_size=20, output_size=1)

# Adjust training parameters
history = nn.train(X, y_true, 
                  epochs=10000,         # Total training iterations
                  learning_rate=0.1,    # Step size for gradient updates
                  store_interval=20)    # How often to store network state for animation
```

## How It Works

### Neural Network Architecture

This implementation uses a simple feedforward neural network with:
- 1 input node (X value)
- 1 hidden layer with configurable neurons (default: 10)
- 1 output node (predicted Y value)
- Sigmoid activation for the hidden layer
- Linear activation for the output layer

### Implementation Details

1. **Initialization**: Weights are initialized with small random values scaled by 0.1, and biases are initialized to zero.

2. **Forward Pass**: Input data is processed through the network to make predictions.

3. **Backward Pass**: Gradients are calculated and weights are updated using basic gradient descent.

4. **Visualization**: The training history is recorded at intervals and used to create an animation showing how the network's predictions improve over time.

## Extending the Project

Some ideas for extending this project:
- Add more hidden layers
- Implement different activation functions
- Add regularization to prevent overfitting
- Support for more complex datasets (2D inputs, classification problems)
- Compare with other curve fitting methods

## License

[MIT License](LICENSE)

## Acknowledgments

This project was created as an educational tool to visualize neural network learning dynamics in a simple, intuitive way.