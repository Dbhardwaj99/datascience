# Description:
# Create a program that implements the perceptron algorithm to perform binary classification on a given dataset.
# The program will train a perceptron model using the provided dataset and then allow users to input new data points for prediction.
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)


# Generate positive class points
num_points = 50
x_positive = np.random.uniform(0, 5, num_points)
y_positive = np.random.uniform(0, 5, num_points)
labels_positive = np.ones(num_points)


# Generate negative class points
x_negative = np.random.uniform(-5, 0, num_points)
y_negative = np.random.uniform(-5, 0, num_points)
labels_negative = np.zeros(num_points)


# Combine positive and negative points
x = np.concatenate((x_positive, x_negative))
y = np.concatenate((y_positive, y_negative))
labels = np.concatenate((labels_positive, labels_negative))

# Print the dataset
dataset = np.column_stack((x, y, labels))

def visualize_perceptron(X, y, weights, bias):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')  # Scatter plot of data points

    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])

    # Generate x values to plot the decision boundary line
    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = -(weights[0] / weights[1]) * x_vals - bias / weights[1]

    plt.plot(x_vals, y_vals, 'g--', label='Decision Boundary')  # Plot the decision boundary line
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Perceptron Algorithm')
    plt.show()


def step_function(t):
    if t >= 0:
        return 1
    return 0


def train_perceptron(dataset, learning_rate):
    X = dataset[:, :-1]  # Input features
    y = dataset[:, -1]   # Labels

    weights = np.array(np.random.rand(2, 1))
    bias = np.random.rand(1)[0]

    num_epochs = 100  # Number of training iterations

    for epoch in range(num_epochs):
        for i in range(len(X)):
            # Compute the linear combination of weights and input features
            linear_combination = np.dot(X[i], weights) + bias

            # Apply the step function to get the predicted output
            predicted_output = step_function(linear_combination)

            # Update the weights and bias based on the prediction error
            weights += learning_rate * (y[i] - predicted_output) * X[i]
            bias += learning_rate * (y[i] - predicted_output)
            visualize_perceptron(X, y, weights, bias)

    return weights, bias



train_perceptron(dataset, learning_rate=0.01)