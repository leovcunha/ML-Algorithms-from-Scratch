import numpy as np


class MultilayerPerceptron:
    """
    Implementation of a Multilayer Perceptron (MLP) neural network from scratch.

    This implementation supports multiple hidden layers with ReLU activation
    and a sigmoid output layer for binary classification.

    Example:
        # Binary Classification Network:
        layer_dims = [4, 5, 3, 1]  # Creates:
        # - Input layer: 4 features
        # - First hidden layer: 5 neurons
        # - Second hidden layer: 3 neurons
        # - Output layer: 1 neuron (binary classification)

        mlp = MultilayerPerceptron(
            layer_dims=[4, 5, 3, 1],
            learning_rate=0.01,
            n_iterations=3000,
            random_seed=3,
            verbose=True
        )

        # Training
        X = np.random.randn(4, 100)  # 100 samples, 4 features each
        y = np.random.randint(0, 2, (1, 100))  # Binary labels
        costs = mlp.fit(X, y)  # Returns list of costs during training

        # Predict
        predictions = mlp.predict(X)  # Returns binary predictions (0/1)
    """

    def __init__(
        self,
        layer_dims,
        learning_rate=0.01,
        n_iterations=3000,
        batch_size=None,
        random_seed=3,
        verbose=False,
    ):
        """
        Initialize MLP parameters.

        Args:
            layer_dims: array containing dimensions of each layer [input_size, hidden1_size, ..., output_size]
            learning_rate: step size for gradient descent (default: 0.01)
            n_iterations: number of training iterations (default: 3000)
            batch_size: size of mini-batches (default: None, use full batch)
            random_seed: seed for weight initialization (default: 3)
            verbose: whether to print training progress (default: False)
        """
        if len(layer_dims) < 2:
            raise ValueError("Network must have at least 2 layers (input and output)")
        if not all(isinstance(dim, int) and dim > 0 for dim in layer_dims):
            raise ValueError("All layer dimensions must be positive integers")
        self.layer_dims = layer_dims  # Store the layer dimensions
        self.parameters = self._initialize_parameters_deep(layer_dims, random_seed)
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.verbose = verbose
        self.L = len(layer_dims) - 1  # number of layers (excluding input)

        # Cache storage
        self.linear_cache = {}  # Stores (A_prev, W, b) for each layer
        self.activation_cache = {}  # Stores Z for each layer

    def _initialize_parameters_deep(self, layer_dims, random_seed):
        """
        Initialize parameters for L-layer network.

        Args:
            layer_dims: array containing dimensions of each layer
            random_seed: seed for reproducible initialization

        Returns:
            parameters: Dictionary containing weights W and biases b for each layer
        """
        np.random.seed(random_seed)
        parameters = {}
        L = len(layer_dims)

        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(
                layer_dims[l], layer_dims[l - 1]
            ) * np.sqrt(
                2.0 / layer_dims[l - 1]
            )  # He initialization
            parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

            assert parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1])
            assert parameters["b" + str(l)].shape == (layer_dims[l], 1)

        return parameters

    def _linear_forward(self, A, W, b, layer):
        """
        Implement linear part of forward propagation layer.

        Args:
            A: activations from previous layer (or input data), shape (prev_layer_size, m)
            W: weights matrix, shape (current_layer_size, prev_layer_size)
            b: bias vector, shape (current_layer_size, 1)
            layer: current layer number for caching

        Returns:
            Z: linear output, shape (current_layer_size, m)
        """
        Z = np.dot(W, A) + b
        assert Z.shape == (W.shape[0], A.shape[1])

        # Store cache for backward pass
        self.linear_cache[layer] = (A, W, b)

        return Z

    def _sigmoid(self, Z, layer):
        """
        Compute sigmoid activation.

        Args:
            Z: linear output from current layer
            layer: current layer number for caching

        Returns:
            A: output activation
        """
        A = 1 / (1 + np.exp(-Z))
        self.activation_cache[layer] = Z
        return A

    def _softmax(self, Z, layer):
        """
        Compute softmax activation.

        Args:
            Z: linear output from current layer
            layer: current layer number for caching

        Returns:
            A: output activation
        """
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Prevent overflow
        A = expZ / np.sum(expZ, axis=0, keepdims=True)
        self.activation_cache[layer] = Z
        return A

    def _relu(self, Z, layer):
        """
        Compute ReLU activation.

        Args:
            Z: linear output from current layer
            layer: current layer number for caching

        Returns:
            A: output activation
        """
        A = np.maximum(0, Z)
        self.activation_cache[layer] = Z
        return A

    def _linear_activation_forward(self, A_prev, W, b, activation, layer):
        """
        Implement forward propagation for LINEAR->ACTIVATION layer

        Args:
            A_prev: activations from previous layer (or input data)
            W: weights matrix for current layer
            b: bias vector for current layer
            activation: "sigmoid" or "relu"
            layer: current layer number

        Returns:
            A: output activation for current layer

        Raises:
            ValueError: if activation is not "sigmoid" or "relu"
        """
        if activation not in ["sigmoid", "relu"]:
            raise ValueError('Activation must be "sigmoid" or "relu"')

        # First compute linear forward
        Z = self._linear_forward(A_prev, W, b, layer)

        # Then apply activation function
        if activation == "sigmoid":
            A = self._sigmoid(Z, layer)
        else:  # activation == "relu"
            A = self._relu(Z, layer)

        assert A.shape == (W.shape[0], A_prev.shape[1])
        return A

    def forward_propagation(self, X):
        """
        Implement forward propagation for [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID

        Args:
            X: input data of shape (input_size, number_of_examples)

        Returns:
            AL: final output activation
        """
        # Clear previous caches
        self.linear_cache = {}
        self.activation_cache = {}

        # First activation is input
        A = X

        # Hidden layers with ReLU
        # Forward propagation step for current layer:
        # 1. LINEAR: Z[l] = W[l]A[l-1] + b[l]
        # 2. RELU: A[l] = max(0, Z[l])
        for l in range(1, self.L):
            A_prev = A
            A = self._linear_activation_forward(
                A_prev, self.parameters[f"W{l}"], self.parameters[f"b{l}"], "relu", l
            )

        # Output layer with softmax
        AL = self._softmax(
            self._linear_forward(
                A, self.parameters[f"W{self.L}"], self.parameters[f"b{self.L}"], self.L
            ),
            self.L,
        )

        assert AL.shape == (self.layer_dims[-1], X.shape[1])
        return AL

    def _compute_cost(self, AL, Y):
        """
        Compute categorical cross-entropy cost.

        Args:
            AL: probability output of forward propagation, shape (n_classes, m)
            Y: true one-hot encoded labels, shape (n_classes, m)

        Returns:
            cost: categorical cross-entropy cost
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(AL + 1e-15))  # Add epsilon to prevent log(0)
        return np.squeeze(cost)

    def _linear_backward(self, dZ, layer):
        """
        Implement linear portion of backward propagation.

        Args:
            dZ: gradient of cost with respect to linear output
            layer: current layer number

        Returns:
            dA_prev: gradient with respect to activation of previous layer
            dW: gradient with respect to weights of current layer
            db: gradient with respect to biases of current layer
        """
        A_prev, W, b = self.linear_cache[layer]
        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape

        return dA_prev, dW, db

    def _relu_backward(self, dA, layer):
        """
        Compute gradient of ReLU activation.

        Args:
            dA: gradient of cost with respect to activation
            layer: current layer number

        Returns:
            dZ: gradient with respect to linear output Z
        """
        Z = self.activation_cache[layer]
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def _linear_activation_backward(self, dA, activation, layer):
        """
        Backward propagation for LINEAR->ACTIVATION layer.

        Args:
            dA: gradient of cost with respect to current layer's activation
            activation: "relu" or "sigmoid"
            layer: current layer number

        Returns:
            dA_prev: gradient with respect to activation of previous layer
            dW: gradient with respect to weights of current layer
            db: gradient with respect to biases of current layer

        Raises:
            ValueError: if activation is not "sigmoid" or "relu"
        """
        if activation not in ["sigmoid", "relu"]:
            raise ValueError('Activation must be "sigmoid" or "relu"')

        if activation == "relu":
            dZ = self._relu_backward(dA, layer)
        else:  # activation == "sigmoid"
            dZ = self._sigmoid_backward(dA, layer)

        dA_prev, dW, db = self._linear_backward(dZ, layer)
        return dA_prev, dW, db

    def backward_propagation(self, AL, Y):
        """
        Implement backward propagation for [LINEAR->RELU]*(L-1)->[LINEAR->SIGMOID].

        Args:
            AL: output of forward propagation (final activation value)
            Y: true binary labels

        Returns:
            grads: Dictionary with gradients for each parameter
        """
        grads = {}
        Y = Y.reshape(AL.shape)

        # Initialize backward propagation with derivative of cost
        # For binary cross-entropy: dAL = -(y/a - (1-y)/(1-a))
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Output layer (SIGMOID)
        # Get gradients for parameters of last layer (L)
        # Output layer (SOFTMAX)
        dZL = AL - Y  # Gradient for softmax + cross-entropy
        grads[f"dA{self.L-1}"], grads[f"dW{self.L}"], grads[f"db{self.L}"] = (
            self._linear_backward(dZL, self.L)
        )

        # Hidden layers (RELU)
        for l in reversed(range(self.L - 1)):
            current_cache = self._linear_activation_backward(
                grads[f"dA{l+1}"], "relu", l + 1
            )
            grads[f"dA{l}"], grads[f"dW{l+1}"], grads[f"db{l+1}"] = current_cache

        return grads

    def _update_parameters(self, grads):
        """
        Update parameters using gradient descent.

        Args:
            grads: Dictionary with gradients for each parameter
        """
        for l in range(self.L):
            self.parameters[f"W{l+1}"] -= self.learning_rate * grads[f"dW{l+1}"]
            self.parameters[f"b{l+1}"] -= self.learning_rate * grads[f"db{l+1}"]

    def fit(self, X, Y):
        """
        Train the neural network.

        Args:
            X: Input data of shape (input_size, m_examples)
            Y: True labels of shape (1, m_examples)

        Returns:
            costs: List of costs during training
        """
        costs = []

        for i in range(self.n_iterations):
            # Forward propagation
            AL = self.forward_propagation(X)

            # Compute cost
            cost = self._compute_cost(AL, Y)

            # Backward propagation
            grads = self.backward_propagation(AL, Y)

            # Update parameters
            self._update_parameters(grads)

            if self.verbose and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
                costs.append(cost)

        return costs

    def predict(self, X):
        """
        Make predictions using trained network.

        Args:
            X: Input data of shape (input_size, m_examples)

        Returns:
            predictions: Binary predictions (0/1) of shape (1, m_examples)
        """
        AL = self.forward_propagation(X)
        predictions = (AL > 0.5).astype(int)
        return predictions
