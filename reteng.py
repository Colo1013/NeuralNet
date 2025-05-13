import numpy as np

class Neurone:
    def __init__(self, activation, inputs_dim):
        # Validazione activation
        activation = activation.lower()
        if activation not in ['sigmoid', 'relu']:
            raise ValueError("activation must be 'sigmoid' or 'relu'")
        self.activation_name = activation

        # Inizializzazioni
        self.delta = 0.0
        self.inputs = None
        self.derivative = None
        self.weights = np.random.randn(inputs_dim)
        self.bias = float(np.random.randn(1))  # bias come scalare

    def predict(self, inputs):
        self.inputs = np.array(inputs, dtype=float)
        z = float(np.dot(self.weights, self.inputs)) + self.bias
        self.output = self._activate(z)
        return self.output

    def _activate(self, z):
        if self.activation_name == 'relu':
            return self._relu(z)
        else:  # 'sigmoid'
            return self._sigmoid(z)

    def aggiorna_pesi(self, learning_rate):
        # Aggiorna pesi e bias come scalari
        for i in range(len(self.weights)):
            grad = learning_rate * self.delta * float(self.inputs[i])
            self.weights[i] -= grad
        self.bias -= learning_rate * self.delta

    def _relu(self, z):
        out = max(z, 0.0)
        self.derivative = 1.0 if z > 0 else 0.0
        return out

    def _sigmoid(self, z):
        out = 1.0 / (1.0 + np.exp(-z))
        self.derivative = out * (1.0 - out)
        return out


class Layer:
    def __init__(self, inputs_dim, activation, dimension):
        self.neurons = [Neurone(activation, inputs_dim) for _ in range(dimension)]
        self.outputs = None

    def predict(self, inputs):
        # Propaga gli input attraverso ogni neurone
        outs = [neuron.predict(inputs) for neuron in self.neurons]
        self.outputs = np.array(outs, dtype=float)
        return self.outputs


class NeuralNetwork:
    def __init__(self, layers, epochs=1000, learning_rate=0.1):
        self.layers = layers
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, Y):
        X = [np.array(x, dtype=float) for x in X]
        Y = [y if isinstance(y, float) else float(y[0]) for y in Y]

        for epoch in range(self.epochs):
            for x, y in zip(X, Y):
                # --- Forward pass ---
                activation = x
                for layer in self.layers:
                    activation = layer.predict(activation)
                output = activation  # array dell'ultimo layer

                # --- Backward pass ---
                # Delta output layer
                last = self.layers[-1]
                for i, neuron in enumerate(last.neurons):
                    neuron.delta = (output[i] - y) * neuron.derivative

                # Delta dei layer precedenti
                for idx in range(len(self.layers) - 2, -1, -1):
                    current = self.layers[idx]
                    next_layer = self.layers[idx + 1]
                    for i, neuron in enumerate(current.neurons):
                        error = sum(n.weights[i] * n.delta for n in next_layer.neurons)
                        neuron.delta = error * neuron.derivative

                # --- Aggiorna pesi e bias ---
                for layer in self.layers:
                    for neuron in layer.neurons:
                        neuron.aggiorna_pesi(self.learning_rate)

    def predict(self, X):
        outputs = []
        for x in X:
            activation = np.array(x, dtype=float)
            for layer in self.layers:
                activation = layer.predict(activation)
            outputs.append(activation)
        return outputs


if __name__ == "__main__":
    # Definizione rete (2 input → 2 neuroni → 1 neurone)
    layer1 = Layer(inputs_dim=2, activation='relu', dimension=2)
    layer2 = Layer(inputs_dim=2, activation='sigmoid', dimension=1)
    network = NeuralNetwork([layer1, layer2], epochs=10000, learning_rate=0.1)

    # Dati XOR
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[0.0], [1.0], [1.0], [0.0]]

    # Training
    network.fit(X, Y)

    # Predizione
    preds = network.predict(X)
    print("Output previsto:")
    for inp, out in zip(X, preds):
        print(f"{inp} -> {out}")
