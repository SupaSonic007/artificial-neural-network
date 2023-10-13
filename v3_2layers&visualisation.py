import numpy as np
import decimal
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt

class ArtificialNeuralNetwork:
    """
    A class to represent an Artificial Neural Network.

    Attributes
    ----------
    synaptic_weights : ndarray
        The weights of the synapses in the neural network.
    """

    def __init__(self, inputs: int = 1, outputs: int = 1) -> None:
        """
        Initializes the ArtificialNeuralNetwork object with random synaptic weights.

        Parameters
        ----------
        inputs : int, optional
            The number of input neurons, by default 1
        outputs : int, optional
            The number of output neurons, by default 1

        Returns
        -------
        None

        """

        # Connects input neurons to output neurons
        self.input_to_output  = 2 * np.random.random((inputs, outputs)) - 1
        self.layers = [self.input_to_output]

    def add_layer(self, layer_size: int = 1, layer_position: int = -1) -> None:
        """
        Adds a layer to the neural network.

        Parameters
        ----------
        layer_size : int, optional
            The number of nodes in the layer, by default 1

        layer_position : int, optional
            The position of the layer in the neural network, by default appends to one layer before the output layer

        Returns
        -------
        None
        """
        # Add's a layer to the neural network
        layer = 2 * np.random.random((layer_size, self.layers[-1].shape[1])) - 1

        # If no layer position is given, add layer to the end of the neural network
        if not layer_position: layer_position = len(self.layers)

        while layer_position < 0: layer_position += len(self.layers) + 1

        # adjust size of layer outputs if a next layer exists
        # Insert layer in between two layers, adjust output side
        if layer_position < len(self.layers):

            next_nodes = self.layers[layer_position].shape[1]

            layer = np.resize(layer, (layer_size, next_nodes))

        # adjust input side
        if layer_position != 0 and layer_position <= len(self.layers):
            prev_nodes  = self.layers[layer_position - 1].shape[0]
            while layer_position < 0:
                layer_position += len(self.layers)
            self.layers[layer_position - 1] = np.resize(self.layers[layer_position - 1], (prev_nodes, layer_size))

        self.layers.insert(layer_position, layer)        

    def sigmoid(self, x) -> float:
        """
        Maps a value from 0 to 1 using the sigmoid function.

        Parameters
        ----------
        x : float
            The input value.

        Returns
        -------
        float
            The sigmoid of the input value.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x) -> float:
        """
        Computes the derivative of the sigmoid function.

        Parameters
        ----------
        x : float
            The input value.

        Returns
        -------
        float
            The derivative of the sigmoid of the input value.
        """
        return x * (1-x)

    def think(self, inputs) -> float:
        """
        Computes the dot product of the inputs and synaptic weights, and applies the sigmoid function.

        Parameters
        ----------
        inputs : ndarray
            The input values.

        Returns
        -------
        float
            The output of the neural network.
        """
        inputs = inputs.astype(float)
        output = None
        for layer in range(len(self.layers)):
                if layer == 0:
                    output = self.sigmoid(np.dot(inputs, self.layers[0]))
                else:
                    output = self.sigmoid(np.dot(output, self.layers[layer]))

        pprint(output)

        return output

    def train(self, training_inputs: np.ndarray, training_outputs: np.ndarray, number_of_iterations: int) -> None:
        """
        Trains the neural network using the given training inputs and outputs.

        It calculates the error margin by comparing the output to the expected output and adjusts the synapses based on this error margin (back propagation).

        Parameters
        ----------
        training_inputs : ndarray
            The inputs that the model will train with.
        training_outputs : ndarray
            The expected outputs of the model.
        number_of_iterations : int
            The number of training iterations to run.

        Returns
        -------
        None
        """
        for _ in range(number_of_iterations):
            output = None
            deltas = list()
            for layer in range(len(self.layers) - 1):
                
                # Calculate the output of the current layer
                if layer == 0:
                    output = self.sigmoid(np.dot(training_inputs, self.layers[0]))
                else:
                    output = self.sigmoid(np.dot(output, self.layers[layer]))

                # Calculate the error of the current layer
                if layer == 0:
                    error = training_outputs - output
                else:
                    error = deltas[layer - 1].dot(self.layers[layer].T)
                
                # Calculate the delta of the current layer - how much each node contributed to the error
                delta = error * self.sigmoid_derivative(output)
                deltas.append(delta)

                # Calculate the adjustments to the synaptic weights
                if layer == 0:
                    adjustments = np.dot(training_inputs.T, delta)
                else:
                    adjustments = np.dot(output.T, delta)

                self.layers[layer] += adjustments

    def get_nodes_per_layer(self):
        """
        Returns the number of nodes in each layer of the neural network.

        Returns
        -------
        tuple
            The number of nodes in the each layer, respectively.
        """

        # If there is only one layer, return the shape of the layer
        if len(self.layers) == 1:
            return [[self.layers[0].shape[0], self.layers[0].shape[1]]]
        
        # If there are multiple layers, return the shape of each layer
        nodeLayers = list(layer.shape[0] for layer in self.layers)
        nodeLayers.append(self.layers[-1].shape[1])
        return nodeLayers
    
    def visualise(self):
        """
        Visualises the neural network using networkx and matplotlib.

        Returns
        -------
        None
        """

        import networkx as nx
        import matplotlib.pyplot as plt

        graph = nx.Graph()

        layers = self.get_nodes_per_layer()
        
        # Add nodes
        for layer in range(len(layers)):
            for node in range(layers[layer]):
                graph.add_node((layer, node), data=layer)

        # Add edges
        # Num of edges = Num of Layers - 1
        for layer in range(len(layers) - 1):
            # For each node, connect to every node in the next layer
            for node in range(layers[layer]):
                for next_node in range(layers[layer + 1]):
                    graph.add_edge((layer, node), (layer + 1, next_node))

        plt.figure(figsize=(10, 10))
        pos = nx.multipartite_layout(graph, subset_key="data")
        nx.draw(graph, pos, with_labels=True)

        plt.show()


if __name__ == '__main__':

    ann = ArtificialNeuralNetwork(6 , 1)

    ann.add_layer(16)
    ann.add_layer(16)


    training_inputs = np.array([[1, 0, 0, 1, 1, 0],
                                [0, 1, 1, 0, 0, 0],
                                [1, 0, 1, 0, 1, 1],
                                [0, 0, 0, 1, 0, 1]])

    # training_outputs = np.array([[1, 0, 0],
    #                              [0, 1, 1],
    #                              [1, 0, 1],
    #                              [0, 0, 0]])

    training_outputs = np.array([[1],
                                 [0],
                                 [1],
                                 [0]])
    print(ann.layers)

    ann.train(training_inputs, training_outputs, 10000)

    _input = np.array([1, 0, 1, 0, 0, 0])

    output = ann.think(_input)

    print(ann.layers)
    result = np.rint(output)
    print('\n')
    pprint(f"The answer for {_input} is: {result}")

    ann.visualise()