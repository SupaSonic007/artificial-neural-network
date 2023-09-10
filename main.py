import numpy as np
import decimal
from pprint import pprint


class ArtificialNeuralNetwork:
    """
    A class to represent an Artificial Neural Network.

    Attributes
    ----------
    synaptic_weights : ndarray
        The weights of the synapses in the neural network.
    """

    def __init__(self) -> None:
        """
        Initializes the ArtificialNeuralNetwork object with random synaptic weights.
        """
        self.synaptic_weights = 2 * np.random.random((4, 1)) - 1

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
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
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
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error *
                                 self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments


if __name__ == '__main__':

    ann = ArtificialNeuralNetwork()

    print("Weights Pre-Training")
    pprint(ann.synaptic_weights)

    training_inputs = np.array([[1, 0, 0, 1],
                                [0, 1, 1, 0],
                                [1, 0, 1, 0],
                                [0, 0, 0, 1]])

    training_outputs = np.array([[1],
                                 [0],
                                 [1],
                                 [0]])

    ann.train(training_inputs, training_outputs, 10000)
    print('\n')
    print("Weights post-Training")
    pprint(ann.synaptic_weights)

    _input = np.array([1, 0, 1, 0])

    output = ann.think(_input)

    result = decimal.Decimal(output[0]).quantize(
        decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP)

    print('\n')
    pprint(f"The answer for {_input} is: {result}")
