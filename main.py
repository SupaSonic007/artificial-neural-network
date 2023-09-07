import numpy as np
import decimal
from pprint import pprint

class ArtificialNeuralNetwork():

    def __init__(self) -> None:
        self.synaptic_weights = 2 * np.random.random((4, 1)) - 1

    def sigmoid(self, x) -> float:
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x) -> float:
        return x * (1-x)
    
    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def train(self, training_inputs, training_outputs, number_of_iterations):
        for _ in range(number_of_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
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

    result = decimal.Decimal(output[0]).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP)

    print('\n')
    pprint(f"The answer for {_input} is: {result}")
