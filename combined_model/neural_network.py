import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, outputs, bias, discount, alpha, reward_good, reward_bad):
        self.reward_good = reward_good
        self.reward_bad = reward_bad
        self.discount = discount
        self.alpha = alpha
        self.input = inputs
        self.output = outputs
        self.weights = np.random.randn(outputs, inputs)
        self.bias = bias
        
    def feedforward(self, X):
        self.output = np.dot(self.weights, X) + self.bias
        return self.output
    
    def backprop(self, state_prime_hrr, state_hrr, y):
        if(y == reward_good):
            pass
#             self.error = (self.reward_bad + self.discount *  - previous_value
#             weights = 1
#             error = reward_good - (np.dot(weights, convolve(reward_tkn, current_state)) + bias)
#             weights = 1
#         else:
#             pass