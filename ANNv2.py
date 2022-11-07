import random
import numpy as np

LAYERS = [2, 2, 1]
TRAINING_RATE = 0.5

# A NeuralNet consists of Layer
class NeuralNet:
    def __init__(self, layer_structure):
        self.layers = []
        
        # Create the layers
        previous_layer = None
        for layer_size in layer_structure:
            layer = Layer(layer_size)
            self.layers.append(layer)
            
            if previous_layer: # If this isn't the first layer
                # Init weights for the previous layers to this one
                for n in layer.neurons:
                    n.w = [1] * len(previous_layer.neurons) # TEMP not random numbers
                    
                    
            previous_layer = layer
            
    def feedforward(self, inputs):
        assert len(inputs) == len(self.layers[0].neurons)
        
        # Init the first layer with the inputs
        for i in range(len(inputs)):
            self.layers[0].neurons[i].y = inputs[i]

        # Call each layer in turn to feedforward
        previous_layer = self.layers[0]
        for layer in self.layers[1:]:
            layer.feedforward(previous_layer)
            previous_layer = layer
            
    def backprop(self, desired):
        assert len(desired) == len(self.layers[-1].neurons)
        
        # Compute the error at the output layer
        for i in range(len(desired)):
            self.layers[-1].neurons[i].compute_error(desired[i])
            
        for 
            
    def get_output(self,):
        output = []
        for neuron in self.layers[-1].neurons:
            output.append(neuron.y)
            #TEMP
            print(neuron.error)
        return output
    
# A Layer consists of Neurons
class Layer:
    def __init__(self, neuronAmount):
        self.neurons = []
        for i in range(neuronAmount):
            self.neurons.append(Neuron())
            
    def feedforward(self, previous_layer):
        for neuron in self.neurons:
            neuron.compute(previous_layer.neurons)

    # def compute_errors(self, ???) #WIP

    def __str__(self):
        return f"LAYER:{self.neuronAmount}"

# A Neuron consists of base variables
class Neuron:
    def __init__(self):
        self.w = [1] # An array of the weights
        self.b = 0 # The bias
        self.x = 0 # The weighted sum of the incoming neurons
        self.y = 0 # The output of this node
        self.error = 0 # The error at this node
        
    def compute(self, previous_neurons):
        assert len(self.w) == len(previous_neurons)
        inputs = []
        for previous_n in previous_neurons:
            inputs.append(previous_n.y)
        
        self.x = np.sum(np.multiply(self.w, inputs)) + self.b # TEMP just pass through for now, sigmoid later
        self.y = self.x # TEMP
        return self.y # TEMP again, no activation func
    
    def compute_error(self, d):
        self.error = d - self.y
        return self.error
    
    def __str__(self):
         return f"NEURON: {self.w} {self.b} {self.x} {self.y}"

nn = NeuralNet(LAYERS)
nn.feedforward([1] * LAYERS[0]) #TEMP
print(nn.get_output())
nn.backprop([1])
print(nn.get_output())