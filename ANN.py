class NeuralNet:
    def __init__(self, layerAmount):
        self.layerAmount = layerAmount
        self.layers = []
    def createLayers(self):
        for x in range(self.layerAmount):
            self.layers.append()
    def backprop():
        pass

class layer:
    def __init__(self, neuronAmount, type = ""):
        self.neurons = []
        self.neuronAmount = neuronAmount
        self.type = type
    def createNeurons(self):
        for x in range(self.neuronAmount):
            pass

class neuron:
    def __init__(self, weight, bias, givingTo, receivingFrom):
        self.weight = weight
        self.bias = bias
        self.givingTo = givingTo
        self.receivingFrom = receivingFrom