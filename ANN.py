import math
from random import seed
from random import random
seed(10)
class NeuralNet:
    def __init__(self, layerAmount, layers, dataset = []):
        self.layerAmount = layerAmount
        self.layers = layers
        self.dataset = dataset
    def createLayers(self):
        for x in range(self.layerAmount):
            if x==0:
                self.layers[x].createNeurons(self.layers[x+1], None)
            elif x==self.layerAmount-1:
                self.layers[x].createNeurons(None, self.layers[x-1])
            else:
                self.layers[x].createNeurons(self.layers[x+1], self.layers[x-1])
            print(self.layers[x])
            for y in self.layers[x].neurons:
                print(y)

    # def initial_values(self, inputs):
    #     for layerInd in range(self.layerAmount):
    #         if self.layers[layerInd].type=="input":
    #             for n in len(self.layers[layerInd].neurons):
    #                 self.layers[layerInd].neurons[n].value = inputs[n]
    #         else:
    #             for n in len(self.layers[layerInd].neurons):
    #                 self.layers[layerInd].neurons[n].value = random()
    

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x)) 
    
    def forwardPass(self, inputs, real):
        for layerInd in range(self.layerAmount):
            neuronNum=0
            print("AT LAYER: ", layerInd, " which is of type: ", self.layers[layerInd].type, "and has a neuron amount of", self.layers[layerInd].neuronAmount)
            if self.layers[layerInd].type=="input":
                for n in range(len(self.layers[layerInd].neurons)):
                    self.layers[layerInd].neurons[n].value = inputs[n]
                    print("Neuron #:", n, " of layer type ", self.layers[layerInd].type, " is of value ", self.layers[layerInd].neurons[n].value)
            else:
                for neuron in self.layers[layerInd].neurons:
                    sum = 0
                    prevNeuronNum = 0
                    for previous_neuron in neuron.receivingFrom.neurons:
                        print("WEIGHTS OF PREVIOUS NEURON #" , prevNeuronNum, " ARE: ", previous_neuron.weights)
                        sum = sum + previous_neuron.value*previous_neuron.weights[neuronNum]
                        prevNeuronNum += 1
                    print("The input to the sigmoid is:",sum)
                    neuron.value = self.sigmoid(sum)
                    print("Neuron #:", neuronNum, " of layer type ", self.layers[layerInd].type, " is of value ", neuron.value)
                    neuronNum+=1
                    
                if self.layers[layerInd].type=="output":
                    output = self.layers[layerInd].neurons[0].value
        
        print("The output of this case is:", output)
        print("The desired output of this case is:", real)
        print("The error of this case is:", output-real)

        return output-real                               

    def meanSquared(self, error):
        error = 0.5*(error**2)
        return error

    def calculateError(self):
        error = 0
        caseNum = 0
        for case in self.dataset:
            caseNum += 1
            print("CURRENTLY ON CASE ", caseNum, "/", len(self.dataset))
            error += self.forwardPass(case[0], case[1])
            
        print("OVERALL ERROR IS:", error)
        return error
    
    def backprop(self, error):
         pass
    def __str__(self):
        return f"{self.layerAmount}, {self.layers}"
class layer:
    def __init__(self, neuronAmount, type=""):
        self.neurons = []
        self.type = type
        self.neuronAmount = neuronAmount
    def createNeurons(self, next_layer, previous_layer):
        self.neurons = [0]*self.neuronAmount
        for x in range(self.neuronAmount):
            self.neurons[x]=neuron([random(),random()], 1, next_layer, previous_layer)
            
            
    def __str__(self):
        return f"LAYER:{self.neuronAmount}, {self.type}"
class neuron:
    def __init__(self, weights, bias, givingTo, receivingFrom):
        self.weights = weights
        self.value = 0
        self.bias = bias
        self.givingTo = givingTo
        self.receivingFrom = receivingFrom
    def __str__(self):
         return f"NEURON: {self.weights}, {self.givingTo}, {self.receivingFrom}"


input_layer = layer(2, "input")
h1 = layer(2, "hidden")
output_layer = layer(1, "output")

net = NeuralNet(3, [input_layer, h1, output_layer], [[[1,0],0], [[0,0],0], [[0,1],0], [[1,1],1]])

net.createLayers()
net.calculateError()

