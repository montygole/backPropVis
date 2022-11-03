from distutils.log import error
import math
from random import seed
import random
import xdrlib
#Set Seed for random()
seed(29)
class NeuralNet:
    def __init__(self, layerAmount, layers, dataset = []):
        self.layerAmount = layerAmount
        self.layers = layers
        self.dataset = dataset
        #errors which will be input to cost function
        self.errors = []
    def createLayers(self):
        for x in range(self.layerAmount):
            #if input layer
            if x==0:
                self.layers[x].createNeurons(self.layers[x+1], None)
            #if output layer
            elif x==self.layerAmount-1:
                self.layers[x].createNeurons(None, self.layers[x-1])
            #if hidden layer
            else:
                self.layers[x].createNeurons(self.layers[x+1], self.layers[x-1])

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x)) 
    
    def derivativeSigmoid(self, x):
        return (math.exp(-x))/((1+math.exp(-x))**2)

    def forwardPass(self, inputs, real):
        total_error = 0
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
                    neuron.input = sum
                    print("The input to the sigmoid is:",sum)
                    neuron.value = self.sigmoid(sum)
                    print("Neuron #:", neuronNum, " of layer type ", self.layers[layerInd].type, " is of value ", neuron.value)
                    neuronNum+=1
                    
                if self.layers[layerInd].type=="output":
                    for outputNeuron in range(len(self.layers[layerInd].neurons)):
                        output = self.layers[layerInd].neurons[outputNeuron].value
                        relative_error = output-real[outputNeuron]
                        self.layers[layerInd].neurons[outputNeuron].error = relative_error
                        total_error+=self.layers[layerInd].neurons[outputNeuron].error

                    
        outputs = []
        
        for neuron in self.layers[len(self.layers)-1].neurons:
            outputs.append(neuron.value)
        print("The output of this case is:", outputs)
        print("The desired output of this case is:", real)
        print("The error of this case is:", abs(total_error))
        total_error = 0.5*(total_error**2)
        self.errors.append(total_error)
        return total_error

    def storeErrorSignal(self):
        for layer in reversed(self.layers):
            if layer.type=="output":
                print("THE ERROR SIGNAL OF NEURONS ON LAYER: OUTPUT ARE")
                for neuron in layer.neurons:
                    neuron.errorSignals[0] = neuron.error
                    print(neuron.errorSignals)
                
            else:
                print("THE ERROR SIGNAL OF NEURONS ON LAYER:", layer.type, "ARE")
                
                for neuron in range(len(layer.neurons)):

                    print("INPUT X/VALUE IS:", layer.neurons[neuron].value)
                    x=self.derivativeSigmoid(layer.neurons[neuron].input)
                    for weight in range(len(layer.neurons[neuron].weights)):
                        errorSig = 0
                        w = layer.neurons[neuron].weights[weight]
                        print(w)
                        for signal in layerToRight.neurons[weight].errorSignals:
                            print(signal)
                            errorSig+=w*signal
                            print(w*signal)
                        layer.neurons[neuron].errorSignals[weight]= x*errorSig                 
                        
                    
            layerToRight=layer


    # def calculateError(self):
    #     error = 0
    #     real = 
    #     error = 0.5*self.forwardPass(case[0], case[1])**2
        
    #     print("OVERALL ERROR IS:", error)
    #     self.errors.append(error)
    #     return error
    
    def backprop(self, rate): #YOU ARE MAKING BACKPROP RUN PER CASE, WTIH ERROR PER CASE< NOT SUMMED OF EACH CASE ERROR!
        print("BEGGINING BACKPROPAGATION \n *****************")
        self.storeErrorSignal()
        for layer in reversed(self.layers):
            if layer.type != "output":
                for neuron in layer.neurons:
                    for weight in range(len(neuron.weights)):
                        neuron.weights[weight]+=rate*neuron.givingTo.neurons[weight].value*neuron.errorSignals[weight]
                
        #Stopping
        #return error
        # if error < 0.6 or callAmount > 10:
            
        # else:
        #     self.backprop(self.calculateError, rate, callAmount+1)

    def train(self, cases, rate, callthreshold=100):
        callAmount = 0
        error = 1
        while error > 0.01: 
            if callAmount > callthreshold:
                return self
            
            for case in cases:
                e = self.forwardPass(case[0], case[1])
                self.backprop(rate)
            callAmount+=1
            error = e
            print("******************************************************ERROR IS: ", error, callAmount)
        return self

            

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
            self.neurons[x]=neuron([], 1, next_layer, previous_layer)
            if self.type != "output":
                r = random.uniform(-0.5, 0.5)
                self.neurons[x].weights = [r]*next_layer.neuronAmount
                self.neurons[x].errorSignals = [0]*next_layer.neuronAmount
            else:
                self.neurons[x].error = 0
                self.neurons[x].errorSignals = [0]

    def __str__(self):
        return f"LAYER:{self.neuronAmount}, {self.type}"
class neuron:
    def __init__(self, weights, bias, givingTo, receivingFrom):
        self.weights = weights
        self.value = 0
        self.bias = bias
        self.input = 0
        self.givingTo = givingTo
        self.receivingFrom = receivingFrom
        self.errorSignals = []
    def __str__(self):
         return f"NEURON: {self.weights}, {self.givingTo}, {self.receivingFrom}"


input_layer = layer(2, "input")
h1 = layer(2, "hidden")
h2 = layer(2, "hidden")

# h3 = layer(1, "hidden")

output_layer = layer(1, "output")

net = NeuralNet(4, [input_layer,h1, h2,output_layer], [[[1, 0],[1]],[[0, 0],[0]]])

net.createLayers()

net.train(net.dataset, 0.9, 5000)

# for x in net.errors:
#     print("ERRORS *****************")
#     print(x)

net.forwardPass([1, 0], [1])