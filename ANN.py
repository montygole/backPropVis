##########################
#Creates, trains, and tests an Artificial neural network with variable amount of layers, and neurons per layer
##########################
from distutils.log import error
import math
from random import seed
import random
import re
import xdrlib
#Set Seed for random()
#SEED AT 29: working
seed(29)

#Global variable to tell vis.py to show updates!
updateVisNodeValues = False
caseNum = 0
updateVisWeights = False

class NeuralNet:
    def __init__(self, layerAmount, layers, dataset = []):
        self.layerAmount = layerAmount
        self.layers = layers
        self.dataset = dataset
        self.errors = [] #errors which will be input to cost function
    def createLayers(self):
        for x in range(self.layerAmount):
            
            if x==0: #if input layer
                self.layers[x].createNeurons(self.layers[x+1], None)
            
            elif x==self.layerAmount-1: #if output layer
                self.layers[x].createNeurons(None, self.layers[x-1])
            
            else: #if hidden layer
                self.layers[x].createNeurons(self.layers[x+1], self.layers[x-1])

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x)) 
    
    def derivativeSigmoid(self, x):
        return (math.exp(-x))/((1+math.exp(-x))**2)
    
    def inverseSigmoid(self, x):
        return math.log(x/(1-x))

    def forwardPass(self, inputs, real, store=[]):
        #This function does the forward pass, and gives each neuron (except input) a new value based on its incoming weights.
        #Parameters: input data, desired output data, 
        #Returns squared error of output neurons


        total_error = 0 #Stores squared error of errors on each output neuron
        for layerInd in range(self.layerAmount):
            #Going into each layer
            neuronNum=0
            print("AT LAYER: ", layerInd, " which is of type: ", self.layers[layerInd].type, "and has a neuron amount of", self.layers[layerInd].neuronAmount)
            if self.layers[layerInd].type=="input": #Set input neurons to inputs for case
                for n in range(len(self.layers[layerInd].neurons)):
                    self.layers[layerInd].neurons[n].value = inputs[n]
                    print("Neuron #:", n, " of layer type ", self.layers[layerInd].type, " is of value ", self.layers[layerInd].neurons[n].value)
            else: 
                #FOr each neuron which isn't in the input lyer, 
                #aggregte sums of previous neurons mulied by their respective wieghts
                #and feed into activation. Set neuron value to activation output

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
                    
                if self.layers[layerInd].type=="output": #aggregate outputs
                    for outputNeuron in range(len(self.layers[layerInd].neurons)):
                        output = self.layers[layerInd].neurons[outputNeuron].value
                        relative_error = output-real[outputNeuron]
                        self.layers[layerInd].neurons[outputNeuron].error = relative_error
                        total_error+=self.layers[layerInd].neurons[outputNeuron].error

                    
        outputs = []
        
        for neuron in self.layers[len(self.layers)-1].neurons:
            outputs.append(neuron.value)

        store.append([outputs, real, total_error])

        total_error = 0.5*(total_error**2) #Squared error

        self.errors.append(total_error)
        return total_error

    def storeErrorSignal(self):
        #Begins at output layer, and finds the error signal for each neuron based on the next closest (previous) layer to the output
        #Return nothing

        for layer in reversed(self.layers):
            if layer.type=="output":
                print("THE ERROR SIGNAL OF NEURONS ON LAYER: OUTPUT ARE")   
                for neuron in layer.neurons:
                    neuron.errorSignals[0] = -1*neuron.error
                    print(neuron.errorSignals)
                
            elif layer.type=="hidden":
                print("THE ERROR SIGNAL OF NEURONS ON LAYER:", layer.type, "ARE")
                
                for neuron in range(len(layer.neurons)):
                    x=self.derivativeSigmoid(layer.neurons[neuron].input) #X stores derivative of sigmoid function wrt previous input to neuron
                    for weight in range(len(layer.neurons[neuron].weights)):
                        errorSig = 0
                        w = layer.neurons[neuron].weights[weight]
                        print(w)
                        for signal in layerToRight.neurons[weight].errorSignals:
                            errorSig+=w*signal
                            print("WEIGHT*PREVIOUS ERROR IS:",w*signal)
                        layer.neurons[neuron].errorSignals[weight]=(x*errorSig)
            elif layer.type=="input":
                print("THE ERROR SIGNAL OF NEURONS ON LAYER:", layer.type, "ARE")
                
                for neuron in range(len(layer.neurons)):

                    print("INPUT X/VALUE IS:", layer.neurons[neuron].value)
                    x=self.derivativeSigmoid(layer.neurons[neuron].value)
                    for weight in range(len(layer.neurons[neuron].weights)):
                        errorSig = 0
                        w = layer.neurons[neuron].weights[weight]
                        print(w)
                        for signal in layerToRight.neurons[weight].errorSignals:
                            #print(signal)
                            errorSig+=w*signal
                            print("WEIGHT*PREVIOUS ERROR IS:",w*signal)
                        layer.neurons[neuron].errorSignals[weight]= (x*errorSig)
                        print("AT WEIGHT", layer.neurons[neuron].weights[weight], "THE ERRORSIG IS", (x*errorSig))
                        
            
            layerToRight=layer
    
    def backprop(self, rate): 
        #Performs backrpopagation
        #Changes each weight based on its gradient relative to the error
        #Retruns
        print("BEGGINING BACKPROPAGATION \n *****************")
        self.storeErrorSignal()
        for layer in reversed(self.layers):
            if layer.type != "output":
                for neuron in layer.neurons:
                    for weight in range(len(neuron.weights)):
                        print("GRADIENT:", rate*neuron.givingTo.neurons[weight].value*neuron.errorSignals[weight])
                        neuron.weights[weight]+= rate*neuron.givingTo.neurons[weight].value*neuron.errorSignals[weight]

    def train(self, cases, rate, callthreshold=100):
        #Commits forward pass, then backpropogation. Stops at call threshold or error threshold
        #Returns a trained neural network object
        callAmount = 0
        error = 1
        while abs(error) > 0.4: 
            if callAmount > callthreshold:
                break
            error = 1
            for case in cases:
                error += self.forwardPass(case[0], case[1])
                self.backprop(rate)
            callAmount+=1
        return self

            

    def __str__(self):
        return f"{self.layerAmount}, {self.layers}"
class layer:
    def __init__(self, neuronAmount, type=""):
        self.neurons = []
        self.bias=1
        self.type = type
        self.neuronAmount = neuronAmount
    def createNeurons(self, next_layer, previous_layer):
        #Create new neuron object based on layer's neuron amount.
        #Sets weights, and gives signs of position in network to neurons
        self.neurons = [0]*self.neuronAmount
        for x in range(self.neuronAmount): 
            self.neurons[x]=neuron([], 1, next_layer, previous_layer)
            if self.type != "output":
                r = random.uniform(0.5, 1) #Random number between (x, y)
                self.neurons[x].weights = [r]*next_layer.neuronAmount #Sets random weight for each neuron in next layer
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

#To be implemeneted in vis.py
# input_layer = layer(2, "input")
# hidden_layer1 = layer(8, "hidden")
# output_layer = layer(1, "output")
# net = NeuralNet(2, [input_layer,output_layer], [[[1, 0],[0]],[[0, 1],[0]],[[1, 1],[1]],[[0, 0],[0]]])
# net.createLayers()
# net.train(net.dataset, 0.2, 1200)

#CODE BELOW IS FOR MY OWN TESTING OF THE ANN
# x_data = list(range(len(net.errors)))

# from matplotlib import pyplot as plt
# plt.scatter(x_data, net.errors)
# plt.show()
# results = []

# net.forwardPass([0, 0], [0], results)
# net.forwardPass([1, 0], [0], results)
# net.forwardPass([1, 1], [1], results)
# net.forwardPass([0, 1], [0], results)

# for result in results:
#     print(result)