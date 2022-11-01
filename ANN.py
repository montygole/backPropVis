from distutils.log import error
import math
from random import seed
from random import random
import xdrlib
seed(200)
class NeuralNet:
    def __init__(self, layerAmount, layers, dataset = []):
        self.layerAmount = layerAmount
        self.layers = layers
        self.dataset = dataset
        self.errors = []
    def createLayers(self):
        for x in range(self.layerAmount):
            if x==0:
                self.layers[x].createNeurons(self.layers[x+1], None)
            elif x==self.layerAmount-1:
                self.layers[x].createNeurons(None, self.layers[x-1])
            else:
                self.layers[x].createNeurons(self.layers[x+1], self.layers[x-1])
            print(self.layers[x])
            # for y in self.layers[x].neurons:
            #     print("NEURON OF TYPE: ", self.layers[x].type, "is giving to a neuron of type: ", y.givingTo.type)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x)) 
    
    def derivativeSigmoid(self, x):
        return (math.exp(-x))/((1+math.exp(-x))**2)

    def inverseSigmoid(self, x):
       
        x = x*0.99999999999
        print("INPUT TO INVERSE IS:", x)
        return -math.log((1 / (x)) - 1)
       # return -(math.log((1/x)-1))

    def meanSquared(self, error):
        error = 0.5*(error**2)
        return error

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
        print("The error of this case is:", total_error)

        return self.meanSquared(total_error)

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
                    x=self.derivativeSigmoid(self.inverseSigmoid(layer.neurons[neuron].value))
                    for weight in range(len(layer.neurons[neuron].weights)):
                        errorSig = 0
                        w = layer.neurons[neuron].weights[weight]
                        for signal in layerToRight.neurons[weight].errorSignals:
                            errorSig+=w*signal
                        layer.neurons[neuron].errorSignals[weight]= x*errorSig
                    print(layer.neurons[neuron].errorSignals)                   
                        
                    
            layerToRight=layer


    def calculateError(self):
        error = 0
        caseNum = 0
        for case in self.dataset:
            caseNum += 1
            print("CURRENTLY ON CASE ", caseNum, "/", len(self.dataset))
            error += self.forwardPass(case[0], case[1])
            
        print("OVERALL ERROR IS:", error)
        self.errors.append(error)
        return error
    
    def backprop(self, error, rate, callAmount):
        print("BEGGINING BACKPROPAGATION \n *****************")
        self.storeErrorSignal()
        for layer in reversed(self.layers):
            print("PERFORMING ON LAYER OF TYPE:", layer.type)
            if layer.type=="output":
                for neuron in range(len(layer.neurons)):
                    print("PERFORMING ON WEIGHTS CONNECTED TO NEURON WITH VALUE:", layer.neurons[neuron].value)
                    for prevN in layer.neurons[neuron].receivingFrom.neurons:
                        #print("NEURON OF TYPE:", layer.neurons[neuron].receivingFrom.type, ", RELATIVE TO NEURON WITH VALUE:", layer.neurons[neuron].value, "CURRENTLY HAS A WEIGHT OF:", prevN.weights[neuron])
                        prevN.weights[neuron]+=rate*layer.neurons[neuron].error*self.derivativeSigmoid(prevN.weights[neuron]*prevN.value)
            else:
                for neuron in range(len(layer.neurons)):
                    for weight in range(len(layer.neurons[neuron].weights)):
                        layer.neurons[neuron].weights[weight]+=rate*layer.neurons[neuron].errorSignals[weight]*layer.neurons[neuron].value

                
        #Stopping
        if self.calculateError() < 0.3 or callAmount > 50:
            return error
        else:
            self.backprop(self.calculateError, rate, callAmount+1)


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
                self.neurons[x].weights = [random()]*next_layer.neuronAmount
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
        self.givingTo = givingTo
        self.receivingFrom = receivingFrom
        self.errorSignals = []
    def __str__(self):
         return f"NEURON: {self.weights}, {self.givingTo}, {self.receivingFrom}"


input_layer = layer(2, "input")
h1 = layer(2, "hidden")
# h2 = layer(3, "hidden")
# h3 = layer(1, "hidden")

output_layer = layer(1, "output")

net = NeuralNet(3, [input_layer,h1,output_layer], [[[1,0],[0]], [[0,0],[0]], [[0,1],[0]], [[1,1],[1]]])

net.createLayers()

net.backprop(net.calculateError(), 0.5, 0)

for x in net.errors:
    print(x)

net.forwardPass([1,1], [1])
net.forwardPass([1,0], [0])
