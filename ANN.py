##########################
#Creates, trains, and tests an Artificial neural network with variable amount of layers, and neurons per layer
##########################
from tkinter import *
from distutils.log import error
import math
from random import seed
import random
import re
import xdrlib
#Set Seed for random()
#SEED AT 29: working
seed(29)

NODE_DISPLAY_SIZE = 40
NODE_DISPLAY_XPAD = 40
NODE_DISPLAY_YPAD = 20
NODE_DISPLAY_HEIGHT = NODE_DISPLAY_SIZE+NODE_DISPLAY_YPAD

#Global variable to tell vis.py to show updates!
updateVisNodeValues = False
caseNum = 0
updateVisWeights = False

# From: https://stackoverflow.com/a/65983607
def rgbtohex(r,g,b):
    return f'#{r:02x}{g:02x}{b:02x}'

EST_MAX = 1.2
def min_max_normalize(x):
    return (x-(-EST_MAX))/(EST_MAX-(-EST_MAX))

class NeuralNet:
    def __init__(self, layerAmount, layers, dataset = []):
        self.layerAmount = layerAmount
        self.layers = layers
        self.dataset = dataset
        self.errors = [] #errors which will be input to cost function
        self.weights_linear = []
        
        # Display variables
        self.max_layer_size = 0
        for l in self.layers:
            self.max_layer_size = max(self.max_layer_size, l.neuronAmount)
        self.max_height = self.max_layer_size*NODE_DISPLAY_HEIGHT
        self.max_height = self.max_layer_size*NODE_DISPLAY_HEIGHT
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

    def draw(self, parent, x, y):
        # Draws first layer
        w_index = 0
        y0 = y + 1/2 * (self.max_layer_size - len(self.layers[0].neurons)) * NODE_DISPLAY_HEIGHT
        previous_node_centers = self.layers[0].draw(parent, x, y0, [], self.weights_linear)
        
        # Draws the rest
        for i, layer in enumerate(self.layers[1:], start = 1):
            x0 = x+i*NODE_DISPLAY_HEIGHT
            y0 = y + 1/2 * (self.max_layer_size - len(layer.neurons)) * NODE_DISPLAY_HEIGHT
            previous_node_centers = layer.draw(parent,
                                       x0,
                                       y0,
                                       previous_node_centers,
                                       self.weights_linear
                                   )
            print(previous_node_centers)

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
                  
    def draw(self, parent, x, y, previous_node_centers, weights):

        current_node_centers = []
        for i, node in enumerate(self.neurons):
            x0 = x
            y0 = y + i*NODE_DISPLAY_HEIGHT
            
            node.draw(parent, 
                          x0,
                          y0,
                          NODE_DISPLAY_SIZE/2
                      )
            
            current_node_centers.append((x0, y0))
            
            # Draw the connecting lines between layers
            for i, prev in enumerate(previous_node_centers):
                for neuron in self.neurons: #WIP
                    print(len(self.neurons))
                    if self.type == "output":
                        for receivingFromNeuron in neuron.receivingFrom.neurons:
                            for weight in receivingFromNeuron.weights:
                                color = rgbtohex(r = round((1-min_max_normalize(weight))*255), g = 0, b = 0)
                                parent.create_line(prev[0], prev[1], x0, y0, 
                                            fill=color)
                    else:
                        for weight in neuron.weights:
                            
                            color = rgbtohex(r = round((1-min_max_normalize(weight))*255), g = 0, b = 0)
                            parent.create_line(prev[0], prev[1], x0, y0, 
                                        fill=color)
        
        return current_node_centers

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
    
    def draw(self, parent, x, y, halfwidth):
        parent.create_oval(
                x - halfwidth, y - halfwidth,
                x + halfwidth, y + halfwidth,
                fill = 'white'
            )
        
    def __str__(self):
         return f"NEURON: {self.weights}, {self.givingTo}, {self.receivingFrom}"


def create_visualization():
    root = Tk()
    root.geometry('500x500')
    
    root_canvas = Canvas(root)
    root_canvas.pack(expand=Y, fill=BOTH)
    root_canvas.configure(bg='white')
    
    net.draw(root_canvas, NODE_DISPLAY_SIZE, NODE_DISPLAY_SIZE)
    
    root.mainloop()
     
input_layer = layer(2, "input")
hidden_layer1 = layer(5, "hidden")
output_layer = layer(1, "output")
layer_structure = [input_layer, hidden_layer1, output_layer]
net = NeuralNet(len(layer_structure), layer_structure, [[[1, 0],[0]],[[0, 1],[0]],[[1, 1],[1]],[[0, 0],[0]]])
net.createLayers()
# net.train(net.dataset, 0.2, 1200)
create_visualization() #DEBUG #TEMP













































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