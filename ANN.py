##########################
#Creates, and trains an Artificial neural network with variable amount of layers, and neurons per layer
#Also visualizes the backpropagation process with Tkinter, and matplotlib
##########################
from tkinter import *
import math
from random import seed
import random
from matplotlib import pyplot as plt

#Set Seed for random()
seed(29)
TRAINING_RATE = 2

# Display variables
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
NODE_DISPLAY_SIZE = 40
NODE_DISPLAY_XPAD = NODE_DISPLAY_SIZE * 4
NODE_DISPLAY_YPAD = NODE_DISPLAY_SIZE * 2
NODE_DISPLAY_HEIGHT = NODE_DISPLAY_SIZE+NODE_DISPLAY_YPAD
LINE_YOFFSET = 5
LINE_WIDTH_WEIGHT = 5
LINE_WIDTH_ERRORSIG = 5

#Global vars for running once per case
caseNum = 0

# Converts 0-255 to a hex value
# From: https://stackoverflow.com/a/65983607
def rgbtohex(r,g,b):
    return f'#{r:02x}{g:02x}{b:02x}'

class NeuralNet:
    def __init__(self, layerAmount, layers, dataset = []):
        self.layerAmount = layerAmount
        self.layers = layers
        self.dataset = dataset
        self.errors = [] #errors which will be input to cost function
        self.weights_linear = []
        
        #training control vars
        self.casesNum = 0
        self.prevError = 0
        # Display variables
        self.max_layer_size = 0
        for l in self.layers:
            self.max_layer_size = max(self.max_layer_size, l.neuronAmount)
        self.max_height = self.max_layer_size*NODE_DISPLAY_HEIGHT
        self.max_height = self.max_layer_size*NODE_DISPLAY_HEIGHT
    def createLayers(self):
        for x in range(self.layerAmount):
            
            if x==0: #if input layer only give its neurons a next_layer
                self.layers[x].createNeurons(self.layers[x+1], None)
            
            elif x==self.layerAmount-1: #if output layer only give its neruons a previous_layer
                self.layers[x].createNeurons(None, self.layers[x-1])
            
            else: #if hidden layer give both a next_layer, and a previous_layer
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
            #print("AT LAYER: ", layerInd, " which is of type: ", self.layers[layerInd].type, "and has a neuron amount of", self.layers[layerInd].neuronAmount)
            if self.layers[layerInd].type=="input": #Set input neurons to inputs for case
                for n in range(len(self.layers[layerInd].neurons)):
                    self.layers[layerInd].neurons[n].value = inputs[n]
                    #print("Neuron #:", n, " of layer type ", self.layers[layerInd].type, " is of value ", self.layers[layerInd].neurons[n].value)
            else: 
                #FOr each neuron which isn't in the input lyer, 
                #aggregte sums of previous neurons mulied by their respective wieghts
                #and feed into activation. Set neuron value to activation output

                for neuron in self.layers[layerInd].neurons:
                    sum = 0
                    prevNeuronNum = 0
                    for previous_neuron in neuron.receivingFrom.neurons:
                        sum = sum + previous_neuron.value*previous_neuron.weights[neuronNum]
                        prevNeuronNum += 1
                    neuron.input = sum
                    neuron.value = self.sigmoid(sum)
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
                for neuron in layer.neurons:
                    neuron.errorSignals[0] = -1*neuron.error
                
            elif layer.type=="hidden":
                
                for neuron in range(len(layer.neurons)):
                    x=self.derivativeSigmoid(layer.neurons[neuron].input) #X stores derivative of sigmoid function wrt previous input to neuron
                    for weight in range(len(layer.neurons[neuron].weights)):
                        errorSig = 0
                        w = layer.neurons[neuron].weights[weight] #w stands for weight
                        for signal in layerToRight.neurons[weight].errorSignals:
                            errorSig+=w*signal
                        layer.neurons[neuron].errorSignals[weight]=(x*errorSig)

            elif layer.type=="input":
                for neuron in range(len(layer.neurons)):
                    x=self.derivativeSigmoid(layer.neurons[neuron].value)
                    for weight in range(len(layer.neurons[neuron].weights)):
                        errorSig = 0
                        w = layer.neurons[neuron].weights[weight]
                        for signal in layerToRight.neurons[weight].errorSignals:
                            errorSig+=w*signal
                        layer.neurons[neuron].errorSignals[weight]= (x*errorSig)
                        
            
            layerToRight=layer
    
    def backprop(self, rate): 
        #Performs backrpopagation
        #Changes each weight based on its gradient wrt to the error
        #Returns nothing

        self.storeErrorSignal()
        for layer in reversed(self.layers):
            if layer.type != "output":
                for neuron in layer.neurons:
                    for weight in range(len(neuron.weights)):
                        neuron.weights[weight]+= rate*neuron.givingTo.neurons[weight].value*neuron.errorSignals[weight]

    
    def train(self, cases, rate):
        #Commits forward pass, then backpropogation
        #Returns a trained neural network object
        if self.casesNum == len(cases):
            self.casesNum = 0
        case = cases[self.casesNum]
        self.prevError += self.forwardPass(case[0], case[1])
        self.backprop(rate)
        self.casesNum +=1

        #Plot errors for each case from error formula (0.5*(sum(error)))
        x_data = list(range(len(net.errors)))
        plt.scatter(x_data, net.errors)
        plt.title("Error per case")
        plt.xlabel("Epoch")
        plt.ylabel("Error (0.5 * (error-desired)^2)")
        plt.pause(0.00000000000000005)  #How long of a pause in between updates so the user can click the window? This should be near 0 for our purposes
        return self

    # Draw the network
    def draw(self, parent, x, y):
        #This function draws layers of the network

        # Draw the first layer
        y0 = y + 1/2 * (self.max_layer_size - len(self.layers[0].neurons)) * NODE_DISPLAY_HEIGHT
        previous_node_centers = self.layers[0].draw(parent, x, y0, [])
        
        # Draw the following layers
        for i, layer in enumerate(self.layers[1:], start = 1):
            x0 = x+i*NODE_DISPLAY_XPAD
            y0 = y + 1/2 * (self.max_layer_size - len(layer.neurons)) * NODE_DISPLAY_HEIGHT
            previous_node_centers = layer.draw(parent,
                                       x0,
                                       y0,
                                       previous_node_centers
                                   )

    # Train the network on one instance and draw the network
    def train_and_draw(self):
        self.train(self.dataset, TRAINING_RATE)
        self.canvas.create_rectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, fill='white')
        self.draw(self.canvas, 70, 50)
        self.canvas.after(100, func=self.train_and_draw)

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
        #Sets weights, and gives signs of position in network to neurons via next_layer and previous_layer
        self.neurons = [0]*self.neuronAmount
        for x in range(self.neuronAmount): 
            self.neurons[x]=neuron([], 1, next_layer, previous_layer)
            if self.type != "output":
                #self.neurons[x].weights = [r]*next_layer.neuronAmount #Sets random weight for each neuron in next layer
                self.neurons[x].weights = [0]*next_layer.neuronAmount
                for i in range(next_layer.neuronAmount):
                    r = random.uniform(-0.5, .5) #Random number between (x, y)
                    self.neurons[x].weights[i] = r
                self.neurons[x].errorSignals = [0]*next_layer.neuronAmount
            else:
                self.neurons[x].error = 0
                self.neurons[x].errorSignals = [0]
                  
    # Draw this layer
    def draw(self, parent, x, y, previous_node_centers):
        
        current_node_centers = [] # retain the current layer's node's positions
        for i, neuron in enumerate(self.neurons):
            x0 = x
            y0 = y + i*NODE_DISPLAY_HEIGHT
            
            neuron.draw(parent, 
                          x0,
                          y0,
                          NODE_DISPLAY_SIZE/2
                      )
            
            current_node_centers.append((x0, y0))
            
            if self.type != "input":
                # For each neuron's incoming connections
                for j, receivingFromNeuron in enumerate(neuron.receivingFrom.neurons):
                    prev = previous_node_centers[-j]

                    weight = receivingFromNeuron.weights[i]
                    weight_255 =max(min(255, round(128*abs(weight) + 128)), 0)
                    color = rgbtohex(r = weight_255, g = weight_255, b = weight_255)
                    
                    parent.create_line(prev[0], prev[1], x0, y0, 
                               fill=color, width=LINE_WIDTH_WEIGHT)
                    
                    parent.create_text(
                            x0-NODE_DISPLAY_SIZE+50, y0-50*j+30,
                            text = f'w: {weight:.4f}'
                        )
                    
                    # Draw each error signal
                    for errorSig in neuron.errorSignals:
                        errorColours=128-math.ceil(128*(errorSig))
                        color = rgbtohex(r = 255-errorColours, g = errorColours, b = 0)
                        parent.create_line(prev[0], prev[1]+LINE_YOFFSET, x0, y0+LINE_YOFFSET, 
                                       fill=color, width=LINE_WIDTH_ERRORSIG)
        
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
    
    # Draw this neuron as a circle
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
    root.resizable(False, False)
    
    root_canvas = Canvas(root)
    root_canvas.pack(expand=Y, fill=BOTH)
    root_canvas.configure(bg='white')
    
    net.canvas = root_canvas
    net.train_and_draw()
    
    root.mainloop()

# Setup some example layers
input_layer = layer(2, "input")
hidden_layer1 = layer(3, "hidden")
output_layer = layer(1, "output")
layer_structure = [input_layer, hidden_layer1, output_layer]

# Initialize the network
net = NeuralNet(len(layer_structure), layer_structure, [[[1, 0],[1]],[[0, 1],[1]],[[1, 1],[0]],[[0, 0],[0]]])
net.createLayers()

# Run the visualization
create_visualization()