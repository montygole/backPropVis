# =============================================================================
# Displays the structure of a neural network using tkinter.
# =============================================================================

from functools import partial
from tkinter import *
import matplotlib.pyplot as plt

# The number of nodes per layer
LAYERS = [7, 4, 2, 4, 7]

# The size of the nodes displayed in layers
NODE_DISPLAY_SIZE = 50

# Represents a neuron in a network
class NodeVis:
    def __init__(self):
        # TODO indivudial display paramters can be set here
        pass
        
# Represents a layer of a network
class LayerVis:
    def __init__(self, count):
        self.nodes = []
        for i in range(count):
            nodes.append(NodeVis())

#TEMP #DEBUG
def callback_me(arg, event):
    print('I am: ', arg.canvas.winfo_rootx())

# Creates a clickable canvas
# DEBUG note, this *was* intended to be used for interactivity
# but tkinter doesn't play well with overlaying canvases.
# However, this is still kept as it is an easy way to align the shapes
def create_clickable_canvas_circle(parent, size, callback):
    c = Canvas(parent, width=size, height=size)
    c.bind('<Button-1>', callback)
    c.create_oval(2, 2, size, size,
                outline='', fill='')
    return c

# DEBUG: related to create_clickable_canvas_circle
class NodeButton:
    def __init__(self, parent, size):
        self.canvas = create_clickable_canvas_circle(
                    parent, size, partial(callback_me, self)
                  )
# DEBUG: related to create_clickable_canvas_circle
def draw_layer(parent, size):
    layer_canvas = Canvas(parent)
    layer_canvas.configure(bg='red') #TEMP
    
    for i in range(size):
        nb = NodeButton(layer_canvas, NODE_DISPLAY_SIZE)
        nb.canvas.pack()
        
    return layer_canvas


# TODO should be packed into a main

root = Tk()
root.geometry('500x500')

root_canvas = Canvas(root)
root_canvas.pack(expand=Y, fill=BOTH)
root_canvas.configure(bg='cyan') # TEMP color to help with viz


# TODO: this name is not the best
# Here it represents canvases that are used to position the shapes
layers = []

for count in LAYERS: # Creates layers based on the constant
    layers.append(draw_layer(root_canvas, count))

for l in layers: # Positions them as tkinter objects
    l.pack(padx=20, side=LEFT)

root.update() # Called to allow packing to happen so positions can be set

for l in layers: # Removes the canvases from display, since we only need the positions
    l.pack_forget()

# Now, draw the shapes onto a single canvas using the positions after packing
# Draw layers 0 to one short of the end
for i in range(len(layers)-1):
    l0 = layers[i]
    l1 = layers[i+1]
    for c0 in l0.winfo_children():
        x0 = c0.winfo_rootx()-root.winfo_x()
        y0 = c0.winfo_rooty()-root.winfo_y()
        halfwidth = NODE_DISPLAY_SIZE / 2
        
        root_canvas.create_oval(
                x0 - halfwidth, y0 - halfwidth,
                x0 + halfwidth, y0 + halfwidth,
                fill = '#F00'
            )
        
        # Draw connections
        for c1 in l1.winfo_children():
            root_canvas.create_line(
                x0, y0, 
                c1.winfo_rootx()-root.winfo_x(), c1.winfo_rooty()-root.winfo_x(), 
                width=1)
            
# Draw the final layer
l0 = layers[-1]
for c0 in l0.winfo_children():
    x0 = c0.winfo_rootx()-root.winfo_x()
    y0 = c0.winfo_rooty()-root.winfo_y()
    halfwidth = NODE_DISPLAY_SIZE / 2
    
    root_canvas.create_oval(
            x0 - halfwidth, y0 - halfwidth,
            x0 + halfwidth, y0 + halfwidth,
            fill = '#F00'
        )

root.mainloop()