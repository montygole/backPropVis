from tkinter import *

# The number of nodes per layer
LAYERS = [5, 4, 3, 4, 5]

# The size of the nodes displayed in layers
NODE_DISPLAY_SIZE = 40
NODE_DISPLAY_XPAD = 20
NODE_DISPLAY_YPAD = 20

# 
class NetworkVis:
    def __init__(self, layer_structure):
        self.layers = [] # The layers of the network
        self.nodes_linear = [] # The nodes visuals of the net in a linear array
        self.weights_linear = [] # The weights of the net in a linear array
        
        for i, node_count in enumerate(layer_structure):
            layer = LayerVis(node_count)
            self.layers.append(layer)
            self.nodes_linear += layer.nodes
            
    def draw(self, parent, x, y):
        previous_node_centers = self.layers[0].draw(parent, x, y, [])
        
        for i, layer in enumerate(self.layers[1:], start = 1):
            x0 = x+i*(NODE_DISPLAY_SIZE+NODE_DISPLAY_XPAD)
            y0 = y
            previous_node_centers = layer.draw(parent,
                                       x0,
                                       y0,
                                       #TODO map weights to colors
                                       previous_node_centers
                                   )
        
                    
# Represents a layer of a network
class LayerVis:
    def __init__(self, count):
        self.nodes = []
        for i in range(count):
            self.nodes.append(NodeVis())
            
    def draw(self, parent, x, y, previous_node_centers):
        
        node_centers = []
        for i, node in enumerate(self.nodes):
            x0 = x
            y0 = y + i*(NODE_DISPLAY_SIZE+NODE_DISPLAY_YPAD)
            
            node.draw(parent, 
                          x0,
                          y0,
                          NODE_DISPLAY_SIZE/2
                      )
            
            node_centers.append((x0, y0))
            
            for prev in previous_node_centers:
                
                parent.create_line(prev[0], prev[1], x0, y0)
            
        
        return node_centers

# Represents a neuron in a network
class NodeVis:
    def __init__(self):
        pass
    
    def draw(self, parent, x, y, halfwidth):
        parent.create_oval(
                x - halfwidth, y - halfwidth,
                x + halfwidth, y + halfwidth,
                fill = 'white'
            )

# TODO should be packed into a main
def create_visualization(layer_structure):
    root = Tk()
    root.geometry('500x500')
    
    root_canvas = Canvas(root)
    root_canvas.pack(expand=Y, fill=BOTH)
    root_canvas.configure(bg='white')
    
    nn_vis = NetworkVis(layer_structure)
    nn_vis.draw(root_canvas, NODE_DISPLAY_SIZE/2, NODE_DISPLAY_SIZE/2)
    
    root.mainloop()
    
create_visualization(LAYERS)