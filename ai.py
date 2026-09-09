import math
import numpy as np
import random

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    if x > 100: return 1.0
    if x < -100: return -1.0
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

class node():
    def __init__(self, wheights, bias):
        self.wheights = wheights
        self.bias = bias
        self.result = 0

    def giveResult(self):
        return self.result
    
    def giveWheights(self):
        return self.wheights

    def giveBias(self):
        return self.bias    

    def logic(self, entries):
        total = 0
        for i in range(len(entries)):
            total += entries[i] * self.wheights[i]
        total += self.bias
        self.result = tanh(total)
        return self.result
    
    def mutation(self, mut):
        for i in range(len(self.wheights)):
            if random.randint(0, 10) < 2:
                self.wheights[i] += random.uniform(-5.0, 5.0) * mut
        
        if random.randint(0, 10) < 2:
            self.bias += random.uniform(-5.0, 5.0) * mut


class layer():
    def __init__(self, size, wheights, bias):
        self.size = size
        self.nodes = []
        self.results = []
        self.wheights = wheights
        self.bias = bias

    def giveWheights(self):
        return self.wheights

    def giveBias(self):
        return self.bias

    def createLayer(self):
        for i in range(self.size):
            self.nodes.append(node(self.wheights[i], self.bias[i]))
    
    def updateData(self):
        self.wheights.clear()
        self.bias.clear()
        for i in range(self.size):
            self.wheights.append(self.nodes[i].giveWheights())
            self.bias.append(self.nodes[i].giveBias())

    def layerLogic(self, entries):
        self.results.clear()
        for i in range(self.size):
            self.results.append(self.nodes[i].logic(entries))
        return self.results
    
    def mutLayer(self, mut):
        for i in range(self.size):
            self.nodes[i].mutation(mut)


class Network():
    def __init__(self, layers_count, per_layer, wheights, bias, entry_size, exit_size):
        self.layers_count = layers_count
        self.per_layer = per_layer
        self.entry_size = entry_size
        self.layers = []
        self.exit_size = exit_size

        if wheights is not None:
            self.wheights = wheights
        else:
            self.wheights = []
            for i in range(self.layers_count):
                templayer = []
                if i == 0:
                    for x in range(self.per_layer):
                        temp = []
                        for t in range(self.entry_size):
                            temp.append(random.uniform(-5.0, 5.0))
                        templayer.append(temp)
                else:     
                    for x in range(self.per_layer):
                        temp = []
                        for t in range(self.per_layer):
                            temp.append(random.uniform(-5.0, 5.0))
                        templayer.append(temp)
                self.wheights.append(templayer) 
            
            templayer = []
            for x in range(self.exit_size):
                temp = []
                for t in range(self.per_layer):
                    temp.append(random.uniform(-5.0, 5.0))
                templayer.append(temp)
            self.wheights.append(templayer)

        if bias is not None:
            self.bias = bias
        else:
            self.bias = []
            for i in range(self.layers_count):
                templayer = []
                # Correction: ajout de self. devant per_layer
                for t in range(self.per_layer): 
                    templayer.append(random.uniform(-5.0, 5.0))
                self.bias.append(templayer)
            
            templayer = []
            for t in range(self.exit_size):
                templayer.append(random.uniform(-5.0, 5.0))
            self.bias.append(templayer)
        
    def createNetwork(self):
        for i in range(self.layers_count):
            self.layers.append(layer(self.per_layer, self.wheights[i], self.bias[i]))
            self.layers[i].createLayer()
            
        a = self.layers_count
        self.layers.append(layer(self.exit_size, self.wheights[a], self.bias[a]))
        self.layers[a].createLayer()
        
    def mutNetwork(self, mut):
        for i in range(self.layers_count + 1):
            self.layers[i].mutLayer(mut)
    
    def logic(self, entries):
        for i in range(self.layers_count + 1):
            entries = self.layers[i].layerLogic(entries)
        return entries

    def sync_data(self):
        self.wheights = []
        self.bias = []
        for i in range(self.layers_count + 1):
            self.layers[i].updateData() # Demande à la couche de se mettre à jour
            self.wheights.append(self.layers[i].giveWheights())
            self.bias.append(self.layers[i].giveBias())
            
    def export_json(self):
        self.sync_data()
        return {
            "layers_count": self.layers_count,
            "per_layer": self.per_layer,
            "entry_size": self.entry_size,
            "exit_size": self.exit_size,
            "wheights": self.wheights,
            "bias": self.bias
        }