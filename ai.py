import math
import numpy as np
from random import randint

def relu(x) :
    return np.maximum(0, x)

def tanh(x) :
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

class node() :
    def __init__(self, wheights, bias) :
        self.wheights = wheights
        self.bias = bias
        self.result = 0

    def giveResult(self) :
        return self.result
    
    def giveWheights(self) :
        return self.wheights

    def giveBias(self) :
        return self.bias
    
    def logic(self, entries) :
        total = 0
        for i in range(len(entries)) :
            total += entries[i] * self.wheights[i]
        total += self.bias
        self.result = tanh(total)
        return self.result
    
    def mutation(self, mut) :
        for i in range(len(self.wheights)) :
            if random.randint(0, 10) < 2 :
                self.wheights[i] += random.randint(-5, 5) * mut
        self.bias += random.randint(-5, 5) * mut


class layer() :
    def __init__(self, size, wheights, bias, entry_size) :
        self.size = size
        self.nodes = []
        self.results = []
        if wheights is not NONE :
            self.wheights = wheights
        else :
            self.wheights = []
            for i in range(size) :
                temp = []
                for t in range(entry_size) :
                    temp.append(random.randint(-5, 5))
                self.wheights.append(temp)
        
        if bias is not NONE :
            self.bias = bias
        else :
            self.bias = []
            for i in range(size) :
                self.bias.append(random.randint(-5, 5))
    
    def 

    def giveWheights(self) :
        return self.wheights

    def giveBias(self) :
        return self.bias

    def createLayer(self) :
        for i in range(self.size) :
            self.nodes.append(node(self.wheights[i], self.bias[i]))
    
    def updateData(self) :
        self.wheights.clear()
        self.bias.clear()
        for i in range(self.size) :
            self.wheights.append(self.nodes[i].giveWheights())
            self.bias.append(self.nodes[i].giveBias())

    def layerLogic(self, entries) :
        for i in range(self.size) :
            self.results.append(self.nodes[i].logic(entries))

class Network() :
    def __init__(self, layers, per_layers, wheights, bias) :
