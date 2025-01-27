#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class NeuralNetwork():
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        #creamos pesos entre los nodos
        #distribución normal -> 0.0 es la media, pow es la desviacion estandard = self.hnodes^-0.5 y self.hnodes,self.inodes es la matriz
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #activation function is the sigmoid funciton
        self.activation_function = lambda x: scipy.special.expit(x)
        
        
    def train(self,inputs_list,targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # error is the (target actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weight recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot( ( output_errors * final_outputs *(1.0 - final_outputs) ) , np.transpose(hidden_outputs))
         # update the weights for the links between the input and hiddenlayers
        self. wih += self.lr * np.dot(( hidden_errors * hidden_outputs *(1.0 - hidden_outputs) ), np.transpose(inputs))
        
        pass
    
    
    def query(self,inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list,ndmin=2).T #crec fa matriu de dos dimensions i la transposa per a que vector quedi vertical
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
       
        return final_outputs 


# In[90]:


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)


# In[91]:


with open(r"C:\Users\Usuario\Downloads\mnist_dataset\mnist_train.csv","r") as data_file:
    data_list = data_file.readlines()
print(data_list[0])


# In[92]:


for record in data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # create the target output values (all 0.01, except the desired label which is 0.99)
    targets = np.zeros(output_nodes) + 0.01
    # all_values[0] is the target label for this record
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass


# In[93]:


with open(r"C:\Users\Usuario\Downloads\mnist_dataset\mnist_test.csv","r") as data_test_file:
    test_data_list = data_test_file.readlines()


# In[94]:


all_values = test_data_list[0].split(",")
print(all_values[0])


# In[95]:


image_array = np.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array,cmap = "Greys", interpolation = "None")


# In[96]:


n.query((np.asfarray(all_values[1:])/255*0.99)+0.01)


# In[97]:


#pagina 160


# In[98]:


# test the neural network
# scorecard for how well the network performs, initially empty
scorecard = []


# In[99]:


# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    print(correct_label, "correct label")
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    print(label, "network's answer")
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to
        scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to
        scorecard
        scorecard.append(0)
    


# In[100]:


print(scorecard)


# In[101]:


# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print ("performance = ", (scorecard_array.sum() /scorecard_array.size)*100,"%")


# In[ ]:





# In[ ]:




