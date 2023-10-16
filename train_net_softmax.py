# coding: utf-8
import sys, os
sys.path.append(os.pardir)
from net_softmax import TwoLayerNet_softmax 
from dataset import generate_XOR_easy,generate_linear
from plot import show_result,plot_loss,plot_acc
import numpy as np
from functions import cross_entropy_error

num_pt = 100
x,y = generate_linear(n=100)
lyy = 1-y
prob_y = np.append(lyy,y,axis = 1)

(x_train, t_train) = (x,prob_y)

network_softmax = TwoLayerNet_softmax(input_size=2, hidden_size=20, output_size=2)

iters_num = 40000
train_size = x_train.shape[0]
learning_rate = 0.01

train_loss_list = []
train_acc_list = []

for i in range(iters_num):
    
    grad = network_softmax.gradient(x_train, t_train)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
      network_softmax.params[key] = network_softmax.params[key] - learning_rate * grad[key]
    
    loss = network_softmax.loss(x_train, t_train)
    train_loss_list.append(loss)

    train_acc = network_softmax.accuracy(x_train, t_train)
    train_acc_list.append(train_acc)
    
    if i % 5000 == 0:
      print("epoch = ",i," loss = ",loss,"train_acc = {:.0%}".format(train_acc))


#show_result(x_train,t_train,network_sigmoid.predictions)
plot_loss(train_loss_list)
plot_acc(train_acc_list)


#================== tesing ==================
test_y = network_softmax.predict(x_train)
i = 1
for ele in test_y:
   format01 = "{:.5f}".format(float(ele[0]))
   format02 = "{:.5f}".format(float(ele[1]))
   print("prediction ",i,": ",format01,format02)
   i = i + 1

test_error = cross_entropy_error(test_y, t_train)

test_y_class = np.argmax(test_y, axis=1)
test_acc = np.sum(test_y_class == t_train) / float(x_train.shape[0])

print("test loss = {:.5f}".format(test_error)," test acc = {:.0%}".format(test_acc))

test_y_dot = np.zeros((num_pt, 1))
for idx in range(0,num_pt):
  if test_y[idx][0]<test_y[idx][1]:
    test_y_dot[idx] = 1


show_result(x,y,test_y_dot)
#%%