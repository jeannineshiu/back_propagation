# coding: utf-8
import sys, os
sys.path.append(os.pardir)
from net_sigmoid import TwoLayerNet_sigmoid
from dataset import generate_XOR_easy,generate_linear
from plot import show_result,plot_loss,plot_acc
import numpy as np
from functions import sum_squared_error

#(x_train, t_train) = generate_XOR_easy()
(x_train, t_train) = generate_linear(n=100)

#num_pt = 21
num_pt = 100

network_sigmoid = TwoLayerNet_sigmoid(input_size=2, hidden_size=100, output_size=1,num_pt=num_pt)
#network_sigmoid = TwoLayerNet_sigmoid(input_size=2, hidden_size=100, output_size=1)

iters_num = 10000
#train_size = x_train.shape[0]
learning_rate = 0.01

train_loss_list = []
train_acc_list = []

for i in range(iters_num):
    
    grad = network_sigmoid.gradient(x_train, t_train)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
      network_sigmoid.params[key] = network_sigmoid.params[key] + learning_rate * grad[key]
    
    loss = network_sigmoid.loss(x_train, t_train)
    train_loss_list.append(loss)

    train_acc = network_sigmoid.accuracy(x_train, t_train)
    train_acc_list.append(train_acc)
    
    if i % 500 == 0:
      print("epoch = ",i," loss = ",loss,"train_acc = {:.0%}".format(train_acc))


#show_result(x_train,t_train,network_sigmoid.predictions)
plot_loss(train_loss_list)
plot_acc(train_acc_list)


#================== tesing ==================
test_y = network_sigmoid.predict(x_train)
a_flatten = test_y.reshape(-1)
i = 1
for ele in a_flatten:
    print("prediction ",i," : {:.5f}".format(float(ele)))
    i += 1

test_error = sum_squared_error(test_y, t_train)

test_y_class = np.zeros((num_pt, 1))
for idx in range(0,num_pt):
    if test_y[idx]>=0.5:
       test_y_class[idx] = 1
test_acc = np.sum(test_y_class == t_train) / float(x_train.shape[0])

print("test loss = {:.5f}".format(test_error)," test acc = {:.0%}".format(test_acc))

show_result(x_train,t_train,test_y_class)
#%%