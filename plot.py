import matplotlib.pyplot as plt
import numpy as np

def show_result(x,y,pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth',fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result',fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()

def plot_loss(loss_list):
    x = np.arange(len(loss_list))
    plt.plot(x, loss_list, label='loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.ylim(min(loss_list), max(loss_list)+2)
    plt.legend(loc='upper right')
    plt.show()

def plot_acc(acc_list):
    x = np.arange(len(acc_list))
    plt.plot(x, acc_list, label='acc')
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.ylim(0, 1.0)
    plt.legend(loc='upper right')
    plt.show()
