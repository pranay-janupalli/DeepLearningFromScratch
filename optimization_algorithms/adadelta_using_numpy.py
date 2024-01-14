import numpy as np 
import pandas as pd

data=pd.read_csv("clean_weather.csv",index_col=0)
data.ffill()

PREDICTORS = ["tmax", "tmin", "rain"]
TARGET = "tmax_tomorrow"

# Ensure we get the same split every time
np.random.seed(0)
data=data.dropna()
split_data = np.split(data, [int(.7 * len(data)), int(.85 * len(data))])
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = [[d[PREDICTORS].to_numpy(), d[[TARGET]].to_numpy()] for d in
                                                            split_data]


class NeuralNet:
    def __init__(self,input_dim):
        self.initialize_parameters(input_dim)
    
    def initialize_parameters(self,num):
        self.w1=np.random.randn(3,num)
        self.bias1=np.random.randn(1,num)
        self.w2=np.random.randn(num,1)
        self.bias2=np.random.randn(1,1)
        # used by optimizer
        self.avg_w1=np.ones(self.w1.shape)
        self.avg_bias1=np.ones(self.bias1.shape)
        self.avg_w2=np.ones(self.w2.shape)
        self.avg_bias2=np.ones(self.bias2.shape)
        return None
    
    @staticmethod
    def calculate(inputs,weights,bias):
        return inputs @ weights + bias 
    
    def loss(self,targets,):
        return np.mean((self.last_layer_result-targets)**2)
    
    def set_learning_rate(self,lr,epsilon):
        self.lr=lr 
        self.epsilon=epsilon


    
    def __call__(self,inputs):
        self.inputs=inputs
        self.first_layer_result= self.calculate(inputs,self.w1,self.bias1)
        self.last_layer_result= self.calculate(self.first_layer_result,self.w2,self.bias2)
        return self.last_layer_result 
    
    def calculate_gradients(self,target):
        d_error_wrt_d_pred = 2*(self.last_layer_result-target)
        d_last_wrt_d_w2 = self.first_layer_result
        d_last_wrt_d_bias2 = 1
        d_last_wrt_d_first_layer= self.w2 
        d_first_layer_wrt_d_w1 = self.inputs
        d_first_layer_wrt_d_bias1= 1 
        batches=target.shape[0]
        self.grad_w2= (d_last_wrt_d_w2/batches).T @ d_error_wrt_d_pred   # nXB @ BX1
        self.grad_bias2= d_last_wrt_d_bias2 * np.mean(d_error_wrt_d_pred,axis=0)         #Bx1
        self.grad_w1= ((d_first_layer_wrt_d_w1/batches).T @ d_error_wrt_d_pred) @ (d_last_wrt_d_first_layer).T #first term.T @ last @(inteemediate).T
        self.grad_bias1= d_first_layer_wrt_d_bias1 * d_last_wrt_d_first_layer @ np.mean(d_error_wrt_d_pred,axis=0) 

    def calculate_square_of_past_grads(self):
        self.avg_w1=0.95*self.avg_w1+0.05*self.grad_w1**2
        self.avg_bias1=0.95*self.avg_bias1+0.05*self.bias1**2
        self.avg_w2=0.95*self.avg_w2+0.05*self.grad_w2**2
        self.avg_bias2=0.95*self.avg_bias2+0.05*self.bias2**2

    def back_propogation(self,target):
        self.calculate_gradients(target)
        self.calculate_square_of_past_grads()
        # update weights
        self.w1=self.w1 - ((self.lr/np.sqrt(self.avg_w1+self.epsilon))*self.grad_w1)
        self.bias1=self.bias1 - ((self.lr/np.sqrt(self.avg_bias1+self.epsilon))*self.grad_bias1)
        self.w2=self.w2 - ((self.lr/np.sqrt(self.avg_w2+self.epsilon))*self.grad_w2)
        self.bias2=self.bias2 - ((self.lr/np.sqrt(self.avg_bias2+self.epsilon))*self.grad_bias2)

    def learning_rate_scheduler(self,epoch):
        self.lr=(1/(self.lr_decay+epoch))*self.lr 
        return self.lr 
    
    def update_lr(self,epoch):
        self.learning_rate_scheduler(epoch)
    
    def train(self,data,targets,epochs,valid_data,valid_targets):
        predictions=self(valid_data)
        print(f"Loss of epoch: 0 is {self.loss(valid_targets)}")
        for i in range(1,epochs):
            step=2
            for index in range(0,len(data),step):
                predictions=self(data[index:index+step,:])
                self.back_propogation(targets[index:index+step,:])
            predictions=self(valid_data)
            print(f"Loss of epoch: {i} is {self.loss(valid_targets)}")
        print("training is done")

model=NeuralNet(5)
model.set_learning_rate(1e-5,0.00001)
model.train(train_x,train_y,20,valid_x,valid_y)