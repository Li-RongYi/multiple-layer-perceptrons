import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Relu():
    def f(self,data):
        return np.maximum(0,data)
    def df(self,data):
        return np.array([1 if i>=0 else 0 for i in data])

class Sigmoid():
    def f(self,data):
        return 1.0 / (1.0 + np.exp(-data))
    def df(self,data):
        return data * (1.0 - data)

class Tanh():
    def f(self,data):
        return (np.exp(data) - np.exp(-data))/(np.exp(data) + np.exp(-data))
    def df(self,data):
        return 1-(data*data)


class MLP_Model():
    def __init__(self,n_inputs=4,hidden_layer=[],label=[]):
        self.n_inputs=n_inputs
        self.label=label
        self.n_outputs=len(label)
        self.deltas=[]
        self.input = []
        self.output = []
        self.hidden_layer=[]
        self.epoch=[]
        self.train_error=[]
        self.validation_error=[]
        self.bias=[]

        #input layer and first hidden layer
        self.hidden_layer.append(np.random.uniform(-2,2,(n_inputs,hidden_layer[0])))
        self.bias.append(np.random.uniform(-2,2,hidden_layer[0]))
        #hidden layers
        for i in range(len(hidden_layer) - 1):
            self.hidden_layer.append(np.random.uniform(-2,2,(hidden_layer[i],hidden_layer[i+1])))
            self.bias.append(np.random.uniform(-2,2,(hidden_layer[i+1])))
        #last hidden layer and output layer
        self.hidden_layer.append(np.random.uniform(-2,2,(hidden_layer[-1], self.n_outputs)))
        self.bias.append(np.random.uniform(-2,2,(self.n_outputs)))
        #print(self.hidden_layer)
        #bias


    def forward_propagate(self,data):
        self.input.clear()
        self.output.clear()
        self.input.append(data)
        new_output=data
        for layer,bias in zip(self.hidden_layer,self.bias):
            new_output = np.matmul(new_output,layer)+bias
            new_output = self.activation.f(new_output)
            self.output.append(new_output)

        self.input.extend(self.output[:-1])
        return new_output

    def backward_propagate(self,expected):
        self.deltas.clear()
        derivative = self.activation.df(self.output[-1])
        self.deltas.append((expected-self.output[-1])*derivative)
        for layer,output in zip(reversed(self.hidden_layer[1:]),reversed(self.output[:-1])):
            delta = []
            derivative = self.activation.df(output)
            for i in layer:
                delta.append(np.sum(i*self.deltas[-1]))
            self.deltas.append(np.array(delta)*derivative)
        self.deltas.reverse()

        #update weight
        for i in range(len(self.hidden_layer)):
            #derivative = self.activation.df(self.output[i])
            for j in range(len(self.hidden_layer[i])):
                self.hidden_layer[i][j]+=self.learning_rate*self.deltas[i]*self.input[i][j]


            self.bias[i]+=self.learning_rate*self.deltas[i]

    def train(self,n_epoch,data,label,learning_rate=0.05,activation='Relu'):
        #np.seterr(divide='ignore', invalid='ignore')
        self.epoch.clear()
        self.train_error.clear()
        self.validation_error.clear()
        text=""

        self.learning_rate=learning_rate
        if activation=='Relu':
            self.activation=Relu()
        elif activation=='Sigmoid':
            self.activation = Sigmoid()
        elif activation=='Tanh':
            self.activation = Tanh()

        print(activation)

        train_data,validation_data,train_label,validation_label=train_test_split(data,label,test_size=0.2)
        try:
            for _ in range(n_epoch):
                validation_error = 0
                for data, label in zip(validation_data, validation_label):
                    output = self.forward_propagate(data)
                    expected = np.array([0 for i in range(self.n_outputs)])
                    expected[self.label.index(label)] = 1
                    validation_error += sum([(output[i] - expected[i]) ** 2 for i in range(self.n_outputs)])

                train_error=0
                for data,label in zip(train_data,train_label):
                    output=self.forward_propagate(data)
                    expected = np.array([0 for i in range(self.n_outputs)])
                    expected[self.label.index(label)]=1
                    train_error+=sum([(output[i]-expected[i])**2 for i in range(self.n_outputs)])
                    self.backward_propagate(expected)

                if _ % 1 ==0:
                    train_error = train_error/float(len(train_label))/self.n_outputs
                    validation_error = validation_error/float(len(validation_label))/self.n_outputs
                    # if len(self.validation_error) > 0 and validation_error > self.validation_error[-1]:
                    #     return True,text
                    if train_error <= 0.01:
                        return
                    text+=("n_epoch:%d learning_rate:%.3f\ntrain_error:%.4f validation_error:%.4f\n\n"%(_,self.learning_rate,train_error,validation_error))
                    self.epoch.append(_)
                    self.train_error.append(train_error)
                    self.validation_error.append(validation_error)
            return True,text
        except Exception as e:
            return False,str(e)



    def predict(self,data):
        predictions=[]
        for i in data:
            output=list(self.forward_propagate(i))
            predictions.append(self.label[output.index(max(output))])
        return predictions

    def show(self):
        if len(self.train_error):
            print(self.hidden_layer)
            plt.title('error_curve')
            plt.plot(self.epoch,self.train_error,label='train_error')
            plt.plot(self.epoch, self.validation_error,label='validation_error')
            plt.legend()
            plt.show()



