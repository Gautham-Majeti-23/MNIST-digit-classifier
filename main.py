import mlx.core as np
import pandas as pd
from pathlib import Path
from grayscale_converter import image_to_vector

def linear_function(w, x, b):
    '''
    z = [X][W] + [B]
    '''
    return np.matmul(w, x) + b

def activation_function(x):
    '''
    ReLU
    '''
    return np.maximum(0, x)

def total_output(w, x, b):
    '''
    ReLU(z)
    '''
    return activation_function(linear_function(w, x, b))

def activation_gradient(a):
    '''
    ReLU'
    '''
    return (a > 0).astype(np.float32)

# Train set.
train_set = pd.read_csv('mnist_train.csv')

X_train = np.array(train_set.drop(columns=['label']).values) / 255
Y = train_set['label'].values

# For computing error function
Y_train = []
for i, a in enumerate(Y):
    Y_train.append(np.array([1 if j == a else 0 for j in range(10)], dtype=np.float32))

# Test set.
test_set = pd.read_csv('mnist_test.csv')

X_test = np.array(test_set.drop(columns=['label']).values) / 255
Y_test = np.array(test_set['label'].values)

class Network:
    def __init__(self, layer_counts):
        '''
        Initialises the values of parameters and depths. 

        Structure of layer_wise_wts:
            Dimension 1: Layers
            Dimension 2: Neurons
            Dimension 3: Weights
        
        Structure of layer_wise_biases:
            Dimension 1: Layers
            Dimension 2: Biases
        '''
        self.layer_counts = layer_counts

        self.layer_wise_wts = []
        self.layer_wise_biases = []

        for i, a in enumerate(self.layer_counts):
            if i == 0: continue
            layer_wts = np.random.normal(shape=(a, self.layer_counts[i - 1])) * np.sqrt(2 / self.layer_counts[i - 1])
            layer_biases = np.zeros(a)
            self.layer_wise_wts.append(layer_wts)
            self.layer_wise_biases.append(layer_biases)

    def fit(self, data, labels, lr = 0.01):
        '''
        Training the model.
        '''
        for i, a in enumerate(data):
            predictions = self.forward(a)
            output = np.softmax(predictions[0][-1])
            error = output - labels[i]
            self.backprop(predictions[0], lr, error)

    def forward(self, x):
        '''
        Passes inputs through linear and activation and gives output in linear func (for easier backprop implementation) which we'll check with training labels.

        Return: outputs of all layers, predicted output
        '''
        outputs = [x]
        prev_layer_output = x
        for i, a in enumerate(self.layer_wise_wts):
            biases = self.layer_wise_biases[i]
            
            if i < len(self.layer_wise_wts) - 1:
                prev_layer_output = total_output(a, prev_layer_output, biases)
            else:
                prev_layer_output = linear_function(a, prev_layer_output, biases)

            outputs.append(prev_layer_output)

        return outputs, np.argmax(np.softmax(prev_layer_output))

    def backprop(self, prs, lr, err):
        '''
        Back Propagation

        checks gradient of every single param.
        '''
        delta = err
        for i in range(len(self.layer_wise_wts) - 1, -1, -1):
            layer_input = prs[i]

            wt_grad_tracker = delta[:, None] * layer_input[None, :]
            bias_grad_tracker = delta

            if i > 0:
                next_delta = np.matmul(self.layer_wise_wts[i].T, delta) * activation_gradient(prs[i])

            self.layer_wise_wts[i] -= lr * wt_grad_tracker
            self.layer_wise_biases[i] -= lr * bias_grad_tracker

            if i > 0:
                delta = next_delta

        np.eval(self.layer_wise_wts, self.layer_wise_biases)


nn = Network([784, 64, 10])
nn.fit(X_train, Y_train)

a = 0

for i in range(len(X_test)):
    if nn.forward(X_test[i])[1] == Y_test[i]:
        a += 1

print(a / len(X_test) * 100) # ~ 95%



'''
Template for testing your own 28 * 28 pixel with the network
'''
path = Path('your path')
test = image_to_vector(path) # Set invert arg to True if picture has white background.

# test
nn.forward(np.array(t))[1]
