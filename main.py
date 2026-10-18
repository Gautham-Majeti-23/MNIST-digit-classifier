from pathlib import Path
import numpy as np
import pandas as pd
from grayscale_converter import image_to_vector
from nn import Network, softmax

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

nn = Network([784, 128, 64, 10])
nn.fit(X_train, Y_train)

a = 0
for i in range(len(X_test)):
    if nn.forward(X_test[i])[1] == Y_test[i]:
        a += 1
print(a / len(X_test) * 100)

path = Path('pixil-frame-0.png')
t = image_to_vector(path) / 255
output = nn.forward(np.array(t))

print(np.array(softmax(output[0][-1])) * 100, int(output[1]))
