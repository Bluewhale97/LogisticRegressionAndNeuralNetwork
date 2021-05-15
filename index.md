## Introduction

Logistic regression is a classifier for two classes. To build a general architecture of a learning algorithm, we should initialize parameters, calculate the cost function and gradient as well as use an optimization algorithm.

In this article, we will be looking at buidling a logistic classifier to recognize cats, which will step through how to do this with a neural network mindset about deep learning.

## 1. Packages

We should implement some packages for assignments. Numpy is the fundamental package for computation in Python, h5py is a common package to interact with a dataset that is stored on an H5 file and matplotlib is a library to plot graphs, PIL and scipy are used to test the model with new data.

Now import packages:

```python
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *

%matplotlib inline
%load_ext autoreload
%autoreload 2
```

## 2. Overview of the Problem set

We are given a dataset data.h5 containing a training set of m_train images labeled as cat or non-cat, a test set of m_test images labeled as cat or non-cat and each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels(RGB). Thus each image is square (height = num_px) and (width=num_px).

Now we are going to build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.

Load the data:
```python
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
```
We added "_orig" at the end of image datasets (train and test) because we are going to preprocess them. After preprocessing, we will end up with train_set_x and test_set_x (the labels train_set_y and test_set_y don't need any preprocessing).

Visualize an example of an image:
```python
# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
```

See the picture: it is a cat:

![image](https://user-images.githubusercontent.com/71245576/118326781-7214d880-b4d3-11eb-8c17-6e290d973ddd.png)

Specifically, train_set_x_orig is a numpy-array of shape (m_train, num_px, num_px, 3). For instance, you can access m_train by writing train_set_x_orig.shape[0]

Let's grab more information:

```python
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]


print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
```

The result shows as expected:

![image](https://user-images.githubusercontent.com/71245576/118331183-3e877d80-b4d6-11eb-9de4-8f02bcbfbfca.png)

Reshape the training and test data sets so that images of size(num_px, num_px, 3) are flattened into single vectors of shape(num_px*num_px*3,1)

A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b ∗ c ∗ d, a) is to use:

```python
X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
```

```python
# Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Check that the first 10 pixels of the second image are in the correct place
assert np.alltrue(train_set_x_flatten[0:10, 1] == [196, 192, 190, 193, 186, 182, 188, 179, 174, 213]), "Wrong solution. Use (X.shape[0], -1).T."
assert np.alltrue(test_set_x_flatten[0:10, 1] == [115, 110, 111, 137, 129, 129, 155, 146, 145, 159]), "Wrong solution. Use (X.shape[0], -1).T."

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
```

The result shows:

![image](https://user-images.githubusercontent.com/71245576/118332335-ecdff280-b4d7-11eb-8af2-389eb8b7879c.png)

To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the pixel value is actually a vector of three numbers ranging from 0 to 255.

We now standardize our dataset:

```python
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.
```
## 3. Logistic regression as simple neural network

We actually can understand the logistic regression as a very simple neural network, 

For example, when we recognize the image from cats:

![image](https://user-images.githubusercontent.com/71245576/118333656-139f2880-b4da-11eb-988b-4743ec07606f.png)

The mathematical expression of the algorithm is: for one example 𝑥(𝑖):

![image](https://user-images.githubusercontent.com/71245576/118333746-41846d00-b4da-11eb-8139-6ca4a79ea624.png)

The cost then is computed by summing over all training examples:

![image](https://user-images.githubusercontent.com/71245576/118333773-4d702f00-b4da-11eb-82fb-a230f4c44c8f.png)

## 4. Building the algorithm

The main steps for building a neural network are: define the model structure such as number of input features, initialize the model's parameters as well as looping such as calculate current loss, gradient and updata parameters. Finally we integrate the 3 separated parts into one function we call model().

### 4.1 Define the model structure

Implement a sigmoid() which compute sigmoid(z) = 1/(1+e^-z) for z= 𝑤𝑇𝑥+𝑏.

```python
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s= 1/(1+np.exp(-z))
    
    return s
 ```
Now test for this function:
```python
 x = np.array([0.5, 0, 2.0])
output = sigmoid(x)
print(output)
```

The result is:

![image](https://user-images.githubusercontent.com/71245576/118334237-2c5c0e00-b4db-11eb-9f46-f025b584665f.png)

### 4.2 Initializing parameters

To initialize our parameters, we have to initialize w as a vector of zeros. The function np.zeros() in the Numpy library helps it.

```python
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    """

    w = np.zeros((dim,1))
    b = 0.0 


    return w, b
```

The instance to verify:

```python
dim = 2
w, b = initialize_with_zeros(dim)

assert type(b) == float
print ("w = " + str(w))
print ("b = " + str(b))

initialize_with_zeros_test(initialize_with_zeros)
```

The result:

![image](https://user-images.githubusercontent.com/71245576/118335195-18191080-b4dd-11eb-94d1-bb91341d14e2.png)

### 4.3 Forward and backward propagation

Now parameters are initialized, we can do the "forward" and "backward" propagation steps for learning the parameters.

We need to implement a function propagate() that computes the cost function and its gradient, there are two formulas we should use:

![image](https://user-images.githubusercontent.com/71245576/118335282-50205380-b4dd-11eb-8ef8-474336e187a1.png)

The cost function is: 

![image](https://user-images.githubusercontent.com/71245576/118335958-bc4f8700-b4de-11eb-8440-bd4ca8558196.png)

Let's write the propagation code:

```python

"""
Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
"""

def propagate(w, b, X, Y):
 
    
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -1/m*(np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T))
    
    dw = 1/m*np.dot(X,(A-Y).T)
    db = 1/m*np.sum(A-Y)

    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
 ```
 
 The instance:
 ```python
 w =  np.array([[1.], [2.]])
b = 2.
X =np.array([[1., 2., -1.], [3., 4., -3.2]])
Y = np.array([[1, 0, 1]])
grads, cost = propagate(w, b, X, Y)

assert type(grads["dw"]) == np.ndarray
assert grads["dw"].shape == (2, 1)
assert type(grads["db"]) == np.float64


print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

propagate_test(propagate)
```

The result is:

![image](https://user-images.githubusercontent.com/71245576/118364609-1ac04800-b567-11eb-8e83-7f92ff02cd82.png)

### 4.4 Optimazation

We have initialized the parameters and compute a cost function and its gradient, now we can update the parameters using gradient descent. Our goal here is to learn w and b by minimizing the cost function J. For a parameter 𝜃, the update rule is 𝜃=𝜃-𝛼 𝑑𝜃, where 𝛼 is the learning rate.

```python
def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        # (≈ 1 lines of code)
        # Cost and gradient calculation 
        # grads, cost = ...
        grads, cost = propagate(w,b,X,Y)
        
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        # w = ...
        # b = ...
        w = w-learning_rate*dw
        b = b-learning_rate*db
        
        # YOUR CODE ENDS HERE
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs
 ```
 
 The instance:
 ```python
 params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print("Costs = " + str(costs))

optimize_test(optimize)
```

Result shows:

![image](https://user-images.githubusercontent.com/71245576/118365003-e188d780-b568-11eb-98fc-afc55bda967f.png)

Now, let's predict the labels for a dataset from the trained w and b. We need to calculate the predicted value and convert the entries of a into 0 or 1, stores the predictions in a vector Y_prediction. 

```python
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T,X)+b)
    
    
    for i in range(A.shape[1]):
     
        if A[0,i]>=0.5:
            Y_prediction[0,i]=1
        else:
            Y_prediction[0,i]=0
    
    return Y_prediction
 ```
 
 Call the statement for an instance:
 
 ```python
 w = np.array([[0.1124579], [0.23106775]])
b = -0.3
X = np.array([[1., -1.1, -3.2],[1.2, 2., 0.1]])
print ("predictions = " + str(predict(w, b, X)))

predict_test(predict)
```
The result:

![image](https://user-images.githubusercontent.com/71245576/118365263-f2861880-b569-11eb-87e1-54c44d44b385.png)

## 5. Merging all functions

We did implement separated parts of a logistic regression prediction, now we can see how the overall model is structured by putting together all the building blocks.

```python
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    # (≈ 1 line of code)   
    # initialize parameters with zeros 
    # w, b = ...
    w, b = initialize_with_zeros(X_train.shape[0])
    #(≈ 1 line of code)
    # Gradient descent 
    # parameters, grads, costs = ...
    parameters, grads, costs =  optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    # Retrieve parameters w and b from dictionary "parameters"

    w = parameters["w"]
    b = parameters["b"]
    # Predict test/train set examples (≈ 2 lines of code)
 
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)


    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
 ```
 
 Tested the model and all tests passed, now we run the logistic regression model:
 ```python
 logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
 ```
 The result:
 
![image](https://user-images.githubusercontent.com/71245576/118365902-88bb3e00-b56c-11eb-9e6e-d9845c3f0a69.png)

Training accuracy is close to 100%. This is a good sanity check: your model is working and has high enough capacity to fit the training data. Test accuracy is 70%. It is actually not bad for this simple model, given the small dataset we used and that logistic regression is a linear classifier. 

Here is an example of a picture that was wrongly classified:
```python
index = 1
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(logistic_regression_model['Y_prediction_test'][0,index])].decode("utf-8") +  "\" picture.")
```

The result:

![image](https://user-images.githubusercontent.com/71245576/118366022-08490d00-b56d-11eb-862e-28de0a5ac722.png)

now plot the cost function and the gradients:

```python
# Plot learning curve (with costs)
costs = np.squeeze(logistic_regression_model['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
plt.show()
```
You can see the cost decreasing. It shows that the parameters are being learned. However, you see that you could train the model even more on the training set. Try to increase the number of iterations in the cell above and rerun the cells. You might see that the training set accuracy goes up, but the test set accuracy goes down. This is called overfitting.

![image](https://user-images.githubusercontent.com/71245576/118366076-46463100-b56d-11eb-94c0-09c154bbad64.png)

## 6. Further analysis

In order for Gradient Descent to work you must choose the learning rate wisely. The learning rate  𝛼  determines how rapidly we update the parameters. If the learning rate is too large we may "overshoot" the optimal value. Similarly, if it is too small we will need too many iterations to converge to the best values. That's why it is crucial to use a well-tuned learning rate.

Let's compare the learning curve of our model with several choices of learning rates. Run the cell below. This should take about 1 minute. Feel free also to try different values than the three we have initialized the learning_rates variable to contain, and see what happens.

```python
learning_rates = [0.01, 0.001, 0.0001]
models = {}

for lr in learning_rates:
    print ("Training a model with learning rate: " + str(lr))
    models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for lr in learning_rates:
    plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
```

See the result:

![image](https://user-images.githubusercontent.com/71245576/118366204-c8365a00-b56d-11eb-8251-6d544f01bc38.png)

Now, let's test with our own image, 

```python
# change this to the name of your image file
my_image = "my_image.jpg"   

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
```

The result shows:

![image](https://user-images.githubusercontent.com/71245576/118366364-e56b2880-b56d-11eb-8eea-11d7b81095e3.png)

## Reference

Logistic regression with a neural network mindset, retrieved from https://www.coursera.org/learn/neural-networks-deep-learning/programming/thQd4/logistic-regression-with-a-neural-network-mindset/lab


