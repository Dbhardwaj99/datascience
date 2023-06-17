**Perceptron** is a linear supervised machine learning algorithm. It is used for binary classification. This article will introduce you to a very important binary classifier, the perceptrons, which forms the basis for the most popular machine learning models nowadays – the neural networks.

## Introduction

**Perceptron Learning Algorithm** is also understood as an Artificial Neuron or neural network unit that helps to detect certain input data computations in business intelligence. The perceptron learning algorithm is treated as the most straightforward Artificial Neural network. It is a supervised learning algorithm of binary classifiers. Hence, it is a single-layer neural network with four main parameters, **i.e.**, input values, weights and Bias, net sum, and an activation function.

## What is the Perceptron Learning Algorithm?

**There are four significant steps in a perceptron learning algorithm:**

![[Capture-2023-06-14-213921.png]]

**An overview of this algorithm is illustrated in the following Figure:**

![Perceptron Rule](https://www.scaler.com/topics/images/perceptron-rule.webp)

**In a more standardized notation, the perceptron learning algorithm is as follows:**

```plaintext
P <-- inputs with label 1
N <-- inputs with label 0
Initialise w randomly;
while !converge do:
$\hspace{2em}$ Pick random x $\in P \cup N;$
$\hspace{2em}$ if x $\in$ P and w.x $<$ 0 then
$\hspace{3em}$ w = w+x
$\hspace{2em}$ end
$\hspace{3em}$ if x $\in$ N and w.x $\ge$ 0 then
$\hspace{3em}$ w = w-x
$\hspace{2em}$ end
end
```

We aim to find the w vector that can perfectly classify positive and negative inputs in a dataset. w is initialised with a random vector. We are then iterative overall positive and negative samples (PUN). Now, if an input x belongs to P, w.x should be greater than or equal to 0. And if x belongs to N, w.x should be lesser than or equal to 0. Only when these conditions are not met do we update the weights?

## Basic Components of Perceptron

Frank Rosenblatt invented the perceptron learning algorithm.

![Basic Components of Perceptron](https://www.scaler.com/topics/images/basic-components-of-perceptron.webp)

**It is a binary classifier and consists of three main components. These are:**

![Basic Components of Perceptron Function](https://www.scaler.com/topics/images/basic-components-of-perceptron-function.webp)

1. **Input Nodes or Input Layer:** Primary component of Perceptron learning algorithm, which accepts the initial input data into the model. Each input node contains an actual value.
2. **Weight and Bias:** The weight parameter represents the strength of the connection between units. Bias can be considered as the line of intercept in a linear equation.
3. **Activation Function:** Final and essential components help determine whether the neuron will fire. The activation function can be primarily considered a step function. There are various types of activation functions used in a perceptron learning algorithm. Some of them are the sign function, step function, sigmoid function, etc.

## Types of Perceptron Models

**Based on the number of layers, perceptrons are broadly classified into two major categories:**

1. **Single Layer Perceptron Model:**  
    It is the simplest Artificial Neural Network (ANN) model. A single-layer perceptron model consists of a feed-forward network and includes a threshold transfer function for thresholding on the Output. The main objective of the single-layer perceptron model is to classify linearly separable data with binary labels.
    
2. **Multi-Layer Perceptron Model:**  
    The multi-layer perceptron learning algorithm has the same structure as a single-layer perceptron but consists of an additional one or more hidden layers, unlike a single-layer perceptron, which consists of a single hidden layer. The distinction between these two types of perceptron models is shown in the Figure below.
    

![Perceptron Models](https://www.scaler.com/topics/images/perceptron-models.webp)

![[Capture-2023-06-14-214115.png]]
![[Capture-2023-06-14-214217.png]]

So whatever the w vector may be, as long as it makes an angle less than 90 degrees with the positive example data vectors (x ∈∈ P) and an angle more than 90 degrees with the negative example data vectors (x ∈∈ N), we are cool. So ideally, it should look something like this:

![Geometry of the solution space](https://www.scaler.com/topics/images/geometry-of-the-solution-space.webp)
![[Capture-2023-06-14-214316.png]]
So the angle between w and x should be less than 90 when x belongs to the P class, and the angle between them should be more than 90 when x belongs to the N class. Pause and convince yourself that the above statements are true and you believe them.

## Perceptron Learning Algorithm: Implementation of AND Gate

**The steps for this implementation are as follows:**

1. **Import all the required libraries:**

```python
#import required library
import tensorflow as tf
```

2. **Define Vector Variables for Input and Output:**

```python
#input1, input2 and bias
train_in = [
    [1., 1.,1],
    [1., 0,1],
    [0, 1.,1],
    [0, 0,1]]
 
#output
train_out = [
[1.],
[0],
[0],
[0]]
```

3. **Define the Weight Variable:**

```python
#weight variable initialized with random values using random_normal()
w = tf.Variable(tf.random_normal([3, 1], seed=12))
```

4. **Define placeholders for Input and Output:**

```python
#Placeholder for input and Output
x = tf.placeholder(tf.float32,[None,3])
y = tf.placeholder(tf.float32,[None,1])
```

5. **Calculate Output and Activation Function:**  
    ![Perceptron Learning Algorithm: Implementation of AND Gate](https://www.scaler.com/topics/images/perceptron-learning-algorithm-implementation-of-and-gate.webp)

```python
#calculate output 
output = tf.nn.relu(tf.matmul(x, w))
```

6. **Calculate the Cost or Error:**

```python
#Mean Squared Loss or Error
loss = tf.reduce_sum(tf.square(output - y))
```

7. **Minimize Error:**

```python
#Minimize loss using GradientDescentOptimizer with a learning rate of 0.01
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

8. **Initialize all the variables:**

```python
#Initialize all the global variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
```

9. **Training Perceptron learning algorithm in Iterations:**

```python
training_epochs = 1000

#Compute cost w.r.t to input vector for 1000 epochs
 
for epoch in range(training_epochs):
    sess.run(train, {x:train_in,y:train_out})
    cost = sess.run(loss,feed_dict={x:train_in,y:train_out})
    if i > 990:
        print('Epoch--',epoch,'--loss--',cost)

```

**Output:**

Following is the final Output obtained after my perceptron model has been trained.

```plaintext
Epoch-- 991 --loss-- 0.0003835174
Epoch-- 992 --loss-- 0.00038088957
Epoch-- 993 --loss-- 0.0003782803
Epoch-- 994 --loss-- 0.0003756886
Epoch-- 995 --loss-- 0.0003731146
Epoch-- 996 --loss-- 0.00037055893
Epoch-- 997 --loss-- 0.00036801986
Epoch-- 998 --loss-- 0.00036549888
Epoch-- 999 --loss-- 0.00036299432
```

In the above code, you can observe how we are feeding train_in (input set of AND Gate) and train_out (output set of AND gate) to placeholders x and y respectively using feed_dict for calculating the cost or Error.

## Perceptron With Scikit-Learn

The perceptron learning algorithm is readily available in the scikit-learn Python machine learning library via the Perceptron class. Some of the important configurable parameters for this class are – the learning rate (eta0) which has a default value of 1.0, and training epochs (max_iter) which have a default value of 1000. Early stopping, which has False as its default value and type of regularization (penalty), which has a default value of None and can have 'l2', 'l1', and 'elastic net' as its values.

![[Capture-2023-06-14-214352.png]]

Now, we will demonstrate the implementation of the Perceptron learning algorithm with a working example. We will generate a synthetic classification dataset for this purpose. We use the make_classification() function to create a dataset with 1,000 examples, each with 20 input variables.

```python
# Evaluate a perceptron model on a synthetic dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
# Define the dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# Define the model
model = Perceptron()
# Define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# Evaluate the model
scores = cross_val_score(model, X, y, scoring=’accuracy’, cv=cv, n_jobs=-1)
# Summarize the results
print(‘Mean Accuracy: %.3f (%.3f)’ % (mean(scores), std(scores)))
```

**Output:**

```plaintext
Mean Accuracy: 0.847 (0.052)
```

The above code example evaluates the Perceptron algorithm on the synthetic dataset and prints the average accuracy across three repeats of 10-fold cross-validation.

Next, we show how to call a trained Perceptron algorithm on a new dataset using the predict() function and perform the final prediction, thus demonstrating an end-to-end training cum inference pipeline for a Perceptron classifier.

```python
# Predict with a perceptron model on the dataset
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
# Define the dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# Define the model
model = Perceptron()
# Fit model
model.fit(X, y)
# Define new data
row = [0.12777556,-3.64400522,-2.23268854,-1.82114386,1.75466361,0.1243966,1.03397657,2.35822076,1.01001752,0.56768485]
# Make a prediction
yhat = model.predict([row])
# Summarize the prediction
print(‘Predicted Class: %d’ % yhat)
```

**Output:**

```plaintext
Predicted Class: 1
```

## Tune Perceptron Hyperparameters

Next, let's look at how to tune hyperparameters for a perceptron learning algorithm. Hyperparameter tuning is part and parcel of any machine learning algorithm and is tuned specifically to a particular dataset. A large learning rate helps the model to learn faster but might result in lower accuracy, whereas a lower learning rate can result in better accuracy but might take more time to train.

Testing learning rates on a log scale between a small value such as 1e-4 (or even smaller) and 1.0 is a common technique used for this purpose. We demonstrate this with the following example code.

```python
# grid search learning rate for the Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = Perceptron()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['eta0'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))
```

Running the above example will evaluate all the possible combinations of the configurations using repeated cross-validation.  
Due to the stochastic nature of the algorithm, the results might vary. So it is recommended to run the code a few times.  
Following are some sample results with different values of the learning rate.

```plaintext
Mean Accuracy: 0.857
Config: {'eta0': 0.0001}
>0.857 with: {'eta0': 0.0001}
>0.857 with: {'eta0': 0.001}
>0.853 with: {'eta0': 0.01}
>0.847 with: {'eta0': 0.1}
>0.847 with: {'eta0': 1.0}
```

Another important hyperparameter is the number of epochs for model training. This will also vary depending on the training data. For this also, we explore a range of values on a log scale between 1 and 1e+4.

```python
# define grid
grid = dict()
grid['max_iter'] = [1, 10, 100, 1000, 10000]
```

In the previous example, the results show that the learning rate of 0.0001 performs the best. So we use this learning rate to illustrate grid searching of the number of training epochs in the example code below.

```python
# grid search total epochs for the Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import Perceptron
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
# define model
model = Perceptron(eta0=0.0001)
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['max_iter'] = [1, 10, 100, 1000, 10000]
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))
```

Running the above example evaluates various combinations of configurations using repeated cross-validations.

A sample result for the above code is given below. We see here that epochs 10 to 10,000 result in approximately the same accuracy. An interesting exploration in this direction would be to explore tuning the learning rate and the number of training epochs at the same time.

```python
Mean Accuracy: 0.857
Config: {'max_iter': 10}
>0.850 with: {'max_iter': 1}
>0.857 with: {'max_iter': 10}
>0.857 with: {'max_iter': 100}
>0.857 with: {'max_iter': 1000}
>0.857 with: {'max_iter': 10000}
```

## SONAR Data Classification Using Single Layer Perceptrons

SONAR data is available for free [here](https://datahub.io/machine-learning/sonar%23resource-sonar). We will first understand this data and perform classification on this data using single-layer perceptrons.

The dataset consists of 208 patterns obtained by bouncing sonar signals off a metal cylinder (naval mine) and rock at various angles and under various conditions. A naval mine is a self-contained explosive device placed in water to damage or destroy surface ships or submarines. So, our objective is to build a model that can predict whether the object is a naval mine or rock based on our data set.

**Let's look at a glimpse of this dataset:**

![SONAR Data Classification Using Single Layer Perceptrons](https://www.scaler.com/topics/images/sonar-data-classification-using-single-layer-perceptrons.webp)

The overall procedure is very similar to learning the AND gate function, with few differences. The overall procedure flow for the classification of the SONAR data set using the Single Layer Perceptron learning algorithm is shown below.

![SONAR data set using Single Layer Perceptron learning algorithm](https://www.scaler.com/topics/images/sonar-data-set-using-single-layer-perceptron-learning-algorithm.webp)

**The following are the steps:**

1. **Import all the required Libraries:**  
	First, we start with importing all the required libraries as listed below:  
	    matplotlib library: It provides functions for plotting the graph.  
	    tensorflow library: It provides functions for implementing Deep Learning Model.  
	    pandas, numpy and sklearn library: It provides functions for pre-processing the data.
1. **Read and Pre-process the data set:**

```python
#Read the sonar dataset
df = pd.read_csv("sonar.csv")
print(len(df.columns))
X = df[df.columns[0:60]].values
y = df[df.columns[60]]
 
#encode the dependent variable as it has two categorical values
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
Y = one_hot_encode(y)
```

3. **Function for One Hot Encoder:**

```python
#function for applying one_hot_encoder
def one_hot_encode(labels):
  n_labels = len(labels)
  n_unique_labels = len(np.unique(labels))
  one_hot_encode = np.zeros((n_labels,n_unique_labels))
  one_hot_encode[np.arange(n_labels), labels] = 1
  return one_hot_encode
```

4. **Dividing data set into Training and Test Subset:**

```python
#Divide the data in training and test the subset
X,Y = shuffle(X,Y,random_state=1)
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.20, random_state=42)
```

5. **Define Variables and Placeholders:**  
    Here, I will define variables for the following entities:  
    Learning Rate: The amount by which the weight will be adjusted.  
    Training Epochs: No. of iterations  
    Cost History: An array that stores the cost values in successive epochs.  
    Weight: Tensor variable for storing weight values  
    Bias: Tensor variable for storing bias values

```python
#define all the variables to work with the tensors
learning_rate = 0.1
training_epochs = 1000
 
cost_history = np.empty(shape=[1],dtype=float)
 
n_dim = X.shape[1]
n_class = 2
 
x = tf.placeholder(tf.float32,[None,n_dim])
W = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros([n_class])
```

6. **Calculate the Cost or Error:**

```python
y_ = tf.placeholder(tf.float32,[None,n_class])
y = tf.nn.softmax(tf.matmul(x, W)+ b)
cost_function = tf.reduce_mean(-tf.reduce_sum((y_ * tf.log(y)),reduction_indices=[1]))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
```

7. **Training the Perceptron learning algorithm in Successive Epochs:**

```python
#Minimizing the cost for each epoch
for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={x:train_x,y_:train_y})
    cost = sess.run(cost_function,feed_dict={x: train_x,y_: train_y})
    cost_history = np.append(cost_history,cost)
    print('epoch : ', epoch,  ' - ', 'cost: ', cost)
```

8. **Validation of the Model based on the Test Subset:**

```python
#Run the trained model on test subset
pred_y = sess.run(y, feed_dict={x: test_x})
 
#calculate the correct predictions
correct_prediction = tf.equal(tf.argmax(pred_y,1), tf.argmax(test_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(&amp;amp;quot;Accuracy: &amp;amp;quot;,sess.run(accuracy))
```

**Output:**

Following is the Output that you will get once the training has been completed:

```plaintext
epoch :  986    -     cost:   0.423809 
epoch :  987    -     cost:   0.423757 
epoch :  988    -     cost:   0.423704 
epoch :  989    -     cost:   0.423652 
epoch :  990    -     cost:   0.4236 
epoch :  991    -     cost:   0.423548 
epoch :  992    -     cost:   0.423496 
epoch :  993    -     cost:   0.423444 
epoch :  994    -     cost:   0.423392 
epoch :  995    -     cost:   0.42334 
epoch :  996    -     cost:   0.423288 
epoch :  997    -     cost:   0.423236 
epoch :  998    -     cost:   0.423185 
epoch :  999    -     cost:   0.423133

Accuracy:   0.833333

```

As you can see, we got an accuracy of 83.34%, which is decent enough. Now, let us observe how the cost or Error has been reduced in successive epochs by plotting a graph of **Cost vs. No. Of Epochs**:

![Graph of Cost vs. No. Of Epochs](https://www.scaler.com/topics/images/graph-of-cost-vs-no-of-epochs.webp)

**Complete Code for SONAR Data Classification Using Single Layer Perceptron**

```python
#import the required libraries
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import  shuffle
from sklearn.model_selection import train_test_split
 
#define the one hot encode function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode
 
#Read the sonar dataset
df = pd.read_csv(&amp;amp;quot;sonar.csv&amp;amp;quot;)
print(len(df.columns))
X = df[df.columns[0:60]].values
y=df[df.columns[60]]
#encode the dependent variable containing categorical values
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
Y = one_hot_encode(y)
 
#Transform the data in training and testing
X,Y = shuffle(X,Y,random_state=1)
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.20, random_state=42)
 
 
#define and initialize the variables to work with the tensors
learning_rate = 0.1
training_epochs = 1000
 
#Array to store cost obtained in each epoch
cost_history = np.empty(shape=[1],dtype=float)
 
n_dim = X.shape[1]
n_class = 2
 
x = tf.placeholder(tf.float32,[None,n_dim])
W = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros([n_class]))
 
#initialize all variables.
init = tf.global_variables_initializer()
 
#define the cost function
y_ = tf.placeholder(tf.float32,[None,n_class])
y = tf.nn.softmax(tf.matmul(x, W)+ b)
cost_function = tf.reduce_mean(-tf.reduce_sum((y_ * tf.log(y)),reduction_indices=[1]))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
 
#initialize the session
sess = tf.Session()
sess.run(init)
mse_history = []
 
#calculate the cost for each epoch
for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={x:train_x,y_:train_y})
    cost = sess.run(cost_function,feed_dict={x: train_x,y_: train_y})
    cost_history = np.append(cost_history,cost)
    print('epoch : ', epoch,  ' - ', 'cost: ', cost)
 
pred_y = sess.run(y, feed_dict={x: test_x})
 
#Calculate Accuracy
correct_prediction = tf.equal(tf.argmax(pred_y,1), tf.argmax(test_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(&amp;amp;quot;Accuracy: &amp;amp;quot;,sess.run(accuracy))
 
plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()
```

## Limitations of the Perceptron Model

**A perceptron model has the following limitations:**

1. The Output of a perceptron can only be a binary number (0 or 1) due to the hard limit transfer function. Thus it is difficult to use in problems other than binary classification, like regression or multiclass classification.
2. Perceptron can only be used to classify linearly separable sets of input vectors. If input vectors are non-linear, it is not easy to classify them properly.

## Future of Perceptron

Perceptrons have a very bright and significant future as it is a very intuitive and interpretable model and helps to interpret the data well. Artificial neurons form the backbone of perceptrons, and they are the future of state-of-the-art and highly popular neural network models. Thus, with the growing popularity of artificial intelligence and neural networks nowadays, perceptron learning algorithms play a very significant role.

## Conclusion

- Perceptron learning algorithm is a linear supervised machine learning algorithm. It forms the basis of a neural network, the most famous machine learning algorithm nowadays.
- In this article, we have discussed what a perceptron learning algorithm, its essential components, and the types of a perceptron learning algorithm is.
- Geometry of solution space is also presented in this article. In addition, the Python code implementation of AND gate and hyperparameter tuning is also discussed here.
- The article concludes after presenting SONAR data classification using a single-layer perceptron, the perceptron model's limitations, and the Perceptron's future.