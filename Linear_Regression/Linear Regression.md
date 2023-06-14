Linear Regression is a linear method to predict a real-valued output for a given vector of inputs. It is defined as

$$
f(x) = w_0 + \sum_{i=1}^{D} w_i x_i.
$$

Alternatively, if do not have a bias term $w_0$, it can be represented as

$$
\begin{aligned}
f(x) =& \sum_{i=1}^{D} w_i x_i \\
=& w ^\top x.
\end{aligned}
$$

Here $w$ is the weight vector and $x$ is a vector of inputs.

Given the above model formulation, our objective is to utilize the training data, to learn a predictor. Assume that we are provided with a datastet

$$
(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots , (x^{(N)}, y^{(N)})
$$

where, the input and the output are defined as, $x^{(n)} \in \mathbb{R}^D$ and  $y^{(n)} \in \mathbb{R}$ respectively. 

Our strategy for linear regression will be as follows:

1. First, we define a quantitative measure of how well a given set of parameters $w$ fits to the a dataset. This is called a loss function.
2. Second, we find the parameters $w$ that minimise that measure on the training data.
3. Third we hope that those parameters perform well on test data.

To improve the generalisation performance we will modify this procedure to include regularisation and model selection methods.

# 2. Quality of Fit

Here are two common ways to measure how well the loss fits to training data.

Mean Absolute Error

$$
\mathrm{MAE} = \frac{1}{N} \sum_{n=1}^{N} \vert y^{(n)} - f(x^{(n)}) \vert
$$

Mean Square Error

$$
\mathrm{MSE} = \frac{1}{N} \sum_{n=1}^{N} \left( y^{(n)} - f(x^{(n)}) \right)^{2}
$$

Sometimes we sum the loss over the training data, sometimes (like here) we take the mean. It doesn't matter for now, but watch out for this distinction when we look at regularisers.

Let's try to understand these loss functions by looking at some example datasets.

## Example 1

Suppose we have a 1-D input and 1-D output, and our regression function is defined as

$$
f(x) =  w x.
$$

Also, let us have three data points defined.

$$
\begin{array}{lrr}
n & x^{(n)} & y^{(n)}  \\
\hline
1 & 1 &  0.5 \\
2 & 2 &  2.5 \\ 
3 &  3 &  3 \\   
\end{array}
$$

Observe the following figure where we have plotted the data points.

![[dat1.svg]]

### Plotting the MSE

Let us first try to understand what happens with the MSE. For each value of $n$, $(y^{(n)}-w x^{(n)})^2$ will be an upwards sloping quadratic. For $n=1$ the minimum will be at $w=0.5$ and the slope will be relatively shallow. For $n=2$ the minimum will be at $w=1.25$ and the slope will be a bit steeper. For $n=3$ the minimum will be at $w=1$ and the slope will even steeper.

The MSE is the mean of these curves. A visualisation is shown below.
![[mae_mean1.svg]]

Here, $n = 1, n = 2$ and $n=3$ are the three data points that we mentioned above.

### Plotting the MAE

Similarly, if we plot the absolute error curves for all three and plot the MAE curve, we get the plot shown below. As can be observed, the curves are different from the plot we obtained before for MSE. Specifically, instead of a parabola-like curve, we obtain more V-shape-like curves. This difference is because, for MAE, we take the absolute values of the difference between our true output and predicted output. 
![[mae_mean1.svg]]
## Example 2

Now, we move on to a different set of data points. The new dataset is almost similar to the previous dataset we discussed. The only difference between the two datasets is that we changed the output label for the first point. The new dataset is given below.

$$
\begin{array}{lrr}
n & x^{(n)} & y^{(n)}  \\
\hline
1 & 1 &  0 \\
2 & 2 &  2.5 \\ 
3 &  3 &  3 \\   
\end{array}
$$

Visualizing the new dataset, we see that the only difference from the previous dataset is that the $n = 1$ datapoint is shifted downwards.
![[data2.svg]]

### Plotting the MSE

When we plot the MSE error for the new dataset, we observe that the curve for the first data point (blue curve) is shifted to the left. As a result, the optimal value of the weight decreases slightly. We can observe this by visualizing the final MSE curve, which also shifts slightly to the left. 

The four curves are shown below.
![[mse_mean2.svg]]

### Plotting the MAE

Similarly, when we plot the absolute error for the individual data points of the new dataset, we see that the curve for the first data point (the blue curve) is shifted to the left. When me plot the mean of all the curves i.e. the MAE cure, we see that while the curve is different from the MAE curve that we saw previously for the first dataset, the optimal value is the weight is still same as before. This is because, even though the curve for the first data point changed, the slope of all the curves at the optimal weight value  and the right hand side region to the optima is still same as before. As a result  we did not see any change in the MAE curve at the optimal weight value and the right hand side region to the optimal weight. 

Here are the four curves shown below
![[mae_mean2.svg]]

## Discussion

What's better? There's no clear answer. It's a property of the MSE that single "outlier" datapoints can significantly change the solution. That might be a good thing if you really don't want to have large errors for specific points, though!

## Example 3

Consider the following set of data points. The input is on the x-axis, while the output is on the y-axis. The true curve is shown in green. Then, for each point, with a probability of $0.5$ we will add some amount of Gaussian noise. To start with, the noise is zero, hence all points sit exactly on the true line.
![[data_0.5_0.svg]]

If we apply noise with a standard deviation of $0.1$, we observe that some of the data points have slightly deviated from the true line. A visualization for the same is shown below.
![[data_0.5_0.1.svg]]

If the standard deviation of the input data is further increased to $1.0$, we see from the visualization below that our dataset is relatively noisier, as $50\%$ of the data points are not on the true line.
![[data_0.5_1.0.svg]]

### MSE and MAE

Now, let us plot two curves - one that minimizes **the MSE and another that minimizes **the MAE.

For the case where standard deviation was $0$, we get the plot shown below.
![[fits_0.5_0.svg]]

We repeat the same experiment as above but slightly change the input data by increasing the standard deviation of the noise a little by a factor of $0.1$. A visualization of our results is shown below.
![[fits_0.5_0.1.svg]]

Here, we observe that the MSE curve is shifting slowly below the true curve because, by chance, the noise tends to be negative for points to the right. The MAE curve remains glued to the true data for now.

As a next step, we further increase the standard deviation of the noise to $1.0$. 
![[fits_0.5_1.0.svg]]

From the plot of our results above, we can see that the MSE curve tries quite hard to accommodate the points that have had noise added to them. Due to this relatively higher 'sensitivity', the prediction curve for MSE deviates from the "true" function, while the MAE aligns perfectly to the original regression line.

## Example 4

In the previous example, we added noise with a probability of $0.5$. Instead, let us see what happens if we add noise to *all* the points i.e. with probability $1$.

For standard deviation $0.1$, we get the following plot.
![[data_1_0.1.svg]]

For standard deviation equal to $1.0$, we see that our dataset now is extremely noisy. Almost all the data points are not aligned with the true line. Although, it is worth taking a note that even in this particular situation, the data points still follow a specific trend, which we would want our predictive model to estimate as accurately as possible.
![[data_1_1.0.svg]]

### MSE and MAE

Let us now plot the curves that minimize the MSE and the MAE.

When the standard deviation is $0.1$, the obtain the plot shown below. Particularly, we observe that the two curves diverging only slightly.
![[fits_1_0.1.svg]]

However, when the standard deviation is $1.0$, we see from our results below that both the curves are quite far from the true curve we had shown before. 
![[fits_1_1.0.svg]]

We specifically observe that the MSE retains *some* pattern that the true output tends to increase when $x$ is larger. Theory suggests that this behavior can be attributed to the higher sensitivity of the MSE relative to the MAE.

A question may arise that since the data is so arbitrarily distributed, a regression line that shows no pattern *might* be a better candidate than the one that still somehow resembles the true function line. The answer to this lies in the fact that even though the data is *noisy*, there is still a very well-defined and strong relationship between the input and the output samples.

Thus, we can conclude that, in general, MSE performs better than MAE for generalization on test data. Therefore, in practice, when no other domain information is available, it is always advisable to use the Mean square error as your loss function for the given regression task. 

# 3. A closed-form solution for the MSE

Let us assume for now that the input $x$ is 1-D and therefore $x \in \mathbb{R}$, the output $y$ is also 1-D and therefore $y \in \mathbb{R}$. The parameter $w$ is also 1-D and hence $w \in \mathbb{R}$.

Now, we intend to find the minimiser of the residual sum of squares (RSS), defined as

$$
\mathrm{RSS} (w) = \sum_{n=1}^{N} \left( y^{(n)} - w x^{(n)} \right)^{2}.
$$

The derivative of this function w.r.t $w$ is

$$
\frac{\mathrm{d}\mathrm{RSS}(w)}{\mathrm{d}w} = \sum_{n=1}^{N} 2 \left( y^{(n)} - w x^{(n)} \right) (-x^{(n)}).
$$

At the best value $w^*$, the derivative will be zero. Thus, we want to solve

$$
\frac{\mathrm{d}\mathrm{RSS}(w^*)}{\mathrm{d}w} = \sum_{n=1}^{N} 2 \left( y^{(n)} - w^* x^{(n)} \right) (-x^{(n)}) = 0.
$$

Solving the above equation, we get

$$
\sum_{n=1}^{N} y^{(n)} x^{(n)} = \sum_{n=1}^{N} w^* (x^{(n)})^2.
$$

Finally, after further simplifications, we obtain the following equation for the optimal weights,

$$
w^* = \frac{\sum_{n=1}^{N} y^{(n)} x^{(n)}}{\sum_{n=1}^{N}  (x^{(n)})^2}.
$$

Now, let us investigate the relationship between $w^*$ and other variables.

- If the output values $y^{(n)}$ doubles, the value of $w^*$ also doubles.
- If the input values $x^{(n)}$ doubles, then the value of $w^*$ halves.
- If *both* the input and the output value double, then the value of $w^*$ remains unchanged.

# Residual Sum of Squares in D dimensions

As compared to the previous section, let us now define our output values such that $y^{(n)} \in \mathbb{R}$, our input values such that $x^{(n)} \in \mathbb{R}^D$ and our parameters $w \in \mathbb{R}^D$.

Therefore, the RSS function is

$$
\mathrm{RSS} (w) = \sum_{n=1}^{N} \left( y^{(n)} - w^\top x^{(n)} \right)^{2}.
$$

Now, let us design matrices that contain all the different values. That is, let us define a vector $y=(y^{(1)}, y^{(2)}, \cdots, y^{(N)})$ of length $N$ that contains all of the output values.

Similarly, let us construct a matrix $X$ of size $N \times D$, which consists of all the input vectors  $x^{(1)}, x^{(2)} \dots x^{(N)}$ stacked on top of each other. Put another way, the $n$-th row of $X$ is $x^{(n)}$.

Then, the product $Xw$ is be a length $N$ vector consisting of the dot product of each row of $x$ with $w$. That is,

$$
Xw = (w^\top x^{(1)}, w^\top x^{(2)} \dots w^\top x^{(N)}).
$$

Using $y$ and $X$, we can rewrite the RSS function as

$$
\mathrm{RSS} (w) = \Vert y - Xw \Vert^{2}.
$$

This can be rewritten as

$$
\mathrm{RSS} (w) = (y - Xw)^T (y - Xw).
$$

This comes from the fact that for any vector , we have $\Vert a \Vert ^2 = a^\top a$.

The gradient of this function is

$$
\nabla \mathrm{RSS}(w) = \frac{\mathrm{d}\mathrm{RSS}(w)}{\mathrm{d}w} = -2 X^\top (y - Xw).
$$

At the optimum $w^*$, we have $\nabla \mathrm{RSS}(w^*) = 0$. Opening up the RHS, we need

$$
X^\top y = X^\top Xw.
$$

Solving this, we get

$$
\boxed{w^* = (X^\top X)^{-1} X^\top y.}
$$

Note that the above equation is **one of the most important equations** for this course.

The algorithm for this solution, consists of the following steps:

- Compute $X^\top X$
- Compute $X^\top y$
- Solve the equation $(X^\top X)w = (X^\top y)$ for $w$.

We don't suggest taking an inverse. Even though it is mathematically correct, numerically it can be unstable, due to compounding errors in floating point calculations. The better correct approach is to simply use a `solve` operation that directly takes a matrix $A$ and a vector $b$ and returns a vector $x$ such that $A x = b$.