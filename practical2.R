##**********************************************************************
##  DLAI4 (MATH4267) practical worksheets                           ####
##  Code to accompany written material
##  James Liley                                                         
##  October 2023                                                              
##**********************************************************************

# Functions
source("./practical2_functions.R")


##**********************************************************************
## 1. Backpropagation                                               ####
##**********************************************************************

# Given a neural network as specified in the last workshop,
#  Write a function which computes the derivatives of the output of a single neuron in terms of its weights and bias.
# See function dparam1 and dparam1_empirical.

set.seed(23293)

# Decide some weights and biases
w=rnorm(5); b=rnorm(1);

# Activation function and derivative (check what functions logistic and dlogistic do)
phi=logistic
dphi=dlogistic

# Cost function and derivative
cost=function(y) (y-1)^2
dCdY=function(y) 2*(y-1)

# Input of neuron
input=rnorm(5)

# Output of neuron
y=neuron(input,w,b,phi)

# Compute derivatives explicitly using our function dparam1
delta=dparam1(w,b,phi,input,dCdY,dphi)

# Compute derivatives empirically using dparam1_empirical
delta_empirical=dparam1_empirical(w,b,phi,input,cost)



# Extend the function from the first part to compute derivatives of the output of a 
# neural network in terms of its weights and biases.
# Start with with a network with two inputs and two layers, with five neurons 
#  in the first layer and one neuron in the output layer, and logistic activations. 
#  Train this network to try and learn the function ...
# You will need to generate some training and test data. Each time the weights are updated, record the performance of the network on your test data. Plot this performance. 

# Generate some data
n=10000
x1=runif(n,-2,2)
x2=runif(n,-2,2)
y=as.numeric( (x1^2 + x2^2 < 1))
dat=cbind(x1,x2,y)

train_dat=dat[(1:floor(n/2)),]
test_dat=dat[(1+floor(n/2)):n,]





# New neural network
nhl=1 # Number of hidden layers
wd=5 # Width
params=random_network(2,nhl,wd,hphi = logistic,wsd=0.1,bsd=0.1)
params[[nhl+1]]$phi=logistic # change final layer to logistic activation
dphi=list(); for (i in 1:(1+nhl)) dphi[[i]]=dlogistic;



# Train
n_epoch=5 # Number of 'epochs': times to loop through the dataset.
gamma=1/3
track=list() # Keep track of weight values
for (e in 1:n_epoch) {
  track[[e]]=params
  print(e)
  for (k in 1:dim(train_dat)[1]) {
    input=matrix(train_dat[k,1:2])
    # Objective: squared difference
    dCdY=function(y) 2*(y-train_dat[k,3])
    dpar=dparam(params,input,dCdY,dphi)
    params=update_params(params,dpar,gamma)
  }
}


# Evaluate on test data
ytest_pred=network(t(test_dat[,1:2]),parameters=params)

# Plot output
par(mfrow=c(1,2))
xcol=c('black','red')[1+(ytest_pred>0.5)] # red for predicted 1, black for 0
plot(test_dat[,1],test_dat[,2],col=xcol,xlab=expression(x[1]),ylab=expression(x[2]))
# so-so

# Plot change in parameters with time
t0=unlist(lapply(track,function(x) x[[2]]$w[1,1]))
plot(t0,type='b',xlab='Epoch',ylab='Weight value')




# Experiment with the previous item using networks of different dimensions. How does performance change with wider networks? Deeper networks? How does convergence of performance on test data change?

# Rerun the previous block of code from 'New neural network' onwards





##**********************************************************************
## 2. Vanishing and exploding gradients                             ####
##**********************************************************************

# Initialise a network with two inputs, ten layers of five neurons each, 
#  and a one-layer output neuron, with logistic activations.


# New neural network
nhl=10 # Number of hidden layers
wd=5 # Width
params=random_network(2,nhl,wd,hphi = logistic,wsd=0.1,bsd=0.1)
params[[nhl+1]]$phi=logistic # change final layer to logistic activation
dphi=list(); for (i in 1:(1+nhl)) dphi[[i]]=dlogistic;


# Try to fit the function from the previous section (finding a circle). 
#  Track the changes to the weights of a neuron in the first layer and 
#  in the tenth layer. What happens?

# Train
n_epoch=10 # Number of 'epochs': times to loop through the dataset.
gamma=1/3
track=list() # Keep track of weight values
for (e in 1:n_epoch) {
  track[[e]]=params
  print(e)
  for (k in 1:dim(train_dat)[1]) {
    input=matrix(train_dat[k,1:2])
    # Objective: squared difference
    dCdY=function(y) 2*(y-train_dat[k,3])
    dpar=dparam(params,input,dCdY,dphi)
    params=update_params(params,dpar,gamma)
  }
}

# Plot change in parameters with time
par(mfrow=c(1,2))

# Neuron in first layer (barely changes)
t0=unlist(lapply(track,function(x) x[[1]]$w[1,1]))
plot(t0,type='b',xlab='Epoch',ylab='Weight value')

# Neuron in final hidden layer
t1=unlist(lapply(track,function(x) x[[11]]$w[1,1]))
plot(t1,type='b',xlab='Epoch',ylab='Weight value')



##**********************************************************************
## 3. More complex fitting of neural networks                       ####
##**********************************************************************

# Revisit the example from keras on the MNIST dataset from the first workshop. 
#  We used a `dropout' layer. Experiment with using no dropout layers, or using 
#  more dropout layers, and training with more or fewer epochs. What do you find?

# I have reproduced the code here. Please mess around with it as you see fit.


# Load libraries.
library(keras3)
library(dplyr) # Needed for pipe operator

# Load the MNIST dataset. 
mnist = dataset_mnist()

# Load the various parts of the MNIST dataset
x_train = mnist$train$x
y_train = mnist$train$y
x_test = mnist$test$x
y_test = mnist$test$y

# Reshape the dataset
x_train = array_reshape(x_train, c(nrow(x_train), 784))
x_test = array_reshape(x_test, c(nrow(x_test), 784))
x_train = x_train / 255
x_test = x_test / 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, 10)

# Initialise model
input = layer_input(shape=c(784))
output = input %>% 
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')
model=keras_model(input,output)

# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


# Train the model
history = model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

# Evaluate the model on new data
model %>% evaluate(x_test, y_test)

# Generate predictions on new data  
y_test_pred_prob = model %>% predict(x_test) 
y_test_pred=apply(y_test_pred_prob,1,function(x) which.max(x)-1)

#### Investigate discrepancies
# Convert y_test into the same format as y_test_pred
y_test_comp=y_test  %*% (0:9) # now y_test_comp[j]=i, where i is the number in the jth picture.
y_test_pred_vec=as.integer(y_test_pred)

# Look at a few randomly chosen MNIST entries
ns=3 # number of samples
mnist_sample=sample(dim(x_test)[1],ns)
par(mfrow=c(1,ns))
for (i in 1:ns) {
  mat=matrix(x_test[mnist_sample[i],],28,28)
  mat=mat[1:28,28:1]
  image(mat,main=y_test_comp[mnist_sample[i]])
}

# Look at a few samples where our predictor got it wrong
wrong=which(y_test_comp!=y_test_pred_vec)
wrong_sample=sample(wrong,ns)
par(mfrow=c(1,ns))
for (i in 1:ns) {
  mat=matrix(x_test[wrong_sample[i],],28,28)
  mat=mat[1:28,28:1]
  image(mat,main=paste0(
    y_test_comp[wrong_sample[i]], # Correct answer
    "/",
    y_test_pred_vec[wrong_sample[i]] # Predicted answer
  ))
}

