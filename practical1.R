##**********************************************************************
##  DLAI4 (MATH4267) practical worksheets                           ####
##  Code to accompany written material
##  James Liley                                                         
##  October 2023                                                              
##**********************************************************************


# Functions
source("./practical1_functions.R")


##**********************************************************************
## 1. Activation functions                                          ####
##**********************************************************************

# Draw plot
res=100 # Resolution; this many x values
x=seq(-3,3,length=res) # Sequence from -3 to 3 of length 'res'
plot(0,type="n", # This plots nothing: it just sets up a blank canvas...
     xlim=c(-3,3),ylim=c(-3,3), # X and Y limits
     xlab="Input",ylab="Output" # X and Y labels
)
# Draw all the functions. Modulate width (lwd) to make sure they are visible.
lines(x,heaviside(x),col=1,lwd=5)
lines(x,logistic(x),col=2)
lines(x,ReLU(x),col=3,lwd=3)
lines(x,SLU(x),col=4)
lines(x,ramp(x,alpha=2),col=5)
lines(x,pReLU(x,alpha=1/2),col=6)
# Draw a legend
legend("topleft", # placement
       c("Heaviside","Logistic","ReLU","SLU","Ramp","pReLU"), # Labels
       lwd=c(5,1,3,1,1,1), # Line widths
       col=1:6, # Colours
       bty="n" # No bounding box
)




##**********************************************************************
## 2. Basic neurons                                                 ####
##**********************************************************************

## No procedures here

##**********************************************************************
## 3. What neurons and neural networks can do                       ####
##**********************************************************************

## Visualise a simple neuron with weights c(1,2) and bias -1 with logistic activation:
w0=c(1,2) # Specify weights
b0=-1 # Specify bias
visualise2d(neuron,w=w0,b=b0,phi=logistic)


## Visualise a very complicated network with complex weights
## See function random_network

sc=5; res=200
xpar=random_network(2,5,50,hphi = logistic,wsd=5,bsd=5)
visualise2d(network,xlim=sc*c(-1,1),xres=res, yres=res, ylim=sc*c(-1,1),parameters=xpar)



##**********************************************************************
## 4. The MNIST benchmark                                           ####
##**********************************************************************

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

