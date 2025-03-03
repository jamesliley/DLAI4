##**********************************************************************
##  DLAI4 (MATH4267) practical worksheets                           ####
##  Code to accompany written material
##  James Liley                                                         
##  October 2023                                                              
##**********************************************************************


##**********************************************************************
## Activation functions                                          ####
##**********************************************************************

##' @name pmaxm
##' @description Parallel maximum, maintaining dimensions of X if X is a matrix
##' @param x input 1; scalar, vector or matrix
##' @param y input 2: either dimensions the same as x or a scalar
##' @return pmax(x) of the same dimensions as x
pmaxm=function(x,y) {
  if ("matrix" %in% class(x)) matrix(pmax(x,y),dim(x)) else pmax(x,y)
}

##' @name pminm
##' @description Parallel minimum, maintaining dimensions of X if X is a matrix
##' @param x input 1; scalar, vector or matrix
##' @param y input 2: either dimensions the same as x or a scalar
##' @return pmin(x) of the same dimensions as x
pminm=function(x,y) {
  if ("matrix" %in% class(x)) matrix(pmin(x,y),dim(x)) else pmin(x,y)
}



##' @name heaviside
##' @description Implements Heaviside activation function
##' @param x input; scalar, vector, or matrix
##' @return object of same form as x 
heaviside=function(x) {
  # This is a quick hack: x>0 will return an object of the same dimensions as 'x' with TRUE or FALSE entries.
  # 'as.numeric' converts TRUE to 1 and FALSE to 0.
  if ("matrix" %in% class(x)) matrix(as.numeric(x>0),dim(x)) else as.numeric(x>0)
}

##' @name logistic
##' @description Implements logistic activation function
##' @param x input; scalar, vector, or matrix
##' @return object of same form as x 
logistic=function(x) {
  1/(1+exp(-x))
}

##' @name dlogistic
##' @description Implements derivative of logistic function
##' @param x input; scalar, vector, or matrix
##' @return object of same form as x 
dlogistic = function(x) 
  exp(-x)/((1 + exp(-x))^2)


##' @name ReLU
##' @description Implements ReLU activation function
##' @param x input; scalar, vector, or matrix
##' @return object of same form as x 
ReLU=function(x) {
  # The function 'pmax' is 'parallel maximum': it computes a maximum between every pair of scalars in two objects.
  # Comparing x with 0 (a scalar) will return an object of the same dimensions as x.
  pmaxm(x,0)
}

##' @name SLU
##' @description Implements sigmoid linear unit activation function
##' @param x input; scalar, vector, or matrix
##' @return object of same form as x 
SLU=function(x) {
  x/(1+exp(-x))
}

##' @name ramp
##' @description Implements ramp activation function
##' @param x input; scalar, vector, or matrix
##' @param alpha control parameter
##' @return object of same form as x 
ramp=function(x,alpha=1) {
  # Using pmax and pmin together to 'clamp' a result
  pmaxm(pminm(alpha*x,1),-1)
}

##' @name pReLU
##' @description Implements parametric ReLU activation function
##' @param x input; scalar, vector, or matrix
##' @param alpha control parameter
##' @return object of same form as x 
pReLU=function(x,alpha=1) {
  pmax(x,alpha*x)
}



##**********************************************************************
## Basic neurons                                                 ####
##**********************************************************************

##' @name neuron
##' @description Implements a basic neuron
##' @param x matrix input of dimension n x m (n columns, m rows), representing n values of an m-dimensional input.
##' @param w vector of weights of length m; assumed not to be a column vector already
##' @param b bias, a scalar; ADDED not subtracted
##' @param phi activation function; defaults to identity
##' @return output phi()
neuron=function(x,w,b,phi=function(x) x) {
  phi( (t(matrix(w)) %*% x) # input multiplied by weights
       + b  # plus bias
  )
}



##' @name layer
##' @description Implements a layer of neurons
##' @param x matrix input of dimension n x m (n columns, m rows), representing n values of an m-dimensional input.
##' @param w matrix of weights of length m x l (m columns, l rows) where column i represents neuron i, for l neurons total
##' @param b bias, a vector of length l, with ith entry the bias for the ith neuron.
##' @param phi activation function; defaults to identity
##' @param return_S if TRUE, return a list also including S, where Y=phi(S).
##' @return output phi()
layer=function(x,w,b,phi=function(x) x,return_S=FALSE) {
  S=(t(w) %*% x) + b # input multiplied by weightsplus bias
  if (return_S) return(list(Y=phi(S),S=S)) else return(phi(S))
}

##' @name network
##' @description Implements a neural network
##' @param x matrix input of dimension n x m (n columns, m rows), representing n values of an m-dimensional input.
##' @param parameters list of parameters, with length equal to the number of layers. Each list should have three components: a weight matrix W, a bias B, and an activation function phi.
##' @param return_layers set to TRUE to return the outputs from each layer. Default FALSE.
##' @return Either the output of the network, or if return_layers=TRUE, a list containing outputs from each layer PRIOR to being passed through the activation function, and the output.
network=function(x,parameters,return_layers=FALSE) {
  nlayer=length(parameters) # Number of layers
  x_current=x 
  x_all=list()
  for (i in (1:nlayer)) { # Propagate forward through layers.
    p_i=parameters[[i]] # Extract parameters
    x_new=layer(x_current,p_i$w,p_i$b,p_i$phi,return_S=TRUE) # Compute output of layer i
    x_current=x_new$Y
    x_all[[i]]=x_new
  }
  x_all[[nlayer+1]]=x_new$Y
  if (return_layers) return(x_all) else return(x_new$Y) # Return final output.
}


##**********************************************************************
## What neurons and neural networks can do                       ####
##**********************************************************************

##' @name visualise2d
##' @description Allows visualisation of a function of two variables in 2d across a given range. Draws axes in red.
##' @param f a function, assumed to take a first input as a matrix of dimension 2xn. Optional arguments are passed through ...
##' @param xlim x limits, default c(-3,3)
##' @param ylim y limits, default c(-3,3)
##' @param xres x resolution (number of points); defaults to 100
##' @param yres y resolution (number of points); defaults to 100
##' @param colour; changes smoothly across range of x,y. Defaults to grayscale.
##' @param drawplot set to FALSE to not draw a plot
##' @param ... passed to f
##' @return invisibly returns matrix of values of f(x,y).
visualise2d=function(
    f,
    xlim=c(-3,3),
    ylim=c(-3,3),
    xres=100,
    yres=100,
    drawplot=TRUE,
    colour=colorRampPalette(c("white","black"))(100),
    ...
) {
  xr=seq(xlim[1],xlim[2],length=xres) # Sequence of x values
  yr=seq(ylim[1],ylim[2],length=yres) # Sequence of y values
  
  xm=outer(xr,rep(1,yres)) # Matrix of x values
  ym=outer(rep(1,xres),yr) # Matrix of y values
  
  xyv=rbind(as.vector(xm),as.vector(ym)) # Matrix of x and y co-ordinate values
  
  fxyv=f(xyv,...) # Values of function at each co-ordinate
  
  z=matrix(fxyv,xres,yres)
  
  if (drawplot) {
    image(xr,yr,z,
          xlab="X",ylab="Y",
          col=colour)
    abline(h=0,col="red") # Draw x axis in red
    abline(v=0,col="red") # Draw y axis in red
  }
  
  invisible(z) # Returns z only if assigned to a value, e.g. output=visualise2d(...)
}

##' @name random_network
##' @description Specifies paramaters for a random neural network which can be passed as an input to neuron()
##' @param dimx input dimension
##' @param nl number of hidden layers
##' @param wd width of hidden layers (defaults to 10 for each layer). If a scalar assumed to be equal for all hidden layers.
##' @param hphi activation function for hidden layers; default logistic.
##' @param wsd all weights are sampled as rnorm(.,sd=wsd). Default 5
##' @param bsd all biases are sampled as rnorm(.,sd=bsd). Default 5
##' @return list of parameters passable to neuron()
random_network=function(dimx,nl,wd=rep(10,nl),hphi=logistic,wsd=5,bsd=5) {
  if (length(wd)==1) wd=rep(wd,nl)
  
  params=list()
  params[[1]]=list(w=matrix(rnorm(dimx*wd[1],sd=wsd),dimx,wd[1]),b=rnorm(wd[1],sd=bsd),phi=hphi)
  for (i in 2:nl) {
    params[[i]]=list(w=matrix(rnorm(wd[i-1]*wd[i],sd=wsd),wd[i-1],wd[i]),b=rnorm(wd[i],sd=bsd),phi=hphi)
  }
  params[[nl+1]]=list(w=matrix(rnorm(wd[i],sd=wsd),wd[i],1),b=rnorm(1,sd=bsd),phi=function(x) x)
  return(params)
}


