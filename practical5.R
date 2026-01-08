##**********************************************************************
##  DLAI4 (MATH4267) practical worksheets                           ####
##  Code to accompany written material for practical 5
##  James Liley                                                         
##**********************************************************************

# Functions
source("./DLAI4/practical5_functions.R")



##**********************************************************************
## Packages                                                         ####
##**********************************************************************

library(keras3)
library(tensorflow)
library(reticulate)
library(ggplot2)
library(dplyr)
library(readr)
library(Polychrome)


# Set a random seed in R to make it more reproducible 
set.seed(123)

# Set the seed for Keras/TensorFlow
tensorflow::set_random_seed(123)


##**********************************************************************
## Load MNIST dataset                                               ####
##**********************************************************************


mnist<- dataset_mnist()

## Normalise and reshape
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255
x_train <- array_reshape(x_train, c(nrow(x_train), 28*28), order = "F")
x_test <- array_reshape(x_test, c(nrow(x_test), 28*28), order = "F")


##**********************************************************************
## Dimensions of spaces (incl. latent)                              ####
##**********************************************************************

original_dim <- 784L # Dimension of input (22 x 22)
latent_dim <- 2L # Dimension of latent space


intermediate_dim <- 256L # For NN architecture
batch_size<- 128 # For training



##**********************************************************************
## Specify encoder and decoder                                      ####
##**********************************************************************

encoder_inputs <- layer_input(shape = 28 * 28)

x <- encoder_inputs %>%
  layer_dense(intermediate_dim, activation = "relu")

# The encoder maps to a mean (z_mean) and a variance (z_log_var)
z_mean    <- x %>% layer_dense(latent_dim, name = "z_mean")
z_log_var <- x %>% layer_dense(latent_dim, name = "z_log_var")
encoder <- keras_model(encoder_inputs, list(z_mean, z_log_var),
                       name = "encoder")

# Look at encoder
encoder





## Specify decoder: taking an input from the latent space, and returning 
latent_inputs <- layer_input(shape = c(latent_dim))

decoder_outputs <- latent_inputs %>%
  layer_dense(intermediate_dim, activation = "relu") %>%
  layer_dense(original_dim, activation = "sigmoid")

decoder <- keras_model(latent_inputs, decoder_outputs,
                       name = "decoder")

decoder

##**********************************************************************
## Specify VAE structure                                            ####
##**********************************************************************

model_vae <- new_model_class(
  classname = "VAE",
  
  initialize = function(encoder, decoder, ...) {
    super$initialize(...)
    self$encoder <- encoder
    self$decoder <- decoder
    self$sampler <- layer_sampler()
    self$total_loss_tracker <-
      metric_mean(name = "total_loss")
    self$reconstruction_loss_tracker <-
      metric_mean(name = "reconstruction_loss")
    self$kl_loss_tracker <-
      metric_mean(name = "kl_loss")
  },
  
  metrics = mark_active(function() {
    list(
      self$total_loss_tracker,
      self$reconstruction_loss_tracker,
      self$kl_loss_tracker
    )
  }),
  
  train_step = function(data) {
    with(tf$GradientTape() %as% tape, {
      
      c(z_mean, z_log_var) %<-% self$encoder(data)
      z <- self$sampler(z_mean, z_log_var)
      
      reconstruction <- decoder(z)
      reconstruction_loss <-
        loss_binary_crossentropy(data, reconstruction) %>%
        sum(axis = c(1)) %>%
        mean()
      
      kl_loss <- -0.5 * (1 + z_log_var - z_mean^2 - exp(z_log_var))
      total_loss <- reconstruction_loss + mean(kl_loss)
    })
    
    grads <- tape$gradient(total_loss, self$trainable_weights)
    self$optimizer$apply_gradients(zip_lists(grads, self$trainable_weights))
    
    self$total_loss_tracker$update_state(total_loss)
    self$reconstruction_loss_tracker$update_state(reconstruction_loss)
    self$kl_loss_tracker$update_state(kl_loss)
    
    list(total_loss = self$total_loss_tracker$result(),
         reconstruction_loss = self$reconstruction_loss_tracker$result(),
         kl_loss = self$kl_loss_tracker$result())
  }
)





##**********************************************************************
## Compile and fit VAE                                              ####
##**********************************************************************

vae <- model_vae(encoder, decoder)
vae %>% compile(optimizer = optimizer_adam())
vae %>% fit(x_train, epochs = 20,
            shuffle = TRUE)


##**********************************************************************
## Get principal components                                         ####
##**********************************************************************

pc_ex=prcomp(x_train)


##**********************************************************************
## Look at latent space                                             ####
##**********************************************************************

par(mfrow=c(1,2))
ccx = glasbey.colors(11)[2:11]

## VAE

x_test_encoded <- predict(encoder, x_train, batch_size = batch_size)
mu=x_test_encoded[[1]]
xc=mnist$train$y
plot(mu,col=ccx[1+xc],pch=16,cex=0.3,
     xlab="Latent dimension 1",ylab="Latent dimension 2",
     main="VAE (neural network)",
     xlim=c(min(mu[,1]),1.5*max(mu[,1])))
legend("topright",legend=0:9,col=ccx,bty="n",pch=16)


plot(pc_ex$x[,1],pc_ex$x[,2],col=ccx[1+xc],pch=16,cex=0.5,
     xlab="Latent dimension 1",ylab="Latent dimension 2",
     main="Principal components",
     xlim=c(min(pc_ex$x[,1]),1.5*max(pc_ex$x[,1])))
legend("topright",legend=0:9,col=ccx,bty="n",pch=16)



##**********************************************************************
## Generate samples: VAE                                            ####
##**********************************************************************

# Number of digits per row/column
n <- 8

# we will sample n points within [-xmax,xmax] standard deviations
xmax=3
grid_x <- seq(-xmax, xmax, length.out = n)
grid_y <- seq(-xmax, xmax, length.out = n)

digits=list()
ii=1
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    z_sample <- matrix(c(grid_x[i], grid_y[j]), ncol = 2)
    #generate new digits using the predict function with the decoder
    digit=predict(vae$decoder, z_sample) %>% matrix(ncol = 28)
    digits[[ii]]=list(x=grid_x[i],y=grid_y[j],im=digit)
    ii=ii+1
  }
}

par(mfrow=c(1,1))
sc=0.2*((2*xmax)/n)
mu=x_test_encoded[[1]]
xc=mnist$train$y
plot(mu,col=ccx[1+xc],pch=16,cex=0.3,xlim=1.2*c(-xmax,xmax+2),ylim=1.2*c(-xmax,xmax),
     xlab="Latent dimension 1",ylab="Latent dimension 2",
     main="VAE (neural network)")
legend("topright",legend=0:9,col=ccx,bty="n",pch=16)
# Draw an X corresponding to each plotted point
for (i in 1:length(digits)) {
  dd=digits[[i]]
  rx=as.raster(dd$im)
  plot(rx, xleft=dd$x-sc,xright=dd$x+sc,ybottom=dd$y-sc,ytop=dd$y+sc,add=TRUE)
}





##**********************************************************************
## Generate samples: PCs                                            ####
##**********************************************************************


# Number of digits per row/column
n <- 8

# We need to use slightly different X and Y ranges
grid_x <- seq(-5, 10, length.out = n)
grid_y <- seq(-6, 6, length.out = n)

digits=list()
ii=1
for(i in 1:length(grid_x)){
  column <- NULL
  for(j in 1:length(grid_y)){
    x_inp <- c(grid_x[i],grid_y[j])
    #generate new digits using the predict function with the decoder
    vdigit=pr_decode(x_inp,pc_ex,npc=2)
    digit=matrix(vdigit,ncol = 28)
    digits[[ii]]=list(x=grid_x[i],y=grid_y[j],im=digit)
    ii=ii+1
  }
}

par(mfrow=c(1,1))
sc=0.2*((2*xmax)/n)
plot(pc_ex$x[,1],pc_ex$x[,2],col=ccx[1+xc],pch=16,cex=0.5,
     xlab="Latent dimension 1",ylab="Latent dimension 2",
     main="Principal components",
     xlim=c(1.2*min(grid_x),1.5*max(grid_x)),
     ylim=c(1.3*min(grid_y),1.3*max(grid_y)))
legend("topright",legend=0:9,col=ccx,bty="n",pch=16)
# Draw an X corresponding to each plotted point
for (i in 1:length(digits)) {
  dd=digits[[i]]
  rx=as.raster((dd$im-min(dd$im))/(max(dd$im)-min(dd$im)))
  plot(rx, xleft=dd$x-sc,xright=dd$x+sc,ybottom=dd$y-sc,ytop=dd$y+sc,add=TRUE)
}

