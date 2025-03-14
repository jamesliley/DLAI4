##**********************************************************************
##  DLAI4 (MATH4267) practical worksheets                           ####
##  Code to accompany written material
##  James Liley                                                         
##  October 2023                                                              
##**********************************************************************

# Functions
source("./DLAI4/practical4_functions.R")


##**********************************************************************
## 1. Temperature                                                   ####
##**********************************************************************


## Simulate an arbitrary multinomial distribution over ten states
##   (that is, a set of values $p_1, p_2 \dots p_{10}$ such that
##   for states $x_1,x_2, \dots x_{10}$, the probability of a 
##   random variable $X$ over $x_1,x_2 \dots x_{10}$ taking the 
##   value $x_i$ is $P(X=x_i)=p_i$)

pp=runif(10) # pp[i]=p_i
pp=pp/sum(pp) # make sure it is a probability distribution

## Simulate a thousand values of $X$ and plot the frequencies of 
##  each state

X=rmultinom(1,1000,pp)
barplot(t(X))

## Suppose that we express $p_i$ as ... for some normalising factor $K$.
##  Compute the values $q_i$ assuming $K=T=1$. 

qq=-log(pp)

## For the values of $q_i$ from the previous part, recompute the values 
##  $p_i(T)$ (where we now give a dependence on $T$) for $T=0.2,2$. 

temp=0.2
pp1=exp(-qq/temp); 
pp1=pp1/sum(pp1)

temp=2
pp2=exp(-qq/temp); 
pp2=pp2/sum(pp2)

## Re-simulate a thousand values according to the $p_i(T)$ from the 
##  previous part. What happens as $T \to 0$ and $T \to \infty$? 

X1=rmultinom(1,1000,pp1)
X2=rmultinom(1,1000,pp2)
barplot(t(cbind(X,X1,X2)),beside=TRUE)

# As T tends to infinity, the multinomial distribution becomes dominated
#  by the largest value; as T tends to infinity, all probabilities 
#  become equal. 


##**********************************************************************
## 2. LSTM fitting                                                  ####
##**********************************************************************


# Get data
library(keras3)
library(stringr)
#path = get_file(
#  "nietzsche.txt",
#  origin = "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
#)
path=get_file(
  "Stephenie%20Meyer%201.%20Twilight_djvu.txt",
  origin="https://archive.org/stream/StephenieMeyer1.Twilight/Stephenie%20Meyer%201.%20Twilight_djvu.txt"
)
text0 = tolower(readChar(path, file.info(path)$size))
cat("Corpus length (raw):", nchar(text0), "\n")

# trim to the good stuff
text1=substring(text0,
                unlist(gregexpr("i'd never given much thought", text0))[1],
                unlist(gregexpr("more to my throat.", text0))[1]+18)
text=gsub("[\r\n—\\]", "", text1)

cat("Corpus length (text only):", nchar(text), "\n")


# We are going to prepare this data for LSTM training. 
# We will train the LSTM on sequences of text of length `maxlen'. 
# Rather than using all blocks of length 60, we will step through the text
#  and start a new one every 'step' characters
maxlen = 60
step = 3

# Get 'sentences' of length 60
text_indices = seq(1, nchar(text) - maxlen, by = step)
sentences = str_sub(text, text_indices, text_indices + maxlen - 1)
next_chars = str_sub(text, text_indices + maxlen, text_indices + maxlen)

# Print number of sequences
cat("Number of sequences: ", length(sentences), "\n")

# This is going to convert our sentences to a long set of lists of characters 
chars = unique(sort(strsplit(text, "")[[1]]))
cat("Unique characters:", paste(chars,collapse=" "), "\n")
char_indices = 1:length(chars)
names(char_indices) = chars

## Convert these to one-hot encoding (a bit slow)
# For memory considerations, we'll only train on a random 50k sentences.
nsent=50000 # train on this many 


# Set up model. The 'sequential' will make this a sequence-processing model. 
model = keras_model_sequential() %>%
  layer_lstm(units = 128, input_shape = c(maxlen, length(chars))) %>% 
  layer_dense(units = length(chars), activation = "softmax")  ## Output layer: this will predict the next character

# Configuration
optimizer = optimizer_rmsprop(learning_rate = 0.01)
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer
)


## Train the model (and save it). 
xy=get_sentence_subset(50000)
model %>% fit(xy[[1]], xy[[2]], batch_size = 128, epochs = 30)
model %>% save_model_hdf5("twilight.h5")
gc()

# Make some wisdom. Try this with several temperatures:
wisdom(model,nc=250,temperature=0.7)







## LSTMs using words instead




# Get data
library(keras3)
library(stringr)
#path = get_file(
#  "nietzsche.txt",
#  origin = "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
#)
path=get_file(
  "Stephenie%20Meyer%201.%20Twilight_djvu.txt",
  origin="https://archive.org/stream/StephenieMeyer1.Twilight/Stephenie%20Meyer%201.%20Twilight_djvu.txt"
)
text0 = tolower(readChar(path, file.info(path)$size))
cat("Corpus length (raw):", nchar(text0), "\n")

# trim to the good stuff
text1=substring(text0,
                unlist(gregexpr("i'd never given much thought", text0))[1],
                unlist(gregexpr("more to my throat.", text0))[1]+18)
text=gsub("[\r\n—\\]", "", text1)

cat("Corpus length (text only):", nchar(text), "\n")


## Change to words 
text_mod=text
text_mod=gsub("."," .",text_mod,fixed=TRUE) # change full stop to put space first
text_mod=gsub(","," ,",text_mod,fixed=TRUE) # change comma to put space first
text_mod=gsub(" '"," ' ",text_mod,fixed=TRUE) # change quote to put space first
text_mod=gsub("  "," ",text_mod,fixed=TRUE) # remove double space
text_mod=gsub("[-\")()]","",text_mod,fixed=FALSE) # remove hyphens and quotes
text_mod=tolower(text_mod)
words=unlist(strsplit(text_mod," "))
words=words[which(nchar(words)>0)]


## Most frequent words. We will only consider fairly common words.
mwords=table(words)
xwords=names(mwords)[which(mwords>10)]
allwords=xwords[order(mwords[xwords],decreasing=TRUE)]
nwords=length(xwords)
words=words[which(words %in% xwords)]

# We are going to prepare this data for LSTM training. We will use words of length maxlen.
training_set_size=10000
maxlen = 10
tlen=length(words)
train_dat=array(0,dim=c(training_set_size,maxlen,nwords))
gc()

# Get 'sentences' of length maxlen
start_indices = sample(1:(tlen-maxlen),training_set_size)
for (i in 1:maxlen) {
  w=words[start_indices + i - 1]
  train_dat[,i,]=to_one_hot(w)
}

next_chars=to_one_hot(words[start_indices + maxlen])


# Set up model. The 'sequential' will make this a sequence-processing model. 
model = keras_model_sequential() %>%
  layer_lstm(units = 128, input_shape = c(maxlen, nwords)) %>% 
  layer_dense(units = nwords, activation = "softmax")  ## Output layer: this will predict the next character

# Configuration
optimizer = optimizer_rmsprop(learning_rate = 0.01)
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer
)


## Train the model (and save it). 
model %>% fit(train_dat, next_chars, batch_size = 128, epochs = 50)
save_model(model,filepath="twilight_words.keras",overwrite=TRUE)
gc()

# Make some wisdom. Try this with several temperatures:
wisdom_words(model,nc=250,temperature=0.7)


##**********************************************************************
## 3. GANS                                                          ####
##**********************************************************************

## We will set up a GAN to generate images from CIFAR. 



## Train the GAN
cifar10 = dataset_cifar10()
x_train=cifar10$train$x
y_train=cifar10$train$y
x_test=cifar10$test$x
y_test=cifar10$test$y
x_train = x_train[as.integer(y_train) == 7,,,] # pictures of horses
x_train = x_train / 255

# Look at a couple of pictures
par(mfrow=c(2,2))
for (i in 1:4) {
  ix=sample(dim(x_train)[1],1)
  plot(as.raster(x_train[ix,,,]))
}




## Dimensions of images
latent_dim = 32 # Dimension of latent space
height = 32 # 32 pixels
width = 32
channels = 3 # number of colours

## Set up generator
generator_input2 = layer_input(shape = c(latent_dim))
generator_output2 = generator_input2 %>%
  layer_dense(units = 128 * 16 * 16) %>%
  layer_activation_leaky_relu() %>%
  layer_reshape(target_shape = c(16, 16, 128)) %>%
  layer_conv_2d(filters = 256, kernel_size = 5,
                padding = "same") %>%
  layer_conv_2d_transpose(filters = 256, kernel_size = 4,
                          strides = 2, padding = "same") %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(filters = channels, kernel_size = 7,
                activation = "sigmoid", padding = "same")
generator = keras_model(generator_input2, generator_output2)




## Set up discriminator

discriminator_input = layer_input(shape = c(height, width, channels))
discriminator_output = discriminator_input %>%
  layer_conv_2d(filters = 128, kernel_size = 3) %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(filters = 128, kernel_size = 4, strides = 2) %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(filters = 128, kernel_size = 4, strides = 2) %>%
  layer_activation_leaky_relu() %>%
  layer_conv_2d(filters = 128, kernel_size = 4, strides = 2) %>%
  layer_activation_leaky_relu() %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = 1, activation = "sigmoid")
discriminator = keras_model(discriminator_input, discriminator_output)
summary(discriminator)
discriminator_optimizer =optimizer_rmsprop(
  learning_rate = 0.0001, # Learning rate of discriminator
  clipvalue = 1.0)
discriminator %>% compile(
  optimizer = discriminator_optimizer,
  loss = "binary_crossentropy"
)


### This is for training the generator: we hold the discriminator constant when doing this.
freeze_weights(discriminator)
gan_input = layer_input(shape = c(latent_dim))
gan_output = discriminator(generator(gan_input))
gan = keras_model(gan_input, gan_output)
gan_optimizer = optimizer_rmsprop(
  learning_rate = 0.003, # learning rate of generator
  clipvalue = 1.0)
gan %>% compile(
  optimizer = gan_optimizer,
  loss = "binary_crossentropy"
)


iterations = 1000 # Really needs to be more like 10k
batch_size = 5
start = 1
for (step in 1:iterations) {
  
  # Generate random vectors from latent space
  random_latent_vectors = matrix(rnorm(batch_size * latent_dim),
                                 nrow = batch_size, ncol = latent_dim)
  
  
  # Generate fake images
  generated_images = generator %>% predict(random_latent_vectors)
  # To see one:
  # plot(as.raster(generated_images[1,,,]))
  
  # Get some real images
  stop = start + batch_size - 1
  real_images = x_train[start:stop,,,]
  
  # Combine
  rows = nrow(real_images)
  combined_images = array(0, dim = c(rows * 2, dim(real_images)[-1]))
  combined_images[1:rows,,,]= generated_images
  combined_images[(rows+1):(rows*2),,,] = real_images
  labels = rbind(matrix(1, nrow = batch_size, ncol = 1),
                 matrix(0, nrow = batch_size, ncol = 1))
  #labels = labels + (0.5 * array(runif(prod(dim(labels))),
  #                                dim = dim(labels)))
  
  # Train discriminator on images
  d_loss = discriminator %>% train_on_batch(combined_images, labels)
  
  # Now train generator. 
  random_latent_vectors = matrix(rnorm(batch_size * latent_dim),
                                 nrow = batch_size, ncol = latent_dim)
  misleading_targets = array(0, dim = c(batch_size, 1))
  a_loss = gan %>% train_on_batch(
    random_latent_vectors,
    misleading_targets
  )
  
  # Update iterators and show progress
  start = start + batch_size
  if (start > (nrow(x_train) - batch_size))
    start = 1
  if (step %% 10 == 0) {
    cat("Discriminator loss:", d_loss, "\n")
    cat("Adversarial loss:", a_loss, "\n")
    par(mfrow=c(1,2))
    plot(0,type="n",xlim=c(0,1),ylim=c(0,1),
         xlab="",ylab="",xaxt="n",yaxt="n",bty="n",
         main="Real")
    rasterImage(real_images[1,,,], xleft = 0, xright = 1,
                ytop = 0, ybottom = 1, interpolate = FALSE)
    plot(0,type="n",xlim=c(0,1),ylim=c(0,1),
         xlab="",ylab="",xaxt="n",yaxt="n",bty="n",
         main="Generated")
    rasterImage(generated_images[1,,,], xleft = 0, xright = 1,
                ytop = 0, ybottom = 1, interpolate = FALSE)
    
  }
}
save_model(generator, "gan_generator.keras")




##**********************************************************************
## 4. VAEs                                                          ####
##**********************************************************************

library(Polychrome)

mnist<- dataset_mnist()

## Normalise and reshape
x_train <- mnist$train$x/255
x_test <- mnist$test$x/255
x_train <- array_reshape(x_train, c(nrow(x_train), 28*28), order = "F")
x_test <- array_reshape(x_test, c(nrow(x_test), 28*28), order = "F")


##**********************************************************************
## Dimensions of spaces (incl. latent)                                 #
##**********************************************************************

original_dim <- 784L # Dimension of input (22 x 22)
latent_dim <- 2L # Dimension of latent space


intermediate_dim <- 256L # For NN architecture
batch_size<- 128 # For training



##**********************************************************************
## Specify encoder and decoder                                         #
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


## This is a function to resample from the latent distribution
layer_sampler <- new_layer_class(
  classname = "Sampler",
  call = function(z_mean, z_log_var) {
    epsilon <- tf$random$normal(shape = tf$shape(z_mean))
    z_mean + exp(0.5 * z_log_var) * epsilon } # Note reparametrisation trick
)


##**********************************************************************
## Specify VAE structure                                               #
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
## Compile and fit VAE                                                 #
##**********************************************************************

vae <- model_vae(encoder, decoder)
vae %>% compile(optimizer = optimizer_adam())
vae %>% fit(x_train, epochs = 20,
            shuffle = TRUE)



##**********************************************************************
## Look at latent space                                                #
##**********************************************************************

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



##**********************************************************************
## Generate samples: VAE                                               #
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
    digit=predict(vae$decoder, z_sample,verbose=FALSE) %>% matrix(ncol = 28)
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
for (i in 1:length(digits)) {
  dd=digits[[i]]
  rx=as.raster(dd$im)
  plot(rx, xleft=dd$x-sc,xright=dd$x+sc,ybottom=dd$y-sc,ytop=dd$y+sc,add=TRUE)
}

