##**********************************************************************
##  DLAI4 (MATH4267) practical worksheets                           ####
##  Code to accompany written material for practical 3
##  James Liley                                                         
##**********************************************************************

# Functions
source("./DLAI4/practical3_functions.R")


##**********************************************************************
## 1. Convolutions                                                  ####
##**********************************************************************

## Get a simple image using the function get_simple_image() and a 
##  complex image using the function get_complex_image()
circle=get_simple_image()
goya=get_complex_image()

## Draw the images using the function display_image()
par(mfrow=c(1,2))
display_image(circle)
display_image(goya)

## Write a function to convolve two matrices.

# See function conv()

## Define convolutions to find vertical and horizontal edges. Test them on 
##  the simple and complex images, and display the convolved images to show
##  where the horizontal and vertical edges are.

# Vertical
vk=rbind(matrix(-1,5,10),matrix(1,5,10))

# Horizontal
hk=t(vk)

# Test on simple image
par(mfrow=c(2,3))
# Vertical edge finder (white=high)
display_image(circle,main="Plain image")
display_image(vk,main="Kernel (vertical)")
display_image(conv(circle,vk),main="Convolved")

# Horizontal edge finder (white=high)
display_image(circle,main="Plain image")
display_image(hk,main="Kernel (horizontal)")
display_image(conv(circle,hk),main="Convolved")




# Test on complex image
par(mfrow=c(2,3))
# Vertical edge finder (white=high)
display_image(goya,main="Plain image")
display_image(vk,main="Kernel (vertical)")
display_image(conv(goya,vk),main="Convolved")

# Horizontal edge finder (white=high)
display_image(goya,main="Plain image")
display_image(hk,main="Kernel (horizontal)")
display_image(conv(goya,hk),main="Convolved")



## Use the function face() to generate a rough face-like kernel matrix. Test it on 
##  the simple and complex images, and display the convolved images to show
##  where the horizontal and vertical edges are.

fk=face()
cfk=conv(goya,fk) # This is slow - beware.

# Test on complex image
par(mfrow=c(1,3))
# Vertical edge finder (white=high)
display_image(goya,main="Plain image")
display_image(fk,main="Kernel (face)")
display_image(cfk,main="Convolved") 


## Generate a 5x5 constant kernel which blurs an image. Convolve it 
## with the simple and complex images.

# Blurring kernel
blur=matrix(1,5,5)
blur=blur/sum(blur)

# Convolution
cblur=conv(circle,blur)
gblur=conv(goya,blur)

# Test on simple and complex images
par(mfrow=c(2,3))
display_image(circle,main="Plain image")
display_image(blur,main="Kernel (blur)")
display_image(cblur,main="Convolved") 

display_image(goya,main="Plain image")
display_image(blur,main="Kernel (blur)")
display_image(gblur,main="Convolved") 

## Generate a 3x3 kernel which sharpens an image. Convolve it 
## with the simple and complex images.

# Sharpening kernel
sharp=cbind(c(0,-1,0),c(-1,5,-1),c(0,-1,0))

# Convolution
gs=conv(goya,sharp)
gs[which(gs<0)]=0; gs[which(gs>1)]=1

# Test on simple and complex images
par(mfrow=c(1,3))

display_image(goya,main="Plain image")
display_image(sharp,main="Kernel (sharp)")
display_image(gs,main="Convolved") 


##**********************************************************************
## 2. CNNs                                                          ####
##**********************************************************************


## Define a series of directories
base_dir="cats_and_dogs_small/"
train_dir="cats_and_dogs_small/train/"
validation_dir="cats_and_dogs_small/validation"
test_dir="cats_and_dogs_small/test"

train_dogs_dir=paste0(train_dir,"dogs/")
train_cats_dir=paste0(train_dir,"cats/")

validation_dogs_dir=paste0(validation_dir,"dogs/")
validation_cats_dir=paste0(validation_dir,"cats/")

test_dogs_dir=paste0(test_dir,"dogs/")
test_cats_dir=paste0(test_dir,"cats/")

## Keras
library(keras3)



# Functions to get images from the appropriate directories and resize them
train_datagen=image_dataset_from_directory(
  directory = train_dir,
  image_size=c(150,150),
  batch_size=20,
  label_mode="binary"
)
test_datagen=image_dataset_from_directory(
  directory = test_dir,
  image_size=c(150,150),
  batch_size=20,
  label_mode="binary"
)
validation_datagen=image_dataset_from_directory(
  directory = validation_dir,
  image_size=c(150,150),
  batch_size=20,
  label_mode="binary"
)

## The following code generates `simulated' data, which is essentially the same 
##  images rotated and scaled. We will in fact do this with a neural network, 
##  which is a reasonably sensible way to manage this:
data_augmentation <-
  keras_model_sequential(input_shape = c(150,150, 3)) %>%
  layer_random_flip("horizontal") %>% # Flip some models horizontally
  layer_random_rotation(factor = 0.1) # Rotate




# Draw a couple of cats/dogs

# Get a batch
batch <- train_datagen %>%
  as_iterator() %>%
  iter_next()
oldpar=par(mfrow=c(2,2),mar=rep(0.5,4))
for (i in 1:4) {
  plot(0,xlim=c(0,1),ylim=c(0,1),xaxt="n",yaxt="n",xlab="",ylab="",type="n")
  rasterImage(batch[[1]][sample(20,1),,,]/255,0,0,1,1)
}
par(oldpar)


## Set up model
input = layer_input(shape=c(150, 150, 3))
output = input %>% 
  layer_rescaling(scale = 1/255) %>% # rescale RGB values (in [0,255]) to [0,1]
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
model=keras_model(input,output)

## Look at model (note the number of trainable parameters!)
summary(model)

## Compile model, getting ready for training (don't worry about warnings)
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 1e-4),
  metrics = c("acc")
)


### Train the model (and remember history). This is very slow!
history = model %>% fit(
  train_datagen,
  steps_per_epoch = 100,
  epochs = 10,
  validation_data = validation_datagen,
  validation_steps = 50
)

## Save the model (worth doing after all that training)
save_model(model,"cats_and_dogs_small_1.keras")

## This history is typical of overtraining:
plot(history)

## Let's see how we did:
batch <- train_datagen %>%
  as_iterator() %>%
  iter_next()
mp=predict(model,batch[[1]])
par(mfrow=c(2,3))
for (i in 1:6) {
  par(mar=c(1,4,1,1))
  plot(as.raster(batch[[1]][i,,,]/255))
  mpx=mp[i]
  title(main=paste0(c("Cat","Dog")[1 + (mpx>0.5)]))
}
# rubbish


### How can we improve it? Let's add some simulated data

# Let's look at the result of repeatedly applying this to one image
par(mfrow = c(3, 3))
bx=batch[[1]][3, , , ,drop=FALSE] 
for (i in 1:9) {
  bx=bx %>%
    data_augmentation()
  plot(as.raster(bx[1,,,]/255))
}

# Make our new model:
## Set up model
input2 = layer_input(shape=c(150, 150, 3))
output2 = input2 %>% 
  data_augmentation() %>%
  layer_rescaling(scale = 1/255) %>% # rescale RGB values (in [0,255]) to [0,1]
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
model2=keras_model(input2,output2)

# Compile and train
## Compile model, getting ready for training (don't worry about warnings)
model2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 1e-4),
  metrics = c("acc")
)

### Train the model (and remember history). This is very slow!
history2 = model2 %>% fit(
  train_datagen,
  steps_per_epoch = 100,
  epochs = 10,
  validation_data = validation_datagen,
  validation_steps = 50
)

## Save the model (worth doing after all that training)
save_model(model2,"cats_and_dogs_small_2.keras")

## Looking again at history
plot(history2)

## Let's see how we did:
batch <- train_datagen %>%
  as_iterator() %>%
  iter_next()
mp=predict(model2,batch[[1]])
par(mfrow=c(2,3))
for (i in 1:6) {
  par(mar=c(1,4,1,1))
  plot(as.raster(batch[[1]][i,,,]/255))
  mpx=mp[i]
  title(main=paste0(c("Cat","Dog")[1 + (mpx>0.5)]))
}
# slightly better.


## A better way to do this is to use a model which has already been trained to 
##  identify images, and re-train it to identify cats and dogs; this is a bit 
##  more involved, but a guide is available in Chollet et al. 

## Let's do this with the imagenet weights
gc()
base_model= application_xception(
  weights = 'imagenet', # Load weights pre-trained on ImageNet.
  input_shape = c(150, 150, 3),
  include_top = FALSE # Do not include the ImageNet classifier at the top.
)

# Set the base model to be nontrainable
base_model$trainable = FALSE

# We are going to put an extra layer over the big imagenet model
inputs <- layer_input(c(150, 150, 3))
outputs <- inputs %>%
  data_augmentation() %>%
  layer_rescaling(scale = 1/255) %>% # rescale RGB values (in [0,255]) to [0,1]
  base_model(training=FALSE) %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(1)
model_imagenet <- keras_model(inputs, outputs)

# Compile and train
model_imagenet %>%
  compile(optimizer = optimizer_adam(),
          loss = loss_binary_crossentropy(from_logits = TRUE),
          metrics = metric_binary_accuracy())
history_imagenet= model_imagenet %>%  fit(
  train_datagen,
  steps_per_epoch = 30,
  epochs = 5,
  validation_data = validation_datagen,
  validation_steps = 50
)

save_model(model_imagenet,"cats_and_dogs_small_imagenet.keras")

## Now evaluate, and let's see how we did:
## Let's see how we did:
batch <- train_datagen %>%
  as_iterator() %>%
  iter_next()
mp=predict(model_imagenet,batch[[1]])
par(mfrow=c(2,3))
for (i in 1:6) {
  par(mar=c(1,4,1,1))
  plot(as.raster(batch[[1]][i,,,]/255))
  mpx=mp[i]
  title(main=paste0(c("Cat","Dog")[1 + (mpx>0.5)]))
}
# slightly better.
