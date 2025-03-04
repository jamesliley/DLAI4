##**********************************************************************
##  DLAI4 (MATH4267) practical worksheets                           ####
##  Code to accompany written material
##  James Liley                                                         
##  October 2023                                                              
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
library(keras)

## Set up model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
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

## Look at model (note the number of trainable parameters!)
summary(model)

## Compile model, getting ready for training (don't worry about warnings)
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 1e-4),
  metrics = c("acc")
)


## The following code reshapes the images to make them easier to train on.
train_datagen = image_data_generator(rescale = 1/255)
validation_datagen = image_data_generator(rescale = 1/255)
test_datagen = image_data_generator(rescale = 1/255)
train_generator = flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)
validation_generator = flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)
test_generator = flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

## Take a quick look at what is going on with (e.g.) train_generator
batch = generator_next(train_generator)
str(batch)
# This gives a batch of 20 images (in batch[[1]][1:20,,,]) and labels (in batch[[2]][1:20])

# Draw a couple of cats/dogs
par(mfrow=c(1,2))
plot(0,xlim=c(0,1),ylim=c(0,1),xaxt="n",yaxt="n",xlab="",ylab="",type="n")
rasterImage(batch[[1]][1,,,],0,0,1,1)
plot(0,xlim=c(0,1),ylim=c(0,1),xaxt="n",yaxt="n",xlab="",ylab="",type="n")
rasterImage(batch[[1]][2,,,],0,0,1,1)

### Train the model (and remember history). This is very slow!
history = model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 10,
  validation_data = validation_generator,
  validation_steps = 50
)

## Save the model (worth doing after all that training)
model %>% save_model_hdf5("cats_and_dogs_small_1.h5")

## This history is typical of overtraining:
plot(history)


### How can we improve it? Let's add some noise

## The following code generates `simulated' data, which is essentially the same 
##  images rotated and scaled. This effectively enlarges the training sample.
datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

## As examples:
fnames= list.files(train_cats_dir, full.names = TRUE)
img_path = fnames[[3]]
img = image_load(img_path, target_size = c(150, 150))
img_array = image_to_array(img)
img_array = array_reshape(img_array, c(1, 150, 150, 3))
augmentation_generator = flow_images_from_data(
  img_array,
  generator = datagen,
  batch_size = 1
)
op = par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
for (i in 1:4) {
  batch = generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}
par(op)



## Let's also add some dropout layers
model = keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 1e-4),
  metrics = c("acc")
)


## aannnddd.. train it again
datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)
test_datagen = image_data_generator(rescale = 1/255)
train_generator = flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "binary"
)
validation_generator = flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "binary"
)
history = model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 10,
  validation_data = validation_generator,
  validation_steps = 50
)

## and save the new model:
model %>% save_model_hdf5("cats_and_dogs_small_2.h5")


## A better way to do this is to use a model which has already been trained to 
##  identify images, and re-train it to identify cats and dogs; this is a bit 
##  more involved, but a guide is available in Chollet et al. 