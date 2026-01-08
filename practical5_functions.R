##**********************************************************************
##  DLAI4 (MATH4267) practical worksheets                           ####
##  Auxiliary material for practical 5
##  James Liley                                                         
##**********************************************************************

## Resample from the latent distribution
layer_sampler <- new_layer_class(
  classname = "Sampler",
  call = function(z_mean, z_log_var) {
    epsilon <- tf$random$normal(shape = tf$shape(z_mean))
    z_mean + exp(0.5 * z_log_var) * epsilon } # Note reparametrisation trick
)


#' Function to 'decode' a given image with PCs
#' @param inp: vector of values to decode
#' @param pc: principal component object
#' @param npc: number of principal components to use (dim. of latent space)
pr_decode=function(inp,pc,npc=2) {
  m_inp=matrix(inp,1,length(inp))
  pc_rev=t(pc$rotation)[1:npc,] # Principal component matrix for first two components
  return(t(t(m_inp %*% pc_rev) + pc$center))
}

