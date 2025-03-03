require(jpeg)

##' Get complex image. 
##' 
##' @return Matrix representing gray levels for grayscale version of Goya's 'The Witches' sabbath'
get_complex_image=function() {
  ux="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Francisco_de_Goya_y_Lucientes_-_Witches_Sabbath_-_Google_Art_Project.jpg/1024px-Francisco_de_Goya_y_Lucientes_-_Witches_Sabbath_-_Google_Art_Project.jpg"
  loc='~/goya.jpg'
  download.file(ux, loc)
  goya=readJPEG(loc)
  goyabw=goya[,,1] + goya[,,2] + goya[,,3]
  goyabw=goyabw/3
  goyabw=goyabw[dim(goyabw)[1]:1,]
  goyabw=t(goyabw)
  return(goyabw)
}

##' Get simple image
##' 
##' @return Matrix representing gray levels for grayscale version of Goya's 'The Witches' sabbath'
get_simple_image=function() {
  x=seq(-2,2,length=200)
  tmat=outer(x,x,function(x,y) (x^2 + y^2 < 1))
  #image(conv(tmat,ve))
  return(tmat)
}

##' Display an image (in black and white)
##' 
##' Similar to 'image', but black and white and does not display axes.
##' @param mat matrix to display as image
display_image=function(mat,...) {
  bwc=colorRampPalette(c("black","white"))(100)
  image(mat,col=bwc,xaxt="n",yaxt="n",xlab="",ylab="",...)
}


##' Matrix convolution
##' 
##' @param mat1 first matrix, m1 x n1
##' @param mat2 second matrix, m2 x n2
##' @return convolution of matrices, of dimension (m1-m2)
conv=function(mat1,mat2) {
  m1=dim(mat1)[1]; n1=dim(mat1)[2]
  m2=dim(mat2)[1]; n2=dim(mat2)[2]
  xmat=matrix(0,m1-m2,n1-n2)
  for (i in 1:(m1-m2)) {
    for (j in 1:(n1-n2)) {
      xmat[i,j]=sum(mat1[i + (0:(m2-1)),j + (0:(n2-1))]*mat2)
    }
  }
  return(xmat)
}

##' Generates a kernel to find faces (rough)
##' 
##' @param width kernel width
##' @param height kernel height
##' @return matrix of size width x height which roughly looks for faces.
face=function(width=50,height=80) {
  e1=c(0.35,0.6); e2=c(1-e1[1],e1[2]) # eye centres
  er=0.1 # eye radius
  mc=c(0.5,0.3) # mouth centre
  ml=0.4 # mouth length (radians)
  mw=0.1 # mouth width
  fc=c(0.5,0.5) # face centre
  fr=0.4 # face radius
  bl=function(x,y) {
    eye1=((x-e1[1])^2 + (y-e1[2])^2 <= er^2) # eye 1
    eye2=((x-e2[1])^2 + (y-e2[2])^2 <= er^2) # eye 2
    md=((x-mc[1])^2 + (y-mc[2])^2)
    mx=-atan((x-mc[1])/(y-mc[2]))
    mouth= (abs(x-mc[1])< ml/2)& (abs(y-mc[2])< mw/2)
    out=((x-fc[1])^2 + (y-fc[2])^2) > fr^2
    return( (eye1|eye2)|(mouth|out))
  }
  xx=(1:width)/width; yy=(1:height)/height
  z01=outer(xx,yy,bl)
  zw= mean(z01)-z01
  return(zw)
}
