##**********************************************************************
##  DLAI4 (MATH4267) practical worksheets                           ####
##  Code to accompany written material
##  James Liley                                                         
##  October 2023                                                              
##**********************************************************************

##**********************************************************************
## LSTMs                                                            ####
##**********************************************************************

##' @name get_sentence_subset
##' @description Get a subset of sentences. Will use predefined variables.
##' @param nsent number of sentences
##' @return n sentences
get_sentence_subset=function(nsent) {
  x = array(0L, dim = c(nsent, maxlen, length(chars)))
  y = array(0L, dim = c(nsent, length(chars)))
  samp=sample(length(sentences),nsent)
  for (i in 1:nsent) {
    sentence <- strsplit(sentences[[samp[i]]], "")[[1]]
    for (t in 1:length(sentence)) {
      char <- sentence[[t]]
      x[i, t, char_indices[[char]]] = 1
    }
    next_char = next_chars[[samp[i]]]
    y[i, char_indices[[next_char]]] = 1
    if ((i %% 10000)==0) print(paste0("Completed ",i," of ",nsent))
  }
  return(list(x,y))
}



##' @description Samples the next character, given a distribution of predictions and a temperature
##' @param preds predictions; a vector of probabilities summing to 1
##' @param temperature the temperature to use for sampling.
sample_next_char = function(preds, temperature = 1.0) {
  preds = as.numeric(preds)
  preds = log(preds) / temperature
  exp_preds = exp(preds)
  preds = exp_preds / sum(exp_preds)
  which.max(t(rmultinom(1, 1, preds)))
}


##' @description Generate a sequence of characters resembling that of a training text
##' @param model trained LSTM model
##' @param nc number of characters
##' @param temperature temperature at which to sample new characters
##' @param start_index start with a sentence at this start index.
##' @return nothing; prints output.
wisdom=function(model,nc=200,temperature=0.5,start_index = sample(1:(nchar(text) - maxlen - 1), 1)) {
  seed_text = str_sub(text, start_index, start_index + maxlen - 1)
  cat("—- Generating with seed:", seed_text, "\n\n")
  cat("—---- temperature:", temperature, "\n")
  cat(seed_text, "\n")
  generated_text <- seed_text
  for (i in 1:nc) {
    sampled <- array(0, dim = c(1, maxlen, length(chars)))
    generated_chars <- strsplit(generated_text, "")[[1]]
    for (t in 1:length(generated_chars)) {
      char <- generated_chars[[t]]
      sampled[1, t, char_indices[[char]]] <- 1
    }
    preds <- model %>% predict(sampled, verbose = 0)
    next_index <- sample_next_char(preds[1,], temperature)
    next_char <- chars[[next_index]]
    generated_text <- paste0(generated_text, next_char)
    generated_text <- substring(generated_text, 2)
    cat(next_char)
  }
  cat("\n\n")
}

##' @description changes a series of words to one-hot, using a fixed set of words.
##' @param xtext text to change
##' @param ref words
##' @return matrix
to_one_hot=function(xtext,ref=allwords) {
  return(model.matrix(~0+factor(xtext,levels=allwords)))
}

##' @description As for wisdom() but with words
##' @param model trained LSTM model
##' @param nc number of words
##' @param temperature temperature at which to sample new words
##' @param start_index start with a sentence at this start index.
##' @return nothing; prints output.
wisdom_words=function(model,nc=200,temperature=0.5,start_index = sample(length(words)-maxlen, 1)) {
  seed_text = words[start_index:(start_index + maxlen - 1)]
  cat("—- Generating with seed:", seed_text, "\n\n")
  cat("—---- temperature:", temperature, "\n")
  cat(paste0(seed_text,collapse=" "), "\n")
  generated_text <- seed_text
  xinput=array(0,dim=c(1,maxlen,nwords))
  xinput[1,,]=to_one_hot(seed_text)
  for (i in 1:nc) {
#    for (t in 1:length(generated_chars)) {
#      char <- generated_chars[[t]]
#      sampled[1, t, char_indices[[char]]] <- 1
#    }
    preds <- model %>% predict(xinput, verbose = 0)
    next_index <- sample_next_char(preds[1,], temperature)
    next_word <- allwords[[next_index]]
    generated_text <- c(generated_text,next_word)
    cat(paste0(" ",next_word))
    newrow=rep(0,nwords); newrow[match(next_word,allwords)]=1
    xinput[1,,]=rbind(xinput[1,2:maxlen,],newrow)
  }
  cat("\n\n")
}

