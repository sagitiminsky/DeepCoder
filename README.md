###########################################
###iter is the iterative implementation:###
###########################################
  to run this part please select the following parameters:
  s - number of sampling points from the fX(x) dist
  n_s - numebr of sampling points from the fN(n) dist
  my_ftol=1e-6 - the tolarence for the GD algo.
  my_maxIter=5 - maximum number of iteration for the GD
  selec - pre. defined destributions and other parameters
  
####################################
###proj is the DL implementation:###
####################################
  pre_train - select 1 if you need to prefit the model else 0
  pre_comb_model - select 1 if you need to combine models else select 0


  s = n_s - number of samples from the fX(x) and fN(n) distrebutions

  please select the number of nuerons in each layer
  NEURONS_LAYER_1 = 20
  NEURONS_LAYER_2 = 100
  NEURONS_LAYER_3 = 150
  NEURONS_LAYER_4 = 300
  NEURONS_LAYER_5 = 150
  NEURONS_LAYER_6 = 20

  select - pre. defined destributions and other parametes
  
  
##############
###HistEst####
##############
  
  if you are interested to use some other randomized distribution, please import the following: HistEst
  and use this instead of fX(x)
  
##########
###plot###
##########

please use plot to display graph from gathered data
