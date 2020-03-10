# DeepCoder
Analog Mappings for Communication in Real Time. \
In this project there is implementation of an iterative algorithm and deep learning model. The goal is to improve the iterative algorithm using deep learning.


## Iterative Algorithm
In [iter.py](iter.py) file you can find the implemetation of iterative algorithm. \
to run this part please select the following parameters: \
`s` - number of sampling points from the fX(x) distribution \
`n_s` - numebr of sampling points from the fN(n) distribution \
`my_ftol` - the tolarence for the GD algorithm \
`my_maxIter` - maximum number of iteration for the GD \
`select` - pre defined destributions and other parameters 

## Deep Learning Model
In [proj.py](proj.py) file you can find the implemetation of deep learning model. \
`pre_train` - select 1 if you need to prefit the model, else select 0 \
`pre_comb_model` - select 1 if you need to combine models, else select 0 \
`s` - number of samples from the fX(x) distribution \
`n_s` - number of samples from the fN(n) distribution \
please select the number of nuerons in each layer `NEURONS_LAYER_1`, `NEURONS_LAYER_2`, `NEURONS_LAYER_3`, `NEURONS_LAYER_4`, `NEURONS_LAYER_5`, `NEURONS_LAYER_6` \
`select` - pre defined destributions and other parameters 

### Other .py files
[HistEst.py](HistEst.py) - if you are interested to use some other randomized distribution, please import the following: HistEst and use this instead of fX(x) \
[plot.py](plot.py) - please use plot to display graph from gathered data 
