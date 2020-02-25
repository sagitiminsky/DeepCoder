################################################# IMPORTS ##############################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from matplotlib import cm
import datetime
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from pylab import *
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense,Concatenate
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from mpl_toolkits.mplot3d import axes3d
import tensorflow as tf
import tensorflow_probability as tfp
import torch
import random
import collections
from keras.layers import GaussianNoise
from shgo import shgo

import tensorflow_constrained_optimization as tfco

from tensorflow_probability.python.optimizer.linesearch.hager_zhang import hager_zhang

from tensorflow.python.util.all_util import remove_undocumented

########################################### CONSTANTS AND GLOBALS ###########################s###########################

BOUND_X = 7.0
BOUND_N = 4.0
G_graph = 0
H_graph = 1
NE = 10000
NE_G=NE_H=1 # number of iterations of model_comb in NE_COMB
pre_train=0
pre_comb_model=0


s = n_s = 15  # <- this has to be the same samples for the noise and signal

NEURONS_LAYER_1 = 20
NEURONS_LAYER_2 = 100
NEURONS_LAYER_3 = 150
NEURONS_LAYER_4 = 300
NEURONS_LAYER_5 = 150
NEURONS_LAYER_6 = 20

select = 0
""" {1}: 1:1 """


def spiral(t):
    a = 1
    return a * np.sign(t) * np.sqrt(np.sign(t) * t) * np.cos(t), a * np.sqrt(np.sign(t) * t) * np.sin(t)


def gaussian_distribution(x, exp, cov):
    if isinstance(x, float) or x.shape[0] == 1:
        return 1 / np.sqrt(2 * np.pi * cov) * np.exp(-(x - exp) ** 2 / 2 * cov)

    return 1 / np.sqrt((2 * np.pi) ** x.shape[0] * np.linalg.det(cov)) * np.exp(
        -1 / 2 * (x - exp).T @ np.linalg.inv(cov) @ (x - exp))


def switch(c):
    """select mode {1:1 , 1:2 , 2:1 , 2:2}"""
    h_linear = lambda x: np.asarray(x)
    if (c == 0):

        m = k = 1
        MU = np.array([0])
        SIGMA = np.array([[1]])
        dev_m = 1
        dev_k = 1
        my_fX = "gaussian_distribution(x, 0, 1)"
        my_fN = "gaussian_distribution(n, 0, 1)"
        g_init = "g(x)=x"
        fX = lambda x: gaussian_distribution(x, 0, 1)
        fN = lambda n: gaussian_distribution(n, 0, 1)
        g = lambda x: np.asarray(x)
        h_linear = lambda x: np.asarray(x)

        "fX:  integrate from -infty to infty  1/sqrt(2*pi)*e^(-1/2*(x)^2))  "
        """PWR : (1) : integrate from -infty to infty  x^2*1/sqrt(2*pi)*e^(-1/2*(x)^2)) """
        "h = y/2 : integrate from -infty to infty (x*( 1/sqrt(2*pi)*e^(-1/2*(x)^2)) * (1/sqrt(2pi) *e^(-(y-x)^2/2)) ) / integrate from -infty to infty (( 1/sqrt(2*pi)*e^(-1/2*(x)^2)) * (1/sqrt(2pi) *e^(-(y-x)^2/2)))"
        "mse  (0.5)=  integral_x=-infty^infty integral_n=-infty^infty (x-(x+n)/2)^2 * (1/sqrt(2*pi))* e^(-(x^2)/2)) * (1/sqrt(2*pi)) * e^(-(n^2)/2)) dx dn  "

    elif (c == 1):
        m = k = 1
        MU = np.array([0])
        SIGMA = np.array([[1]])
        dev_m = 1
        dev_k = 1
        fX = lambda x: 0.5 * (gaussian_distribution(x, 3, 1) + gaussian_distribution(x, -3, 1))
        fN = lambda n: gaussian_distribution(n, 0, 1)
        g = lambda x: np.asarray(x)

        my_fX = "0.5 * (gaussian_distribution(x, 3, 1) + gaussian_distribution(x, -3, 1))"
        my_fN = "gaussian_distribution(n, 0, 1)"
        g_init = "g(x)=x"

        "fX: integrate from -infty to infty 0.5*( 1/sqrt(2*pi)*e^(-1/2*(x+3)^2) +1/sqrt(2*pi)*e^(-1/2*(x-3)^2))"
        "POWER: (10) integrate from -infty to infty x^2*0.5*( 1/sqrt(2*pi)*e^(-1/2*(x+3)^2) +1/sqrt(2*pi)*e^(-1/2*(x-3)^2))"
        "h =(-3 + y + e^(3 y) (3 + y))/(2 (1 + e^(3 y))) :  integrate from -infty to infty (x*0.5*( 1/sqrt(2*pi)*e^(-1/2*(x+3)^2) +1/sqrt(2*pi)*e^(-1/2*(x-3)^2))  *1/sqrt(2pi) *e^(-(y-x)^2/2) ) / integrate from -infty to infty 0.5*( 1/sqrt(2*pi)*e^(-1/2*(x+3)^2) +1/sqrt(2*pi)*e^(-1/2*(x-3)^2))  *1/sqrt(2pi) *e^(-(y-x)^2/2)"
        """mse (0.61)"""
    if (c == 2):

        m = k = 1
        MU = np.array([0])
        SIGMA = np.array([[1]])
        dev_m = 1
        dev_k = 1
        my_fX = "gaussian_distribution(x, 0, 1)"
        my_fN = "gaussian_distribution(n, 0, 1)"
        g_init = "g(x)=x/3"
        fX = lambda x: gaussian_distribution(x, 0, 1)
        fN = lambda n: gaussian_distribution(n, 0, 1)
        g = lambda x: np.asarray(x / 3)

        "fX:  integrate from -infty to infty  1/sqrt(2*pi)*e^(-1/2*(x)^2))  "
        """PWR : (1) : integrate from -infty to infty  x^2*1/sqrt(2*pi)*e^(-1/2*(x/3)^2)) """
        "h = 3*y/10 : integrate from -infty to infty (x*( 1/sqrt(2*pi)*e^(-1/2*(x/3)^2)) * (1/sqrt(2pi) *e^(-(y-x/3)^2/2)) ) / integrate from -infty to infty (( 1/sqrt(2*pi)*e^(-1/2*(x)^2)) * (1/sqrt(2pi) *e^(-(y-x/3)^2/2)))"
        "mse  (0.9)=  integral_x=-infty^infty integral_n=-infty^infty (x-3(x/3+n)/10)^2 * (1/sqrt(2*pi))* e^(-(x^2)/2)) * (1/sqrt(2*pi)) * e^(-(n^2)/2)) dx dn  "

    elif (c == 3):
        m = 2
        k = 1
        MU = np.array([0])
        SIGMA = np.array([1])
        dev_m = 1
        dev_k = 1
        fX = lambda x: 0.5 * (gaussian_distribution(x, [1, 1], [[1, 0],
                                                                [0, 1]]) + gaussian_distribution(x, [-1, -1], [[1, 0],
                                                                                                               [0, 1]]))

        fN = lambda n: gaussian_distribution(n, 0, 1)

        """PT = 2 the MSE is 1.333 for g(X)=x[0]+x[1]"""

        def g(x):
            if (x[0] == 0): return 0
            return 1 + 1 * np.arctan(x[1] / x[0])

        my_fX = "0.5 * (gaussian_distribution(x, [1,1], [[1, 0],[0, 1]]) + gaussian_distribution(x, [[-1, -1],[1, 0]]))"
        my_fN = "gaussian_distribution(n, 0, 1)"
        g_init = "g(x[0],x[1])=1+1*np.arctan(x[1]/x[0])"


    elif (c == 4):
        m = 1
        k = 2
        dev_m = 1
        dev_k = 1
        MU = np.array([0, 0])
        SIGMA = np.array([[1, 0], [0, 1]])
        fN = lambda n: gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])
        fX = lambda x: 0.5 * (gaussian_distribution(x, 3, 1) + gaussian_distribution(x, -3, 1))
        g = lambda x: np.asarray(
            [1 * np.sign(x) * np.sqrt(np.sign(x) * x) * np.cos(x), 1 * np.sqrt(np.sign(x) * x) * np.sin(x)]).T

        """NOISE POWER  (1): integral_x=-infty^infty integral_y=-infty^infty 1/sqrt(2*pi)*e^(-1/2*(x)^2))  * 1/sqrt(2*pi)*e^(-1/2*(y)^2))  dx dy """

        my_fX = "0.5 * (gaussian_distribution(x, 3, 1) + gaussian_distribution(x, -3, 1))"
        my_fN = "gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])"
        g_init = "g(x)=np.asarray([1*np.sign(x)*np.sqrt(np.sign(x)*x)*np.cos(x) , 1*np.sqrt(np.sign(x)*x)*np.sin(x)]).T"



    elif (c == 5):
        m = 1
        k = 2
        dev_m = 1
        dev_k = 1
        MU = np.array([0, 0])
        SIGMA = np.array([[1, 0], [0, 1]])
        fN = lambda n: gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])
        fX = lambda x: 0.5 * (gaussian_distribution(x, 3, 1) + gaussian_distribution(x, -3, 1))
        g = lambda x: np.asarray([x, x]).T

        my_fX = "0.5 * (gaussian_distribution(x, 3, 1) + gaussian_distribution(x, -3, 1))"
        my_fN = "gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])"
        g_init = "g(x)=np.asarray([x,x]).T"

        "PWR: (20) integrate from -infty to infty 2*x^2*0.5*( 1/sqrt(2*pi)*e^(-1/2*(x+3)^2) +1/sqrt(2*pi)*e^(-1/2*(x-3)^2))"
        "fN:  integrate from -infty to infty  1/sqrt(2*pi)*e^(-1/2*(x)^2))  "
        """ N = integrate from -infty to infty x * 0.5*( 1/sqrt(2*pi)*e^(-1/2*(x+3)^2) +1/sqrt(2*pi)*e^(-1/2*(x-3)^2)  ) * 1/sqrt(2*pi)*e^(-1/2*(y - x)^2)  * 1/sqrt(2*pi)*e^(-1/2*(z - x)^2)  dx"""
        """ D = """
    elif (c == 6):
        m = 1
        k = 2
        dev_m = 1
        dev_k = 1
        MU = np.array([0, 0])
        SIGMA = np.array([[1, 0], [0, 1]])
        fN = lambda n: gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])
        fX = lambda x: gaussian_distribution(x, 0, 1)
        g = lambda x: np.asarray([x, x]).T

        my_fX = "gaussian_distribution(x, 0, 1)"
        my_fN = "gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])"
        g_init = "g(x)=np.asarray([x,x]).T"

        """fX:  integrate from -infty to infty  1/sqrt(2*pi)*e^(-1/2*(x)^2))"""
        "PWR: (2) integrate from -infty to infty 2*x^2*1/sqrt(2*pi)*e^(-1/2*(x)^2))"
        "fN:  integrate from -infty to infty  1/sqrt(2*pi)*e^(-1/2*(x)^2))  "
        """ h : ( (e^(1/3 (-y^2 + y z - z^2)) (y + z))/(6 sqrt(3) π) ) / (e^(1/3 (-y^2 + y z - z^2))/(2 sqrt(3) π)) """

        """mse :  """
    return fN, fX, g, m, k, MU, SIGMA, dev_m, dev_k, my_fX, my_fN, g_init, h_linear


def build_grid_and_spiral(m, k):
    if (m == 1 and k == 1):
        X = np.linspace(-BOUND_X * dev_m, BOUND_X * dev_m, s)
        N = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
        dx = X[1] - X[0]
        dn = N[1] - N[0]

    if (m == 1 and k == 2):
        X = np.linspace(-BOUND_X * dev_m, BOUND_X * dev_m, s)
        n_x = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
        n_y = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
        N_grid = np.asarray(np.meshgrid(n_x, n_y))
        N = np.vstack((np.ravel(N_grid[0].T), np.ravel(N_grid[0])))
        dx = X[1] - X[0]
        dn = max(N.T[1] - N.T[0]) ** k

    if (m == 2 and k == 1):
        N = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
        X_x = np.linspace(-BOUND_X * dev_m, BOUND_X * dev_m, s)
        X_y = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
        X_grid = np.asarray(np.meshgrid(X_x, X_y))
        X = np.vstack((np.ravel(X_grid[0].T), np.ravel(X_grid[0])))
        dn = N[1] - N[0]
        dx = max(X.T[1] - X.T[0]) ** m

    if (m == 2 and k == 2):
        n_x = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
        n_y = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
        N_grid = np.asarray(np.meshgrid(n_x, n_y))
        N = np.vstack((np.ravel(N_grid[0].T), np.ravel(N_grid[0])))

        X_x = np.linspace(-BOUND_X * dev_m, BOUND_X * dev_m, s)
        X_y = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
        X_grid = np.asarray(np.meshgrid(X_x, X_y))
        X = np.vstack((np.ravel(X_grid[0].T), np.ravel(X_grid[0])))

        dn = max(N.T[1] - N.T[0]) ** k
        dx = max(X.T[1] - X.T[0]) ** m

    ####SPIRAL####
    SPIRAL = np.asarray([[spiral(t)[0], spiral(t)[1], t] for t in np.linspace(-25, 25, s)])  # controls the spiral
    a = 1

    ##plot the spiral
    # plt.plot(SPIRAL.T[0],SPIRAL.T[1],color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)
    # plt.show()

    return X, N, dx, dn, SPIRAL


def vec_fX(X):
    return np.array([fX(x) for x in X.T])


def vec_fN(N):
    return np.array([fN(n) for n in N.T])



def H(y_hat):
    """calculate the optimal decoder"""

    def h_num_dx(y_hat, x):

        if (m == k == 1):
            return x * fX(x) * fN(y_hat - g(x)) * dx

    def h_den_dx(y_hat, x):
        if (m == k == 1):
            return fX(x) * fN(y_hat - g(x)) * dx

    def h(y_hat):
        Num = np.sum([h_num_dx(y_hat, x) for x in X.T])
        Den = np.sum([h_den_dx(y_hat, x) for x in X.T])
        return Num / Den

    return h(y_hat)





def plot(X, Y, type_graph,mse,pwr):
    if type_graph == G_graph:
        plt.title("Encoder G  " + str(m) + str('->') + str(k) + "    mse: " + str(mse) + "  PWR constraint: " + str(
            pwr) + "\n" + str(my_fX) + "\n" + str(my_fN) + "\n" + "PT:   " + str(
            PT) +"\n" + "Super epoches:  " + str(
            NE) + "\n" + "NN:  " + str(m) + ":" + str(NEURONS_LAYER_2) + ":" + str(
            NEURONS_LAYER_2) + ":" + str(NEURONS_LAYER_2) + ":" + str(NEURONS_LAYER_2) + ":" + str(k), y=0.95)

    if type_graph == H_graph:
        plt.title("Decoder H  " + str(m) + str('->') + str(k) + "   mse: " + str(mse) + "  PWR constraint: " + str(
            pwr) + "\n" + str(my_fX) + "\n" + str(my_fN) + "\n" + "PT:   " + str(
            PT) + "\n" + "Super epoches:  " + str(
            NE) + "\n" + "NN:  " + str(k) + ":" + str(NEURONS_LAYER_2) + ":" + str(
            NEURONS_LAYER_2) + ":" + str(NEURONS_LAYER_2) + ":" + str(NEURONS_LAYER_2) + ":" + str(m), y=0.95)

    if (k == m == 1):
        if (G_graph == type_graph):
            plt.scatter(X, Y.ravel(), color='green', linestyle='dashed', linewidth=3, marker='o')
            # plt.plot(X, Y.ravel(), color='green', linestyle='dashed', linewidth=3, marker='o')
        if (H_graph == type_graph):
            plt.scatter(X, Y.ravel(), color='green', linestyle='dashed', linewidth=3, marker='o')
            # plt.plot(X, Y.ravel(), color='green', linestyle='dashed', linewidth=3, marker='o')

    plt.show()

# Define value and gradient namedtuple
ValueAndGradient = collections.namedtuple('ValueAndGradient', ['x', 'f', 'df'])

def infinite_sequence(mse,pwr_const):
    return K.abs((K.min([mse + n * (pwr_const) for n in np.arange(-1, 1, 0.1)]) - mse) / (pwr_const))


def my_loss(y_true,y_pred, smooth, thresh,model):

    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]  # evaluation functions


    th=K.constant(thresh)
    x_input=K.constant(X)

    # outputs[0] - X        model.add(Dense(s, activation='relu', input_shape=(m,),name="input_g"))
    # outputs[1]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_1_g"))
    # outputs[2]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_2_g"))
    # outputs[3]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_3_g"))
    # outputs[4]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_4_g"))
    # outputs[5]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_5_g"))
    # outputs[6]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_6_g"))
    # outputs[7] - g(x)     model.add(Dense(k, activation='linear', name="output_g"))
    # outputs[8] - N        model.add(GaussianNoise(s,input_shape=(k,),name="noise"))
    # outputs[9] - Y_hat    model.add(Dense(s, activation='relu',input_shape=(k,), name="input_h"))
    # outputs[10]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_1_h"))
    # outputs[11]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_2_h"))
    # outputs[12]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_3_h"))
    # outputs[13]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_4_h"))
    # outputs[14]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_5_h"))
    # outputs[15]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_6_h"))
    # outputs[16] - X_hat    model.add(Dense(m, activation='linear', name="ouput_h"))

    # mse=K.sum(K.square(y_pred - y_true),axis=-1)

    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    mse = K.sum(K.square(y_pred - y_true), axis=-1)


    pwr_constraint=K.sum(K.square(outputs[7] + 0*y_pred),axis=-1)-th
    pwr_constraint_abs = K.abs(K.sum(K.square(outputs[7] + 0*y_pred),axis=-1) - th) # must have
    pwr_constraint_squared=K.square(K.sum(K.square(functors[7].outputs + 0*y_pred),axis=-1)-th)

    def value_and_gradients_function(x):
        mse = K.sum(K.square(functors[16].outputs - y_true), axis=-1)
        pwr_constraint = K.square(K.sum(K.square(functors[7].outputs), axis=-1) - th)
        return ValueAndGradient(x = x+K.constant(0)*pwr_constraint, f = x * pwr_constraint , df =  mse )

    # # Set initial step size.
    # step_size = tf.constant(0.01)
    # ls_result = tfp.optimizer.linesearch.hager_zhang(
    #     value_and_gradients_function_h, initial_step_size=step_size)
    # # # Evaluate the results.
    # with tf.Session() as session:
    #     results = session.run(ls_result)
    #     # Ensure convergence.
    #     assert results.converged
    #     # If the line search converged, the left and the right ends of the
    #     # bracketing interval are identical.
    #     assert results.left.x == results.right.x
    #     # Print the number of evaluations and the final step size.
    #     print("Final Step Size: %f, Evaluations: %d" % (results.left.x,
    #                                                     results.func_evals))

    # el=ls_result.left.x

    # lamb=tfp.optimizer.linesearch.hager_zhang(value_and_gradients_function,
    #                                      threshold_use_approximate_wolfe_condition=threshold_use_approximate_wolfe_condition,
    #                                      max_iterations=max_iterations,
    #                                      initial_step_size=tf.constant(initial_step_size))
    #
    #
    # middle=(lamb.left.x+lamb.right.x)/2

    # return mse*0+(pwr_constraint)
    #lamb.left is giving very high numbers for g(x)

    #Preform line search

    START=0
    END=10000
    MIDDLE=0.5



    # elems_small=np.arange(-MIDDLE, MIDDLE, DELTA, dtype=float32)
    # elems_big = np.concatenate((np.arange(-1, -0.5, 0.01, dtype=float32), np.arange(0.5, 1, 0.01, dtype=float32)))
    #
    elems_a = np.array( [0.1] , dtype=float32)
    elems_b = np.array( [0.9], dtype=float32)



    return tf.cond(tf.reduce_mean(pwr_constraint) < 0 ,
                   lambda: tf.map_fn(lambda x: mse , elems_a ) ,
                   lambda: tf.map_fn(lambda x: pwr_constraint ,elems_b))



PT = 1  # the power contraint
NUM_OF_EPOCHES=100
lr=1e-8
wd=0

def mean_squared_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.sum(K.square(y_pred - y_true), axis=-1)


def my_dice_loss(smooth, thresh,model):
    """Define wrapper function"""

    def dice(y_true, y_pred):
        return my_loss(y_true, y_pred, smooth, thresh,model)

    return dice


def baseline_model_g(m, k):
    model = Sequential()
    model.add(Dense(s, activation='relu', input_shape=(m,), name="input_g"))
    model.add(Dense(NEURONS_LAYER_1, activation='relu', name="layer_1_g"))
    model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_2_g"))
    model.add(Dense(NEURONS_LAYER_3, activation='relu', name="layer_3_g"))
    model.add(Dense(NEURONS_LAYER_4, activation='relu', name="layer_4_g"))
    model.add(Dense(NEURONS_LAYER_5, activation='relu', name="layer_5_g"))
    model.add(Dense(NEURONS_LAYER_6, activation='relu', name="layer_6_g"))
    model.add(Dense(k, activation='linear', name="output_g"))
    # model.add(GaussianNoise(k , name="noise_"))

    r_optimizer = Adam()  # best results for lr=0.00001,decay=0.01
    model.compile(optimizer=r_optimizer, loss=mean_squared_error)
    return model

def baseline_model_h(m, k):
    model = Sequential()
    model.add(GaussianNoise(k ,input_shape=(k,),name="noise"))
    model.add(Dense(s, activation='relu',name="input_h"))
    model.add(Dense(NEURONS_LAYER_1, activation='relu', name="layer_1_h"))
    model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_2_h"))
    model.add(Dense(NEURONS_LAYER_3, activation='relu', name="layer_3_h"))
    model.add(Dense(NEURONS_LAYER_4, activation='relu', name="layer_4_h"))
    model.add(Dense(NEURONS_LAYER_5, activation='relu', name="layer_5_h"))
    model.add(Dense(NEURONS_LAYER_6, activation='relu', name="layer_6_h"))
    model.add(Dense(m, activation='linear',name="output_h"))
    r_optimizer = Adam()
    model.compile(optimizer=r_optimizer, loss=mean_squared_error)
    return model

def baseline_model(m, k):
    model = Sequential()
    model.add(Dense(s, activation='relu', input_shape=(m,),name="input_g"))
    model.add(Dense(NEURONS_LAYER_1, activation='relu', name="layer_1_g"))
    model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_2_g"))
    model.add(Dense(NEURONS_LAYER_3, activation='relu', name="layer_3_g"))
    model.add(Dense(NEURONS_LAYER_4, activation='relu', name="layer_4_g"))
    model.add(Dense(NEURONS_LAYER_5, activation='relu', name="layer_5_g"))
    model.add(Dense(NEURONS_LAYER_6, activation='relu', name="layer_6_g"))
    model.add(Dense(k,activation='linear', name="output_g"))
    model.add(GaussianNoise(k,name="noise"))
    model.add(Dense(s,name="input_h"))
    model.add(Dense(NEURONS_LAYER_1, activation='relu', name="layer_1_h"))
    model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_2_h"))
    model.add(Dense(NEURONS_LAYER_3, activation='relu', name="layer_3_h"))
    model.add(Dense(NEURONS_LAYER_4, activation='relu', name="layer_4_h"))
    model.add(Dense(NEURONS_LAYER_5, activation='relu', name="layer_5_h"))
    model.add(Dense(NEURONS_LAYER_6, activation='relu', name="layer_6_h"))
    model.add(Dense(m ,activation='linear', name="ouput_h"))
    model_my_loss = my_dice_loss(smooth=1e-5, thresh=PT,model=model)
    model.compile(optimizer=Adam(), loss=mean_squared_error)
    return model









fN, fX, g, m, k, MU, SIGMA, dev_m, dev_k, my_fX, my_fN, g_init, h_linear, = switch(select)

X, N, dx, dn, SPIRAL = build_grid_and_spiral(m, k)

if (m == 2 and k == 1):
    DIST = cdist(X.T, SPIRAL.T[:2].T, metric="euclidean")
    SPIRAL_index = np.argmin(DIST, axis=1)

GX = np.array([g(x) for x in X.T])
Y_hat = np.asarray([gx + n for gx in GX.T for n in N.T]).T
H_opt = np.asarray([H(y_hat) for y_hat in Y_hat.T]).T
Y = np.vstack((np.repeat(GX, n_s), H_opt))
RAW_DATA = np.vstack((Y, np.tile(N, s)))
RAW_DATA = np.vstack((RAW_DATA, np.repeat(X, n_s)))
RAW_DATA = np.vstack((RAW_DATA, np.repeat(GX, n_s) + np.tile(N, s)))
RAW_DATA = np.vstack((RAW_DATA, np.array([H(y_hat) for y_hat in RAW_DATA[4]])))
RAW_DATA = np.vstack((RAW_DATA, np.array([fX(x) for x in RAW_DATA[3]])))
RAW_DATA = np.vstack((RAW_DATA, np.array([fN(n) for n in RAW_DATA[2]])))

"""RAW DATA"""
"""
{0} - g(x)
{1} - h(g(x)+n)
{2} - N
{3} - X
{4} - y_hat=g(x)+n
{5} - H(y_hat)
{6} - fX(x)
{7} - fN(n)

"""
short_fX = np.array([fX(x) for x in X.T])
short_fN = np.array([fN(n) for n in N.T])





# PRE_LEARNING
SHORT_RAW_DATA = np.vstack((X, GX))
""" SHORT RAW DATA"""
"""
{0} - X
{1} - g(X)
"""

predictions_g = GX

# Y_hat = np.asarray([gx + n for gx in GX.T for n in N.T]).T

Y_hat= GX + np.random.normal(0, 1, 1) * np.ones(s)
H_y_hat = np.asarray([H(y_hat) for y_hat in Y_hat.T]).T

predictions_h=H_y_hat


if pre_train:
    # encoder net - g
    model_g = baseline_model_g(m, k)

    # [0] model.add(Dense(s, activation='relu', input_shape=(m,), name="input_g"))
    # [1] model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_1_g"))
    # [2] model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_2_g"))
    # [3] model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_3_g"))
    # [4] model.add(Dense(k, activation='linear', name="output_g"))

    # decoder net - h
    model_h = baseline_model_h(m, k)

    # [0] model.add(Dense(s, activation='relu', input_shape=(k,),name="input_h"))
    # [1] model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_1_h"))
    # [2] model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_2_h"))
    # [3] model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_3_h"))
    # [4] model.add(Dense(m, activation='linear',name="output_h"))




    ## encoder - g(x)
    model_g.fit(X, GX, nb_epoch=NE, sample_weight=short_fX,verbose='0')
    # long_predictions_g=model_g.predict(RAW_DATA[3].T).T

    get_g_output_layer = K.function([model_g.layers[0].input], [model_g.layers[7].output])
    # get_y_hat_output_layer = K.function([model_g.layers[8].input], [model_g.layers[8].output])

    predictions_g=np.array(get_g_output_layer([X.reshape(s,k)])[0]).T[0]
    # y_hat = np.array(get_y_hat_output_layer([predictions_g.reshape(s, k)])[0]).T[0]

    get_h_output_layer = K.function([model_h.layers[1].input], [model_h.layers[8].output])
    # layer_h_output = get_h_output_layer([[[y_hat]]])[0]

    model_h.fit(GX, X, nb_epoch=NE, sample_weight=short_fX,verbose='0')

    # predictions_h = model_h.predict(GX).T[0]

    # print(predictions_g)
    # print(GX)
    #
    # print("------------")
    #
    # print(predictions_h)
    # print(H_y_hat)
    #
    print("-----------\n")

    mse_strt = np.sum([(x - H(g(x) + n)) ** 2 * fX(x) * fN(n) * dx * dn for x in X.T for n in N.T])
    pwr_strt = np.sum([g(x) ** 2 * fX(x) * dx for x in X.T])

    print("mse analytic: " + str(mse_strt))
    print("pwr analytic: " + str(pwr_strt))

    print("-----------\n")


    mse_predictions=np.sum([(x - get_h_output_layer([[[get_g_output_layer([[[x]]])[0][0][0]+n]]])[0][0][0]) ** 2 * fX(x) * fN(n) * dx * dn for x in X.T for n in N.T])
    pwr_predctions  = np.sum([get_g_output_layer([[[x]]])[0][0][0] ** 2 * fX(x) * dx for x in X.T])


    print("mse learning g h - separated: "+str(mse_predictions))
    print("pwr learning g h - separated: "+str(pwr_predctions))

    print("-----------\n")
    type_graph = G_graph
    plot(X, predictions_g, type_graph,mse_predictions,pwr_predctions)

    type_graph = H_graph
    plot(Y_hat, predictions_h, type_graph,mse_predictions,pwr_predctions)




    # SAVE
    PATH_H = "model_h"
    PATH_G = "model_g"

    model_g.save(PATH_G)
    model_h.save(PATH_H)


PATH_H = "model_h"
PATH_G = "model_g"


model_g = baseline_model_g(m, k)
model_h = baseline_model_h(m, k)
comb_model=baseline_model(m,k)



if pre_comb_model:
    print("COMB MODEL")


    # model_g.load_weights(PATH_G,by_name=True)
    # model_h.load_weights(PATH_H,by_name=True)


    # comb_model.load_weights(PATH_G,by_name=True)
    # comb_model.load_weights(PATH_H,by_name=True)
    # comb_model.fit(X.reshape(s,m), X.reshape(s,m),batch_size=1, nb_epoch=NE, verbose=2,sample_weight=short_fX)

    for epoche in range(NE): # train a bit to get the h function from learning
        print("# of epoche: "+ str(epoche))
        comb_model.fit(X.reshape(s,m), X.reshape(s,m),batch_size=1, nb_epoch=NE_G, verbose=2, sample_weight = short_fX)
        comb_model.load_weights(PATH_G, by_name=True)

    # outputs[0] - X        model.add(Dense(s, activation='relu', input_shape=(m,),name="input_g"))
    # outputs[1]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_1_g"))
    # outputs[2]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_2_g"))
    # outputs[3]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_3_g"))
    # outputs[4]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_4_g"))
    # outputs[5]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_5_g"))
    # outputs[6]            model.add(Dense(s, activation='relu', name="layer_6_g"))
    # outputs[7] - g(x)     model.add(Dense(k, activation='linear', name="output_g"))
    # outputs[8] - N        model.add(GaussianNoise(k,input_shape=(k,),name="noise"))
    # outputs[9] - Y_hat    model.add(Dense(s,  name="input_h"))
    # outputs[10]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_1_h"))
    # outputs[11]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_2_h"))
    # outputs[12]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_3_h"))
    # outputs[13]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_4_h"))
    # outputs[14]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_5_h"))
    # outputs[15]            model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_6_h"))
    # outputs[16] - X_hat    model.add(Dense(m, activation='linear', name="ouput_h"))


    get_g_output_layer_comb = K.function([comb_model.layers[0].input], [comb_model.layers[7].output])
    # layer_g_output = get_g_output_layer([[[x]]])[0]

    #GaussianNoise will take effect only in training
    get_y_hat_output_layer_comb = K.function([comb_model.layers[8].input], [comb_model.layers[8].output])
    # layer_y_hat_output = get_y_hat_output_layer([[[x]]])[0]

    get_h_output_layer_comb = K.function([comb_model.layers[9].input], [comb_model.layers[16].output])
    # layer_h_output = get_h_output_layer([[[y_hat]]])[0]

    # predictions_h=comb_model.predict(RAW_DATA[3]).T[0]
    # predictions_g=comb_model.predict(X).T[0]

    predictions_g=np.array(get_g_output_layer_comb([X.reshape(s,m)])[0]).T[0]
    predictions_y_hat=np.array(get_y_hat_output_layer_comb([predictions_g.reshape(s,m)])[0]).T[0]
    predictions_h=np.array(get_h_output_layer_comb([predictions_y_hat.reshape(s,m)])[0]).T[0]


    # a=X-get_h_output_layer_comb([(predictions_g+N).reshape(s,m)])[0].T[0]
    # P=short_fX.reshape(s,m) @ short_fN.reshape(k,s)
    # mse_predictions_new=a@a.T * np.sum(P) * dx * dn
    # pwr_predctions_new=predictions_g*short_fX@predictions_g.T*dx



    mse_predictions=np.sum([(x - get_h_output_layer_comb([[[get_g_output_layer_comb([[[x]]])[0][0][0]+n]]])[0][0][0]) ** 2 * fX(x) * fN(n) * dx * dn for x in X.T for n in N.T])
    pwr_predctions  = np.sum([get_g_output_layer_comb([[[x]]])[0][0][0] ** 2 * fX(x) * dx for x in X.T])

    print("mse learning g h - combined: "+str(mse_predictions))
    print("pwr learning g h - combined: "+str(pwr_predctions))


    # print("\n\nNEW WAY:\n")
    # print("mse learning g h - combined: " + str(mse_predictions_new))
    # print("pwr learning g h - combined: " + str(pwr_predctions_new))


    type_graph = G_graph
    plot(X, predictions_g, type_graph,mse_predictions,pwr_predctions)

    type_graph = H_graph
    plot(predictions_y_hat, predictions_h, type_graph,mse_predictions,pwr_predctions)

    # SAVE
    COMB_MODEL = "comb_model"
    comb_model.save(COMB_MODEL)



get_g_output_layer_comb = K.function([comb_model.layers[0].input], [comb_model.layers[7].output])
# layer_g_output = get_g_output_layer([[[x]]])[0]

#GaussianNoise will take effect only in training
get_y_hat_output_layer_comb = K.function([comb_model.layers[8].input], [comb_model.layers[8].output])
# layer_y_hat_output = get_y_hat_output_layer([[[x]]])[0]

get_h_output_layer_comb = K.function([comb_model.layers[9].input], [comb_model.layers[16].output])




#train some more:


#https://stackoverflow.com/questions/56943862/using-neural-networks-in-optimization-problems
#https://github.com/google-research/tensorflow_constrained_optimization/blob/master/README.md

COMB_MODEL = "comb_model"
comb_model.load_weights(COMB_MODEL, by_name=True)

mse_predictions=np.sum([(x - get_h_output_layer_comb([[[get_g_output_layer_comb([[[x]]])[0][0][0]+n]]])[0][0][0]) ** 2 * fX(x) * fN(n) * dx * dn for x in X.T for n in N.T])
pwr_predctions  = np.sum([get_g_output_layer_comb([[[x]]])[0][0][0] ** 2 * fX(x) * dx for x in X.T])


# now change the loss function and train some more
model_my_loss = my_dice_loss(smooth=1e-5, thresh=PT,model=comb_model)
comb_model.compile(optimizer=Adam(lr=lr,decay=wd),loss=model_my_loss) #best lr=1e-5,decay=0.1


comb_model.fit(X , X ,batch_size=1, nb_epoch=NUM_OF_EPOCHES ,verbose=2, sample_weight = short_fX)

#0  model.add(Dense(s, activation='relu', input_shape=(m,), name="input_g"))
#1  model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_1_g"))
#2  model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_2_g"))
#3  model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_3_g"))
#4  model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_4_g"))
#5  model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_5_g"))
#6  model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_6_g"))
#7  model.add(Dense(k, activation='linear', name="output_g_"))
#8  model.add(GaussianNoise(k, name="noise"))
#9  model.add(Dense(s, name="input_h_"))
#10 model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_1_h"))
#11 model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_2_h"))
#12 model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_3_h"))
#13 model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_4_h"))
#14 model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_5_h"))
#15 model.add(Dense(NEURONS_LAYER_2, activation='relu', name="layer_6_h"))
#16 model.add(Dense(m, activation='linear', name="ouput_h_"))




predictions_g=np.array(get_g_output_layer_comb([X.reshape(s,m)])[0]).T[0]
predictions_y_hat=np.array(get_y_hat_output_layer_comb([predictions_g.reshape(s,m)])[0]).T[0]
predictions_h=np.array(get_h_output_layer_comb([predictions_y_hat.reshape(s,m)])[0]).T[0]


mse_predictions=np.sum([(x - get_h_output_layer_comb([[[get_g_output_layer_comb([[[x]]])[0][0][0]+n]]])[0][0][0]) ** 2 * fX(x) * fN(n) * dx * dn for x in X.T for counter_n,n in enumerate(N.T)])
pwr_predctions  = np.sum([get_g_output_layer_comb([[[x]]])[0][0][0] ** 2 * fX(x) * dx for counter, x in enumerate(X.T)])

print("mse: "+str(mse_predictions))
print("pwr: "+str(pwr_predctions))

type_graph = G_graph
plot(X, predictions_g, type_graph,mse_predictions,pwr_predctions)

type_graph = H_graph
plot(Y_hat, predictions_h, type_graph,mse_predictions,pwr_predctions)


