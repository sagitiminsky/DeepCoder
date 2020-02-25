################################################# IMPORTS ##############################################################

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from matplotlib import cm
import datetime
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
########################################### CONSTANTS AND GLOBALS ######################################################
G=1
H=0
i=0
P_T=1  # the power contraint

BOUND_X=7.0
BOUND_N=4.0
################################################# NOTES ################################################################
""" CSNR = E[y^2] / E[N^2] """
""" SNR = E[x^2] / MSE """
s=10
n_s=7
my_ftol=1e-6
my_maxIter=5
select=0
""" {1}: 1:1 , {3}: 2:1  , {4}: 1:2  , {7}: 2:2"""
c=2
pos=2
"""{0 - linear encoder with linear decoder / Archm. Spiral / Inverse Archm. / VQ + channel coding
         ; 1 -Linear encoder with optimal decoder / Archm. Spiral  + optiaml decoder / Linear encoder + optimal decoder}
         ; 2 - proposed mapping 
         """

def spiral(t):
    a=1
    return a*np.sign(t)*np.sqrt(np.sign(t)*t)*np.cos(t) , a*np.sqrt(np.sign(t)*t)*np.sin(t)


def gaussian_distribution(x,exp, cov):
    if isinstance(x,float) or x.shape[0]==1:
        return 1 / np.sqrt(2 * np.pi * cov) * np.exp(-(x - exp) ** 2 / 2 * cov)

    return 1 / np.sqrt((2 * np.pi)**x.shape[0] * np.linalg.det(cov)) * np.exp(-1/2 * (x-exp).T @ np.linalg.inv(cov) @ (x-exp))


def plot(X_plot,Y_plot,G_H):
    # giving a title to my graph
    global G_opt

    if(k==m==1):
        if G_H == G:
            plt.title("Encoder G:Rm->Rk  mse:  " + str(mse_optimize(G_opt)) + "  PWR constraint:  " + str(
                PWR_constraint(G_opt)))
        else:
            plt.title("Decoder H:Rk->Rm  mse:  " + str(mse_optimize(G_opt)) + "  PWR constraint:  " + str(
                PWR_constraint(G_opt)))

        if(G_H==G):
            plt.plot(X_plot,Y_plot ,color='green', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)
        if(G_H==H):
            plt.scatter(X_plot,Y_plot)

    elif m==1 and k==2:
        if(G_H==G):

            G0 = np.array([gx_0 for gx_0 in G_opt[::2]])
            G1 = np.array([gx_1 for gx_1 in G_opt[1::2]])

            fig1 = plt.figure()
            plt.title("Encoder G:Rm->Rk  mse:  " + str(mse_optimize(G_opt)) + "  PWR constraint:  " + str(PWR_constraint(G_opt)))
            plt.scatter(G0, G1, c=X_plot, cmap=cm.coolwarm,s=12)
            plt.colorbar()

            fig2 = plt.figure()
            plt.title("Encoder G:Rm->Rk  mse:  " + str(mse_optimize(G_opt)) + "  PWR constraint:  " + str(PWR_constraint(G_opt)))
            ax = fig2.gca(projection='3d')
            ax.scatter(G0, G1, X_plot, alpha=0.8, c='blue', s=12)

        if G_H==H:
            fig1 = plt.figure()
            plt.title("Decoder H:Rk->Rm  mse:  " + str(mse_optimize(G_opt)) + "  PWR constraint:  " + str(PWR_constraint(G_opt)))
            plt.scatter(X_plot[0], X_plot[1], c=Y_plot, cmap=cm.coolwarm,s=12)


            fig2 = plt.figure()
            plt.title("Decoder H:Rk->Rm  mse:  " + str(mse_optimize(G_opt)) + "  PWR constraint:  " + str(PWR_constraint(G_opt)))
            ax = fig2.gca(projection='3d')
            ax.scatter(X_plot[0], X_plot[1], Y_plot, alpha=0.8, c='blue', s=12)

    elif m==2 and k==1:
        if(G_H==G):


            X_grid = np.asarray(np.meshgrid(X_plot, X_plot))
            X = np.vstack((np.ravel(X_grid[0].T), np.ravel(X_grid[0])))

            g_plot=Y_plot

            fig1 = plt.figure()
            plt.title("Encoder G:Rm->Rk  mse:  " + str(mse_optimize(G_opt)) + "  PWR constraint:  " + str(PWR_constraint(G_opt)))
            plt.scatter(X[0],X[1], c=g_plot, cmap=cm.coolwarm,s=12)
            plt.colorbar()



            fig2 = plt.figure()
            plt.title("Encoder G:Rm->Rk  mse:  " + str(mse_optimize(G_opt)) + "  PWR constraint:  " + str(PWR_constraint(G_opt)))
            ax = fig2.gca(projection='3d')
            ax.scatter(X[0],X[1], g_plot, alpha=0.8, c='blue', s=12)



        if(G_H==H):



            fig1 = plt.figure()
            plt.title("Decoder H:Rk->Rm  mse:  " + str(mse_optimize(G_opt)) + "  PWR constraint:  " + str(PWR_constraint(G_opt)))
            plt.scatter(Y_plot.T[0], Y_plot.T[1], c=X_plot, cmap=cm.coolwarm,s=12)
            plt.colorbar()


            fig2 = plt.figure()
            plt.title("Decoder H:Rk->Rm  mse:  " + str(mse_optimize(G_opt)) + "  PWR constraint:  " + str(PWR_constraint(G_opt)))
            ax = fig2.gca(projection='3d')
            ax.scatter(Y_plot.T[0],Y_plot.T[1], X_plot, alpha=0.8, c='blue', s=12)


    elif m == 2 and k == 2:
        pass

    plt.show()


def switch(c):
    h_linear = lambda x: np.asarray(x)
    if(c==0):

        m=k=1
        MU = np.array([0])
        SIGMA = np.array([[1]])
        dev_m= 1
        dev_k = 1
        my_fX="gaussian_distribution(x, 0, 1)"
        my_fN = "gaussian_distribution(n, 0, 1)"
        g_init="g(x)=x"
        fX = lambda x:  gaussian_distribution(x, 0, 1)
        fN = lambda n: gaussian_distribution(n, 0, 1)
        g = lambda x : np.asarray(x)
        h_linear=lambda x: np.asarray(x)

        "fX:  integrate from -infty to infty  1/sqrt(2*pi)*e^(-1/2*(x)^2))  "
        """PWR : (1) : integrate from -infty to infty  x^2*1/sqrt(2*pi)*e^(-1/2*(x)^2)) """
        "h = y/2 : integrate from -infty to infty (x*( 1/sqrt(2*pi)*e^(-1/2*(x)^2)) * (1/sqrt(2pi) *e^(-(y-x)^2/2)) ) / integrate from -infty to infty (( 1/sqrt(2*pi)*e^(-1/2*(x)^2)) * (1/sqrt(2pi) *e^(-(y-x)^2/2)))"
        "mse  (0.5)=  integral_x=-infty^infty integral_n=-infty^infty (x-(x+n)/2)^2 * (1/sqrt(2*pi))* e^(-(x^2)/2)) * (1/sqrt(2*pi)) * e^(-(n^2)/2)) dx dn  "

    elif (c==1):
        m=k=1
        MU = np.array([0])
        SIGMA = np.array([[1]])
        dev_m = 1
        dev_k = 1
        fX = lambda x: 0.5 * (gaussian_distribution(x, 3, 1) + gaussian_distribution(x, -3, 1))
        fN = lambda n: gaussian_distribution(n, 0, 1)
        g=lambda x: np.asarray(x)

        my_fX = "0.5 * (gaussian_distribution(x, 3, 1) + gaussian_distribution(x, -3, 1))"
        my_fN = "gaussian_distribution(n, 0, 1)"
        g_init = "g(x)=x"

        "fX: integrate from -infty to infty 0.5*( 1/sqrt(2*pi)*e^(-1/2*(x+3)^2) +1/sqrt(2*pi)*e^(-1/2*(x-3)^2))"
        "POWER: (10) integrate from -infty to infty x^2*0.5*( 1/sqrt(2*pi)*e^(-1/2*(x+3)^2) +1/sqrt(2*pi)*e^(-1/2*(x-3)^2))"
        "h =(-3 + y + e^(3 y) (3 + y))/(2 (1 + e^(3 y))) :  integrate from -infty to infty (x*0.5*( 1/sqrt(2*pi)*e^(-1/2*(x+3)^2) +1/sqrt(2*pi)*e^(-1/2*(x-3)^2))  *1/sqrt(2pi) *e^(-(y-x)^2/2) ) / integrate from -infty to infty 0.5*( 1/sqrt(2*pi)*e^(-1/2*(x+3)^2) +1/sqrt(2*pi)*e^(-1/2*(x-3)^2))  *1/sqrt(2pi) *e^(-(y-x)^2/2)"


        """
        x = -infty ^ infty
        integral_n = -infty ^ infty(x - h(g(x) + n)) ^ 2 * f(x) * n(n) * dx * dn
        g(x) = x, h(x) = (-3 + x + e ^ (3 x)(3 + x)) / (2(1 + e ^ (3 x))), f(x) = 1 / sqrt(2 * pi) * e ^ (
                    -1 / 2 * (x) ^ 2), n(n) = 1 / sqrt(2 * pi) * e ^ (-1 / 2 * (n) ^ 2)"""

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
        g = lambda x: np.asarray(x/3)

        "fX:  integrate from -infty to infty  1/sqrt(2*pi)*e^(-1/2*(x)^2))  "
        """PWR : (1) : integrate from -infty to infty  x^2*1/sqrt(2*pi)*e^(-1/2*(x/3)^2)) """
        "h = 3*y/10 : integrate from -infty to infty (x*( 1/sqrt(2*pi)*e^(-1/2*(x/3)^2)) * (1/sqrt(2pi) *e^(-(y-x/3)^2/2)) ) / integrate from -infty to infty (( 1/sqrt(2*pi)*e^(-1/2*(x)^2)) * (1/sqrt(2pi) *e^(-(y-x/3)^2/2)))"
        "mse  (0.9)=  integral_x=-infty^infty integral_n=-infty^infty (x-3(x/3+n)/10)^2 * (1/sqrt(2*pi))* e^(-(x^2)/2)) * (1/sqrt(2*pi)) * e^(-(n^2)/2)) dx dn  "

    elif (c==3):
        m=2
        k=1
        MU = np.array([0])
        SIGMA = np.array([1])
        dev_m = 1
        dev_k = 1
        fX = lambda x: 0.5 * (gaussian_distribution(x, [1, 1], [[1, 0],
                                                                [0, 1]]) + gaussian_distribution(x, [-1, -1], [[1, 0],
                                                                                                               [0, 1]]))

        fN = lambda n: gaussian_distribution(n, 0, 1)

        """P_T = 2 the MSE is 1.333 for g(X)=x[0]+x[1]"""
        def g(x):
            if (x[0]==0): return 0
            return 1+1*np.arctan(x[1]/x[0])

        my_fX = "0.5 * (gaussian_distribution(x, [1,1], [[1, 0],[0, 1]]) + gaussian_distribution(x, [[-1, -1],[1, 0]]))"
        my_fN = "gaussian_distribution(n, 0, 1)"
        g_init = "g(x[0],x[1])=1+1*np.arctan(x[1]/x[0])"

        """E[x^2]=4  integral_x=-infty^infty integral_y=-infty^infty  0.5*(x^2+y^2)*(  (1 /( 2 * pi) * (e ^ (-1 / 2 * ((x-1) ^ 2+(y-1)^2)))) + (1 / (2 * pi) * (e ^ (-1 / 2 * ((x+1) ^ 2+(y+1)^2)))) )dx dy) """
        """E[n^2] (1) = integral_n=-infty^infty  (n^2* 1 / sqrt(2 * pi) * (e ^ (-1 / 2 * (n) ^ 2) )))dn """


    elif (c==4):
        m=1
        k=2
        dev_m = 1
        dev_k = 1
        MU = np.array([0, 0])
        SIGMA = np.array([[1, 0], [0, 1]])
        fN = lambda n: gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])
        fX = lambda x: 0.5 * (gaussian_distribution(x, 3, 1) + gaussian_distribution(x, -3, 1))
        g = lambda x: np.asarray([1*np.sign(x)*np.sqrt(np.sign(x)*x)*np.cos(x) , 1*np.sqrt(np.sign(x)*x)*np.sin(x)]).T

        """NOISE POWER  (1): E[n^2] (1) = integral_x=-infty^infty integral_y=-infty^infty  ((x^2* 1 / sqrt(2 * pi) * (e ^ (-1 / 2 * (x) ^ 2)))* (y^2* 1 / sqrt(2 * pi) * (e ^ (-1 / 2 * (y) ^ 2) )))dx dy """

        my_fX = "0.5 * (gaussian_distribution(x, 3, 1) + gaussian_distribution(x, -3, 1))"
        my_fN = "gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])"
        g_init = "g(x)=np.asarray([1*np.sign(x)*np.sqrt(np.sign(x)*x)*np.cos(x) , 1*np.sqrt(np.sign(x)*x)*np.sin(x)]).T"



    elif (c==5):
        m=1
        k=2
        dev_m = 1
        dev_k = 1
        MU = np.array([0, 0])
        SIGMA = np.array([[1, 0], [0, 1]])
        fN = lambda n: gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])
        fX = lambda x: 0.5 * (gaussian_distribution(x, 3, 1) + gaussian_distribution(x, -3, 1))
        g = lambda x: np.asarray([x,x]).T


        my_fX = "0.5 * (gaussian_distribution(x, 3, 1) + gaussian_distribution(x, -3, 1))"
        my_fN = "gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])"
        g_init = "g(x)=np.asarray([x,x]).T"


        "PWR: (20) integrate from -infty to infty 2*x^2*0.5*( 1/sqrt(2*pi)*e^(-1/2*(x+3)^2) +1/sqrt(2*pi)*e^(-1/2*(x-3)^2))"
        "fN:  integrate from -infty to infty  1/sqrt(2*pi)*e^(-1/2*(x)^2))  "
        """ N = integrate from -infty to infty x * 0.5*( 1/sqrt(2*pi)*e^(-1/2*(x+3)^2) +1/sqrt(2*pi)*e^(-1/2*(x-3)^2)  ) * 1/sqrt(2*pi)*e^(-1/2*(y - x)^2)  * 1/sqrt(2*pi)*e^(-1/2*(z - x)^2)  dx"""
        """ D = """
    elif (c==6):
        m=1
        k=2
        dev_m = 1
        dev_k = 1
        MU = np.array([0, 0])
        SIGMA = np.array([[1, 0], [0, 1]])
        fN = lambda n: gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])
        fX = lambda x:  gaussian_distribution(x, 0, 1)
        g = lambda x: np.asarray([x,x]).T

        my_fX = "gaussian_distribution(x, 0, 1)"
        my_fN = "gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])"
        g_init = "g(x)=np.asarray([x,x]).T"


        """fX:  integrate from -infty to infty  1/sqrt(2*pi)*e^(-1/2*(x)^2))"""
        "PWR: (2) integrate from -infty to infty 2*x^2*1/sqrt(2*pi)*e^(-1/2*(x)^2))"
        "fN:  integrate from -infty to infty  1/sqrt(2*pi)*e^(-1/2*(x)^2))  "
        """ h : ( (e^(1/3 (-y^2 + y z - z^2)) (y + z))/(6 sqrt(3) π) ) / (e^(1/3 (-y^2 + y z - z^2))/(2 sqrt(3) π)) """

    elif (c == 7):
        m = 2
        k = 2
        dev_m = 1
        dev_k = 1
        MU = np.array([0, 0])
        SIGMA = np.array([[1, 0], [0, 1]])
        fN = lambda n: gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])
        fX = lambda x: 0.5 * (gaussian_distribution(x, [1, 1], [[1, 0],
                                                                [0, 1]]) + gaussian_distribution(x, [-1, -1], [[1, 0],
                                                                                                               [0, 1]]))

        def g(x):
            return np.array([x[0],x[1]])

        my_fX = "gaussian_distribution(x, 0, 1)"
        my_fN = "gaussian_distribution(n, [0, 0], [[1, 0], [0, 1]])"
        g_init = "g(x)=np.asarray([x,x]).T"

    return fN,fX,g,m,k,MU,SIGMA,dev_m,dev_k,my_fX,my_fN,g_init,h_linear



fN,fX,g,m,k,MU,SIGMA,dev_m,dev_k,my_fX,my_fN,g_init,h_linear,=switch(select)

X = np.linspace(-BOUND_X * dev_m, BOUND_X * dev_m, s)
N = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
dx = abs(X[1] - X[0])
dn = abs(N[1] - N[0])

# if(m==1 and k==1):
#     dx=X[1]-X[0]
#     dn = N[1] - N[0]
#
# if (m==1 and k==2):
#     X = np.linspace(-BOUND_X * dev_m, BOUND_X * dev_m, s)
#     n_x=np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
#     n_y = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
#     N_grid=np.asarray(np.meshgrid(n_x,n_y))
#     N=np.vstack((np.ravel(N_grid[0].T),np.ravel(N_grid[0])))
#     dx = X[1] - X[0]
#     dn = max(N.T[1] - N.T[0])
#
#
# if (m==2 and k==1):
#     N = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
#     X_x = np.linspace(-BOUND_X * dev_m, BOUND_X * dev_m, s)
#     X_y = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
#     X_grid = np.asarray(np.meshgrid(X_x, X_y))
#     X = np.vstack((np.ravel(X_grid[0].T), np.ravel(X_grid[0])))
#     dn = N[1] - N[0]
#     dx = max(X.T[1] - X.T[0])
#
# if (m==2 and k==2):
#     n_x = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
#     n_y = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
#     N_grid = np.asarray(np.meshgrid(n_x, n_y))
#     N = np.vstack((np.ravel(N_grid[0].T), np.ravel(N_grid[0])))
#
#     X_x = np.linspace(-BOUND_X * dev_m, BOUND_X * dev_m, s)
#     X_y = np.linspace(-BOUND_N * dev_k, BOUND_N * dev_k, n_s)
#     X_grid = np.asarray(np.meshgrid(X_x, X_y))
#     X = np.vstack((np.ravel(X_grid[0].T), np.ravel(X_grid[0])))
#     dn = max(N.T[1] - N.T[0])
#     dx = max(X.T[1] - X.T[0])

####SPIRAL####
InverseSPIRAL={} #2 to 1
SPIRAL={} # 1 to 2

for t in np.linspace(-10,10,1000):
    xy_point=(spiral(t)[0], spiral(t)[1])
    SPIRAL[t] = xy_point
    InverseSPIRAL[xy_point]=t


##plot the spiral
# plt.title("Spiral mapping 2:1")
# plt.scatter(np.array(list(SPIRAL.values())).T[0],np.array(list(SPIRAL.values())).T[1], c=list(SPIRAL.keys()), cmap=cm.coolwarm, s=12)
# plt.colorbar()
# plt.show()

##plot the Inversespiral
# plt.title("Inverse Spiral mapping 1:2")
# plt.scatter(np.array(list(SPIRAL.values())).T[0],np.array(list(SPIRAL.values())).T[1], c=list(SPIRAL.keys()), cmap=cm.coolwarm, s=12)
# plt.colorbar()
# plt.show()

def get_closest_2_to_1(x,y):
    ret_key = (-1,-1)
    ret_value = -1
    distance = 1e6
    for (key, value) in InverseSPIRAL.items():
        if distance > (key[0] - x) ** 2 + (key[1]-y)**2:
            distance = (key[0] - x) ** 2 + (key[1]-y)**2
            ret_key=key
            ret_value = value

    return (ret_key, ret_value)

def get_closest_1_to_2(x):
    ret_key=-1
    ret_value=(-1,-1)
    distance=1e6
    for (key,value) in SPIRAL.items():
        if distance>(key-x)**2:
            distance=(key-x)**2
            ret_key=key
            ret_value=value

    return (ret_key,ret_value)


################# POWER ##############

def PWR_dx(x):
    if m==2:
        x=np.array(x)

    if(k==1):
        return g(x) ** 2 * fX(x) * dx**m #python does not support '@' for float 64
    else:
        return g(x) @ g(x).T * fX(x) * dx**m


def PWR():
    if m==1:
        return np.sum([PWR_dx(x) for counter,x in enumerate(X.T)])
    else:
        return np.sum([PWR_dx([x,y]) for x in X for y in X])

def PWERX_dx(x):
    if(m==1):
        return x**2 * fX(x) * dx
    else:
        return (x[0]**2+x[1]**2) * fX(np.array([x[0],x[1]])) * dx * dx


def PWRX():
    if m==2:
        return np.sum([PWERX_dx([x,y]) for  x in X for y in X])
    else:
        return np.sum([PWERX_dx(x) for x in X])



def PWRN_dn(n):
    if(k==1):
        return n**2 * fN(n) * dn
    else:
        return (n[0]**2+n[1]**2) * fN(np.array([n[0],n[1]])) * dn * dn


def PWRN():
    if k==2:
        return np.sum([PWRN_dn([n_x,n_y]) for n_x in N for n_y in N])
    else:
        return np.sum([PWRN_dn(n) for  n in N])



##################### h ####################
def h_num_dx(y_hat,x):



    if(m==1):
        return x * fX(x) * fN(y_hat-g(x)) * dx
    else:
        x = np.array(x)
        return x * fX(x) * fN(y_hat-g(x)) * dx**m

def h_den_dx(y_hat,x):
    if m==2:
        x=np.array(x)
    return fX(x) * fN(y_hat-g(x)) * dx**m

def h(y_hat):

    if m==2:

        Num=np.sum([h_num_dx(y_hat,[x,y]) for x in X for y in X],axis=0) #axis=0 so that Num remain with dim 2
        Den = np.sum([h_den_dx(y_hat, [x,y]) for x in X.T for y in X])

    else:

        Num = np.sum([h_num_dx(y_hat, x) for x in X.T])
        Den=np.sum([h_den_dx(y_hat,x) for x in X.T])

    return Num/Den


###################### mse #######################

def mse_sampler(x,n):
    global i
    i+=1
    print("iteration",i,"/",s**m*n_s**k)
    if m==2:
        x = np.array(x)
    if k==2:
        n=np.array(n)

    if m==1:
        return (x-h(g(x)+n))**2 * fX(x) * fN(n) * dx**m * dn**k
    else:
        return (x - h(g(x) + n)) @ (x - h(g(x) + n)).T * fX(x) * fN(n) * dx ** m * dn ** k


def mse(): #MSE
    if m==1 and k==1:
        return np.sum([mse_sampler(x,n) for n in N for x in X ])
    elif m==1 and k==2:
        return np.sum([mse_sampler(x, [n_x,n_y]) for n_x in N for n_y in N for x in X])

    elif m==2 and k==1:
        return np.sum([mse_sampler([x,y], n) for n in N for x in X for y in X])
    elif m==2 and k==2:
        return np.sum([mse_sampler([x,y], [n_x, n_y]) for n_x in N for n_y in N for x in X for y in X])









################################ OPTIMIZATION ##############################################






def mse_optimize_sampler(x,n,LEN_X,counter_x,counter_n):
    global  H_opt
    if(m==1):
        return (x - H_opt[LEN_X * counter_n +counter_x]) ** 2 * fX(x) * fN(n) * dx**m * dn**k
    return (x-H_opt[LEN_X * counter_n +counter_x]) @ (x-H_opt[LEN_X * counter_n +counter_x]).T * fX(x) * fN(n) * dx**m * dn**k

def mse_optimize(G_opt): #MSE
    global N
    if m == 1 and k == 1:
        return np.sum([mse_optimize_sampler(x, n,len(X),counter_x,counter_n) for  counter_n,n in enumerate(N) for counter_x,x in enumerate(X)])

    elif m == 1 and k == 2:
        N_temp=np.array([[n_x,n_y]for n_x in N for n_y in N])
        return np.sum([mse_optimize_sampler(x, n, len(X),counter_x,counter_n) for counter_n,n in enumerate(N_temp) for counter_x,x in enumerate(X)])

    elif m == 2 and k == 1:
        X_temp = np.array([[x, y] for x in X for y in X])
        return np.sum([mse_optimize_sampler(x, n,X_temp.shape[0],counter_x,counter_n) for counter_n,n in enumerate(N) for counter_x,x in enumerate(X_temp)])

    elif m == 2 and k == 2:
        N_temp = np.array([[n_x, n_y] for n_x in N for n_y in N])
        X_temp = np.array([[x, y] for x in X for y in X])
        return np.sum([mse_optimize_sampler(x, n,X_temp.shape[0],counter_x,counter_n) for counter_n,n in enumerate(N_temp) for counter_x,x in enumerate(X_temp)])

####################OPTIMIZATION#########################################

def integrand_encoder_num_optimize(y_hat,x,gx):
    if m == 2:
        x = np.array(x)
    return x * fX(x) * fN(y_hat-gx) * dx**m

def integrand_encoder_den_optimize(y_hat,x,gx):
    if m == 2:
        x = np.array(x)
    return fX(x) * fN(y_hat-gx) * dx**m

def h_optimize(y_hat,G_opt):

    if m==2:
        X_temp = [[x, y] for x in X for y in X]
        Num = np.sum([integrand_encoder_num_optimize(y_hat, x, gx) for x, gx in zip(X_temp, G_opt.T)],axis=0)  # axis=0 so that Num remain with dim 2
        Den = np.sum([integrand_encoder_den_optimize(y_hat, x,gx) for x,gx in zip(X_temp, G_opt.T) ])



    else:
        Num = np.sum([integrand_encoder_num_optimize(y_hat, x, gx) for x, gx in zip(X.T, G_opt.T)])
        Den=np.sum([integrand_encoder_den_optimize(y_hat,x,gx) for x,gx in zip(X.T,G_opt.T)])

    return Num/Den

#####################OPTIMIZATION################################


def PWR_constraint_sampler(x,gx):
    if k==2:
        gx=np.array(gx)
    if m==2:
        x=np.array(x)
    if k==1:
        return gx **2 * fX(x) * dx**m
    else:
        return gx @ gx.T * fX(x) * dx ** m

def PWR_constraint(G_opt):
    X_temp=X
    if m==1 and k==1:
        return np.sum([PWR_constraint_sampler(x,gx) for x,gx in zip(X,G_opt)])
    elif m==2 and k==1:
        X_temp = np.array([[x, y] for x in X for y in X])
        return np.sum([PWR_constraint_sampler(x, gx) for x,gx in zip(X_temp, G_opt)])
    elif m==1 and k==2:
        G_opt_temp=np.array([[gx_0,gx_1] for gx_0,gx_1 in zip(G_opt[::2],G_opt[1::2])])
        return np.sum([PWR_constraint_sampler(x,gx) for x,gx in zip(X_temp,G_opt_temp)])
    elif m==2 and k==2:
        X_temp = np.array([[x, y] for x in X for y in X])
        G_opt_temp = np.array([[gx_0, gx_1] for gx_0, gx_1 in zip(G_opt[:2], G_opt[1:2])])
        return np.sum([PWR_constraint_sampler(x, gx) for x, gx in zip(X_temp, G_opt_temp)])


def constraint(G_opt): #average power constraint
    return P_T-PWR_constraint(G_opt)

#######################OPTIMIZATION#################################################################################################


def sample_G_and_H(G_opt):
    global H_opt,Y_hat

    N_temp=N

    if k==2:
        N_temp=np.array([[n_x,n_y] for n_x in N for n_y in N])

    Y_hat=np.asarray([gx+n for n in N_temp for gx in G_opt.T]).T
    H_opt = np.asarray([h_optimize(y_hat, G_opt) for y_hat in Y_hat.T])


def callback(G_opt):
    global i
    i += 1
    print("iteration :\t", i)
    # iter_mse = mse_optimize(G_opt)

    sample_G_and_H(G_opt)
    # print("iteration :\t", i, "\tmse: \t", iter_mse,"\tPWR constraint:\t",PWR_constraint(G_opt))
    #
    # f.write("iteration :\t" + str(i) + "\nmse: \t" + str(iter_mse) +" PWR :\t"+str(PWR_constraint(G_opt))+"\n\n")
    # f.write("\n\n")





########################################################################################################################



POWER_X_Sqr= PWRX()
POWER_N_Sqr=PWRN()
POWER=PWR()
OptCost=mse()

print('current E[x^2]\t=\t ', POWER_X_Sqr)
print('current E[n^2]\t=\t ', POWER_N_Sqr)
print('current E[g^2(x)]\t=\t ', POWER)
print('current mse: \t ', OptCost)

def linear_encoder(x):
    return c*x

def linear_decoder(y_hat):
    X=c*Y
    global POWER_X_Sqr,POWER_N_Sqr
    return y_hat*c*POWER_X_Sqr/(c**2*POWER_X_Sqr+POWER_N_Sqr)


#linear encoder linear decoder
if pos==0:
    G_opt=np.asarray([linear_encoder(x) for x in X.T])
    Y_hat = np.asarray([gx + n for n in N.T for gx in G_opt.T]).T
    H_opt = np.asarray([linear_decoder(y_hat) for y_hat in Y_hat.T])
    print("POWER :\t",PWR_constraint(G_opt),"\t mse:\t",mse_optimize(G_opt))

#linear encoder optimal decoder
if pos==1:
    G_opt = np.asarray([linear_encoder(x) for x in X.T])
    Y_hat = np.asarray([gx + n for n in N.T for gx in G_opt.T]).T
    H_opt = np.asarray([h_optimize(y_hat,G_opt) for y_hat in Y_hat.T])
    print("POWER :\t", PWR_constraint(G_opt), "\t mse:\t", mse_optimize(G_opt))

#optimal encoder optimal decoder
if pos==2:

    f=open("log.txt","a+")
    start_train_time = time.time()
    f.write("\n############################################### INITIALIZATION #########################################################\n")
    f.write(" m ="+str(m)+"\t k="+str(k)+"\n")
    f.write("Noise:\t N("+str(MU)+","+str(SIGMA)+")\n")
    f.write("dev_m :\t"+str(dev_m)+"\n")
    f.write("dev_k :\t"+str(dev_k)+"\n")
    f.write("fX = \t"+my_fX+"\n")
    f.write("fN = \t"+my_fN+"\n")
    f.write(g_init+"\n")
    f.write("Started at :\t"+str(datetime.datetime.now())+"\n\n")
    f.write("X\t"+str(X)+'\n\n')
    f.write("Power:\t"+str(POWER)+"\n")
    f.write("MSE:\t"+str(OptCost)+"\n")
    f.write("POWER Constraint: \t "+str(P_T)+"\n")
    f.write("ftol is:\t"+str(my_ftol)+"\n")
    f.write("ftol is:\t"+str(my_maxIter)+"\n")
    f.write("Number of sampling points of x:\t"+str(s)+"\t\t"+"Number of sampling points of n:\t"+str(n_s)+"\n\n")


    #Build G_opt
    if m==1 and k==1:
        G_opt = np.asarray([g(x) for x in X.T]).T


    elif (m == 2 and k == 1):
        G_opt = np.asarray([get_closest_2_to_1(x,y)[1] for x in X for y in X]).T



    elif (m == 1 and k == 2):
        G_opt = np.asarray([get_closest_1_to_2(x)[1] for x in X]).T


    else:# m=2 and k=2
        G_opt = np.asarray([g([x,y]) for x in X for y in X]).T


    sample_G_and_H(G_opt)
    # plot
    # plot(X, G_opt, G)
    # plot(Y_hat, H_opt, H)

    f.write("Strating with:\n\n")
    f.write("G_opt:\t" + str(G_opt)+"\n\n")
    # f.write("H_opt:\t" + str(H_opt)+"\n\n")


    cons=({'type':'ineq','fun':constraint})
    # G_opt=np.ravel(G_opt,order='F')

    print("##### STARTING ITERATIONS #####")

    sol=minimize(mse_optimize,G_opt,method='SLSQP',constraints=cons,options={'disp':True ,'ftol':my_ftol ,'maxiter':my_maxIter} ,callback=callback)
    G_opt=sol.x


    print('Iterations finished in {} sec'.format(int(time.time() - start_train_time)))
    f.write('Iterations finished in {} sec'.format(int(time.time() - start_train_time))+"\n\n")
    f.write("finish with:\n\n")
    f.write("G_opt:\t" + str(G_opt)+"\n\n")
    f.write("H_opt:\t" + str(H_opt)+"\n\n")
    f.close()


    sample_G_and_H(G_opt)


    if not (m==2 and k==2):
        plot(X, G_opt, G)
        plot(Y_hat, H_opt, H)
    else:
        print("mse: ",mse_optimize(G_opt))
        print("pwr: ",constraint(G_opt))