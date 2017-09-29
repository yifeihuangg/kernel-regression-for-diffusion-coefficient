#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:31:55 2017

@author: yellow
"""

import numpy as np
import matplotlib.pyplot as plt
import kernel_regression as kr
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error


def get_U(x):
    """
    Returns the potential energy of a function that is a quartic double well of form (x^4-2x^2).

    Parameters
    ----------
    x : float
        The location in the one-dimensional coordinate.
    a : float
        The prefactor for the potential.

    Returns
    -------
    U : float
        Value of the potential.
    """
    xsq = x*x
    print xsq, xsq*xsq, 2*xsq
    return xsq*xsq-2.*xsq

def get_F(x):
    """
    Returns the potential energy of a function that is a quartic double well of form (x^4-2x^2).

    Parameters
    ----------
    x : float
        The location in the one-dimensional coordinate.

    Returns
    -------
    F : float
        Value of the force.
    """
    return -(4*x*x*x-4*x)

def get_D(x):
    """
    Returns the value of the diffusion function at x.

    Parameters
    ----------
    x : float
        The location in the one-dimensional coordinate.

    Returns
    -------
    D : float
        Value of the diffusion function.
    """
    return np.sin(x)*0.5+1

def get_dD(x):
    """
    Returns the value of the divergence of the diffusion function at x.

    Parameters
    ----------
    x : float
        The location in the one-dimensional coordinate.

    Returns
    -------
    dD : float
        Value of the divergence of the diffusion function.
    """
    return np.cos(x)*0.5

def brownian_dynamics(nsteps,x0,force_method,get_divD,get_D,dt=0.001,kT=1.0):
    """
    Runs brownian dynamics.

    Parameters
    ----------
    nsteps : int
        Number of dynamics steps to run.
    x0 : 1d array-like
        Starting coordinate from which to run the dynamics.
    force_method : subroutine
        Subroutine that yields the force of the system.  Must take in an array of the same shape as x0, and return a force vector of the same size.
    dt : float, optional
        Timestep for the dynamics.  Default is 0.001, which is a decent choice for a harmonic with force constant 1.
    get_D : Subroutine
         Subroutine that yields diffusion tensor for the system.
    get_divD: Subroutine
         Subroutine that yields the divergence of D
    kT : float, optional
        Boltzmann factor for the system (k_B * T).  Default is natural units (1.0)


    Returns
    -------
    traj : 2D array
        Two dimensonal array, where the element i,j is the j'th coordinate of the system at timestep i.

    """
    # Set defaults and parameters
    ndim = len(x0) # Find dimensionality of the system
    # Propagate Brownian dynamics according to the Euler-Maruyama method.
    traj = []
    cfg = np.copy(x0)
    sig = np.sqrt(2.* dt) # Perform some algebra ahead of time.
    for j in xrange(int(nsteps)):
        D = get_D(cfg) # Typecast to array for easy math.
        c = np.sqrt(D) # Square root of Diffusion matrix hits the noise
        rando = np.dot(c,np.random.randn(ndim))
        force = np.dot(D,force_method(cfg))
        divD = get_divD(cfg)
        cfg += dt * force + sig * rando + divD * dt
        traj.append(np.copy(cfg))
    return np.array(traj)

def main():
    """
    Simulate a double well.
    """
    skip = 10000
    nsteps = 50000 + skip
    subsampling = 10
    x0 =  np.array([0.0])
    kT = 1.0
    dt = 0.001

    traj = brownian_dynamics(nsteps,x0,get_F,get_dD,get_D,dt=dt,kT=kT)
    #traj has 50000 elements, but y only has 49999 elements, minus 1 element from traj -> x
    # burn-in trajectory drop the first skip = 10000 data points
    #traj = traj[-(nsteps-skip):]
    # time consuming -> from begining to end skip every 10 points
   # traj = traj[::subsampling]
    x = traj[0:-1]
    #get (y_i)^2
    y = np.zeros((len(x)))
    for i in range(len(y)):
        y[i] = (traj[i+1] - traj[i])**2
    # kernel regression part
    kernel = kr.KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    model = kernel.fit(x,y)
    y_kr = model.predict(x)
    y_kr /= dt*subsampling*2
    # y_kr is estimated diffusion coefficient
    plt.figure()
    plt.scatter(x,y_kr,c = 'y', s = 4, label = 'kernel')
    plt.scatter(x, get_D(x), s = 4,label = 'true')
    plt.xlabel('position(x)')
    plt.ylabel('Diffusion coefficient(D)')
    plt.title('Kernel regression')
    plt.legend()
    plt.show()


    
    """
    Estimate error of the estimation
    """
    # make errorbar plot
    xax=np.linspace(-1.7,1.8,num = 17 ) #x-axis values I used in estimate sd
    xax = np.transpose([xax])
    run_times = 10
    #I set up a (runtime*x_values) matrix to store D estimation values, each col correspondes to same x value
    y2 = np.zeros([run_times,len(xax)])
    for i in range(run_times):
        traj = brownian_dynamics(nsteps,x0,get_F,get_dD,get_D,dt=dt,kT=kT)
        #traj = traj[-(nsteps-skip):]
        x = traj[0:-1]
        #get y_i
        y = np.zeros((len(x)))
        for j in range(len(y)):
            y[j] = (traj[j+1] - traj[j])**2
        # each time in loop produce a different trajectory, and get a new kernel regression model, use a new model each time to predict D
        model = kernel.fit(x,y)
        y2[i,] = model.predict(xax)
        y2[i,] /= dt*subsampling*2

    # calculate mean for each col, that's the mean for each x value
    mean = np.mean(y2,axis = 0)
    # calculate sd for each col, that's the sd for each x value
    sd = np.std(y2,axis = 0)
    #plot error bar
    plt.figure()
    plt.errorbar(xax,mean,yerr = sd, linestyle = 'None', marker = '^',label='kernel')
    plt.scatter(x, get_D(x), s = 4, label = 'true')
    plt.legend()
    plt.xlabel('trajectory(x)')
    plt.ylabel('Diffusion coefficient(D)')
    plt.title('Error bar plot of D Estimation')
    plt.show()

    """
    make RMSE plot
    """
    # Here I make up a new y2_true matrix, and each row is D_true, shape is run_times * len(xax)
    D_true = get_D(xax)
    y2_true = np.array([D_true[:,0]]*run_times)
    mse = mean_squared_error(y2_true, y2,multioutput = 'raw_values' )
    rmse = np.sqrt(mse)
    plt.figure()
    plt.plot(xax,rmse)
    plt.xlabel('trajectory(x)')
    plt.ylabel('RMSE(root mean square error)')
    plt.title('RMSE of D Estimation')


 # get RMSE errorbar plot, I need to run loop to get new y2
    MSE = np.zeros([run_times,len(xax)])
    for n in range(run_times):
        for i in range(run_times):
            traj = brownian_dynamics(nsteps,x0,get_F,get_dD,get_D,dt=dt,kT=kT)
            traj = traj[-(nsteps-skip):]
            traj = traj[::subsampling]
            x = traj[0:-1]
            y = np.zeros((len(x)))#get y_i
            for j in range(len(y)):
                y[j] = (traj[j+1] - traj[j])**2
        # each time in loop produce a different trajectory, and get a new kernel regression model, use a new model each time to predict D
        model = kernel.fit(x,y)
        y2[i,] = model.predict(xax)
        y2[i,] /= dt*subsampling*2
        #MSE store mse values for each y2 shape: run_times*len(xax)
        MSE[n,] = mean_squared_error(y2_true, y2,multioutput = 'raw_values' )
    RMSE = np.sqrt(MSE)
    # calculate mean for each col, that's the mean for each x value
    mean_rmse = np.mean(RMSE,axis = 0)
    # calculate sd for each col, that's the sd for each x value
    sd_rmse = np.std(RMSE,axis = 0)
    #plot error bar
    plt.figure()
    plt.errorbar(xax,mean_rmse,yerr = sd_rmse, linestyle = 'None', marker = '^')
    plt.xlabel('trajectory(x)')
    plt.ylabel('RMSE(D)')
    plt.title('Error bar plot of RMSE')
    plt.show()



    return
    #Not sure what's svr and their using conditions but give bad estimation. Not using here
    svr = GridSearchCV(SVR(kernel='rbf'), cv=5,
                   param_grid={"C": [1e-1, 1e0, 1e1, 1e2],
                               "gamma": np.logspace(-2, 2, 10)})
    y_svr = svr.fit(x, y).predict(x)
    plt.scatter(x,y_svr,c = 'y', label = 'svr')
    plt.hold('on')
    plt.scatter(x,get_D(x), label = 'true')
    plt.legend()
    plt.xlabel('data(x)')
    plt.ylabel('target(y)')
    plt.title('svr regression')
    plt.show()
    



if __name__ == "__main__":
    main()
