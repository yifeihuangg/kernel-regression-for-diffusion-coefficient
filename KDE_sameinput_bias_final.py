#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:17:46 2017

@author: yellow
"""
import numpy as np
import matplotlib.pyplot as plt
import kernel_regression as kr



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
#    return np.sin(x)*0.5+1
    return np.sin(-x)*0.5+1

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
#    return np.cos(x)*0.5
    return np.cos(-x)*0.5

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
def get_b(trajs):
    """
    Estimates the E(dx) using KDE method.
    Parameters
    ----------
    trajs: list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.

    Returns
    -------
    model : function(?)
        Estimationg model of E(dx).
    """
    z_trajs = np.zeros([len(trajs),len(trajs[0])-1])
    x_trajs = np.zeros_like(z_trajs)
    kernel = kr.KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    # Store all the y in one dataset and use all the x's and corresponding y's to make model
    for n in xrange(len(trajs)):
        for i in xrange(len(trajs[0])-1):
            z_trajs[n][i] = (trajs[n][i+1] - trajs[n][i])
            x_trajs[n][i] = trajs[n][i]
    # Kernel.fit only takes in input matrix with dim <= 2, so put them into 1d array
    x = x_trajs.flatten()
    z = z_trajs.flatten()
    x_fit = np.zeros([len(x),1])
    for m in xrange(len(x_fit)):
        x_fit[m] = x[m]

    model = kernel.fit(x_fit,z)
    return model

def get_D_KDE(trajs, pos, subsampling=10, dt=0.001,use_drift=True):
    """
    Estimates the position-dependent diffusion constant using KDE method.
    Parameters
    ----------
    trajs : list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.
    pos: list
        Positions where you would like to estimate diffusion coefficient using the model
    subsampling: float, optional
        The subsampling used in get trajs. Default is 10.
    dt: float, optional
         Timestep for the Bronwnian_dynamics.  Default is 0.001, which is a decent choice for a harmonic with force constant 1.
    Returns
    -------
    D : array
        Estimate of the diffusion constant corresponding to pos.
    """
    y_trajs = np.zeros([len(trajs),len(trajs[0])-1])
    x_trajs = np.zeros_like(y_trajs)
    D = np.zeros(len(pos))
    kernel = kr.KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    if use_drift:
        model = get_b(trajs)
    # Store all the y in one dataset and use all the x's and corresponding y's to make model
    for n in xrange(len(trajs)):
        if use_drift:
            b_dt = model.predict(trajs[n])
        for i in xrange(len(trajs[0])-1):
            y_trajs[n][i] = ((trajs[n][i+1] - trajs[n][i])**2)
            if use_drift:
                y_trajs[n][i] -= (b_dt[i])**2
            x_trajs[n][i] = trajs[n][i]
    # Kernel.fit only takes in input matrix with dim <= 2, so put them into 1d array
    x = x_trajs.flatten()
    y = y_trajs.flatten()
    x_fit = np.zeros([len(x),1])
    for m in xrange(len(x_fit)):
        x_fit[m] = x[m]
    model = kernel.fit(x_fit,y)
    x_pos = np.zeros([len(pos),1])
    # model.predict only take (x,1) size input
    for m in xrange(len(pos)):
        x_pos[m] = pos[m]
    D = (model.predict(x_pos))/(dt*2*subsampling)
    return D

def main():
    # Load in trajectories into trajs file.
    trajs = []
    ## NEED CODE HERE
    initial = np.linspace(-2.,2.,101)
    kT = 1.0
#    dt = 0.00001
##    nsteps = 500
#    nsteps = 10000
#    subsampling = 1000
    dt = 0.001
    nsteps = 500
#    nsteps = 100
    subsampling = 10
    for inipoint in initial:
        for times in xrange(2):
            x0 =  np.array([inipoint])
            traj = brownian_dynamics(nsteps,x0,get_F,get_dD,get_D,dt=dt,kT=kT)
            # time consuming -> from begining to end skip every 10 points
            traj = traj[::subsampling]
            trajs.append(traj)

    hist_edges = np.linspace(-2.,2.,101)
    pos = np.zeros(len(hist_edges)-1)
    for m in xrange(len(hist_edges)-1):
        pos[m] = (hist_edges[m]+hist_edges[m+1])/2

    D_wo_drift= get_D_KDE(trajs,pos,subsampling,dt,use_drift=False)
    D_w_drift= get_D_KDE(trajs,pos,subsampling,dt)
    print("DONE!")
    #Plot estimation and true D values
    plt.figure()
    plt.plot(pos, D_w_drift,c = 'y', label = 'kernel w drift')
    plt.plot(pos, D_wo_drift,c = 'r', label = 'kernel w/o drift')
#    plt.hold('on')
    plt.scatter(pos, get_D(pos), label = 'true')
    plt.xlabel('Positions')
    plt.ylabel('Diffusion coefficient(D)')
    plt.title('Kernel regression')
    plt.savefig('KDE')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
