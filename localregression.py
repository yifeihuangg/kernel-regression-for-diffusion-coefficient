#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:37:52 2017

@author: yellow
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spl
from mpl_toolkits.mplot3d import Axes3D
import pyqt_fit.nonparam_regression as smooth
from pyqt_fit import npr_methods
from pyqt_fit import plot_fit
import kernel_regression as kr
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_U(coord):
    """
    Returns the potential energy of a function that is of form (x^4-2x^2+y^2) at coord.

    Parameters
    ----------
    coord : arraylike
            An array-like structure, corresponding to the coordinate of the point at which to evaluate the potential


    Returns
    -------
    U : float
        Value of the potential.
    """
    x = coord[0]
    y = coord[1]
    xsq = x*x
    ysq = y*y
    return xsq*xsq-2.*xsq+ysq

def get_F(coord):
    """
    Returns the force at coord.

    Parameters
    ----------
    coord : arraylike
            An array-like structure, corresponding to the coordinate of the point at which to evaluate force

    Returns
    -------
    F : array
        A numpy array, corresponding to the force vector.
    """
    x = coord[0]
    y = coord[1]
    F = np.zeros(len(coord))
    F[0] = -(4*x*x*x-4*x)
    F[1] = -2*y
    return F

def get_D(coord):
    """
    Returns the value of the diffusion function at coord.

    Parameters
    ----------
    coord : arraylike
            An array-like structure, corresponding to the coordinate of the point at which to evaluate diffusion coefficient

    Returns
    -------
    D : arraylike
        Matrix of the diffusion coefficient.
    """
    x = coord[0]
    y = coord[1]
    D = np.zeros([len(coord),len(coord)])
    xsq = x*x
    ysq = y*y
    D[0] = [5*ysq+5, 0]
    D[1] = [0, 1]
    return D


def get_dD(coord):
    """
    Returns the value of the divergence of the diffusion function at coord.

    Parameters
    ----------
    coord : arraylike
            An array-like structure, corresponding to the coordinate of the point at which to evaluate the divergence

    Returns
    -------
    dD : array
         Matrix of the divergence of the diffusion function.
    """
    x = coord[0]
    y = coord[1]
    dD = np.zeros([len(coord)])
    dD[0] = 0
    dD[1] = 0
    return dD

def brownian_dynamics(nsteps,x0,force_method,get_divD,get_D,dt=0.001,kT=1.0):
    """
    Runs brownian dynamics.

    Parameters
    ----------
    nsteps : int
        Number of dynamics steps to run.
    x0 : array-like
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
    traj : array
        Two dimensonal array, where the element i,j is the j'th coordinate of the system at timestep i.

    """
    ndim = len(x0) # Find dimensionality of the system

    # Propagate Brownian dynamics according to the Euler-Maruyama method.
    traj = []
    cfg = np.copy(x0)
    sig = np.sqrt(2.* dt) # Perform some algebra ahead of time.
    for j in xrange(int(nsteps)):
        D = get_D(cfg) # Typecast to array for easy math.
        c = spl.cholesky(D) # Square root of Diffusion matrix hits the noise
        rando = np.dot(c,np.random.randn(ndim))
        force = np.dot(D,force_method(cfg))
        divD = get_divD(cfg)
        cfg += dt * force + sig * rando + divD * dt
        traj.append(np.copy(cfg))
    return np.array(traj)


def get_D_KDE_pyqt(trajs, subsampling=1, dt=0.001):
    """
    Estimates the position-dependent diffusion constant using KDE method with PyQt-Fit's package.

    Parameters
    ----------
    trajs : list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.
    subsampling: float, optional
        The subsampling used when calculate trajectories. Default is 1.
    dt: float, optional
         Timestep for the Bronwnian_dynamics.  Default is 0.001, which is a decent choice for a harmonic with force constant 1.
    use_drift: optional, True/False
         If True, use drift term in the approximation, if False, not use.
    Returns
    -------
    D:  array
        Estimate of the diffusion constant corresponding to x with KR method.

    """

    dim = np.size(trajs[0][0])
    #if it is 1d dimension, then put all training datas in 1d array
    print('Dimension is %d' %(dim))
    D = np.zeros([dim,dim,len(trajs)*(len(trajs[0])-1)])
    x_trajs = np.zeros([len(trajs), len(trajs[0])-1, dim])
    for m in xrange(len(trajs)):
        for n in xrange(len(trajs[0])-1):
            x_trajs[m][n] = trajs[m][n]
    x = x_trajs.reshape(len(trajs)*(len(trajs[0])-1),dim)
    xt = np.transpose(x)

    for i in xrange(dim):
        for j in xrange(dim):
            y_trajs = np.zeros([len(trajs), len(trajs[0])-1])
            for m in xrange(len(trajs)):
                for n in xrange(len(trajs[0])-1):
                    y_trajs[m][n] = (trajs[m][n+1][i]-trajs[m][n][i])*(trajs[m][n+1][j]-trajs[m][n][j])

            y = y_trajs.flatten()
            k0 = smooth.NonParamRegression(xt, y, method=npr_methods.SpatialAverage())
            k0.fit()
            D[i][j] = (k0(xt))/(dt*2*subsampling)

    return D


def get_D_LOESS(trajs, subsampling=1, dt=0.001):
    """
    Estimates the position-dependent diffusion constant using locally weighted regression method with PyQt-Fit's package.

    Parameters
    ----------
    trajs : list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.
    subsampling: float, optional
        The subsampling used when calculate trajectories. Default is 1.
    dt: float, optional
         Timestep for the Bronwnian_dynamics.  Default is 0.001, which is a decent choice for a harmonic with force constant 1.

    Returns
    -------
    D : array
         Estimate of the diffusion constant corresponding to trajs with LOESS function.
    """
    dim = np.size(trajs[0][0])
    print('Dimension is %d' %(dim))
    D = np.zeros([dim,dim,len(trajs)*(len(trajs[0])-1)])
    x_trajs = np.zeros([len(trajs), len(trajs[0])-1, np.size(trajs[0][0])])
    for m in xrange(len(trajs)):
        for n in xrange(len(trajs[0])-1):
            x_trajs[m][n] = trajs[m][n]
    x = x_trajs.reshape(len(trajs)*(len(trajs[0])-1),dim)
    xt = np.transpose(x)

    for i in xrange(dim):
        for j in xrange(dim):
            y_trajs = np.zeros([len(trajs), len(trajs[0])-1])
            for m in xrange(len(trajs)):
                for n in xrange(len(trajs[0])-1):
                    y_trajs[m][n] = (trajs[m][n+1][i]-trajs[m][n][i])*(trajs[m][n+1][j]-trajs[m][n][j])

            y = y_trajs.flatten()
            k1 = smooth.NonParamRegression(xt,y,method=npr_methods.LocalPolynomialKernel(q=1))
            k1.fit();
            D[i][j] = (k1(xt))/(dt*2*subsampling)

    return D


def get_D_KDE_KR(trajs, subsampling=1, dt=0.001):
    """
    Estimates the position-dependent diffusion constant using KR with kernelregression package.

    Parameters
    ----------
    trajs : list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.
    subsampling: float, optional
        The subsampling used when calculate trajectories. Default is 1.
    dt: float, optional
         Timestep for the Bronwnian_dynamics.  Default is 0.001, which is a decent choice for a harmonic with force constant 1.

    Returns
    -------
    D : array
        Estimate of the diffusion constant corresponding to trajs with kernelregression package.
    x : Matrix N*dim
        Input traning data x of the model, for plotting afterwards.
    """
    kernel = kr.KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    dim = np.size(trajs[0][0])
    print('Dimension is %d' %(dim))
    D = np.zeros([dim,dim,len(trajs)*(len(trajs[0])-1)])
    x_trajs = np.zeros([len(trajs), len(trajs[0])-1, np.size(trajs[0][0])])
    for m in xrange(len(trajs)):
        for n in xrange(len(trajs[0])-1):
            x_trajs[m][n] = trajs[m][n]
    x = x_trajs.reshape(len(trajs)*(len(trajs[0])-1),dim)

    for i in xrange(dim):
        for j in xrange(dim):
            y_trajs = np.zeros([len(trajs), len(trajs[0])-1])
            for m in xrange(len(trajs)):
                for n in xrange(len(trajs[0])-1):
                    y_trajs[m][n] = (trajs[m][n+1][i]-trajs[m][n][i])*(trajs[m][n+1][j]-trajs[m][n][j])

            y = y_trajs.flatten()
            model = kernel.fit(x,y)
            D[i][j] = (model.predict(x))/(dt*2*subsampling)

    return D, x


def main():
    # Load in trajectories into trajs file.
    trajs = []
    initial = np.linspace(-2.,2.,8)
    kT = 1.0
    dt = 0.001
    nsteps = 15
    subsampling = 1

    for xinitial in initial:
        for yinitial in initial:
            for times in xrange(1):
                x0 =  np.array([xinitial, yinitial])
                traj = brownian_dynamics(nsteps,x0,get_F,get_dD,get_D,dt=dt,kT=kT)
                trajs.append(traj)

    # D_KDE_pyqt = get_D_KDE_pyqt(trajs,subsampling,dt)
    D_LOESS = get_D_LOESS(trajs,subsampling,dt)
    D_KR,x = get_D_KDE_KR(trajs, subsampling, dt)
    realD = np.array([get_D(x_i) for x_i in x])

    #Plot estimation and true D values
    plt.figure()
#   scatter plot
    vmax=np.amax(realD[:,0,0])+0.35*np.amax(realD[:,0,0])
    vmin=np.amin(realD[:,0,0])-0.35*np.amax(realD[:,0,0])

    plt.subplot(131,aspect='equal', adjustable='box-forced')
    ax = plt.gca()
    SC_1 = plt.scatter(x[:,0], x[:,1], c=realD[:,0,0],linewidth='0',vmax=vmax,vmin=vmin)
    plt.xlabel('x')
    plt.ylabel('y')
    diver = make_axes_locatable(ax)
    cax = diver.append_axes("right", size="5%", pad = 0.05)
    plt.colorbar(SC_1,cax=cax)
    plt.title('RealD')

    # plt.subplot(2,2,2)
    # SC_2 = plt.scatter(x[:,0], x[:,1], c=D_KDE_pyqt[0,0,:],linewidth='0',vmax=vmax,vmin=vmin)
    # plt.colorbar(SC_2)
    # plt.xlabel('x',fontsize=7)
    # plt.ylabel('y',fontsize=7)
    # ax = plt.gca()
    # ax.xaxis.labelpad = 0
    # ax.yaxis.labelpad = 0
    # plt.title('KDE_pyqt',fontsize=8, y = 1.08)

    plt.subplot(132,aspect='equal', adjustable='box-forced')
    ax = plt.gca()
    SC_3 = plt.scatter(x[:,0], x[:,1], c=D_LOESS[0,0,:],linewidth='0',vmax=vmax,vmin=vmin)
    plt.xlabel('x')
    plt.ylabel('y')
    diver = make_axes_locatable(ax)
    cax = diver.append_axes("right", size="5%", pad = 0.05)
    plt.colorbar(SC_3,cax=cax)
    plt.title('LOESS')

    plt.subplot(133,aspect='equal', adjustable='box-forced')
    SC_4 = plt.scatter(x[:,0], x[:,1], c=D_KR[0,0,:],linewidth='0',vmax=vmax,vmin=vmin)
    ax = plt.gca()
    plt.xlabel('x')
    plt.ylabel('y')
    diver = make_axes_locatable(ax)
    cax = diver.append_axes("right", size="5%", pad = 0.05)
    plt.colorbar(SC_4,cax=cax)
    plt.title('KR')

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    return



if __name__ == "__main__":
    main()
