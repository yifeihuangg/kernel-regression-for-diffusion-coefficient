#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:11:49 2017

@author: yellow
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spl
import kernel_regression as kr
from scipy import *
from dmap.diffusion_map import diffusion_map
from mpl_toolkits.mplot3d import Axes3D


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
    D[0] = [xsq*xsq+ysq*ysq+1, 0]
    D[1] = [0, xsq+0.5*np.sin(x+y)+1]
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
    dD[0] = 4*(x*x*x)
    dD[1] = 0.5*np.cos(x+y)
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

def get_D_KDE_var(trajs, subsampling=1, dt=0.001):
    """
    Estimates the position-dependent diffusion constant using KDE method.
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
    D : matrix
        Estimate of the diffusion constant.
    """
    print('Variable bandwidth')
    dim = np.size(trajs[0][0])
    print('Dimension is %d' %(dim))
    D = np.zeros((dim,dim,len(trajs)*(len(trajs[0])-1)))
    x_trajs = np.zeros([len(trajs), len(trajs[0])-1, np.size(trajs[0][0])])
    for m in xrange(len(trajs)):
        for n in xrange(len(trajs[0])-1):
            x_trajs[m][n] = trajs[m][n]
    x = x_trajs.reshape(len(trajs)*(len(trajs[0])-1),dim)
    print 'starting dmap'
    # print np.shape(x), 'x shape'
    L,pi,k,rho,q_alpha,epses = diffusion_map(x,alpha = 0.0, beta = '-1/d', return_full = True,rho_norm=False,verbosity=True,nneighb=500)
    # np.save('L.npy',L)
    # np.save('epsilon.npy',[epses])
    # np.save('pi.npy',pi)
    # np.save('k.npy',k)
    # np.save('rho.npy',rho)
    for i in xrange(dim):
        for j in xrange(dim):
            y_trajs = np.zeros([len(trajs), len(trajs[0])-1])
            for m in xrange(len(trajs)):
                for n in xrange(len(trajs[0])-1):
                    y_trajs[m][n] = (trajs[m][n+1][i]-trajs[m][n][i])*(trajs[m][n+1][j]-trajs[m][n][j])
            y = y_trajs.flatten()
            Const = (1./(dt*2*subsampling))
            dot_product = k.dot(y)
            ones = np.ones_like(y)
            dot_product_1 = k.dot(ones)
            D[i][j] = (dot_product/dot_product_1)*Const


    return D, x, y

def get_D_KDE_const(trajs, subsampling=1, dt=0.001):
    """
    Estimates the position-dependent diffusion constant using KDE method.
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
    D : array
        Estimate of the diffusion constant corresponding to pos.
    """
    print('Constant bandwidth')
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
    return D,x,y

def main():
    # Load in trajectories into trajs file.
    trajs = []
    initial = np.linspace(-2.,2.,25)
    kT = 1.0
    dt = 0.001
    nsteps = 20
    subsampling = 1
    for xinitial in initial:
        for yinitial in initial:
            for times in xrange(1):
                x0 =  np.array([xinitial, yinitial])
                traj = brownian_dynamics(nsteps,x0,get_F,get_dD,get_D,dt=dt,kT=kT)
                trajs.append(traj)

    trajs = np.array(trajs)
    # print('trajs shape')
    # print(np.shape(trajs))
    dim = len(trajs[0][0])
    D_var,x_trajs, y_trajs= get_D_KDE_var(trajs,subsampling,dt)
    D_const,x,y = get_D_KDE_const(trajs,subsampling,dt)
    # fig,(ax1,ax2) = plt.subplots(2)
    # ax1.scatter(x_trajs[:,0],x_trajs[:,1],c=D_wo_drift[0,0,:])
    # ax2.scatter(pos[:,0],pos[:,1],c=D_wo_drift[0,0,:])
    # plt.show()
    #D_wo_drift = D_wo_drift.reshape(dim,dim,len(pos))
    realD = np.array([get_D(pos_i) for pos_i in x_trajs])
    #Plot estimation and true D values


#   scatter plot
    print('start plotting D[1,1]')
    vmax=np.amax(realD[:,0,0])+0.75*np.amax(realD[:,0,0])
    vmin=np.amin(realD[:,0,0])-0.75*np.amax(realD[:,0,0])

    plt.figure()
    # plt.subplot(131, aspect='equal', adjustable='box-forced')
    SC_1 = plt.scatter(x_trajs[:,0], x_trajs[:,1], c=D_var[0,0,:],linewidth='0',vmax=vmax,vmin=vmin)
    plt.colorbar(SC_1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Var_Band KR')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.figure()
    # plt.subplot(132, aspect='equal', adjustable='box-forced')
    SC_2 = plt.scatter(x_trajs[:,0],x_trajs[:,1], c = D_const[0,0,:],linewidth='0',vmax=vmax, vmin=vmin)
    plt.colorbar(SC_2)
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title('Const_Band KR')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.figure()
    # plt.subplot(133, aspect='equal', adjustable='box-forced')
    SC_3 = plt.scatter(x_trajs[:,0],x_trajs[:,1], c=realD[:,0,0], linewidth='0',vmax=vmax,vmin=vmin)
    plt.colorbar(SC_3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Real D')
    plt.tight_layout()
    plt.show()
    plt.close()



    print('start plotting D[2,2]')
    vmax=np.amax(realD[:,1,1])+0.75*np.amax(realD[:,1,1])
    vmin=np.amin(realD[:,1,1])-0.75*np.amax(realD[:,1,1])

    plt.figure()
    # plt.subplot(131, aspect='equal', adjustable='box-forced')
    SC_1 = plt.scatter(x_trajs[:,0], x_trajs[:,1], c=D_var[1,1,:],linewidth='0',vmax=vmax,vmin=vmin)
    plt.colorbar(SC_1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Var_Band KR')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.figure()
    # plt.subplot(132, aspect='equal', adjustable='box-forced')
    SC_2 = plt.scatter(x_trajs[:,0],x_trajs[:,1], c = D_const[1,1,:],linewidth='0',vmax=vmax, vmin=vmin)
    plt.colorbar(SC_2)
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title('Const_Band KR')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.figure()
    # plt.subplot(133, aspect='equal', adjustable='box-forced')
    SC_3 = plt.scatter(x_trajs[:,0],x_trajs[:,1], c=realD[:,1,1], linewidth='0',vmax=vmax,vmin=vmin)
    plt.colorbar(SC_3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Real D')
    plt.tight_layout()
    plt.show()
    plt.close()


    print('start plotting D[1,2]')
    vmax=np.amax(realD[:,0,1])+np.amax(realD[:,0,1])
    vmin=np.amin(realD[:,0,1])-np.amax(realD[:,0,1])

    plt.figure()
    # plt.subplot(131, aspect='equal', adjustable='box-forced')
    SC_1 = plt.scatter(x_trajs[:,0], x_trajs[:,1], c=D_var[0,1,:],linewidth='0',vmax=vmax,vmin=vmin)
    plt.colorbar(SC_1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Var_Band KR')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.figure()
    # plt.subplot(132, aspect='equal', adjustable='box-forced')
    SC_2 = plt.scatter(x_trajs[:,0],x_trajs[:,1], c = D_const[0,1,:],linewidth='0',vmax=vmax, vmin=vmin)
    plt.colorbar(SC_2)
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title('Const_Band KR')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.figure()
    # plt.subplot(133, aspect='equal', adjustable='box-forced')
    SC_3 = plt.scatter(x_trajs[:,0],x_trajs[:,1], c=realD[:,0,1], linewidth='0',vmax=vmax,vmin=vmin)
    plt.colorbar(SC_3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Real D')
    plt.tight_layout()
    plt.show()
    plt.close()



    print('start plotting D[2,1]')
    vmax=np.amax(realD[:,1,0])+np.amax(realD[:,1,0])
    vmin=np.amin(realD[:,1,0])-np.amax(realD[:,1,0])

    plt.figure()
    # plt.subplot(131, aspect='equal', adjustable='box-forced')
    SC_1 = plt.scatter(x_trajs[:,0], x_trajs[:,1], c=D_var[1,0,:],linewidth='0',vmax=vmax,vmin=vmin)
    plt.colorbar(SC_1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Var_Band KR')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.figure()
    # plt.subplot(132, aspect='equal', adjustable='box-forced')
    SC_2 = plt.scatter(x_trajs[:,0],x_trajs[:,1], c = D_const[1,0,:],linewidth='0',vmax=vmax, vmin=vmin)
    plt.colorbar(SC_2)
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title('Const_Band KR')
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.figure()
    # plt.subplot(133, aspect='equal', adjustable='box-forced')
    SC_3 = plt.scatter(x_trajs[:,0],x_trajs[:,1], c=realD[:,1,0], linewidth='0',vmax=vmax,vmin=vmin)
    plt.colorbar(SC_3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Real D')
    plt.tight_layout()
    plt.show()
    plt.close()




    print("DONE!")
    return



if __name__ == "__main__":
    main()
