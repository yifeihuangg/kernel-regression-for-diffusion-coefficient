import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import kernel_regression as kr
import scipy.linalg as spl
import numbers
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle


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
    D[0] = [0.5*np.cos(x+y)+2, 0]
    D[1] = [0, 0.25*(xsq+ysq)+1]
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
    dD[0] = -0.5*np.sin(x+y)
    dD[1] = 0.5*y
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


def get_D_KDE(trajs, pos, subsampling=1, dt=0.001):
    """
    Estimates the position-dependent diffusion constant using KDE method.
    Parameters
    ----------
    trajs : list of arrays
        List, where each element is a trajectory.  The trajectory is assumed to be a 2D array-like, where each row is a timestep and each column is a coordinate.  Note: each trajectory must have the same timesteps.
    pos: list
        Positions where you would like to estimate diffusion coefficient using the model
    subsampling: float, optional
        The subsampling used when calculate trajectories. Default is 1.
    dt: float, optional
         Timestep for the Bronwnian_dynamics.  Default is 0.001, which is a decent choice for a harmonic with force constant 1.

    Returns
    -------
    D : array
        Estimate of the diffusion constant corresponding to pos.
    """
    kernel = kr.KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
    dim = np.size(trajs[0][0])
    print('Dimension is %d' %(dim))
    D = np.zeros([dim,dim,len(pos)])

    #filling training data x
    x_trajs = np.zeros([len(trajs), len(trajs[0])-1, np.size(trajs[0][0])])
    for m in xrange(len(trajs)):
        for n in xrange(len(trajs[0])-1):
            x_trajs[m][n] = trajs[m][n]
    x = x_trajs.reshape(len(trajs)*(len(trajs[0])-1),dim)
    #filling training data y
    for i in xrange(dim):
        for j in xrange(dim):
            y_trajs = np.zeros([len(trajs), len(trajs[0])-1])
            for m in xrange(len(trajs)):
                for n in xrange(len(trajs[0])-1):
                    y_trajs[m][n] = (trajs[m][n+1][i]-trajs[m][n][i])*(trajs[m][n+1][j]-trajs[m][n][j])
            y = y_trajs.flatten()

            model = kernel.fit(x,y)
            D[i][j] = (model.predict(pos))/(dt*2*subsampling)

    return D

def main():
    # Load in trajectories into trajs file.
    trajs = []
    initial = np.linspace(-2.,2.,21)
    kT = 1.0
    dt = 0.001
    nsteps = 20
    subsampling = 1
    for xinitial in initial:
        for yinitial in initial:
            for times in xrange(1):
                x0 =  np.array([xinitial, yinitial])
                traj = brownian_dynamics(nsteps,x0,get_F,get_dD,get_D,dt=dt,kT=kT)
                # traj = traj[::subsampling]
                trajs.append(traj)


    hist_edges = np.linspace(-2.,2.,101)
    pos = []
    for m in xrange(len(hist_edges)-1):
        for n in xrange(len(hist_edges)-1):
            pos.append ([(hist_edges[m]+hist_edges[m+1])/2,(hist_edges[n]+hist_edges[n+1])/2])
    pos = np.array(pos)

    D_KDE = get_D_KDE(trajs,pos,subsampling,dt)
    realD = np.array([get_D(pos_i) for pos_i in pos])

    print("DONE!")


#Plot estimation and true D values



#plot D[0,0]
#   scatter plot
    plt.figure()
    plt.subplot(2,2,1)
    vmax=np.amax(realD[:,0,0])+0.25*np.amax(realD[:,0,0])
    vmin=np.amin(realD[:,0,0])-0.25*np.amax(realD[:,0,0])
    SC_1 = plt.scatter(pos[:,0], pos[:,1], c=D_KDE[0,0,:],linewidth='0',vmax=vmax,vmin=vmin)
    plt.colorbar(SC_1)
    plt.xlabel('x positions')
    plt.ylabel('y position')
    plt.title('Kernel regression')
    plt.subplot(2,2,2)
    SC_2 = plt.scatter(pos[:,0],pos[:,1], c=realD[:,0,0], linewidth='0',vmax=vmax,vmin=vmin)
    plt.colorbar(SC_2)
    plt.xlabel('x positions')
    plt.ylabel('y position')
    plt.title('Real D')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.savefig('KDE_2D')
    plt.close()

#   3D plt
    X = pos[:,0]
    Y = pos[:,1]
    Z = D_KDE[0,0,:]
    Zreal = realD[:,0,0]

    fig = plt.figure()
    ax = fig.add_subplot(2,2,1, projection='3d')
    SP_1 = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, linewidth = '0', vmax = vmax, vmin = vmin,antialiased=False)
    ax.set_xlabel = ('x position')
    ax.set_ylabel = ('y position')
    ax.set_zlabel = ('D')
    ax.set_title("Kernel Regression")
    plt.colorbar(SP_1)

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    SP_2 = ax.plot_trisurf(X, Y, Zreal, cmap=cm.coolwarm, linewidth = '0', vmax = vmax, vmin = vmin,antialiased=False)
    ax.set_xlabel = ('x position')
    ax.set_ylabel = ('y position')
    ax.set_zlabel = ('D')
    ax.set_title("RealD")
    plt.colorbar(SP_2)
    ax.legend()
    plt.tight_layout()
    plt.show()
    #fig.savefig('3d_scatter')
    plt.close()


# #plot D[0,1]
# #   scatter plot
#     plt.figure()
#     plt.subplot(2,2,1)
#     vmax=np.amax(realD[:,0,1])+0.5
#     vmin=np.amin(realD[:,0,1])-0.5
#     SC_1 = plt.scatter(pos[:,0], pos[:,1], c=D_KDE[0,1,:],linewidth='0',vmax=vmax,vmin=vmin)
#     plt.colorbar(SC_1)
#     plt.xlabel('x positions')
#     plt.ylabel('y position')
#     plt.title('Kernel regression')
#     plt.subplot(2,2,2)
#     SC_2 = plt.scatter(pos[:,0],pos[:,1], c=realD[:,0,1], linewidth='0',vmax=vmax,vmin=vmin)
#     plt.colorbar(SC_2)
#     plt.xlabel('x positions')
#     plt.ylabel('y position')
#     plt.title('Real D')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     #plt.savefig('KDE_2D')
#     plt.close()
#
# #   3D plt
#     X = pos[:,0]
#     Y = pos[:,1]
#     Z = D_KDE[0,1,:]
#     Zreal = realD[:,0,1]
#
#     fig = plt.figure()
#     ax = fig.add_subplot(2,2,1, projection='3d')
#     SP_1 = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, linewidth = '0', vmax = vmax, vmin = vmin,antialiased=False)
#     ax.set_xlabel = ('x position')
#     ax.set_ylabel = ('y position')
#     ax.set_zlabel = ('D')
#     ax.set_title("Kernel Regression")
#     plt.colorbar(SP_1)
#
#     ax = fig.add_subplot(2, 2, 2, projection='3d')
#     SP_2 = ax.plot_trisurf(X, Y, Zreal, cmap=cm.coolwarm, linewidth = '0', vmax = vmax, vmin = vmin,antialiased=False)
#     ax.set_xlabel = ('x position')
#     ax.set_ylabel = ('y position')
#     ax.set_zlabel = ('D')
#     ax.set_title("RealD")
#     plt.colorbar(SP_2)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
#     #fig.savefig('3d_scatter')
#     plt.close()
#
#
#
# # plot D[1,0]
# #   scatter plot
#     plt.figure()
#     plt.subplot(2,2,1)
#     vmax=np.amax(realD[:,1,0])+0.5
#     vmin=np.amin(realD[:,1,0])-0.5
#     SC_1 = plt.scatter(pos[:,0], pos[:,1], c=D_KDE[1,0,:],linewidth='0',vmax=vmax,vmin=vmin)
#     plt.colorbar(SC_1)
#     plt.xlabel('x positions')
#     plt.ylabel('y position')
#     plt.title('Kernel regression')
#     plt.subplot(2,2,2)
#     SC_2 = plt.scatter(pos[:,0],pos[:,1], c=realD[:,1,0], linewidth='0',vmax=vmax,vmin=vmin)
#     plt.colorbar(SC_2)
#     plt.xlabel('x positions')
#     plt.ylabel('y position')
#     plt.title('Real D')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     #plt.savefig('KDE_2D')
#     plt.close()
#
# #   3D plt
#     X = pos[:,0]
#     Y = pos[:,1]
#     Z = D_KDE[1,0,:]
#     Zreal = realD[:,1,0]
#
#     fig = plt.figure()
#     ax = fig.add_subplot(2,2,1, projection='3d')
#     SP_1 = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, linewidth = '0', vmax = vmax, vmin = vmin,antialiased=False)
#     ax.set_xlabel = ('x position')
#     ax.set_ylabel = ('y position')
#     ax.set_zlabel = ('D')
#     ax.set_title("Kernel Regression")
#     plt.colorbar(SP_1)
#
#     ax = fig.add_subplot(2, 2, 2, projection='3d')
#     SP_2 = ax.plot_trisurf(X, Y, Zreal, cmap=cm.coolwarm, linewidth = '0', vmax = vmax, vmin = vmin,antialiased=False)
#     ax.set_xlabel = ('x position')
#     ax.set_ylabel = ('y position')
#     ax.set_zlabel = ('D')
#     ax.set_title("RealD")
#     plt.colorbar(SP_2)
#     ax.legend()
#     plt.tight_layout()
#     plt.show()
#     #fig.savefig('3d_scatter')
#     plt.close()


# plot D[1,1]
#   scatter plot
    plt.figure()
    plt.subplot(2,2,1)
    vmax=np.amax(realD[:,1,1])+0.25*np.amax(realD[:,1,1])
    vmin=np.amin(realD[:,1,1])-0.25*np.amax(realD[:,1,1])
    SC_1 = plt.scatter(pos[:,0], pos[:,1], c=D_KDE[1,1,:],linewidth='0',vmax=vmax,vmin=vmin)
    plt.colorbar(SC_1)
    plt.xlabel('x positions')
    plt.ylabel('y position')
    plt.title('Kernel regression')
    plt.subplot(2,2,2)
    SC_2 = plt.scatter(pos[:,0],pos[:,1], c=realD[:,1,1], linewidth='0',vmax=vmax,vmin=vmin)
    plt.colorbar(SC_2)
    plt.xlabel('x positions')
    plt.ylabel('y position')
    plt.title('Real D')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.savefig('KDE_2D')
    plt.close()

#   3D plt
    X = pos[:,0]
    Y = pos[:,1]
    Z = D_KDE[1,1,:]
    Zreal = realD[:,1,1]

    fig = plt.figure()
    ax = fig.add_subplot(2,2,1, projection='3d')
    SP_1 = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, linewidth = '0', vmax = vmax, vmin = vmin,antialiased=False)
    ax.set_xlabel = ('x position')
    ax.set_ylabel = ('y position')
    ax.set_zlabel = ('D')
    ax.set_title("Kernel Regression")
    plt.colorbar(SP_1)

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    SP_2 = ax.plot_trisurf(X, Y, Zreal, cmap=cm.coolwarm, linewidth = '0', vmax = vmax, vmin = vmin,antialiased=False)
    ax.set_xlabel = ('x position')
    ax.set_ylabel = ('y position')
    ax.set_zlabel = ('D')
    ax.set_title("RealD")
    plt.colorbar(SP_2)

    ax.legend()
    plt.tight_layout()
    plt.show()
    #fig.savefig('3d_scatter')
    plt.close()

    return



if __name__ == "__main__":
    main()
