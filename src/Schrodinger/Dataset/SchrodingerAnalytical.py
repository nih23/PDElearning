import numpy as np
import h5py as h5
import argparse
import os

inv_fthsqrt_pi = 1 / np.pi ** (1. / 4.)
inv_sqrt_two = 1. / np.sqrt(2.)

def phi0(x,f):
    root_f = f ** (1./4.)
    return root_f*inv_fthsqrt_pi * np.exp(- x * f * x / 2.0)

def phi1(x,f):
    return  phi0(x,f) * inv_sqrt_two * (np.sqrt(f) * 2.0 * x)

def Phi0(x, y, f):
    return phi0(x,f) * phi0(y,f)

def Phi1(x, y, f):
    return phi1(x,f) * phi1(y,f)


def Psi(x, y, t,f):
    return inv_sqrt_two * (np.exp(-1j*f*t) * Phi0(x, y,f) + np.exp(-1j*3*f*t)*Phi1(x, y,f))


def write_solution(x, y, t, f, prefix, filebase="step-"):
    sol = Psi(x, y, t, f).T.reshape(200, 200)
    dt = 0.001
    u = sol.real
    v = sol.imag
    u = u.reshape(-1)
    v = v.reshape(-1)
    state = int(t / dt)
    with h5.File(os.path.join(prefix,"{}{}.h5".format(filebase,state)), "w") as f:

        f.create_dataset('real', data=u)
        f.create_dataset('imag', data=v)

def main():

    parser = argparse.ArgumentParser(description='Calculate Analytical Solution for Quantum Oscillator in 2D')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    parser.add_argument('--xdim', action='store', type=int,
                        default=200,
                        help='number of samples in x, arbitrary units')
    parser.add_argument('--ydim', action='store', type=int,
                        default=200,
                        help='number of samples in y, arbitrary units')
    parser.add_argument('--dt', action='store', type=float,
                        default=.001,
                        help='time delta in arbitrary units')
    parser.add_argument('--nsteps', action='store', type=int,
                        default=2000,
                        help='number of time steps')
    parser.add_argument('--prefix', action='store',
                        default="./analytical_results/",type=str,
                        help='prefix to store analytical_results into')
    parser.add_argument('--filebase', action='store',
                        default="step-",type=str,
                        help='file base name to use, e.g. step- will produce step-1.h5 etc.')
    parser.add_argument('--f', action='store',
                        default=1.,type=float,
                        help='Frequency of harmonic oscillator')
    args = parser.parse_args()

    x = np.linspace(-10, 10, args.xdim)
    y = np.linspace(-10,10, args.ydim)

    dt = args.dt
    X, Y = np.meshgrid(x, y)

    X = X.reshape(-1)
    Y = Y.reshape(-1)

    for t in range(0,args.nsteps):
        if t % 200 == 0:
            print("t", t)
        write_solution(X, Y, t*dt,args.f, args.prefix,args.filebase)

if __name__ == '__main__':
    main()
