#!/usr/bin/env python
# Explicit FD Poroelastic Test Case
# 2D Isotropic condition

import numpy as np
import matplotlib.pyplot as plt


# Mesh Parameters
Lx = 1500        # m, X Dimension Length
Lz = 1500        # m, Z Dimension Length
hx = 3           # m, X Dimension Interval
hz = 3           # m, Z Dimension Interval

# Input Parameters
k = 9.8692327e-16 # m**2
phi   = 0.16      # %
rho_f = 937.0     # kg / m**3
rho_s = 2500.0    # kg / m**3
G = 15e09         # Pa
K_dr = 6e09       # Pa
K_s = 36e9        # Pa
K_f = 2.25e9      # Pa
mu_f = 0.0001     # Pa*s
T = 10.0          # -

# Generated Parameters
alpha = 1.0 - K_dr/K_s  # Biot Coefficient
M = ( (alpha - phi)/K_s + phi/K_f )    # Biot Modulus
K_u = K_dr + alpha**2 * M # Undrained Bulk Modulus, Pa
l_u = K_u - (2.0/3.0)*G  # Undrained Lame #1
P_d = K_dr + (4.0/3.0)*G # Drained P Wave Modulus
H = P_d + alpha**2 * M # Undrained P Wave Modulus
D = (k*M*P_d)/(mu_f*H) # Coefficient of Hydraulic Diffusivity
b = mu_f / k # Coefficient of friction / Fluid Mobility, Pa*s / m**2
omega_B = (b*phi)/(T*rho_f) # Biot Frequency
rho_m = (T*rho_f)/phi # Effective Fluid Density, kg / m**3
rho_b = phi*rho_f + (1.0 - phi)*rho_s # Bulk Density, kg / m**3
rho_bar = rho_m*rho_b - rho_f**2 #

# Velocities
Vp = ((K_u + 4.0*G/3.0)/rho_b)**0.5
Vs = (G/rho_b)**0.5

# CFL Criteria
dt = np.min(np.min([hx,hz]) / (Vp**2 - Vs**2)**0.5) # sec

# Run parameters
tn = 0.10 # sec

# ==============================================================================
# Generate Grid(s)
#nx = np.int(Lx/hx)
#nz = np.int(Lz/hz)

#v_x = np.zeros([nx,nz])
#v_z = np.zeros([nx,nz])
#q_x = np.zeros([nx,nz])
#q_z = np.zeros([nx,nz])
#txx = np.zeros([nx,nz])
#tzz = np.zeros([nx,nz])
#txz = np.zeros([nx,nz])
#p = np.zeros([nx,nz])


# ==============================================================================
# Devito Parameters
from poroelastic import demo_model, TimeAxis, RickerSource, PoroelasticWaveSolver, Receiver
from devito import Grid, Function

preset = 'constant-poroelastic'
shape = (Lx, Lz)
spacing = (hx, hz)
space_order = 8
nbpml = 40
dtype = np.float64
origin =  tuple([0. for _ in shape])

# Generate test grid
shape_pml = np.array(shape) + 2 * nbpml
extent = tuple(np.array(spacing) * (shape_pml - 1))
grid = Grid(extent=extent, shape=shape_pml, origin=origin, dtype=dtype)
fun_rho_s = Function(name='rho_s', grid=grid, space_order=space_order, dtype=dtype)

# Use existing method to generate model
model = demo_model(preset, space_order=space_order, shape=shape, nbpml=nbpml,
                   dtype=dtype, spacing=spacing)

# Generate Source
dt = model.critical_dt # sec
t0 = 0.0 # sec
f0 = 50.0 # Hz
time_range = TimeAxis(start=t0, stop=tn, step=dt)
src = RickerSource(name='src', grid=model.grid, f0=f0, time_range=time_range)
# Set source at center of volume
src.coordinates.data[0, :] = np.array(model.domain_size) * .5
src.coordinates.data[:, :] = np.array(model.domain_size) * .5

# Denote receiver array
nrec = 2*shape[0]
rec = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=nrec)
rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)


# ==============================================================================
# Solver Parameters
autotune = False
dse = "advanced"
dle = "advanced"
save = False

# Create solver object to provide relevant operators
solver = PoroelasticWaveSolver(model, source=src, receiver=rec, space_order=space_order, dtype=dtype, dse=dse, dle=dle)
rec1, rec2, vx, vz, qx, qz, txx, tzz, txz, p, summary = solver.forward(autotune=autotune,save=save)



