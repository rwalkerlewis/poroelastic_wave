#!/usr/bin/env python

import numpy as np
from numpy import sin, cos
from scipy.integrate import quad
# Calculate approximations for optimized FD coefficients


N = 4
beta = 1.05

# Numerical Integration
f = lambda xi: (np.cos(xi-1.0))*xi
varphi = lambda m, xi: np.sin((m - 0.5)*xi) - (m - 0.5)*xi

def integrandLHS(xi, m, n):
  return varphi(m, xi)*varphi(n, xi)

def integrandRHS(xi, n):
  return varphi(n, xi)*f(xi)

# Generate LHS
LHS = np.zeros(N)
LHS_sum = np.zeros(N)
RHS = np.zeros(N)

for j in np.arange(0,N):
  n = j+1
  for i in np.arange(0,N):
    m = i+1
    LHS_sum[i], e = quad(integrandLHS, 0, beta, args=(m,n))
  LHS[j] = np.sum(LHS_sum)
  RHS[j], e = quad(integrandRHS, 0, beta, args=(n))

LHS = LHS.reshape([N,1])  
RHS = RHS.reshape([N,1])


# Analytical Attempt
def LHS_Analytical(m,n,b):

  A = -1.0*(((192.*m**3.-288.*m**2+144*m-24.)*n**2 + \
      ((-192.*m**3)+288.*m**2-144.*m+24.)*n - \
      192.*m**5+480*m**4-432.*m**3+168.*m**2-24.*m)* \
      sin((2*b*n-b)/2.)+(((-192*b*m**3)+288*b*m**2 - \
      144*b*m+24*b)*n**3+(288*b*m**3-432*b*m**2+216*b*m-36*b)*n**2 + \
      (192.*b*m**5-480.*b*m**4+336.*b*m**3-24*b*m**2-48*b*m+12*b)*n - \
      96*b*m**5+240*b*m**4-216*b*m**3+84*b*m**2-12*b*m) * \
      cos((2*b*n-b)/2.)+((96*m**2-96*m+24)*n**3+((-96.*m**3) + \
      72*m-24)*n**2+(96*m**3-72*m**2+6)*n-24*m**3+24.*m**2-6*m) * 
      sin(b*n+b*m-b)+(((-96*m**2.)+96*m-24)*n**3+((-96.*m**3) + \
      288.*m**2-216*m+48)*n**2+(96*m**3-216*m**2+144.*m-30)*n-24*m**3+48*m**2-30.*m+6.) * \
      sin(b*n-b*m)+(192.*sin((2.*b*m-b)/2.)+(96.*b-192.*b*m)*cos((2*b*m-b)/2) - \
      64.*b**3*m**3+96.*b**3*m**2-48.*b**3*m+8*b**3)*n**5 + \
      ((-480.*sin((2*b*m-b)/2.))+(480.*b*m-240.*b)*cos((2*b*m-b)/2) + \
      160.*b**3*m**3-240.*b**3*m**2+120.*b**3*m-20*b**3)*n**4 + \
      (((-192.*m**2)+192.*m+432.)*sin((2*b*m-b)/2) + \
      (192.*b*m**3-288*b*m**2-336*b*m+216*b)*cos((2*b*m-b)/2.) + \
      64.*b**3*m**5-160.*b**3*m**4 +160.*b**3.*m**2-100.*b**3*m+18*b**3)*n**3 + \
      ((288*m**2-288*m-168)*sin((2*b*m-b)/2)+((-288*b*m**3)+432*b*m**2+24*b*m-84*b) * \
      cos((2.*b*m-b)/2.)-96.*b**3*m**5+240.*b**3*m**4-160.*b**3*m**3+30.*b**3*m-7.*b**3)*n**2 + \
      (((-144*m**2)+144*m+24)*sin((2*b*m-b)/2) + \
      (144.*b*m**3-216.*b*m**2+48.*b*m+12.*b)*cos((2*b*m-b)/2.) + \
      48.*b**3*m**5-120.*b**3*m**4+100.*b**3*m**3-30*b**3*m**2+b**3) * \
      n+(24.*m**2-24.*m)*sin((2*b*m-b)/2.) + \
      ((-24.*b*m**3)+36.*b*m**2-12.*b*m)*cos((2.*b*m-b)/2.) - \
      8.*b**3*m**5+20.*b**3*m**4 - \
      18.*b**3*m**3+7.*b**3.*m**2-b**3*m) / \
      ((192.*m**2-192.*m+48.)*n**4 + \
      ((-384.*m**2)+384.*m-96.)*n**3 + \
      ((-192.*m**4)+384.*m**3-192.*m+60.)*n**2 + \
      (192.*m**4-384.*m**3+192.*m**2-12.)*n-48.*m**4+96.*m**3-60.*m**2+12.*m)
  return A

