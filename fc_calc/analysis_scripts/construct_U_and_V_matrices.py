# fc_pdc_calc: Calculates the time-ordered solutions of ultrafast frequency
# conversion / parametric down-conversion and evaluates the process parameters.
#     Copyright (C) 2013  Andreas Christ
# 
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.


#! /usr/bin/env python
""" This script construct the U and V matrices of the different solutions from
the files in the raw_data directory. 

The directory in which the raw_data directory resides has to be supplied via
command line. A new directory U_and_V matrices storing the matrices is created
next to it.

We consider 4 different solution matrices:

- Analytic solution (short ana): Created from the analytic formulas in the
  corresponding paper

- First-order numerical solution (num): This solution is identical to the
  analytic one except for the fact that the z-integration is performed
  numerically. This is useful to check the z-integration routine.

- Rigorous solution (rig): This solution is obtained from solving the
  differential equations in the corresponding paper.

- Reconstructed rigorous solution (rig_recon): This solution is identical to
  the rigorous solution except for the fact that the U matrices are constructed
  from the V matrices to get rid of numerical issues in the U matrices.
"""

import sys, os
import numpy as np
import pickle
from scipy import linalg
import matplotlib.pyplot as plt

import lib

print "########################"
print "# Constructing U and V #"
print "########################"

# Path the analysis directory
path = sys.argv[1]


#################################
# Import the process properties #
#################################

# Load process properties via pickle
pkl_file = open(path + 'FC_properties.pkl', 'rb')
fc_properties = pickle.load(pkl_file)
pkl_file.close()

# Calculate necessary arrays:
wRange = np.linspace(fc_properties['w_start'], fc_properties['w_stop'], \
        fc_properties['w_steps'])

zRange = np.linspace(fc_properties['z_start'], fc_properties['z_stop'], \
        fc_properties['z_steps'])

## Calculate Uinitial which also serves as the discretized delta function
## in the calculations
Uinitial = np.identity(wRange.size, dtype=complex) / fc_properties['w_step']


#######################
# Import the raw data #
#######################

# The view options reconstructs the saved floats to the original
# complex numbers
ra_ana = np.loadtxt(path + 'raw_data/ra_ana.txt.gz').view(complex)
ra_num = np.loadtxt(path + 'raw_data/ra_num.txt.gz').view(complex)
rc_ana = np.loadtxt(path + 'raw_data/rc_ana.txt.gz').view(complex)
rc_num = np.loadtxt(path + 'raw_data/rc_num.txt.gz').view(complex)

va_rig = np.loadtxt(path + 'raw_data/va_rig.txt.gz').view(complex)
ua_rig = np.loadtxt(path + 'raw_data/ua_rig.txt.gz').view(complex)
vc_rig = np.loadtxt(path + 'raw_data/vc_rig.txt.gz').view(complex)
uc_rig = np.loadtxt(path + 'raw_data/uc_rig.txt.gz').view(complex)


###########################################################
# Construct analytical and first order numerical solution #
###########################################################

# Construct U and V for the analytical solution
ua_ana , va_ana = lib.construct_u_and_v(ra_ana, fc_properties['w_step'], wRange)
uc_ana , vc_ana = lib.construct_u_and_v(rc_ana, fc_properties['w_step'], wRange)

# Construct U and V for the numerical solution
ua_num, va_num = lib.construct_u_and_v(ra_num, fc_properties['w_step'], wRange)
uc_num, vc_num = lib.construct_u_and_v(rc_num, fc_properties['w_step'], wRange)


##########################################
# Create reconstructed rigorous solution #
##########################################

def construct_rig_solution_ua_and_uc_from_v_rig(v, w_step):
    """ This function constructs the rigorous solutions from a given va_rig.
    This is necessary since we want to the ua_rig and ub_rig matrices without
    their numerical issues. In order for the corresponding tests to pass also
    va_rig and vb_rig have to be reconstructed otherwise the phases don't work
    out.  This is an issue with the SVD algorithm."""
    Uinitial = np.identity(wRange.size, dtype=complex) / w_step

    # Perform SVD to get the modes
    vA, vD, vBh = linalg.svd(v)
    # Correct for w_step to get actual v amplitudes
    vD = vD * w_step
    rD = np.arcsin(vD)
    # Calculate vD and uDp singular values
    # (uDp = singular values of uD without identity)
    vD = np.sin(rD)
    uD = np.cos(rD)

    uDp = 1 - uD


    # There has to be a better way to write this:
    for i in uD:
        if i < 0:
            print """ uD has negative amplitudes. 
            The singular value decomposition of this U performed later on 
            will however return positive ones. This is an inherent problem 
            of the SVD algorithm and has to be kept in mind in the analysis."""

    # Transform into numerical basis again.
    vD = vD / w_step
    uDp = uDp / w_step

    A = vA
    B = np.transpose(np.conjugate(vBh))

    # Construct U_a
    # The trick here is to construct U while treating the delta function 
    # separately since the direct calculation leads to numerical issues.
    u_a = Uinitial - np.dot(A, np.dot(np.diag(uDp), \
        np.conjugate(np.transpose(A))))

    # Construct V_a
    v_a = np.dot(np.dot(A, np.diag(vD)), np.transpose(np.conjugate(B)))

    # Construct U_c 
    # The complete solution is reconstructed from a single-matrix
    # otherwise the phase factors don't work out
    u_c = Uinitial - np.dot(B, np.dot(np.diag(uDp), \
        np.conjugate(np.transpose(B))))

    # Construct V_c
    v_c = np.dot(np.dot(B, np.diag(vD)), np.transpose(np.conjugate(A)))

    return u_a, v_a, u_c, v_c



# Reconstruct the rigorous solution from a given v_rig. This
# enables us later to plot the ua_rig and uc_rig functions 
# without any numerical issues.
ua_rec, va_rec, uc_rec, vc_rec = \
        construct_rig_solution_ua_and_uc_from_v_rig(va_rig, fc_properties['w_step'])


##########################
# Save solutions to file #
##########################

os.mkdir(path + "/U_and_V_matrices")
os.mkdir(path + "/U_and_V_matrices/data")

os.mkdir(path + "/U_and_V_matrices/data/ana")
os.mkdir(path + "/U_and_V_matrices/data/num")
os.mkdir(path + "/U_and_V_matrices/data/rig")
os.mkdir(path + "/U_and_V_matrices/data/rec")

# Save analytic solution
np.savetxt(path + "/U_and_V_matrices/data/ana/va_ana.txt.gz", va_ana.view(float))
np.savetxt(path + "/U_and_V_matrices/data/ana/ua_ana.txt.gz", ua_ana.view(float))
np.savetxt(path + "/U_and_V_matrices/data/ana/vc_ana.txt.gz", vc_ana.view(float))
np.savetxt(path + "/U_and_V_matrices/data/ana/uc_ana.txt.gz", uc_ana.view(float))

# Save first-order numerical solution
np.savetxt(path + "/U_and_V_matrices/data/num/va_num.txt.gz", va_num.view(float))
np.savetxt(path + "/U_and_V_matrices/data/num/ua_num.txt.gz", ua_num.view(float))
np.savetxt(path + "/U_and_V_matrices/data/num/vc_num.txt.gz", vc_num.view(float))
np.savetxt(path + "/U_and_V_matrices/data/num/uc_num.txt.gz", uc_num.view(float))

# Save rigorous solution
# (In principle one could also just copy the files from the raw_data)
np.savetxt(path + "/U_and_V_matrices/data/rig/va_rig.txt.gz", va_rig.view(float))
np.savetxt(path + "/U_and_V_matrices/data/rig/ua_rig.txt.gz", ua_rig.view(float))
np.savetxt(path + "/U_and_V_matrices/data/rig/vc_rig.txt.gz", vc_rig.view(float))
np.savetxt(path + "/U_and_V_matrices/data/rig/uc_rig.txt.gz", uc_rig.view(float))

# Save reconstructed rigorous solution
np.savetxt(path + "/U_and_V_matrices/data/rec/va_rec.txt.gz", va_rec.view(float))
np.savetxt(path + "/U_and_V_matrices/data/rec/ua_rec.txt.gz", ua_rec.view(float))
np.savetxt(path + "/U_and_V_matrices/data/rec/vc_rec.txt.gz", vc_rec.view(float))
np.savetxt(path + "/U_and_V_matrices/data/rec/uc_rec.txt.gz", uc_rec.view(float))



####################################################
# Plot the resulting matrices and save the figures # 
####################################################

os.mkdir(path + "/U_and_V_matrices/pic")

w_start = fc_properties['w_start']
w_stop = fc_properties['w_stop']

# Plot U and V of the analytical solution
fig = plt.figure(figsize=(12,8))

fig.suptitle('Analytic solution', fontsize=20)

plt.subplot(2,3,1)
plt.imshow(np.abs(ua_ana), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_a(\omega,\omega')|$")
plt.colorbar()

plt.subplot(2,3,2)
plt.imshow(np.abs(ua_ana - Uinitial), origin="lower", cmap=plt.cm.hot, extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_a(\omega,\omega')| - \delta(\omega-\omega')$")
plt.colorbar()

plt.subplot(2,3,3)
plt.imshow(np.abs(va_ana), origin="lower", cmap=plt.cm.hot, extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|V_a(\omega, \omega')|$")
plt.colorbar()

plt.subplot(2,3,4)
plt.imshow(np.abs(uc_ana), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_c(\omega,\omega')|$")
plt.colorbar()

plt.subplot(2,3,5)
plt.imshow(np.abs(uc_ana - Uinitial), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_c(\omega,\omega')| - \delta(\omega-\omega')$")
plt.colorbar()

plt.subplot(2,3,6)
plt.imshow(np.abs(vc_ana), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|V_c(\omega, \omega')|$")
plt.colorbar()

plt.savefig(path + '/U_and_V_matrices/pic/analytical_solution.png')
plt.close()


# Plot U and V of the first-order numerical solution
fig = plt.figure(figsize=(12,8))

fig.suptitle('First-order numerical solution', fontsize=20)

plt.subplot(2,3,1)
plt.imshow(np.abs(ua_num), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_a(\omega,\omega')|$")
plt.colorbar()

plt.subplot(2,3,2)
plt.imshow(np.abs(ua_num - Uinitial), origin="lower", cmap=plt.cm.hot, extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_a(\omega,\omega')| - \delta(\omega-\omega')$")
plt.colorbar()

plt.subplot(2,3,3)
plt.imshow(np.abs(va_num), origin="lower", cmap=plt.cm.hot, extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|V_a(\omega, \omega')|$")
plt.colorbar()

plt.subplot(2,3,4)
plt.imshow(np.abs(uc_num), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_c(\omega,\omega')|$")
plt.colorbar()

plt.subplot(2,3,5)
plt.imshow(np.abs(uc_num - Uinitial), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_c(\omega,\omega')| - \delta(\omega-\omega')$")
plt.colorbar()

plt.subplot(2,3,6)
plt.imshow(np.abs(vc_num), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|V_c(\omega, \omega')|$")
plt.colorbar()

plt.savefig(path + '/U_and_V_matrices/pic/first-order_numerical_solution.png')
plt.close()


# Plot U and V of the rigorous solution
fig = plt.figure(figsize=(12,8))

fig.suptitle('Rigorous solution', fontsize=20)

plt.subplot(2,3,1)
plt.imshow(np.abs(ua_rig), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_a(\omega,\omega')|$")
plt.colorbar()

plt.subplot(2,3,2)
plt.imshow(np.abs(ua_rig - Uinitial), origin="lower", cmap=plt.cm.hot, extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_a(\omega,\omega')| - \delta(\omega-\omega')$")
plt.colorbar()

plt.subplot(2,3,3)
plt.imshow(np.abs(va_rig), origin="lower", cmap=plt.cm.hot, extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|V_a(\omega, \omega')|$")
plt.colorbar()

plt.subplot(2,3,4)
plt.imshow(np.abs(uc_rig), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_c(\omega,\omega')|$")
plt.colorbar()

plt.subplot(2,3,5)
plt.imshow(np.abs(uc_rig - Uinitial), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_c(\omega,\omega')| - \delta(\omega-\omega')$")
plt.colorbar()

plt.subplot(2,3,6)
plt.imshow(np.abs(vc_rig), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|V_c(\omega, \omega')|$")
plt.colorbar()

plt.savefig(path + '/U_and_V_matrices/pic/rigorous_solution.png')
plt.close()


# Plot U and V of the reconstructed rigorous solution
fig = plt.figure(figsize=(12,8))

fig.suptitle('Reconstructed rigorous solution', fontsize=20)

plt.subplot(2,3,1)
plt.imshow(np.abs(ua_rec), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_a(\omega,\omega')|$")
plt.colorbar()

plt.subplot(2,3,2)
plt.imshow(np.abs(ua_rec - Uinitial), origin="lower", cmap=plt.cm.hot, extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_a(\omega,\omega')| - \delta(\omega-\omega')$")
plt.colorbar()

plt.subplot(2,3,3)
plt.imshow(np.abs(va_rec), origin="lower", cmap=plt.cm.hot, extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|V_a(\omega, \omega')|$")
plt.colorbar()

plt.subplot(2,3,4)
plt.imshow(np.abs(uc_rec), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_c(\omega,\omega')|$")
plt.colorbar()

plt.subplot(2,3,5)
plt.imshow(np.abs(uc_rec - Uinitial), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|U_c(\omega,\omega')| - \delta(\omega-\omega')$")
plt.colorbar()

plt.subplot(2,3,6)
plt.imshow(np.abs(vc_rec), origin="lower", cmap=plt.cm.hot , extent=(w_start, w_stop, w_start, w_stop) )
plt.title(r"$|V_c(\omega, \omega')|$")
plt.colorbar()

plt.savefig(path + '/U_and_V_matrices/pic/reconstructed_rigorous_solution.png')
plt.close()
