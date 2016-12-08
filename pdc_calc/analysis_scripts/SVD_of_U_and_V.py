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
""" This script performs singular-value-decompositions of the U and V matrices
of the different solutions. A new directory SVD_modes_and_amplitudes is created
storing and visualizing the results.

Attention: The SVD algorithm can only obtain positive singular values. For
frequency conversion processes with negative process amplitudes the wrong
singular values and mode shapes are returned. This happens for conversion
efficiencies after unity for U and conversion efficiencies which went back to 0
after passing unity for V. This is an inherent problem of the SVD algorithm and
there is no easy fix for this.
"""

import sys, os
import numpy as np
import pickle
from scipy import linalg
import matplotlib.pyplot as plt

print "##################"
print "# SVD of U and V #"
print "##################"

# Path the analysis directory
path = sys.argv[1]

#################################
# Import the process properties #
#################################

# Load process properties via pickle
pkl_file = open(path + 'PDC_properties.pkl', 'rb')
pdc_properties = pickle.load(pkl_file)
pkl_file.close()

# Calculate necessary arrays:
wRange = np.linspace(pdc_properties['w_start'], pdc_properties['w_stop'], \
        pdc_properties['w_steps'])

zRange = np.linspace(pdc_properties['z_start'], pdc_properties['z_stop'], \
        pdc_properties['z_steps'])

## Calculate Uinitial which also serves as the discretized delta function
## in the calculations
Uinitial = np.identity(wRange.size, dtype=complex) / pdc_properties['w_step']


###############################
# Import the U and V matrices #
###############################

# Load analytic solution
va_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/va_ana.txt.gz').view(complex)
ua_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/ua_ana.txt.gz').view(complex)
vb_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/vb_ana.txt.gz').view(complex)
ub_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/ub_ana.txt.gz').view(complex)

# Load first-order numerical solution
va_num = np.loadtxt(path + 'U_and_V_matrices/data/num/va_num.txt.gz').view(complex)
ua_num = np.loadtxt(path + 'U_and_V_matrices/data/num/ua_num.txt.gz').view(complex)
vb_num = np.loadtxt(path + 'U_and_V_matrices/data/num/vb_num.txt.gz').view(complex)
ub_num = np.loadtxt(path + 'U_and_V_matrices/data/num/ub_num.txt.gz').view(complex)

# Load rigorous solution
va_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/va_rig.txt.gz').view(complex)
ua_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/ua_rig.txt.gz').view(complex)
vb_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/vb_rig.txt.gz').view(complex)
ub_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/ub_rig.txt.gz').view(complex)

# Save reconstructed rigorous solution
va_rec = np.loadtxt(path + 'U_and_V_matrices/data/rec/va_rec.txt.gz').view(complex)
ua_rec = np.loadtxt(path + 'U_and_V_matrices/data/rec/ua_rec.txt.gz').view(complex)
vb_rec = np.loadtxt(path + 'U_and_V_matrices/data/rec/vb_rec.txt.gz').view(complex)
ub_rec = np.loadtxt(path + 'U_and_V_matrices/data/rec/ub_rec.txt.gz').view(complex)


##################################################################
# Perform SVD of all solution to obtain the amplitudes and modes #
##################################################################

# This seems a bit backwards because we used the SVD to construct some
# solutions in the beginning however performing an SVD again is more 
# consistent.


# Notation:
# We write the decomposition of U as follows:
# uA, uD, uBh = svd_u(U)
# And the decomposition of V as:
# vA, vD, vBh = svd_v(V)
# This means the first letter gives if we talk about the U or V matrices
#
# A:  Labels the left unitary matrix of the SVD storing the obtained 
#     broadband modes, which is transposed so that the individual modes can
#     be accessed via A[i]
# D:  Labels the inner matrix which only has real positive diagonal entries.
#     (Except for the fact that after the SVD the separate treatment of the
#     delta function in the case of U matrices can return negative values).
#     Only an array with these values is returned.
# Bh: Labels the right unitary matrix of the SVD storing the second set of
#     obtained broadband modes, where the individual modes can directly be
#     accessed via Bh[j]


def svd_u(u, w_step):
    """ Gives the modes and singular values obtained from the u_functions.
    (with the proper normalization.)

    This functions is specially written for the u matrices to cope with the
    involved delta functions which otherwise would lead to numerical issues.
    """
    Uinitial = np.identity(wRange.size, dtype=complex) / w_step

    # Never ever directly diagonalize u the delta function Uinitial will 
    # screw up the svd algorithm.
    uA, uDp, uBh = linalg.svd(u - Uinitial)

    # Normalize basis modes
    uA = uA / np.sqrt(w_step)
    uBh = uBh / np.sqrt(w_step)

    # Turn uA in order to make the modes accessible via uA[i]
    uA = np.transpose(uA)

    # It is vital to explicitly copy the arrays vA and vBh into new arrays,
    # since the representation of the imaginary numbers in the arrays returned
    # from the svd algorithm interferes with the views used to save the data.
    # This is circumvented via this explicit copy operation. The problem also
    # occurs for transposed arrays and I am not aware of an easy fix, except
    # changing all the save and load methods.
    uA2 = uA.copy()
    uBh2 = uBh.copy()

    # Normalize uDp:
    uDp = uDp * w_step
    # Calculate the singular values
    uD = 1 + uDp

    return uA2, uD, uBh2


def svd_v(v, w_step):
    """ Gives the modes and singular values obtained from v. 
    (with the proper normalization.)"""

    vA, vD, vBh = linalg.svd(v)

    # Normalize basis modes
    vA = vA / np.sqrt(w_step)
    vBh = vBh / np.sqrt(w_step)

    # Turn vA in order to make the modes accessible via vA[i]
    vA = np.transpose(vA)

    # It is vital to explicitly copy the arrays vA and vBh into new arrays,
    # since the representation of the imaginary numbers in the arrays returned
    # from the svd algorithm interferes with the views used to save the data.
    # This is circumvented via this explicit copy operation. The problem also
    # occurs for transposed arrays and I am not aware of an easy fix, except
    # changing all the save and load methods.
    vA2 = vA.copy()
    vBh2 = vBh.copy()

    # Normalize vD:
    vD = vD * w_step

    return vA2, vD, vBh2


# Calculate Schmidt values and modes for the analytical solution
uaA_ana, uaD_ana, uaBh_ana = svd_u(ua_ana, pdc_properties['w_step'])
vaA_ana, vaD_ana, vaBh_ana = svd_v(va_ana, pdc_properties['w_step'])

ubA_ana, ubD_ana, ubBh_ana = svd_u(ub_ana, pdc_properties['w_step'])
vbA_ana, vbD_ana, vbBh_ana = svd_v(vb_ana, pdc_properties['w_step'])

# Calculate Schmidt values and modes for the numerical solution
uaA_num, uaD_num, uaBh_num = svd_u(ua_num, pdc_properties['w_step'])
vaA_num, vaD_num, vaBh_num = svd_v(va_num, pdc_properties['w_step'])

ubA_num, ubD_num, ubBh_num = svd_u(ub_num, pdc_properties['w_step'])
vbA_num, vbD_num, vbBh_num = svd_v(vb_num, pdc_properties['w_step'])

# Calculate Schmidt values and modes for the rigorous solution
uaA_rig, uaD_rig, uaBh_rig = svd_u(ua_rig, pdc_properties['w_step'])
vaA_rig, vaD_rig, vaBh_rig = svd_v(va_rig, pdc_properties['w_step'])

ubA_rig, ubD_rig, ubBh_rig = svd_u(ub_rig, pdc_properties['w_step'])
vbA_rig, vbD_rig, vbBh_rig = svd_v(vb_rig, pdc_properties['w_step'])

# Calculate Schmidt values and modes for the reconstructed rigorous solution
uaA_rec, uaD_rec, uaBh_rec = svd_u(ua_rec, pdc_properties['w_step'])
vaA_rec, vaD_rec, vaBh_rec = svd_v(va_rec, pdc_properties['w_step'])

ubA_rec, ubD_rec, ubBh_rec = svd_u(ub_rec, pdc_properties['w_step'])
vbA_rec, vbD_rec, vbBh_rec = svd_v(vb_rec, pdc_properties['w_step'])



# There are some numerical issues with the U_rig matrices We hence take a look
# at the decomposition of UUT and UTU which is *much* more stable. (Looking at
# the reconstructed solution obviously helps as well, since it uses the modes
# from the V matrix).


## Calculate the singular values and modes for all combinations of UUT and UTU
print "Calculating SVD of UUT and UTU"
uuta_rig = np.dot(ua_rig, np.conjugate(np.transpose(ua_rig))) * pdc_properties['w_step']
uutb_rig = np.dot(ub_rig, np.conjugate(np.transpose(ub_rig))) * pdc_properties['w_step']
utua_rig = np.dot(np.conjugate(np.transpose(ua_rig)), ua_rig) * pdc_properties['w_step']
utub_rig = np.dot(np.conjugate(np.transpose(ub_rig)), ub_rig) * pdc_properties['w_step']


uutaA_rig, uutaD_rig, uutaBh_rig = svd_u(uuta_rig, pdc_properties['w_step'])
uutbA_rig, uutbD_rig, uutbBh_rig = svd_u(uutb_rig, pdc_properties['w_step'])
utuaA_rig, utuaD_rig, utuaBh_rig = svd_u(utua_rig, pdc_properties['w_step'])
utubA_rig, utubD_rig, utubBh_rig = svd_u(utub_rig, pdc_properties['w_step'])


# From the diagonal elements of UUT and UTU obtain the corresponding
# U diagonal elements:
uutaD_rig = np.sqrt(uutaD_rig)
utuaD_rig = np.sqrt(utuaD_rig)
uutbD_rig = np.sqrt(uutbD_rig)
utubD_rig = np.sqrt(utubD_rig)



############################
# Save the results to file #
############################


os.mkdir(path + "/SVD_modes_and_amplitudes/")
os.mkdir(path + "/SVD_modes_and_amplitudes/data/")

os.mkdir(path + "/SVD_modes_and_amplitudes/data/ana")
os.mkdir(path + "/SVD_modes_and_amplitudes/data/num")
os.mkdir(path + "/SVD_modes_and_amplitudes/data/rig")
os.mkdir(path + "/SVD_modes_and_amplitudes/data/rec")
os.mkdir(path + "/SVD_modes_and_amplitudes/data/uut")
os.mkdir(path + "/SVD_modes_and_amplitudes/data/utu")

# The Schmidt amplitudes are already real so the view(float) trick to 
# save imaginary values is not required in their case

# Save analytic solution
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/uaA_ana.txt.gz", uaA_ana.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/uaD_ana.txt.gz", uaD_ana)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/uaBh_ana.txt.gz", uaBh_ana.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/vaA_ana.txt.gz", vaA_ana.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/vaD_ana.txt.gz", vaD_ana)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/vaBh_ana.txt.gz", vaBh_ana.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/ubA_ana.txt.gz", ubA_ana.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/ubD_ana.txt.gz", ubD_ana)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/ubBh_ana.txt.gz", ubBh_ana.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/vbA_ana.txt.gz", vbA_ana.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/vbD_ana.txt.gz", vbD_ana)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/vbBh_ana.txt.gz", vbBh_ana.view(float))

# Save numerical first-order solution
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/uaA_num.txt.gz", uaA_num.view(float)) 
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/uaD_num.txt.gz", uaD_num)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/uaBh_num.txt.gz", uaBh_num.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/vaA_num.txt.gz", vaA_num.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/vaD_num.txt.gz", vaD_num)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/vaBh_num.txt.gz", vaBh_num.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/ubA_num.txt.gz", ubA_num.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/ubD_num.txt.gz", ubD_num)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/ubBh_num.txt.gz", ubBh_num.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/vbA_num.txt.gz", vbA_num.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/vbD_num.txt.gz", vbD_num)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/vbBh_num.txt.gz", vbBh_num.view(float))

# Save rigorous solution
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/uaA_rig.txt.gz", uaA_rig.view(float)) 
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/uaD_rig.txt.gz", uaD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/uaBh_rig.txt.gz", uaBh_rig.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/vaA_rig.txt.gz", vaA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/vaD_rig.txt.gz", vaD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/vaBh_rig.txt.gz", vaBh_rig.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/ubA_rig.txt.gz", ubA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/ubD_rig.txt.gz", ubD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/ubBh_rig.txt.gz", ubBh_rig.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/vbA_rig.txt.gz", vbA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/vbD_rig.txt.gz", vbD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/vbBh_rig.txt.gz", vbBh_rig.view(float))

# Save reconstructed rigorous solution
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/uaA_rec.txt.gz", uaA_rec.view(float)) 
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/uaD_rec.txt.gz", uaD_rec)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/uaBh_rec.txt.gz", uaBh_rec.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/vaA_rec.txt.gz", vaA_rec.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/vaD_rec.txt.gz", vaD_rec)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/vaBh_rec.txt.gz", vaBh_rec.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/ubA_rec.txt.gz", ubA_rec.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/ubD_rec.txt.gz", ubD_rec)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/ubBh_rec.txt.gz", ubBh_rec.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/vbA_rec.txt.gz", vbA_rec.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/vbD_rec.txt.gz", vbD_rec)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/vbBh_rec.txt.gz", vbBh_rec.view(float))


# Save the modes and amplitudes from the UUT matrices
np.savetxt(path + "/SVD_modes_and_amplitudes/data/uut/uutaA_rig.txt.gz", uutaA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/uut/uutaD_rig.txt.gz", uutaD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/uut/uutaBh_rig.txt.gz", uutaBh_rig.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/uut/uutbA_rig.txt.gz", uutbA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/uut/uutbD_rig.txt.gz", uutbD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/uut/uutbBh_rig.txt.gz", uutbBh_rig.view(float))

# Save the modes and amplitudes from the UTU matrices
np.savetxt(path + "/SVD_modes_and_amplitudes/data/utu/utuaA_rig.txt.gz", utuaA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/utu/utuaD_rig.txt.gz", utuaD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/utu/utuaBh_rig.txt.gz", utuaBh_rig.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/utu/utubA_rig.txt.gz", utubA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/utu/utubD_rig.txt.gz", utubD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/utu/utubBh_rig.txt.gz", utubBh_rig.view(float))



################################################################
# Plot the resulting modes and amplitudes and save the figures # 
################################################################

os.mkdir(path + "/SVD_modes_and_amplitudes/pic")


# Plot SVD amplitudes
plt.figure(figsize=(12,8))

N = 10
ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars

plt.subplot(221)
rects1 = plt.bar(ind, vaD_ana[:N], width, color='r')
rects2 = plt.bar(ind+width, vaD_num[:N], width, color='b')
rects3 = plt.bar(ind+2*width, vaD_rig[:N], width, color='g')
rects4 = plt.bar(ind+3*width, vaD_rec[:N], width, color='y')
plt.ylabel(r'$\sinh(r_k)$', fontsize=20)
plt.title('SVD Va', fontsize=20)
plt.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('ana', 'num', 'rig', 'rec'), loc = "lower right")


plt.subplot(222)
rects1 = plt.bar(ind, vbD_ana[:N], width, color='r')
rects2 = plt.bar(ind+width, vbD_num[:N], width, color='b')
rects3 = plt.bar(ind+2*width, vbD_rig[:N], width, color='g')
rects4 = plt.bar(ind+3*width, vbD_rec[:N], width, color='c')
plt.ylabel(r'$\sinh(r_k)$', fontsize=20)
plt.title('SVD vb', fontsize=20)
plt.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('ana', 'num', 'rig', 'rec'), loc = "lower right")


width = 0.15       # the width of the bars


plt.subplot(223)
rects1 = plt.bar(ind, uaD_ana[:N], width, color='r')
rects2 = plt.bar(ind+width, uaD_num[:N], width, color='b')
rects3 = plt.bar(ind+2*width, uaD_rig[:N], width, color='y')
rects4 = plt.bar(ind+3*width, uaD_rec[:N], width, color='g')
rects5 = plt.bar(ind+4*width, uutaD_rig[:N], width, color='c')
rects6 = plt.bar(ind+5*width, utuaD_rig[:N], width, color='c')
plt.ylabel(r'$\cosh(r_k)$', fontsize=20)
plt.title('SVD Ua', fontsize=20)
plt.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]), ('ana', 'num', 'rig', 'rec', 'uut rig', 'utu rig'), loc = "lower right")


plt.subplot(224)
rects1 = plt.bar(ind, ubD_ana[:N], width, color='r')
rects2 = plt.bar(ind+width, ubD_num[:N], width, color='b')
rects3 = plt.bar(ind+2*width, ubD_rig[:N], width, color='y')
rects4 = plt.bar(ind+3*width, ubD_rec[:N], width, color='g')
rects5 = plt.bar(ind+4*width, uutbD_rig[:N], width, color='c')
rects6 = plt.bar(ind+5*width, utubD_rig[:N], width, color='c')
plt.ylabel(r'$\cosh(r_k)$', fontsize=20)
plt.title('SVD ub', fontsize=20)
plt.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]), ('ana', 'num', 'rig', 'rec', 'uut rig', 'utu rig'), loc = "lower right")


plt.savefig(path + '/SVD_modes_and_amplitudes/pic/svd_amplitudes.png')
plt.close()


## Plot SVD modes


# Analytic solution
fig = plt.figure(figsize=(15,7))

plt.subplot(2,2,1)
plt.plot(np.abs(vaA_ana[0]), linewidth=4,  color='r')
plt.plot(np.abs(vaA_ana[1]), linewidth=4,  color='b')
plt.plot(np.abs(vaA_ana[2]), linewidth=4,  color='g')
plt.plot(np.abs(vaA_ana[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs A (ana / V)")

plt.subplot(2,2,2)
plt.plot(np.abs(vbA_ana[0]), linewidth=4,  color='r')
plt.plot(np.abs(vbA_ana[1]), linewidth=4,  color='b')
plt.plot(np.abs(vbA_ana[2]), linewidth=4,  color='g')
plt.plot(np.abs(vbA_ana[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs B (ana / V)")

plt.subplot(2,2,3)
plt.plot(np.abs(uaA_ana[0]), linewidth=4,  color='r')
plt.plot(np.abs(uaA_ana[1]), linewidth=4,  color='b')
plt.plot(np.abs(uaA_ana[2]), linewidth=4,  color='g')
plt.plot(np.abs(uaA_ana[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs A (ana / U)")

plt.subplot(2,2,4)
plt.plot(np.abs(ubA_ana[0]), linewidth=4,  color='r')
plt.plot(np.abs(ubA_ana[1]), linewidth=4,  color='b')
plt.plot(np.abs(ubA_ana[2]), linewidth=4,  color='g')
plt.plot(np.abs(ubA_ana[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs B (ana / U)")

plt.savefig(path + '/SVD_modes_and_amplitudes/pic/svd_modes_ana.png')
plt.close()


# First-order numerical solution
fig = plt.figure(figsize=(15,7))

plt.subplot(2,2,1)
plt.plot(np.abs(vaA_num[0]), linewidth=4,  color='r')
plt.plot(np.abs(vaA_num[1]), linewidth=4,  color='b')
plt.plot(np.abs(vaA_num[2]), linewidth=4,  color='g')
plt.plot(np.abs(vaA_num[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs A (num / V)")

plt.subplot(2,2,2)
plt.plot(np.abs(vbA_num[0]), linewidth=4,  color='r')
plt.plot(np.abs(vbA_num[1]), linewidth=4,  color='b')
plt.plot(np.abs(vbA_num[2]), linewidth=4,  color='g')
plt.plot(np.abs(vbA_num[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs B (num / V)")

plt.subplot(2,2,3)
plt.plot(np.abs(uaA_num[0]), linewidth=4,  color='r')
plt.plot(np.abs(uaA_num[1]), linewidth=4,  color='b')
plt.plot(np.abs(uaA_num[2]), linewidth=4,  color='g')
plt.plot(np.abs(uaA_num[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs A (num / U)")

plt.subplot(2,2,4)
plt.plot(np.abs(ubA_num[0]), linewidth=4,  color='r')
plt.plot(np.abs(ubA_num[1]), linewidth=4,  color='b')
plt.plot(np.abs(ubA_num[2]), linewidth=4,  color='g')
plt.plot(np.abs(ubA_num[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs B (num / U)")

plt.savefig(path + '/SVD_modes_and_amplitudes/pic/svd_modes_num.png')
plt.close()


# Rigorous solution
fig = plt.figure(figsize=(15,7))

plt.subplot(2,2,1)
plt.plot(np.abs(vaA_rig[0]), linewidth=4,  color='r')
plt.plot(np.abs(vaA_rig[1]), linewidth=4,  color='b')
plt.plot(np.abs(vaA_rig[2]), linewidth=4,  color='g')
plt.plot(np.abs(vaA_rig[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs A (rig / V)")

plt.subplot(2,2,2)
plt.plot(np.abs(vbA_rig[0]), linewidth=4,  color='r')
plt.plot(np.abs(vbA_rig[1]), linewidth=4,  color='b')
plt.plot(np.abs(vbA_rig[2]), linewidth=4,  color='g')
plt.plot(np.abs(vbA_rig[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs B (rig / V)")

plt.subplot(2,2,3)
plt.plot(np.abs(uaA_rig[0]), linewidth=4,  color='r')
plt.plot(np.abs(uaA_rig[1]), linewidth=4,  color='b')
plt.plot(np.abs(uaA_rig[2]), linewidth=4,  color='g')
plt.plot(np.abs(uaA_rig[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs A (rig / U)")

plt.subplot(2,2,4)
plt.plot(np.abs(ubA_rig[0]), linewidth=4,  color='r')
plt.plot(np.abs(ubA_rig[1]), linewidth=4,  color='b')
plt.plot(np.abs(ubA_rig[2]), linewidth=4,  color='g')
plt.plot(np.abs(ubA_rig[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs B (rig / U)")


plt.savefig(path + '/SVD_modes_and_amplitudes/pic/svd_modes_rig.png')
plt.close()


# Reconstructed rigorous solution
fig = plt.figure(figsize=(15,7))

plt.subplot(2,2,1)
plt.plot(np.abs(vaA_rec[0]), linewidth=4,  color='r')
plt.plot(np.abs(vaA_rec[1]), linewidth=4,  color='b')
plt.plot(np.abs(vaA_rec[2]), linewidth=4,  color='g')
plt.plot(np.abs(vaA_rec[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs A (rec / V)")

plt.subplot(2,2,2)
plt.plot(np.abs(vbA_rec[0]), linewidth=4,  color='r')
plt.plot(np.abs(vbA_rec[1]), linewidth=4,  color='b')
plt.plot(np.abs(vbA_rec[2]), linewidth=4,  color='g')
plt.plot(np.abs(vbA_rec[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs B (rec / V)")

plt.subplot(2,2,3)
plt.plot(np.abs(uaA_rec[0]), linewidth=4,  color='r')
plt.plot(np.abs(uaA_rec[1]), linewidth=4,  color='b')
plt.plot(np.abs(uaA_rec[2]), linewidth=4,  color='g')
plt.plot(np.abs(uaA_rec[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs A (rec / U)")

plt.subplot(2,2,4)
plt.plot(np.abs(ubA_rec[0]), linewidth=4,  color='r')
plt.plot(np.abs(ubA_rec[1]), linewidth=4,  color='b')
plt.plot(np.abs(ubA_rec[2]), linewidth=4,  color='g')
plt.plot(np.abs(ubA_rec[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs B (rec / U)")

plt.savefig(path + '/SVD_modes_and_amplitudes/pic/svd_modes_rec.png')
plt.close()
