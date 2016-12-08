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


###############################
# Import the U and V matrices #
###############################

# Load analytic solution
va_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/va_ana.txt.gz').view(complex)
ua_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/ua_ana.txt.gz').view(complex)
vc_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/vc_ana.txt.gz').view(complex)
uc_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/uc_ana.txt.gz').view(complex)

# Load first-order numerical solution
va_num = np.loadtxt(path + 'U_and_V_matrices/data/num/va_num.txt.gz').view(complex)
ua_num = np.loadtxt(path + 'U_and_V_matrices/data/num/ua_num.txt.gz').view(complex)
vc_num = np.loadtxt(path + 'U_and_V_matrices/data/num/vc_num.txt.gz').view(complex)
uc_num = np.loadtxt(path + 'U_and_V_matrices/data/num/uc_num.txt.gz').view(complex)

# Load rigorous solution
va_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/va_rig.txt.gz').view(complex)
ua_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/ua_rig.txt.gz').view(complex)
vc_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/vc_rig.txt.gz').view(complex)
uc_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/uc_rig.txt.gz').view(complex)

# Save reconstructed rigorous solution
va_rec = np.loadtxt(path + 'U_and_V_matrices/data/rec/va_rec.txt.gz').view(complex)
ua_rec = np.loadtxt(path + 'U_and_V_matrices/data/rec/ua_rec.txt.gz').view(complex)
vc_rec = np.loadtxt(path + 'U_and_V_matrices/data/rec/vc_rec.txt.gz').view(complex)
uc_rec = np.loadtxt(path + 'U_and_V_matrices/data/rec/uc_rec.txt.gz').view(complex)


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
    uA, uDp, uBh = linalg.svd(Uinitial - u)

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
    uD = 1 - uDp

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
uaA_ana, uaD_ana, uaBh_ana = svd_u(ua_ana, fc_properties['w_step'])
vaA_ana, vaD_ana, vaBh_ana = svd_v(va_ana, fc_properties['w_step'])

ucA_ana, ucD_ana, ucBh_ana = svd_u(uc_ana, fc_properties['w_step'])
vcA_ana, vcD_ana, vcBh_ana = svd_v(vc_ana, fc_properties['w_step'])

# Calculate Schmidt values and modes for the numerical solution
uaA_num, uaD_num, uaBh_num = svd_u(ua_num, fc_properties['w_step'])
vaA_num, vaD_num, vaBh_num = svd_v(va_num, fc_properties['w_step'])

ucA_num, ucD_num, ucBh_num = svd_u(uc_num, fc_properties['w_step'])
vcA_num, vcD_num, vcBh_num = svd_v(vc_num, fc_properties['w_step'])

# Calculate Schmidt values and modes for the rigorous solution
uaA_rig, uaD_rig, uaBh_rig = svd_u(ua_rig, fc_properties['w_step'])
vaA_rig, vaD_rig, vaBh_rig = svd_v(va_rig, fc_properties['w_step'])

ucA_rig, ucD_rig, ucBh_rig = svd_u(uc_rig, fc_properties['w_step'])
vcA_rig, vcD_rig, vcBh_rig = svd_v(vc_rig, fc_properties['w_step'])

# Calculate Schmidt values and modes for the reconstructed rigorous solution
uaA_rec, uaD_rec, uaBh_rec = svd_u(ua_rec, fc_properties['w_step'])
vaA_rec, vaD_rec, vaBh_rec = svd_v(va_rec, fc_properties['w_step'])

ucA_rec, ucD_rec, ucBh_rec = svd_u(uc_rec, fc_properties['w_step'])
vcA_rec, vcD_rec, vcBh_rec = svd_v(vc_rec, fc_properties['w_step'])



# There are some numerical issues with the U_rig matrices We hence take a look
# at the decomposition of UUT and UTU which is *much* more stable. (Looking at
# the reconstructed solution obviously helps as well, since it uses the modes
# from the V matrix).


## Calculate the singular values and modes for all combinations of UUT and UTU
print "Calculating SVD of UUT and UTU"
uuta_rig = np.dot(ua_rig, np.conjugate(np.transpose(ua_rig))) * fc_properties['w_step']
uutc_rig = np.dot(uc_rig, np.conjugate(np.transpose(uc_rig))) * fc_properties['w_step']
utua_rig = np.dot(np.conjugate(np.transpose(ua_rig)), ua_rig) * fc_properties['w_step']
utuc_rig = np.dot(np.conjugate(np.transpose(uc_rig)), uc_rig) * fc_properties['w_step']


uutaA_rig, uutaD_rig, uutaBh_rig = svd_u(uuta_rig, fc_properties['w_step'])
uutcA_rig, uutcD_rig, uutcBh_rig = svd_u(uutc_rig, fc_properties['w_step'])
utuaA_rig, utuaD_rig, utuaBh_rig = svd_u(utua_rig, fc_properties['w_step'])
utucA_rig, utucD_rig, utucBh_rig = svd_u(utuc_rig, fc_properties['w_step'])


# From the diagonal elements of UUT and UTU obtain the corresponding
# U diagonal elements:
uutaD_rig = np.sqrt(uutaD_rig)
utuaD_rig = np.sqrt(utuaD_rig)
uutcD_rig = np.sqrt(uutcD_rig)
utucD_rig = np.sqrt(utucD_rig)



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

np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/ucA_ana.txt.gz", ucA_ana.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/ucD_ana.txt.gz", ucD_ana)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/ucBh_ana.txt.gz", ucBh_ana.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/vcA_ana.txt.gz", vcA_ana.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/vcD_ana.txt.gz", vcD_ana)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/ana/vcBh_ana.txt.gz", vcBh_ana.view(float))

# Save numerical first-order solution
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/uaA_num.txt.gz", uaA_num.view(float)) 
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/uaD_num.txt.gz", uaD_num)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/uaBh_num.txt.gz", uaBh_num.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/vaA_num.txt.gz", vaA_num.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/vaD_num.txt.gz", vaD_num)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/vaBh_num.txt.gz", vaBh_num.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/ucA_num.txt.gz", ucA_num.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/ucD_num.txt.gz", ucD_num)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/ucBh_num.txt.gz", ucBh_num.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/vcA_num.txt.gz", vcA_num.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/vcD_num.txt.gz", vcD_num)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/num/vcBh_num.txt.gz", vcBh_num.view(float))

# Save rigorous solution
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/uaA_rig.txt.gz", uaA_rig.view(float)) 
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/uaD_rig.txt.gz", uaD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/uaBh_rig.txt.gz", uaBh_rig.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/vaA_rig.txt.gz", vaA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/vaD_rig.txt.gz", vaD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/vaBh_rig.txt.gz", vaBh_rig.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/ucA_rig.txt.gz", ucA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/ucD_rig.txt.gz", ucD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/ucBh_rig.txt.gz", ucBh_rig.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/vcA_rig.txt.gz", vcA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/vcD_rig.txt.gz", vcD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rig/vcBh_rig.txt.gz", vcBh_rig.view(float))

# Save reconstructed rigorous solution
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/uaA_rec.txt.gz", uaA_rec.view(float)) 
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/uaD_rec.txt.gz", uaD_rec)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/uaBh_rec.txt.gz", uaBh_rec.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/vaA_rec.txt.gz", vaA_rec.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/vaD_rec.txt.gz", vaD_rec)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/vaBh_rec.txt.gz", vaBh_rec.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/ucA_rec.txt.gz", ucA_rec.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/ucD_rec.txt.gz", ucD_rec)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/ucBh_rec.txt.gz", ucBh_rec.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/vcA_rec.txt.gz", vcA_rec.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/vcD_rec.txt.gz", vcD_rec)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/rec/vcBh_rec.txt.gz", vcBh_rec.view(float))


# Save the modes and amplitudes from the UUT matrices
np.savetxt(path + "/SVD_modes_and_amplitudes/data/uut/uutaA_rig.txt.gz", uutaA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/uut/uutaD_rig.txt.gz", uutaD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/uut/uutaBh_rig.txt.gz", uutaBh_rig.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/uut/uutcA_rig.txt.gz", uutcA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/uut/uutcD_rig.txt.gz", uutcD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/uut/uutcBh_rig.txt.gz", uutcBh_rig.view(float))

# Save the modes and amplitudes from the UTU matrices
np.savetxt(path + "/SVD_modes_and_amplitudes/data/utu/utuaA_rig.txt.gz", utuaA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/utu/utuaD_rig.txt.gz", utuaD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/utu/utuaBh_rig.txt.gz", utuaBh_rig.view(float))

np.savetxt(path + "/SVD_modes_and_amplitudes/data/utu/utucA_rig.txt.gz", utucA_rig.view(float))
np.savetxt(path + "/SVD_modes_and_amplitudes/data/utu/utucD_rig.txt.gz", utucD_rig)
np.savetxt(path + "/SVD_modes_and_amplitudes/data/utu/utucBh_rig.txt.gz", utucBh_rig.view(float))



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
plt.ylabel(r'$\sin(r_k)$', fontsize=20)
plt.title('SVD Va', fontsize=20)
plt.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('ana', 'num', 'rig', 'rec'), loc = "lower right")


plt.subplot(222)
rects1 = plt.bar(ind, vcD_ana[:N], width, color='r')
rects2 = plt.bar(ind+width, vcD_num[:N], width, color='b')
rects3 = plt.bar(ind+2*width, vcD_rig[:N], width, color='g')
rects4 = plt.bar(ind+3*width, vcD_rec[:N], width, color='c')
plt.ylabel(r'$\sin(r_k)$', fontsize=20)
plt.title('SVD Vc', fontsize=20)
plt.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('ana', 'num', 'rig', 'rec'), loc = "lower right")


width = 0.15       # the width of the bars


plt.subplot(223)
rects1 = plt.bar(ind, uaD_ana[:N], width, color='r')
rects2 = plt.bar(ind+width, uaD_num[:N], width, color='b')
rects3 = plt.bar(ind+2*width, uaD_rig[:N], width, color='y')
rects4 = plt.bar(ind+3*width, uaD_rec[:N], width, color='g')
rects5 = plt.bar(ind+4*width, uutaD_rig[:N], width, color='c')
rects6 = plt.bar(ind+5*width, utuaD_rig[:N], width, color='c')
plt.ylabel(r'$\cos(r_k)$', fontsize=20)
plt.title('SVD Ua', fontsize=20)
plt.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]), ('ana', 'num', 'rig', 'rec', 'uut rig', 'utu rig'), loc = "lower right")


plt.subplot(224)
rects1 = plt.bar(ind, ucD_ana[:N], width, color='r')
rects2 = plt.bar(ind+width, ucD_num[:N], width, color='b')
rects3 = plt.bar(ind+2*width, ucD_rig[:N], width, color='y')
rects4 = plt.bar(ind+3*width, ucD_rec[:N], width, color='g')
rects5 = plt.bar(ind+4*width, uutcD_rig[:N], width, color='c')
rects6 = plt.bar(ind+5*width, utucD_rig[:N], width, color='c')
plt.ylabel(r'$\cos(r_k)$', fontsize=20)
plt.title('SVD Uc', fontsize=20)
plt.legend( (rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]), ('ana', 'num', 'rig', 'rec', 'uut rig', 'utu rig'), loc = "lower right")


plt.savefig(path + '/SVD_modes_and_amplitudes/pic/svd_amplitudes.png')
plt.close()


## Plot SVD modes

# *Attention*: at high conversion efficiencies the modes obtained from the U
# and V matrices do not match each other directly any more (vaA[i] != uaA[i]).
# The reason for this behaviour is the sin and cos behaviour of U and V.  In
# the low gain regime the highest amplitude of V corresponds to the highest one
# of U. However when cos and sin cannot be approximated by a linear function
# any more this direct correspondence can fail. This effect is directly visible
# by checking if cos(r_k)^2 + sin(r_k)^2 = 1. If not the mode functions have to
# be sorted manually.  This behaviour, while seemingly confusing, is to be
# expected since the singular value decomposition always sorts the obtained
# modes by the magnitude of the corresponding Schmidt values.  One can still
# verify the mode correspondence between the U and V matrices by plotting
# several modes of each matrix and manually matching the mode functions.


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
plt.plot(np.abs(vcA_ana[0]), linewidth=4,  color='r')
plt.plot(np.abs(vcA_ana[1]), linewidth=4,  color='b')
plt.plot(np.abs(vcA_ana[2]), linewidth=4,  color='g')
plt.plot(np.abs(vcA_ana[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs C (ana / V)")

plt.subplot(2,2,3)
plt.plot(np.abs(uaA_ana[0]), linewidth=4,  color='r')
plt.plot(np.abs(uaA_ana[1]), linewidth=4,  color='b')
plt.plot(np.abs(uaA_ana[2]), linewidth=4,  color='g')
plt.plot(np.abs(uaA_ana[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs A (ana / U)")

plt.subplot(2,2,4)
plt.plot(np.abs(ucA_ana[0]), linewidth=4,  color='r')
plt.plot(np.abs(ucA_ana[1]), linewidth=4,  color='b')
plt.plot(np.abs(ucA_ana[2]), linewidth=4,  color='g')
plt.plot(np.abs(ucA_ana[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs C (ana / U)")

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
plt.plot(np.abs(vcA_num[0]), linewidth=4,  color='r')
plt.plot(np.abs(vcA_num[1]), linewidth=4,  color='b')
plt.plot(np.abs(vcA_num[2]), linewidth=4,  color='g')
plt.plot(np.abs(vcA_num[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs C (num / V)")

plt.subplot(2,2,3)
plt.plot(np.abs(uaA_num[0]), linewidth=4,  color='r')
plt.plot(np.abs(uaA_num[1]), linewidth=4,  color='b')
plt.plot(np.abs(uaA_num[2]), linewidth=4,  color='g')
plt.plot(np.abs(uaA_num[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs A (num / U)")

plt.subplot(2,2,4)
plt.plot(np.abs(ucA_num[0]), linewidth=4,  color='r')
plt.plot(np.abs(ucA_num[1]), linewidth=4,  color='b')
plt.plot(np.abs(ucA_num[2]), linewidth=4,  color='g')
plt.plot(np.abs(ucA_num[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs C (num / U)")

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
plt.plot(np.abs(vcA_rig[0]), linewidth=4,  color='r')
plt.plot(np.abs(vcA_rig[1]), linewidth=4,  color='b')
plt.plot(np.abs(vcA_rig[2]), linewidth=4,  color='g')
plt.plot(np.abs(vcA_rig[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs C (rig / V)")

plt.subplot(2,2,3)
plt.plot(np.abs(uaA_rig[0]), linewidth=4,  color='r')
plt.plot(np.abs(uaA_rig[1]), linewidth=4,  color='b')
plt.plot(np.abs(uaA_rig[2]), linewidth=4,  color='g')
plt.plot(np.abs(uaA_rig[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs A (rig / U)")

plt.subplot(2,2,4)
plt.plot(np.abs(ucA_rig[0]), linewidth=4,  color='r')
plt.plot(np.abs(ucA_rig[1]), linewidth=4,  color='b')
plt.plot(np.abs(ucA_rig[2]), linewidth=4,  color='g')
plt.plot(np.abs(ucA_rig[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs C (rig / U)")


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
plt.plot(np.abs(vcA_rec[0]), linewidth=4,  color='r')
plt.plot(np.abs(vcA_rec[1]), linewidth=4,  color='b')
plt.plot(np.abs(vcA_rec[2]), linewidth=4,  color='g')
plt.plot(np.abs(vcA_rec[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs C (rec / V)")

plt.subplot(2,2,3)
plt.plot(np.abs(uaA_rec[0]), linewidth=4,  color='r')
plt.plot(np.abs(uaA_rec[1]), linewidth=4,  color='b')
plt.plot(np.abs(uaA_rec[2]), linewidth=4,  color='g')
plt.plot(np.abs(uaA_rec[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs A (rec / U)")

plt.subplot(2,2,4)
plt.plot(np.abs(ucA_rec[0]), linewidth=4,  color='r')
plt.plot(np.abs(ucA_rec[1]), linewidth=4,  color='b')
plt.plot(np.abs(ucA_rec[2]), linewidth=4,  color='g')
plt.plot(np.abs(ucA_rec[3]), linewidth=4,  color='c')
plt.legend( ('0', '1', '2', '3') )
plt.title("Compare modes abs C (rec / U)")

plt.savefig(path + '/SVD_modes_and_amplitudes/pic/svd_modes_rec.png')
plt.close()
