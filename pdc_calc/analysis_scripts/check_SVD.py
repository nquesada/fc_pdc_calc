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
""" This scripts checks if the individual Schmidt modes and amplitudes obtained
from the SVD of the U and V matrices form a canonical transformation.

The directory in which the SVD resides has to be supplied via command line. A
new directory check_SVD is created next to it.
"""

import sys, os
import numpy as np
import pickle
import matplotlib.pyplot as plt

print "########################"
print "# Check SVD of U and V #"
print "########################"

# Path to the analysis directory
path = sys.argv[1]

# Load process properties via pickle
pkl_file = open(path + 'PDC_properties.pkl', 'rb')
pdc_properties = pickle.load(pkl_file)
pkl_file.close()

# Obtain necessary values:
w_step = pdc_properties['w_step']

##################
# Import the SVD #
##################

# The view options reconstructs the saved floats to the original complex
# numbers

# Analytic solution
uaA_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/uaA_ana.txt.gz').view(complex)
uaD_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/uaD_ana.txt.gz')
uaBh_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/uaBh_ana.txt.gz').view(complex)

ubA_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/ubA_ana.txt.gz').view(complex)
ubD_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/ubD_ana.txt.gz')
ubBh_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/ubBh_ana.txt.gz').view(complex)

vaA_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/vaA_ana.txt.gz').view(complex)
vaD_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/vaD_ana.txt.gz')
vaBh_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/vaBh_ana.txt.gz').view(complex)

vbA_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/vbA_ana.txt.gz').view(complex)
vbD_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/vbD_ana.txt.gz')
vbBh_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/vbBh_ana.txt.gz').view(complex)


# First-order numerical solution
uaA_num = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/num/uaA_num.txt.gz').view(complex)
uaD_num = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/num/uaD_num.txt.gz')
uaBh_num = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/num/uaBh_num.txt.gz').view(complex)

ubA_num = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/num/ubA_num.txt.gz').view(complex)
ubD_num = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/num/ubD_num.txt.gz')
ubBh_num = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/num/ubBh_num.txt.gz').view(complex)

vaA_num = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/num/vaA_num.txt.gz').view(complex)
vaD_num = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/num/vaD_num.txt.gz')
vaBh_num = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/num/vaBh_num.txt.gz').view(complex)

vbA_num = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/num/vbA_num.txt.gz').view(complex)
vbD_num = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/num/vbD_num.txt.gz')
vbBh_num = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/num/vbBh_num.txt.gz').view(complex)


# Rigorous solution
uaA_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/uaA_rig.txt.gz').view(complex)
uaD_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/uaD_rig.txt.gz')
uaBh_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/uaBh_rig.txt.gz').view(complex)

ubA_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/ubA_rig.txt.gz').view(complex)
ubD_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/ubD_rig.txt.gz')
ubBh_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/ubBh_rig.txt.gz').view(complex)

vaA_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/vaA_rig.txt.gz').view(complex)
vaD_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/vaD_rig.txt.gz')
vaBh_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/vaBh_rig.txt.gz').view(complex)

vbA_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/vbA_rig.txt.gz').view(complex)
vbD_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/vbD_rig.txt.gz')
vbBh_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/vbBh_rig.txt.gz').view(complex)


# Reconstructed rigorous solution
uaA_rec = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rec/uaA_rec.txt.gz').view(complex)
uaD_rec = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rec/uaD_rec.txt.gz')
uaBh_rec = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rec/uaBh_rec.txt.gz').view(complex)

ubA_rec = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rec/ubA_rec.txt.gz').view(complex)
ubD_rec = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rec/ubD_rec.txt.gz')
ubBh_rec = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rec/ubBh_rec.txt.gz').view(complex)

vaA_rec = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rec/vaA_rec.txt.gz').view(complex)
vaD_rec = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rec/vaD_rec.txt.gz')
vaBh_rec = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rec/vaBh_rec.txt.gz').view(complex)

vbA_rec = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rec/vbA_rec.txt.gz').view(complex)
vbD_rec = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rec/vbD_rec.txt.gz')
vbBh_rec = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rec/vbBh_rec.txt.gz').view(complex)


# uut amplitudes and modes from the rigorous solution
uutaA_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/uut/uutaA_rig.txt.gz').view(complex)
uutaD_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/uut/uutaD_rig.txt.gz')
uutaBh_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/uut/uutaBh_rig.txt.gz').view(complex)

uutbA_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/uut/uutbA_rig.txt.gz').view(complex)
uutbD_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/uut/uutbD_rig.txt.gz')
uutbBh_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/uut/uutbBh_rig.txt.gz').view(complex)


# utu amplitudes and modes from the rigorous solution
utuaA_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/utu/utuaA_rig.txt.gz').view(complex)
utuaD_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/utu/utuaD_rig.txt.gz')
utuaBh_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/utu/utuaBh_rig.txt.gz').view(complex)

utubA_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/utu/utubA_rig.txt.gz').view(complex)
utubD_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/utu/utubD_rig.txt.gz')
utubBh_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/utu/utubBh_rig.txt.gz').view(complex)


##################################
## Check the Schmidt amplitudes ##
##################################
# This should always give 1. All solution except the rigorous one are
# specifically constructed to exactly fulfill this criteria. The result of the
# rigorous solution however is a good test for the numerical accuracy of the
# iteration procedure in the calculation.

# Analytic solution
aD_ana = uaD_ana**2 - vaD_ana**2 
cD_ana = ubD_ana**2 - vbD_ana**2 

# First-order numerical solution
aD_num = uaD_num**2 - vaD_num**2 
cD_num = ubD_num**2 - vbD_num**2 

# Rigorous solution
aD_rig = uaD_rig**2 - vaD_rig**2 
aD2_rig = uutaD_rig**2 - vaD_rig**2 
aD3_rig = utuaD_rig**2 - vaD_rig**2 
cD_rig = ubD_rig**2 - vbD_rig**2 
cD2_rig = uutbD_rig**2 - vbD_rig**2 
cD3_rig = utubD_rig**2 - vbD_rig**2 

# Reconstructed rigorous solution
aD_rec = uaD_rec**2 - vaD_rec**2 
cD_rec = ubD_rec**2 - vbD_rec**2 

print "Ana: ua^2 - va^2 = ",  aD_ana[:5]
print "Ana: ub^2 - vb^2 = ",  cD_ana[:5]

print "Num: ua^2 - va^2 = ",  aD_num[:5]
print "Num: ub^2 - vb^2 = ",  cD_num[:5]

print "Rig: ua^2 - va^2 = ",  aD_rig[:5]
print "Rig: uuta^2 - va^2 = ",  aD2_rig[:5]
print "Rig: utua^2 - va^2 = ",  aD3_rig[:5]
print "Rig: ub^2 - vb^2 = ",  cD_rig[:5]
print "Rig: uutb^2 - vb^2 = ",  cD2_rig[:5]
print "Rig: utub^2 - vb^2 = ",  cD3_rig[:5]

print "Rec: ua^2 - va^2 = ",  aD_rec[:5]
print "Rec: ub^2 - vb^2 = ",  cD_rec[:5]


# Write results to file:
os.mkdir(path + "check_SVD/")
f = open(path + 'check_SVD/SVD_error_analysis.txt', 'w')

f.write("####################\n")
f.write("## Error Analysis ##\n")
f.write("####################\n")
f.write("\n")

f.write("## Test if the Schmidt amplitudes form a canonical transform:\n")
f.write("## (Should give one) \n \n")

f.write("# Analytical solution:\n")
f.write("ua^2 - va^2 = " + str(aD_ana[:5]) + "\n")
f.write("ub^2 - vb^2 = " + str(aD_ana[:5]) + "\n")
f.write("# First-order numerical solution:\n")
f.write("ua^2 - va^2 = " + str(aD_num[:5]) + "\n")
f.write("ub^2 - vb^2 = " + str(aD_num[:5]) + "\n")
f.write("# Rigorous solution:\n")
f.write("ua^2 - va^2 = " + str(aD_rig[:5]) + "\n")
f.write("uuta^2 - va^2 = " + str(aD2_rig[:5]) + "\n")
f.write("utua^2 - va^2 = " + str(aD3_rig[:5]) + "\n")
f.write("ub^2 - vb^2 = " + str(aD_rig[:5]) + "\n")
f.write("uutb^2 - vb^2 = " + str(aD2_rig[:5]) + "\n")
f.write("utub^2 - vb^2 = " + str(aD3_rig[:5]) + "\n")
f.write("# Reconstructed rigorous solution:\n")
f.write("ua^2 - va^2 = " + str(aD_rec[:5]) + "\n")
f.write("ub^2 - vb^2 = " + str(aD_rec[:5]) + "\n")
f.write("\n\n")

f.close()


## Plot the results
os.mkdir(path + "/check_SVD/pic")
os.mkdir(path + "/check_SVD/pic/ana")
os.mkdir(path + "/check_SVD/pic/num")
os.mkdir(path + "/check_SVD/pic/rig")
os.mkdir(path + "/check_SVD/pic/rig_uut")
os.mkdir(path + "/check_SVD/pic/rig_utu")
os.mkdir(path + "/check_SVD/pic/rec")

plt.figure(figsize=(22,7))

N = 10
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

plt.subplot(241)
rects1 = plt.bar(ind, aD_ana[:N], width, color='r')
plt.ylabel(r'$u^2 - v^2 = 1$', fontsize=15)
plt.title('A beam ana', fontsize=20)

plt.subplot(242)
rects1 = plt.bar(ind, aD_num[:N], width, color='r')
plt.ylabel(r'$u^2 - v^2 = 1$', fontsize=15)
plt.title('A beam num', fontsize=20)

plt.subplot(243)
rects1 = plt.bar(ind, aD_rig[:N], width, color='r')
rects2 = plt.bar(ind+width, aD2_rig[:N], width, color='b')
rects3 = plt.bar(ind+2*width, aD3_rig[:N], width, color='g')
plt.ylabel(r'$u^2 - v^2 = 1$', fontsize=15)
plt.title('A beam rig', fontsize=20)
plt.legend( (rects1[0], rects2[0], rects3[0]), ('ua', 'uuta', 'utua'), loc = "lower right")

plt.subplot(244)
rects1 = plt.bar(ind, aD_rec[:N], width, color='r')
plt.ylabel(r'$u^2 - v^2 = 1$', fontsize=15)
plt.title('A beam rec', fontsize=20)

plt.subplot(245)
rects1 = plt.bar(ind, cD_ana[:N], width, color='r')
plt.ylabel(r'$u^2 - v^2 = 1$', fontsize=15)
plt.title('C beam ana', fontsize=20)

plt.subplot(246)
rects1 = plt.bar(ind, cD_num[:N], width, color='r')
plt.ylabel(r'$u^2 - v^2 = 1$', fontsize=15)
plt.title('C beam num', fontsize=20)

plt.subplot(247)
rects1 = plt.bar(ind, cD_rig[:N], width, color='r')
rects2 = plt.bar(ind+width, cD2_rig[:N], width, color='b')
rects3 = plt.bar(ind+2*width, cD3_rig[:N], width, color='g')
plt.ylabel(r'$u^2 - v^2 = 1$', fontsize=15)
plt.title('C beam rig', fontsize=20)
plt.legend( (rects1[0], rects2[0], rects3[0]), ('ua', 'uuta', 'utua'), loc = "lower right")

plt.subplot(248)
rects1 = plt.bar(ind, cD_rec[:N], width, color='r')
plt.ylabel(r'$u^2 - v^2 = 1$', fontsize=15)
plt.title('A beam rec', fontsize=25)

plt.savefig(path + "check_SVD/pic/schmidt_amplitudes_test.png")
plt.close()



#############################
## Check the Schmidt modes ##
#############################
# All the plotted modes in each plot should be identical.  However a canonical
# transformation only requires that only two modes each should be identical
# (see paper). Currently however I am not aware of any case where this happens.

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


# The range of optical modes which should be plotted.
modeNumberList = np.arange(5)

# Analytic solution
for modeNumber in modeNumberList:
    fig = plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    uaA_plt, = plt.plot(np.abs(uaA_ana[modeNumber]), linewidth=4, linestyle="dashed", alpha=0.5)
    uaBh_plt, = plt.plot(np.abs(uaBh_ana[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    vaA_plt, = plt.plot(np.abs(vaA_ana[modeNumber]), linewidth=4, linestyle="dashdot", alpha=0.5)
    vbBh_plt, = plt.plot(np.abs(vbBh_ana[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    plt.legend( (uaA_plt, uaBh_plt, vaA_plt, vbBh_plt), ('uaA', 'uaBh', 'vaA', 'vbBh') )
    plt.title("Compare modes abs A (ana)")

    plt.subplot(1,2,2)
    ubA_plt, = plt.plot(np.abs(ubA_ana[modeNumber]), linewidth=4, linestyle="dashed", alpha=0.5)
    ubBh_plt, = plt.plot(np.abs(ubBh_ana[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    vbA_plt, = plt.plot(np.abs(vbA_ana[modeNumber]), linewidth=4, linestyle="dashdot", alpha=0.5)
    vaBh_plt, = plt.plot(np.abs(vaBh_ana[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    plt.legend( (ubA_plt, ubBh_plt, vbA_plt, vaBh_plt), ('ubA', 'ubBh', 'vbA', 'vaBh') )
    plt.title("Compare modes abs C (ana)")

    plt.savefig(path + "check_SVD/pic/ana/schmidt_modes_test_ana_" + str(modeNumber) + ".png")
    plt.close()


# First-order numerical solution
for modeNumber in modeNumberList:
    fig = plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    uaA_plt, = plt.plot(np.abs(uaA_num[modeNumber]), linewidth=4, linestyle="dashed", alpha=0.5)
    uaBh_plt, = plt.plot(np.abs(uaBh_num[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    vaA_plt, = plt.plot(np.abs(vaA_num[modeNumber]), linewidth=4, linestyle="dashdot", alpha=0.5)
    vbBh_plt, = plt.plot(np.abs(vbBh_num[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    plt.legend( (uaA_plt, uaBh_plt, vaA_plt, vbBh_plt), ('uaA', 'uaBh', 'vaA', 'vbBh') )
    plt.title("Compare modes abs A (num)")

    plt.subplot(1,2,2)
    ubA_plt, = plt.plot(np.abs(ubA_num[modeNumber]), linewidth=4, linestyle="dashed", alpha=0.5)
    ubBh_plt, = plt.plot(np.abs(ubBh_num[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    vbA_plt, = plt.plot(np.abs(vbA_num[modeNumber]), linewidth=4, linestyle="dashdot", alpha=0.5)
    vaBh_plt, = plt.plot(np.abs(vaBh_num[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    plt.legend( (ubA_plt, ubBh_plt, vbA_plt, vaBh_plt), ('ubA', 'ubBh', 'vbA', 'vaBh') )
    plt.title("Compare modes abs C (num)")

    plt.savefig(path + "check_SVD/pic/num/schmidt_modes_test_num_" + str(modeNumber) + ".png")
    plt.close()


# Rigorous solution
for modeNumber in modeNumberList:
    fig = plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    uaA_plt, = plt.plot(np.abs(uaA_rig[modeNumber]), linewidth=4, linestyle="dashed", alpha=0.5)
    uaBh_plt, = plt.plot(np.abs(uaBh_rig[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    vaA_plt, = plt.plot(np.abs(vaA_rig[modeNumber]), linewidth=4, linestyle="dashdot", alpha=0.5)
    vbBh_plt, = plt.plot(np.abs(vbBh_rig[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    plt.legend( (uaA_plt, uaBh_plt, vaA_plt, vbBh_plt), ('uaA', 'uaBh', 'vaA', 'vbBh') )
    plt.title("Compare modes abs A (rig)")

    plt.subplot(1,2,2)
    ubA_plt, = plt.plot(np.abs(ubA_rig[modeNumber]), linewidth=4, linestyle="dashed", alpha=0.5)
    ubBh_plt, = plt.plot(np.abs(ubBh_rig[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    vbA_plt, = plt.plot(np.abs(vbA_rig[modeNumber]), linewidth=4, linestyle="dashdot", alpha=0.5)
    vaBh_plt, = plt.plot(np.abs(vaBh_rig[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    plt.legend( (ubA_plt, ubBh_plt, vbA_plt, vaBh_plt), ('ubA', 'ubBh', 'vbA', 'vaBh') )
    plt.title("Compare modes abs C (rig)")

    plt.savefig(path + "check_SVD/pic/rig/schmidt_modes_test_rig_" + str(modeNumber) + ".png")
    plt.close()


# Rigorous solution with uut modes
for modeNumber in modeNumberList:
    fig = plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    uutaA_plt, = plt.plot(np.abs(uutaA_rig[modeNumber]), linewidth=4, linestyle="dashed", alpha=0.5)
    uutaBh_plt, = plt.plot(np.abs(uutaBh_rig[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    vaA_plt, = plt.plot(np.abs(vaA_rig[modeNumber]), linewidth=4, linestyle="dashdot", alpha=0.5)
    vbBh_plt, = plt.plot(np.abs(vbBh_rig[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    plt.legend( (uutaA_plt, uutaBh_plt, vaA_plt, vbBh_plt), ('uutaA', 'uutaBh', 'vaA', 'vbBh') )
    plt.title("Compare modes abs A (rig)")

    plt.subplot(1,2,2)
    uutbA_plt, = plt.plot(np.abs(uutbA_rig[modeNumber]), linewidth=4, linestyle="dashed", alpha=0.5)
    uutbBh_plt, = plt.plot(np.abs(uutbBh_rig[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    vbA_plt, = plt.plot(np.abs(vbA_rig[modeNumber]), linewidth=4, linestyle="dashdot", alpha=0.5)
    vaBh_plt, = plt.plot(np.abs(vaBh_rig[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    plt.legend( (ubA_plt, ubBh_plt, vbA_plt, vaBh_plt), ('ubA', 'ubBh', 'vbA', 'vaBh') )
    plt.title("Compare modes abs C (rig)")

    plt.savefig(path + "check_SVD/pic/rig_uut/schmidt_modes_test_rig_uut_" + str(modeNumber) + ".png")
    plt.close()


# Rigorous solution with utu modes
for modeNumber in modeNumberList:
    fig = plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    utuaA_plt, = plt.plot(np.abs(utuaA_rig[modeNumber]), linewidth=4, linestyle="dashed", alpha=0.5)
    utuaBh_plt, = plt.plot(np.abs(utuaBh_rig[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    vaA_plt, = plt.plot(np.abs(vaA_rig[modeNumber]), linewidth=4, linestyle="dashdot", alpha=0.5)
    vbBh_plt, = plt.plot(np.abs(vbBh_rig[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    plt.legend( (utuaA_plt, utuaBh_plt, vaA_plt, vbBh_plt), ('utuaA', 'utuaBh', 'vaA', 'vbBh') )
    plt.title("Compare modes abs A (rig)")

    plt.subplot(1,2,2)
    utubA_plt, = plt.plot(np.abs(utubA_rig[modeNumber]), linewidth=4, linestyle="dashed", alpha=0.5)
    utubBh_plt, = plt.plot(np.abs(utubBh_rig[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    vbA_plt, = plt.plot(np.abs(vbA_rig[modeNumber]), linewidth=4, linestyle="dashdot", alpha=0.5)
    vaBh_plt, = plt.plot(np.abs(vaBh_rig[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    plt.legend( (ubA_plt, ubBh_plt, vbA_plt, vaBh_plt), ('ubA', 'ubBh', 'vbA', 'vaBh') )
    plt.title("Compare modes abs C (rig)")

    plt.savefig(path + "check_SVD/pic/rig_utu/schmidt_modes_test_rig_utu_" + str(modeNumber) + ".png")
    plt.close()



# Reconstructed rigorous solution
for modeNumber in modeNumberList:
    fig = plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    uaA_plt, = plt.plot(np.abs(uaA_rec[modeNumber]), linewidth=4, linestyle="dashed", alpha=0.5)
    uaBh_plt, = plt.plot(np.abs(uaBh_rec[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    vaA_plt, = plt.plot(np.abs(vaA_rec[modeNumber]), linewidth=4, linestyle="dashdot", alpha=0.5)
    vbBh_plt, = plt.plot(np.abs(vbBh_rec[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    plt.legend( (uaA_plt, uaBh_plt, vaA_plt, vbBh_plt), ('uaA', 'uaBh', 'vaA', 'vbBh') )
    plt.title("Compare modes abs A (rec)")

    plt.subplot(1,2,2)
    ubA_plt, = plt.plot(np.abs(ubA_rec[modeNumber]), linewidth=4, linestyle="dashed", alpha=0.5)
    ubBh_plt, = plt.plot(np.abs(ubBh_rec[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    vbA_plt, = plt.plot(np.abs(vbA_rec[modeNumber]), linewidth=4, linestyle="dashdot", alpha=0.5)
    vaBh_plt, = plt.plot(np.abs(vaBh_rec[modeNumber]), linewidth=4, linestyle="dotted", alpha=0.5)
    plt.legend( (ubA_plt, ubBh_plt, vbA_plt, vaBh_plt), ('ubA', 'ubBh', 'vbA', 'vaBh') )
    plt.title("Compare modes abs C (rec)")

    plt.savefig(path + "check_SVD/pic/rec/schmidt_modes_test_rec_" + str(modeNumber) + ".png")
    plt.close()
