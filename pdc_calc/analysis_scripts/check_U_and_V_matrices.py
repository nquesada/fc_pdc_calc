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
""" This script tests if the obtained U and V matrices form a canonical
transformation. 

The directory in which the U_and_V_matrices directory resides has to be
supplied via command line. A new directory check_U_and_V_matrices storing the
results is created next to it.
"""

import sys, os
import numpy as np
import pickle
import matplotlib.pyplot as plt

print "####################"
print "# Checking U and V #"
print "####################"

# Path the analysis directory
path = sys.argv[1]

# Load process properties via pickle
pkl_file = open(path + 'PDC_properties.pkl', 'rb')
pdc_properties = pickle.load(pkl_file)
pkl_file.close()

# Calculate necessary arrays:
w_step = pdc_properties['w_step']

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

# The view options reconstructs the saved floats to the original
# complex numbers
ua_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/ua_ana.txt.gz').view(complex)
ub_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/ub_ana.txt.gz').view(complex)
va_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/va_ana.txt.gz').view(complex)
vb_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/vb_ana.txt.gz').view(complex)

ua_num = np.loadtxt(path + 'U_and_V_matrices/data/num/ua_num.txt.gz').view(complex)
ub_num = np.loadtxt(path + 'U_and_V_matrices/data/num/ub_num.txt.gz').view(complex)
va_num = np.loadtxt(path + 'U_and_V_matrices/data/num/va_num.txt.gz').view(complex)
vb_num = np.loadtxt(path + 'U_and_V_matrices/data/num/vb_num.txt.gz').view(complex)

ua_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/ua_rig.txt.gz').view(complex)
ub_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/ub_rig.txt.gz').view(complex)
va_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/va_rig.txt.gz').view(complex)
vb_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/vb_rig.txt.gz').view(complex)

ua_rec = np.loadtxt(path + 'U_and_V_matrices/data/rec/ua_rec.txt.gz').view(complex)
ub_rec = np.loadtxt(path + 'U_and_V_matrices/data/rec/ub_rec.txt.gz').view(complex)
va_rec = np.loadtxt(path + 'U_and_V_matrices/data/rec/va_rec.txt.gz').view(complex)
vb_rec = np.loadtxt(path + 'U_and_V_matrices/data/rec/vb_rec.txt.gz').view(complex)


############################################################################
## Define the 4x2 different test cases for the Ua, Ub and Va, vb matrices ##
############################################################################

# Each test always consists of subtracting two matrices which should give 0 for
# a canonical transformation. For debugging purposes each test returns these
# two matrices, the norm of the differences between the matrices (should be
# zero) which we label the error of the test and finally the relative error
# which is the error normalized via the two initial matrices.

def test1(u, v):
    mat_A = np.dot(u, np.conjugate(np.transpose(u))) * w_step
    mat_A = mat_A - Uinitial
    mat_B = np.dot(v, np.conjugate(np.transpose(v))) * w_step

    error = np.linalg.norm(mat_A - mat_B)
    mat_A_norm = np.linalg.norm(mat_A)
    mat_B_norm = np.linalg.norm(mat_B)
    relative_error = error/(0.5 * (mat_A_norm + mat_B_norm))

    return(mat_A, mat_B, error, relative_error) 


def test2(u1, v1, u2, v2):
    mat_A = np.dot(u1, np.transpose(v2)) * w_step
    mat_B = np.dot(v1, np.transpose(u2)) * w_step

    error = np.linalg.norm(mat_A - mat_B)
    # Additionally subtract Uinital for the normalization
    # of the relative error to account for the fact that the
    # delta function in the numerator cancels itself
    mat_A_norm = np.linalg.norm(mat_A - Uinitial)
    mat_B_norm = np.linalg.norm(mat_B - Uinitial)
    relative_error = error/(0.5 * (mat_A_norm + mat_B_norm))

    return(mat_A, mat_B, error, relative_error) 


# Inverse canonical transformation
def test3(u, v):
    mat_A = np.dot(np.conjugate(np.transpose(u)), u) * w_step
    mat_A = mat_A - Uinitial
    mat_B = np.transpose(np.dot(np.conjugate(np.transpose(v)), v)) * w_step

    error = np.linalg.norm(mat_A - mat_B)
    mat_A_norm = np.linalg.norm(mat_A)
    mat_B_norm = np.linalg.norm(mat_B)
    relative_error = error/(0.5*(mat_A_norm + mat_B_norm))

    return(mat_A, mat_B, error, relative_error) 


def test4(u1, v1, u2, v2):
    mat_A = np.dot(np.conjugate(np.transpose(u1)), v1) * w_step
    mat_B = np.transpose(np.dot(np.conjugate(np.transpose(u2)), v2))*w_step

    error = np.linalg.norm(mat_A - mat_B)
    # Additionally subtract Uinital for the normalization
    # of the relative error to account for the fact that the
    # delta function in the numerator cancels itself
    mat_A_norm = np.linalg.norm(mat_A - Uinitial)
    mat_B_norm = np.linalg.norm(mat_B - Uinitial)
    relative_error = error/(0.5*(mat_A_norm + mat_B_norm))

    return(mat_A, mat_B, error, relative_error) 



#########################
## Check the solutions ##
#########################

# Check the analytical solution
(ana_test1a_mat_A, ana_test1a_mat_B, ana_test1a_error, ana_test1a_relative_error) = \
        test1(ua_ana, va_ana)
(ana_test1c_mat_A, ana_test1c_mat_B, ana_test1c_error, ana_test1c_relative_error) = \
        test1(ub_ana, vb_ana)

(ana_test2a_mat_A, ana_test2a_mat_B, ana_test2a_error, ana_test2a_relative_error) = \
        test2(ua_ana, va_ana, ub_ana, vb_ana)
(ana_test2c_mat_A, ana_test2c_mat_B, ana_test2c_error, ana_test2c_relative_error) = \
        test2(ub_ana, vb_ana, ua_ana, va_ana)

(ana_test3a_mat_A, ana_test3a_mat_B, ana_test3a_error, ana_test3a_relative_error) = \
        test3(ua_ana, vb_ana)
(ana_test3c_mat_A, ana_test3c_mat_B, ana_test3c_error, ana_test3c_relative_error) = \
        test3(ub_ana, va_ana)

(ana_test4a_mat_A, ana_test4a_mat_B, ana_test4a_error, ana_test4a_relative_error) = \
        test4(ua_ana, va_ana, ub_ana, vb_ana)
(ana_test4c_mat_A, ana_test4c_mat_B, ana_test4c_error, ana_test4c_relative_error) = \
        test4(ub_ana, vb_ana, ua_ana, va_ana)

print ""
print "#Relative Error ana solution:"
print "relative error test1a (ana):", ana_test1a_relative_error
print "relative error test1c (ana):", ana_test1c_relative_error
print "relative error test2a (ana):", ana_test2a_relative_error
print "relative error test2c (ana):", ana_test2c_relative_error
print "relative error test3a (ana):", ana_test3a_relative_error
print "relative error test3c (ana):", ana_test3c_relative_error
print "relative error test4a (ana):", ana_test4a_relative_error
print "relative error test4c (ana):", ana_test4c_relative_error
print ""


# Check error of the numerical solution
(num_test1a_mat_A, num_test1a_mat_B, num_test1a_error, num_test1a_relative_error) = \
        test1(ua_num, va_num)
(num_test1c_mat_A, num_test1c_mat_B, num_test1c_error, num_test1c_relative_error) = \
        test1(ub_num, vb_num)

(num_test2a_mat_A, num_test2a_mat_B, num_test2a_error, num_test2a_relative_error) = \
        test2(ua_num, va_num, ub_num, vb_num)
(num_test2c_mat_A, num_test2c_mat_B, num_test2c_error, num_test2c_relative_error) = \
        test2(ub_num, vb_num, ua_num, va_num)

(num_test3a_mat_A, num_test3a_mat_B, num_test3a_error, num_test3a_relative_error) = \
        test3(ua_num, vb_num)
(num_test3c_mat_A, num_test3c_mat_B, num_test3c_error, num_test3c_relative_error) = \
        test3(ub_num, va_num)

(num_test4a_mat_A, num_test4a_mat_B, num_test4a_error, num_test4a_relative_error) = \
        test4(ua_num, va_num, ub_num, vb_num)
(num_test4c_mat_A, num_test4c_mat_B, num_test4c_error, num_test4c_relative_error) = \
        test4(ub_num, vb_num, ua_num, va_num)


print "#Relative Error num solution:"
print "relative error test1a (num):", num_test1a_relative_error
print "relative error test1c (num):", num_test1c_relative_error
print "relative error test2a (num):", num_test2a_relative_error
print "relative error test2c (num):", num_test2c_relative_error
print "relative error test3a (num):", num_test3a_relative_error
print "relative error test3c (num):", num_test3c_relative_error
print "relative error test4a (num):", num_test4a_relative_error
print "relative error test4c (num):", num_test4c_relative_error
print ""


# Check error of the rigorous solution
(rig_test1a_mat_A, rig_test1a_mat_B, rig_test1a_error, rig_test1a_relative_error) = \
        test1(ua_rig, va_rig)
(rig_test1c_mat_A, rig_test1c_mat_B, rig_test1c_error, rig_test1c_relative_error) = \
        test1(ub_rig, vb_rig)

(rig_test2a_mat_A, rig_test2a_mat_B, rig_test2a_error, rig_test2a_relative_error) = \
        test2(ua_rig, va_rig, ub_rig, vb_rig)
(rig_test2c_mat_A, rig_test2c_mat_B, rig_test2c_error, rig_test2c_relative_error) = \
        test2(ub_rig, vb_rig, ua_rig, va_rig)

(rig_test3a_mat_A, rig_test3a_mat_B, rig_test3a_error, rig_test3a_relative_error) = \
        test3(ua_rig, vb_rig)
(rig_test3c_mat_A, rig_test3c_mat_B, rig_test3c_error, rig_test3c_relative_error) = \
        test3(ub_rig, va_rig)

(rig_test4a_mat_A, rig_test4a_mat_B, rig_test4a_error, rig_test4a_relative_error) = \
        test4(ua_rig, va_rig, ub_rig, vb_rig)
(rig_test4c_mat_A, rig_test4c_mat_B, rig_test4c_error, rig_test4c_relative_error) = \
        test4(ub_rig, vb_rig, ua_rig, va_rig)


print "#Relative Error rig solution:"
print "relative error test1a (rig):", rig_test1a_relative_error
print "relative error test1c (rig):", rig_test1c_relative_error
print "relative error test2a (rig):", rig_test2a_relative_error
print "relative error test2c (rig):", rig_test2c_relative_error
print "relative error test3a (rig):", rig_test3a_relative_error
print "relative error test3c (rig):", rig_test3c_relative_error
print "relative error test4a (rig):", rig_test4a_relative_error
print "relative error test4c (rig):", rig_test4c_relative_error
print ""

# Check error of the rigorous solution with the reconstructed ua and ub
(rec_test1a_mat_A, rec_test1a_mat_B, rec_test1a_error, \
        rec_test1a_relative_error) = test1(ua_rec, va_rec)
(rec_test1c_mat_A, rec_test1c_mat_B, rec_test1c_error, \
        rec_test1c_relative_error) = test1(ub_rec, vb_rec)

(rec_test2a_mat_A, rec_test2a_mat_B, rec_test2a_error, \
        rec_test2a_relative_error) \
        = test2(ua_rec, va_rec, ub_rec, vb_rec)
(rec_test2c_mat_A, rec_test2c_mat_B, rec_test2c_error,\
        rec_test2c_relative_error) \
        = test2(ub_rec, vb_rec, ua_rec, va_rec)

(rec_test3a_mat_A, rec_test3a_mat_B, rec_test3a_error, \
        rec_test3a_relative_error) = test3(ua_rec, vb_rec)
(rec_test3c_mat_A, rec_test3c_mat_B, rec_test3c_error, \
        rec_test3c_relative_error) = test3(ub_rec, va_rec)

(rec_test4a_mat_A, rec_test4a_mat_B, rec_test4a_error, \
        rec_test4a_relative_error) = \
        test4(ua_rec, va_rec, ub_rec, vb_rec)
(rec_test4c_mat_A, rec_test4c_mat_B, rec_test4c_error, \
        rec_test4c_relative_error) = \
        test4(ub_rec, vb_rec, ua_rec, va_rec)


print "#Relative Error rec solution:"
print "relative error test1a (rec):", rec_test1a_relative_error
print "relative error test1c (rec):", rec_test1c_relative_error
print "relative error test2a (rec):", rec_test2a_relative_error
print "relative error test2c (rec):", rec_test2c_relative_error
print "relative error test3a (rec):", rec_test3a_relative_error
print "relative error test3c (rec):", rec_test3c_relative_error
print "relative error test4a (rec):", rec_test4a_relative_error
print "relative error test4c (rec):", rec_test4c_relative_error
print ""


## Write the errors to the text file
os.mkdir(path + "/check_U_and_V_matrices/")
f = open(path + '/check_U_and_V_matrices/U_and_V_error_analysis.txt', 'w')

f.write("####################\n")
f.write("## Error Analysis ##\n")
f.write("####################\n")
f.write("\n")

f.write("## Test if the four solutions are canonical transforms:\n")
f.write("## (The number give an relative error (should be close to zero) \n \n")

f.write("## Analytical solution:\n")
f.write("ana test1a: " + str(ana_test1a_relative_error) + "\n") 
f.write("ana test1c: " + str(ana_test1c_relative_error) + "\n") 
f.write("ana test2a: " + str(ana_test2a_relative_error) + "\n") 
f.write("ana test2c: " + str(ana_test2c_relative_error) + "\n") 
f.write("ana test3a: " + str(ana_test3a_relative_error) + "\n") 
f.write("ana test3c: " + str(ana_test3c_relative_error) + "\n") 
f.write("ana test4a: " + str(ana_test4a_relative_error) + "\n") 
f.write("ana test4c: " + str(ana_test4c_relative_error) + "\n") 
f.write("\n")


f.write("## First-order numerical solution:\n")
f.write("num test1a: " + str(num_test1a_relative_error) + "\n") 
f.write("num test1c: " + str(num_test1c_relative_error) + "\n") 
f.write("num test2a: " + str(num_test2a_relative_error) + "\n") 
f.write("num test2c: " + str(num_test2c_relative_error) + "\n") 
f.write("num test3a: " + str(num_test3a_relative_error) + "\n") 
f.write("num test3c: " + str(num_test3c_relative_error) + "\n") 
f.write("num test4a: " + str(num_test4a_relative_error) + "\n") 
f.write("num test4c: " + str(num_test4c_relative_error) + "\n") 
f.write("\n")

f.write("## Rigorous solution:\n")
f.write("rig test1a: " + str(rig_test1a_relative_error) + "\n") 
f.write("rig test1c: " + str(rig_test1c_relative_error) + "\n") 
f.write("rig test2a: " + str(rig_test2a_relative_error) + "\n") 
f.write("rig test2c: " + str(rig_test2c_relative_error) + "\n") 
f.write("rig test3a: " + str(rig_test3a_relative_error) + "\n") 
f.write("rig test3c: " + str(rig_test3c_relative_error) + "\n") 
f.write("rig test4a: " + str(rig_test4a_relative_error) + "\n") 
f.write("rig test4c: " + str(rig_test4c_relative_error) + "\n") 
f.write("\n")

f.write("## Reconstructed rigorous solution:\n")
f.write("rec test1a: " + str(rec_test1a_relative_error) + "\n") 
f.write("rec test1c: " + str(rec_test1c_relative_error) + "\n") 
f.write("rec test2a: " + str(rec_test2a_relative_error) + "\n") 
f.write("rec test2c: " + str(rec_test2c_relative_error) + "\n") 
f.write("rec test3a: " + str(rec_test3a_relative_error) + "\n") 
f.write("rec test3c: " + str(rec_test3c_relative_error) + "\n") 
f.write("rec test4a: " + str(rec_test4a_relative_error) + "\n") 
f.write("rec test4c: " + str(rec_test4c_relative_error) + "\n") 

f.close()


## Visualize the test cases.
os.mkdir(path + "/check_U_and_V_matrices/pic")

os.mkdir(path + "/check_U_and_V_matrices/pic/ana")
os.mkdir(path + "/check_U_and_V_matrices/pic/num")
os.mkdir(path + "/check_U_and_V_matrices/pic/rig")
os.mkdir(path + "/check_U_and_V_matrices/pic/rec")

# In the resulting figures the first row should be identical to the second one.
# The third row plots the differences which should evaluate to zero when the
# state is a canonical transformation.  These figures are very useful for
# debugging purposes.

def plot_mat_difference(matA, matA_name, matB, matB_name, sub_directory):
    """ Plot the difference between 2 matrices, where the 
    names have to be supplied as well."""
    plt.figure(figsize=(20,12))

    plt.subplot(3,3,1)
    plt.imshow(np.real(matA),origin="lower")
    plt.title(matA_name + " real")
    plt.colorbar()

    plt.subplot(3,3,2)
    plt.imshow(np.imag(matA),origin="lower")
    plt.title(matA_name + " imag")
    plt.colorbar()

    plt.subplot(3,3,3)
    plt.imshow(np.abs(matA),origin="lower")
    plt.title(matA_name + " abs")
    plt.colorbar()

    plt.subplot(3,3,4)
    plt.imshow(np.real(matB),origin="lower")
    plt.title(matB_name + " real")
    plt.colorbar()

    plt.subplot(3,3,5)
    plt.imshow(np.imag(matB),origin="lower")
    plt.title(matB_name + " imag")
    plt.colorbar()

    plt.subplot(3,3,6)
    plt.imshow(np.abs(matB),origin="lower")
    plt.title(matB_name + " abs")
    plt.colorbar()

    plt.subplot(3,3,7)
    plt.imshow(np.real(matA - matB),origin="lower")
    plt.title(matA_name + "-" + matB_name +  " real")
    plt.colorbar()

    plt.subplot(3,3,8)
    plt.imshow(np.imag(matA - matB),origin="lower")
    plt.title(matA_name + "-" + matB_name +  " imag")
    plt.colorbar()

    plt.subplot(3,3,9)
    plt.imshow(np.abs(matA - matB),origin="lower")
    plt.title(matA_name + "-" + matB_name +  " abs")
    plt.colorbar()

    plt.savefig(path + "check_U_and_V_matrices/pic/" + sub_directory + '/' + matA_name + "-" + matB_name)

    # This close() is important otherwise the figure stays in memory.
    plt.close()


# The minus signs make the different matrices appear identical in the plots.
# This eases the visual comparison.
plot_mat_difference(ana_test1a_mat_A, "ana_test1a_a", ana_test1a_mat_B, "ana_test1a_b", "ana")
plot_mat_difference(ana_test1c_mat_A, "ana_test1c_a", ana_test1c_mat_B, "ana_test1c_b", "ana")
plot_mat_difference(ana_test2a_mat_A, "ana_test2a_a", ana_test2a_mat_B, "ana_test2a_b", "ana")
plot_mat_difference(ana_test2c_mat_A, "ana_test2c_a", ana_test2c_mat_B, "ana_test2c_b", "ana")
plot_mat_difference(ana_test3a_mat_A, "ana_test3a_a", ana_test3a_mat_B, "ana_test3a_b", "ana")
plot_mat_difference(ana_test3c_mat_A, "ana_test3c_a", ana_test3c_mat_B, "ana_test3c_b", "ana")
plot_mat_difference(ana_test4a_mat_A, "ana_test4a_a", ana_test4a_mat_B, "ana_test4a_b", "ana")
plot_mat_difference(ana_test4c_mat_A, "ana_test4c_a", ana_test4c_mat_B, "ana_test4c_b", "ana")

plot_mat_difference(num_test1a_mat_A, "num_test1a_a", num_test1a_mat_B, "num_test1a_b", "num")
plot_mat_difference(num_test1c_mat_A, "num_test1c_a", num_test1c_mat_B, "num_test1c_b", "num")
plot_mat_difference(num_test2a_mat_A, "num_test2a_a", num_test2a_mat_B, "num_test2a_b", "num")
plot_mat_difference(num_test2c_mat_A, "num_test2c_a", num_test2c_mat_B, "num_test2c_b", "num")
plot_mat_difference(num_test3a_mat_A, "num_test3a_a", num_test3a_mat_B, "num_test3a_b", "num")
plot_mat_difference(num_test3c_mat_A, "num_test3c_a", num_test3c_mat_B, "num_test3c_b", "num")
plot_mat_difference(num_test4a_mat_A, "num_test4a_a", num_test4a_mat_B, "num_test4a_b", "num")
plot_mat_difference(num_test4c_mat_A, "num_test4c_a", num_test4c_mat_B, "num_test4c_b", "num")

plot_mat_difference(rig_test1a_mat_A, "rig_test1a_a", rig_test1a_mat_B, "rig_test1a_b", "rig")
plot_mat_difference(rig_test1c_mat_A, "rig_test1c_a", rig_test1c_mat_B, "rig_test1c_b", "rig")
plot_mat_difference(rig_test2a_mat_A, "rig_test2a_a", rig_test2a_mat_B, "rig_test2a_b", "rig")
plot_mat_difference(rig_test2c_mat_A, "rig_test2c_a", rig_test2c_mat_B, "rig_test2c_b", "rig")
plot_mat_difference(rig_test3a_mat_A, "rig_test3a_a", rig_test3a_mat_B, "rig_test3a_b", "rig")
plot_mat_difference(rig_test3c_mat_A, "rig_test3c_a", rig_test3c_mat_B, "rig_test3c_b", "rig")
plot_mat_difference(rig_test4a_mat_A, "rig_test4a_a", rig_test4a_mat_B, "rig_test4a_b", "rig")
plot_mat_difference(rig_test4c_mat_A, "rig_test4c_a", rig_test4c_mat_B, "rig_test4c_b", "rig")

plot_mat_difference(rec_test1a_mat_A, "rec_test1a_a", rec_test1a_mat_B, "rec_test1a_b", "rec")
plot_mat_difference(rec_test1c_mat_A, "rec_test1c_a", rec_test1c_mat_B, "rec_test1c_b", "rec")
plot_mat_difference(rec_test2a_mat_A, "rec_test2a_a", rec_test2a_mat_B, "rec_test2a_b", "rec")
plot_mat_difference(rec_test2c_mat_A, "rec_test2c_a", rec_test2c_mat_B, "rec_test2c_b", "rec")
plot_mat_difference(rec_test3a_mat_A, "rec_test3a_a", rec_test3a_mat_B, "rec_test3a_b", "rec")
plot_mat_difference(rec_test3c_mat_A, "rec_test3c_a", rec_test3c_mat_B, "rec_test3c_b", "rec")
plot_mat_difference(rec_test4a_mat_A, "rec_test4a_a", rec_test4a_mat_B, "rec_test4a_b", "rec")
plot_mat_difference(rec_test4c_mat_A, "rec_test4c_a", rec_test4c_mat_B, "rec_test4c_b", "rec")
