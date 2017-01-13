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
""" This file contains the function calc_PDC with calculates
the analytic and rigorous solution of a given parametric dwon-conversion
process as specified in the corresponding paper. It saves the data
in a hdf5 file for further analysis.

Details on the math can be found in the corresponding paper."""

import numpy as np
import tables


############################
# Definitions and formulas #
############################

## Define PDC functions

# x = omega
# y = omega'

# Phasematching function
L=2.23013

# as defined for E_a: x = omega, y = omega', k_p = A, k_s = B, k_i = C
def dk(x, y, A, B, C):
    """ Return the k-vector mismatch between the three waves. """
    return (A*(x + y) - B*(x) - C*(y))

# Pump distribution
# Note that the the pump distribution is not normalized to replicate
# the formula in the paper.
def alpha(x, sigma_x):
    """ Return the amplitude of the pump distribution alpha center at 0
    with given width sigma and frequency x"""
    # Gaussian pump
    return np.exp(-x**2 / (2*sigma_x**2) )


# Analytical phasematching function
def pm(x, y, z_start, z_stop, A, B, C):
    """ Analytic phasematching function for a waveguide of length
    z_stop - z_start. It is however assumed that the waveguide
    starts at z_stop=-z_start and ends at z_start.  """
    gamma=0.193
    return L * np.exp(-gamma*(L*dk(x, y, A, B, C)/2)**2) 
        


# Analytical first-order solution for Va (A beam)
# (I cannot name it f like in the paper since this is occupied by the f
# function from the differential equation)
def ra_analytical(x, y, z_start, z_stop, coupling, A, B, C, pump_width):
    """ Analytical solution for the A beam of first-order parametric
    down-conversion."""
    return -1j * np.conjugate(coupling) * np.conjugate(alpha(x+y, pump_width)) \
            * pm(x, y, z_start, z_stop, A, B, C)

# Analytical first-order solution for Vb (B beam)
# (I cannot name it f like in the paper since this is occupied by the f
# function from the differential equation)
def rb_analytical(x, y, z_start, z_stop, coupling, A, B, C, pump_width):
    """ Analytical solution for the B beam of first-order PDC.
        Note: rb_analytical = x-y switdch of ra_analytical"""
    return -1j * np.conjugate(coupling) * np.conjugate(alpha(y+x, pump_width)) \
            * pm(y, x, z_start, z_stop, A, B, C)

# Numerical phi function
def grating(z):
    return np.exp(-0.1*z**2)

def phi_num(x, y, z, A, B, C):
    """ Return the value of the phi function in the differential equation"""
    gamma=0.193
    return np.exp(-1j * dk(x, y, A, B, C)*z)*np.exp(-z*z/(L*L*gamma))/np.sqrt(np.pi*gamma)



    # Additional hypergrating to get rid of sinc side peaks:
    # (Currently not used)
    # return grating(z) * np.exp(-1j * dk(x, y, A, B, C)*z)


# f function with prefactor 
def f_a(x, y, z, coupling, A, B, C, pump_width):
    """ Body of the differential equation for A 
    (on the right of the differential equation)"""
    return -1j * coupling * alpha(x+y, pump_width) \
            * phi_num(x, y, z, A, B, C)


def f_b(x, y, z, coupling, A, B, C, pump_width):
    """ Body of the differential equation for C 
    (on the right of the differential equation)"""
    # This function is just the x-y switched version of f_a
    return f_a(y, x, z, coupling, A, B, C, pump_width)



# The big function creation the analytic and rigorous solution for PDC
def calc_PDC(coupling, w_start, w_stop, w_steps, z_start, z_stop, z_steps,\
        A, B, C, pump_width, save_directory):
    """ Solve PDC in three different ways:
    1) Analytical first-order solution
    2) Numerical first-order solution (identical to analytic solution, but 
       features a numerical z-integration)
    3) Rigorous solution Vaa iterative Euler approach
    
    All data is saved in a hdf5 file which can get very big."""

    ## Calculate vectors for the iteration process
    wRange = np.linspace(w_start, w_stop , w_steps)
    zRange = np.linspace(z_start, z_stop , z_steps)
    
    print "wRange:", wRange
    print "zRange:", zRange

    # Calculate the step sizes
    w_step = wRange[1] - wRange[0]
    z_step = zRange[1] - zRange[0]

    print "zSampling:", zRange.size
    print "wSampling:", wRange.size

    ## Generate Initial Matrices to store the results for Ua, Ub, Va and Vb:
    #   In all generated matrices "dtype=complex" has to be explicitly stated,
    #   otherwise the applied matrix operations cannot handle complex numbers
    print "# Generating Ua, Ub, Va and Vb initial #"

    Vinitial = np.zeros( (wRange.size, wRange.size), dtype=complex)
    # This is the definition of a discretized delta function (note the /w_step)
    Uinitial = np.identity(wRange.size, dtype=complex) / w_step

    # Create and open Hdf5 File
    ## Attention do not change the naming, it's exact nature is used in the 
    ## analysis scripts.
    directory_name = save_directory + "/PDC_calc"
    evaluation_parameters = "---PAR--pow_" + "%.3f" % coupling \
            + "--w_" + "%.1f" % w_start + "_" + "%.1f" % w_stop + "_" \
            + "%.4d" % w_steps + \
            "--z_" +  "%.1f" % z_start + "_" + "%.1f" % z_stop + "_" \
            + "%.4d" % z_steps
    pdc_parameters = "---PDC--A_" + "%.1f" % A + "--B_" + "%.1f" % B \
            + "--C_" + "%.1f" % C + "--pumpWidth_" + "%.5f" % pump_width


    filename = directory_name + evaluation_parameters + pdc_parameters + ".h5"
    

    h5file = tables.open_file(filename, mode = "w", title = "PDC-Bog-Prop")
    
    # Create subgroup for Va, Vb, Ua and Ub
    groupVa = h5file.create_group("/", "Va", "z-Position")
    groupUa = h5file.create_group("/", "Ua", "z-Position")
    groupVb = h5file.create_group("/", "Vb", "z-Position")
    groupUb = h5file.create_group("/", "Ub", "z-Position")

    # Generate initial Va and Ua Arrays

    # For some reason pytables does not want identifiers with a "."
    # consequently the zPos is identified by  0,1,2,3 and the actual position
    # has to be recalculated with help of the given z_start and z_step 
    print "-> Generating Initial Va"
    print "-> Generating Va initial array at z =", 
    for i in np.arange(zRange.size):
        h5file.create_array(groupVa, "zPos" + str(i), Vinitial, "z-Position") 
        print i,
    print "finished."

    print "-> Generating Initial Ua"
    print "-> Generating Ua initial array at z =",
    for i in np.arange(zRange.size):
        h5file.create_array(groupUa, "zPos" + str(i), Uinitial, "z-Position") 
        print i,
    print "finished."


    print "-> Generating Initial Vb"
    print "-> Generating Vb initial array at z =", 
    for i in np.arange(zRange.size):
        h5file.create_array(groupVb, "zPos" + str(i), Vinitial, "z-Position") 
        print i,
    print "finished."

    print "-> Generating Initial Ub"
    print "-> Generating Ub initial array at z =",
    for i in np.arange(zRange.size):
        h5file.create_array(groupUb, "zPos" + str(i), Uinitial, "z-Position") 
        print i,
    print "finished."


    ## Create f functions 
    print "# Generating f functions #"

    # Note the flipped Y, X in order to cope with numpys  array definition
    # The matrix multiplication used in our solution will only work with this
    # exact notation
    # Please refer to the memo for the in's and out's of this procedure
    Y, X = np.meshgrid(wRange,wRange)

    groupFa = h5file.create_group("/", "fa", "z Position")
    for i in np.arange(zRange.size):
        famatrix = f_a(X, Y, zRange[i], coupling, A, B, C, pump_width)
        h5file.create_array(groupFa, "zPos" + str(i), famatrix, "zPos-Set") 
        print "-> fa array z =", zRange[i], "finished"

    print ""

    # Actually fb is just a transposed fa 
    groupFb = h5file.create_group("/", "fb", "z Position")
    for i in np.arange(zRange.size):
        fbmatrix = f_b(X, Y, zRange[i], coupling, A, B, C, pump_width)
        h5file.create_array(groupFb, "zPos" + str(i), fbmatrix, "zPos-Set") 
        print "-> fb array z =", zRange[i], "finished"



    ## Create numerical first-order solution
    print "# Generating Ra, Rb numerical first-order solution. #"

    groupRafirstItNum = h5file.create_group("/", "RafirstItNum", "z-Position")
    RafirstItTmp = Vinitial.copy()
    for i in np.arange(zRange.size):

        if i == 0 or i == 1:
            if i == 0:
                # At the first data point we cannot perform the trapezoid routine
                # and instead opt for a pseudo midstep method. Just to be clear 
                # this isn't a midstep method but gives sufficient results for 
                # the first step 
                fatmpNode_0 = h5file.get_node('/fa', 'zPos'+str(i))
                fatmp_0 = fatmpNode_0.read()
                RafirstItTmp_0 = z_step*np.dot(fatmp_0, Uinitial)*w_step
                RafirstItTmp = RafirstItTmp_0
                h5file.create_array(groupRafirstItNum, "zPos" + str(i), RafirstItTmp, "zPos Set") 

            if i == 1:
                ## Trapezoid integration routine for the second data point
                fatmpNode_0 = h5file.get_node('/fa', 'zPos'+str(i-1))
                fatmpNode_1 = h5file.get_node('/fa', 'zPos'+str(i))
                fatmp_0 = fatmpNode_0.read()
                fatmp_1 = fatmpNode_1.read()
                RafirstItTmp_0 = z_step*np.dot(fatmp_0, Uinitial)*w_step
                RafirstItTmp_1 = z_step*np.dot(fatmp_1, Uinitial)*w_step
                RafirstItTmp = 0.5 * (RafirstItTmp_0 + RafirstItTmp_1)
                h5file.create_array(groupRafirstItNum, "zPos" + str(i), RafirstItTmp, "zPos Set") 

        else:
            ## Trapezoid integration routine 
            RafirstItTmp_Node = h5file.get_node('/RafirstItNum', 'zPos'+str(i-1))
            RafirstItTmp = RafirstItTmp_Node.read()

            fatmpNode_0 = h5file.get_node('/fa', 'zPos'+str(i-1))
            fatmpNode_1 = h5file.get_node('/fa', 'zPos'+str(i))
            fatmp_0 = fatmpNode_0.read()
            fatmp_1 = fatmpNode_1.read()
            RafirstItTmp_0 = z_step*np.dot(fatmp_0, Uinitial)*w_step
            RafirstItTmp_1 = z_step*np.dot(fatmp_1, Uinitial)*w_step
            RafirstItTmp += 0.5 * (RafirstItTmp_0 + RafirstItTmp_1)
            h5file.create_array(groupRafirstItNum, "zPos" + str(i), RafirstItTmp, "zPos Set") 
            print "-> RafirstItNum z =", zRange[i], "finished"


    groupRbfirstItNum = h5file.create_group("/", "RbfirstItNum", "z Position")
    RbfirstItTmp = Vinitial.copy()
    for i in np.arange(zRange.size):

        if i == 0 or i == 1:
            if i == 0:
                # At the first data point we cannot perform the trapezoid routine
                # and instead opt for a pseudo midstep method. Just to be clear 
                # this isn't a midstep method but gives sufficient results for 
                # the first step 
                fbtmpNode_0 = h5file.get_node('/fb', 'zPos'+str(i))
                fbtmp_0 = fbtmpNode_0.read()
                RbfirstItTmp_0 = z_step*np.dot(fbtmp_0, Uinitial)*w_step
                RbfirstItTmp = RbfirstItTmp_0
                h5file.create_array(groupRbfirstItNum, "zPos" + str(i), RbfirstItTmp, "zPos Set") 

            if i == 1:
                ## Trapezoid integration routine for the second data point
                fbtmpNode_0 = h5file.get_node('/fb', 'zPos'+str(i-1))
                fbtmpNode_1 = h5file.get_node('/fb', 'zPos'+str(i))
                fbtmp_0 = fbtmpNode_0.read()
                fbtmp_1 = fbtmpNode_1.read()
                RbfirstItTmp_0 = z_step*np.dot(fbtmp_0, Uinitial)*w_step
                RbfirstItTmp_1 = z_step*np.dot(fbtmp_1, Uinitial)*w_step
                RbfirstItTmp = 0.5 * (RbfirstItTmp_0 + RbfirstItTmp_1)
                h5file.create_array(groupRbfirstItNum, "zPos" + str(i), RbfirstItTmp, "zPos Set") 

        else:
            ## Trapezoid integration routine 
            RbfirstItTmp_Node = h5file.get_node('/RbfirstItNum', 'zPos'+str(i-1))
            RbfirstItTmp = RbfirstItTmp_Node.read()

            fbtmpNode_0 = h5file.get_node('/fb', 'zPos'+str(i-1))
            fbtmpNode_1 = h5file.get_node('/fb', 'zPos'+str(i))
            fbtmp_0 = fbtmpNode_0.read()
            fbtmp_1 = fbtmpNode_1.read()
            RbfirstItTmp_0 = z_step*np.dot(fbtmp_0, Uinitial)*w_step
            RbfirstItTmp_1 = z_step*np.dot(fbtmp_1, Uinitial)*w_step
            RbfirstItTmp += 0.5 * (RbfirstItTmp_0 + RbfirstItTmp_1)
            h5file.create_array(groupRbfirstItNum, "zPos" + str(i), RbfirstItTmp, "zPos Set") 
            print "-> RbfirstItNum z =", zRange[i], "finished"



    ## Create analytical first-order solutions Ra and Rb which will be used
    ## to reconstruct Va/Ua and Vb/Ub in the analysis
    ## (These correspond the f function in the exponent of the unitary
    ## transformation creating the analytical solution in the paper)
    print "# Generating Ra, Rb analytical first-order solution. #"
    groupRafirstItAna= h5file.create_group("/", "RafirstItAna", "z Position")
    for i in np.arange(zRange.size):
        RafirstItAna = ra_analytical(X, Y, z_start, zRange[i], coupling, A, B, C, pump_width)
        h5file.create_array(groupRafirstItAna, "zPos" + str(i), RafirstItAna , "zPos Set") 
        print "-> RafirstItAna z =", zRange[i], "finished"

    groupRbfirstItAna= h5file.create_group("/", "RbfirstItAna", "z Position")
    for i in np.arange(zRange.size):
        RbfirstItAna = rb_analytical(X, Y, z_start, zRange[i], coupling, A, B, C, pump_width)
        h5file.create_array(groupRbfirstItAna, "zPos" + str(i), RbfirstItAna, "zPos Set") 
        print "-> RbfirstItAna z =", zRange[i], "finished"


    ## Calculate rigorous solution
    print "# Generating rigorous solution. #"
    print ""
    print "############################"
    print "# Start Iteration process. #"
    print "############################"


    print "## Iterating Vb and Ua ##"
    # This value should never be reached
    maxIterations = 100
    for iteration in np.arange(maxIterations):
        print ""
        print "# Iteration:", iteration
        print "-> Computing Vb"
        VdiffNormMean = 0

        for i in np.arange(zRange.size):
            if (i == 0 or i == 1):
                if i == 0:
                    # At the first data point we cannot perform the trapezoid routine
                    # and instead opt for a pseudo midstep method. Just to be clear 
                    # this isn't a midstep method but gives sufficient results for 
                    # the first step 
                     
                    # Load old Vb data
                    VbTmpNode = h5file.get_node('/Vb', 'zPos'+str(i))
                    VbTmp = VbTmpNode.read()

                    fbtmpNode_0 = h5file.get_node('/fb', 'zPos'+str(i))
                    UaTmpNode_0 = h5file.get_node('/Ua', 'zPos'+str(i))
                    fbtmp_0 = fbtmpNode_0.read()
                    UaTmp_0 = UaTmpNode_0.read()

                    VbNewTmp_0 = z_step*np.dot(fbtmp_0, np.conjugate(UaTmp_0))*w_step
                    VbNewTmp = VbNewTmp_0

                    # Calculate difference between old and new Vb
                    VdiffNorm = np.linalg.norm(VbNewTmp-VbTmp) / np.linalg.norm(VbNewTmp)
                    VdiffNormMean += VdiffNorm
                    # Save new Vb
                    VbTmpNode[:] = VbNewTmp

                if i == 1:
                    ## Trapezoid integration routine for the second data point
                    VbTmpNode = h5file.get_node('/Vb', 'zPos'+str(i))
                    VbTmp = VbTmpNode.read()

                    fbtmpNode_0 = h5file.get_node('/fb', 'zPos'+str(i-1))
                    fbtmpNode_1 = h5file.get_node('/fb', 'zPos'+str(i))
                    UaTmpNode_0 = h5file.get_node('/Ua', 'zPos'+str(i-1))
                    UaTmpNode_1 = h5file.get_node('/Ua', 'zPos'+str(i))
                    fbtmp_0 = fbtmpNode_0.read()
                    fbtmp_1 = fbtmpNode_1.read()
                    UaTmp_0 = UaTmpNode_0.read()
                    UaTmp_1 = UaTmpNode_1.read()

                    VbNewTmp_0 = z_step*np.dot(fbtmp_0, np.conjugate(UaTmp_0))*w_step
                    VbNewTmp_1 = z_step*np.dot(fbtmp_1, np.conjugate(UaTmp_1))*w_step
                    VbNewTmp = 0.5 * (VbNewTmp_0 + VbNewTmp_1)

                    # Calculate difference between old and new Vb
                    VdiffNorm = np.linalg.norm(VbNewTmp-VbTmp) / np.linalg.norm(VbNewTmp)
                    VdiffNormMean += VdiffNorm

                    # Save new Vb
                    VbTmpNode[:] = VbNewTmp

            else:
                ## Trapezoid integration routine 
                VbNewTmpNode = h5file.get_node('/Vb', 'zPos'+str(i-1))
                VbTmpNode = h5file.get_node('/Vb', 'zPos'+str(i))
                VbNewTmp = VbNewTmpNode.read()
                VbTmp = VbTmpNode.read()

                fbtmpNode_0 = h5file.get_node('/fb', 'zPos'+str(i-1))
                fbtmpNode_1 = h5file.get_node('/fb', 'zPos'+str(i))
                UaTmpNode_0 = h5file.get_node('/Ua', 'zPos'+str(i-1))
                UaTmpNode_1 = h5file.get_node('/Ua', 'zPos'+str(i))
                fbtmp_0 = fbtmpNode_0.read()
                fbtmp_1 = fbtmpNode_1.read()
                UaTmp_0 = UaTmpNode_0.read()
                UaTmp_1 = UaTmpNode_1.read()

                VbNewTmp_0 = z_step*np.dot(fbtmp_0, np.conjugate(UaTmp_0))*w_step
                VbNewTmp_1 = z_step*np.dot(fbtmp_1, np.conjugate(UaTmp_1))*w_step
                VbNewTmp += 0.5 * (VbNewTmp_0 + VbNewTmp_1)

                # Calculate difference between old and new Vb
                VdiffNorm = np.linalg.norm(VbNewTmp-VbTmp) / np.linalg.norm(VbNewTmp)
                VdiffNormMean += VdiffNorm

                # Save new Vb
                VbTmpNode[:] = VbNewTmp


        VdiffNormMean = VdiffNormMean/zRange.size
        print "-> VdiffNormMean =", VdiffNormMean 


        print "-> Computing Ua"
        UdiffNormMean = 0
        UaNewTmp = Uinitial.copy()
        for i in np.arange(zRange.size):
            if (i == 0):
                if i == 0:
                    # At the first data point we cannot perform the trapezoid routine
                    # and instead opt for a pseudo midstep method. Just to be clear 
                    # this isn't a midstep method but gives sufficient results for 
                    # the first step 
                    
                    UaTmpNode = h5file.get_node('/Ua', 'zPos'+str(i))
                    UaTmp = UaTmpNode.read()

                    fatmpNode_0 = h5file.get_node('/fa', 'zPos'+str(i))
                    VbTmpNode_0 = h5file.get_node('/Vb', 'zPos'+str(i))
                    fatmp_0 = fatmpNode_0.read()
                    VbTmp_0 = VbTmpNode_0.read()

                    UaNewTmp_0 = Uinitial + z_step*np.dot(fatmp_0, np.conjugate(VbTmp_0))*w_step
                    UaNewTmp = UaNewTmp_0

                    # Calculate difference between old and new Ua
                    UdiffNorm = np.linalg.norm(UaNewTmp-UaTmp) / np.linalg.norm(UaNewTmp - Uinitial)
                    UdiffNormMean += UdiffNorm
                    # Save new Ua
                    UaTmpNode[:] = UaNewTmp

                if i == 1:
                    ## Trapezoid integration routine for the second data point
                    UaTmpNode = h5file.get_node('/Ua', 'zPos'+str(i))
                    UaTmp = UaTmpNode.read()

                    fatmpNode_0 = h5file.get_node('/fa', 'zPos'+str(i-1))
                    fatmpNode_1 = h5file.get_node('/fa', 'zPos'+str(i))
                    VbTmpNode_0 = h5file.get_node('/Vb', 'zPos'+str(i-1))
                    VbTmpNode_1 = h5file.get_node('/Vb', 'zPos'+str(i))
                    fatmp_0 = fatmpNode_0.read()
                    fatmp_1 = fatmpNode_1.read()
                    VbTmp_0 = VbTmpNode_0.read()
                    VbTmp_1 = VbTmpNode_1.read()

                    UaNewTmp_0 = z_step*np.dot(fatmp_0, np.conjugate(VbTmp_0))*w_step
                    UaNewTmp_1 = z_step*np.dot(fatmp_1, np.conjugate(VbTmp_1))*w_step
                    UaNewTmp = Uinitial + 0.5 * (UaNewTmp_0 + UaNewTmp_1)

                    # Calculate difference between old and new Ua
                    UdiffNorm = np.linalg.norm(UaNewTmp-UaTmp) / np.linalg.norm(UaNewTmp - Uinitial)
                    UdiffNormMean += UdiffNorm

                    # Save new Ua
                    UaTmpNode[:] = UaNewTmp

            else:
                ## Trapezoid integration routine 
                UaNewTmpNode = h5file.get_node('/Ua', 'zPos'+str(i-1))
                UaTmpNode = h5file.get_node('/Ua', 'zPos'+str(i))
                UaNewTmp = UaNewTmpNode.read()
                UaTmp = UaTmpNode.read()

                fatmpNode_0 = h5file.get_node('/fa', 'zPos'+str(i-1))
                fatmpNode_1 = h5file.get_node('/fa', 'zPos'+str(i))
                VbTmpNode_0 = h5file.get_node('/Vb', 'zPos'+str(i-1))
                VbTmpNode_1 = h5file.get_node('/Vb', 'zPos'+str(i))
                fatmp_0 = fatmpNode_0.read()
                fatmp_1 = fatmpNode_1.read()
                VbTmp_0 = VbTmpNode_0.read()
                VbTmp_1 = VbTmpNode_1.read()


                UaNewTmp_0 = z_step*np.dot(fatmp_0, np.conjugate(VbTmp_0))*w_step
                UaNewTmp_1 = z_step*np.dot(fatmp_1, np.conjugate(VbTmp_1))*w_step
                UaNewTmp = UaNewTmp + 0.5 * (UaNewTmp_0 + UaNewTmp_1)

                #Calculate Difference between old and new Ua
                UdiffNorm = np.linalg.norm(UaNewTmp-UaTmp) / np.linalg.norm(UaNewTmp - Uinitial)
                UdiffNormMean += UdiffNorm
                # Save new Ua
                UaTmpNode[:] = UaNewTmp


        UdiffNormMean = UdiffNormMean/zRange.size
        print "UdiffNormMean =", UdiffNormMean 


        if UdiffNormMean <= 1e-6 and VdiffNormMean <= 1e-6:
            print iteration, "iterations needed to converge."
            print ""
            print ""
            break 


    ## This part is mostly a copy of the previous one.
    print "## Iterating Va and Ub ##"
    # This value should never be reached
    maxIterations = 100
    for iteration in np.arange(maxIterations):
        print ""
        print "# Iteration:", iteration
        print "-> Computing Va"

        VdiffNormMean = 0

        for i in np.arange(zRange.size):
            if (i == 0 or i == 1):
                if i == 0:
                    # At the first data point we cannot perform the trapezoid routine
                    # and instead opt for a pseudo midstep method. Just to be clear 
                    # this isn't a midstep method but gives sufficient results for 
                    # the first step 
                    
                    # Load old Va data
                    VaTmpNode = h5file.get_node('/Va', 'zPos'+str(i))
                    VaTmp = VaTmpNode.read()

                    fatmpNode_0 = h5file.get_node('/fa', 'zPos'+str(i))
                    UbTmpNode_0 = h5file.get_node('/Ub', 'zPos'+str(i))
                    fatmp_0 = fatmpNode_0.read()
                    UbTmp_0 = UbTmpNode_0.read()

                    VaNewTmp_0 = z_step*np.dot(fatmp_0, np.conjugate(UbTmp_0))*w_step
                    VaNewTmp = VaNewTmp_0

                    # Calculate difference between old and new Va
                    VdiffNorm = np.linalg.norm(VaNewTmp-VaTmp) / np.linalg.norm(VaNewTmp)
                    VdiffNormMean += VdiffNorm
                    # Save new Va
                    VaTmpNode[:] = VaNewTmp

                if i == 1:
                    ## Trapezoid integration routine for the second data point
                    VaTmpNode = h5file.get_node('/Va', 'zPos'+str(i))
                    VaTmp = VaTmpNode.read()

                    fatmpNode_0 = h5file.get_node('/fa', 'zPos'+str(i-1))
                    fatmpNode_1 = h5file.get_node('/fa', 'zPos'+str(i))
                    UbTmpNode_0 = h5file.get_node('/Ub', 'zPos'+str(i-1))
                    UbTmpNode_1 = h5file.get_node('/Ub', 'zPos'+str(i))
                    fatmp_0 = fatmpNode_0.read()
                    fatmp_1 = fatmpNode_1.read()
                    UbTmp_0 = UbTmpNode_0.read()
                    UbTmp_1 = UbTmpNode_1.read()

                    VaNewTmp_0 = z_step*np.dot(fatmp_0, np.conjugate(UbTmp_0))*w_step
                    VaNewTmp_1 = z_step*np.dot(fatmp_1, np.conjugate(UbTmp_1))*w_step
                    VaNewTmp = 0.5 * (VaNewTmp_0 + VaNewTmp_1)

                    # Calculate difference between old and new Va
                    VdiffNorm = np.linalg.norm(VaNewTmp-VaTmp) / np.linalg.norm(VaNewTmp)
                    VdiffNormMean += VdiffNorm

                    # Save new Va
                    VaTmpNode[:] = VaNewTmp

            else:
                ## Trapezoid integration routine 
                VaNewTmpNode = h5file.get_node('/Va', 'zPos'+str(i-1))
                VaTmpNode = h5file.get_node('/Va', 'zPos'+str(i))
                VaNewTmp = VaNewTmpNode.read()
                VaTmp = VaTmpNode.read()

                fatmpNode_0 = h5file.get_node('/fa', 'zPos'+str(i-1))
                fatmpNode_1 = h5file.get_node('/fa', 'zPos'+str(i))
                UbTmpNode_0 = h5file.get_node('/Ub', 'zPos'+str(i-1))
                UbTmpNode_1 = h5file.get_node('/Ub', 'zPos'+str(i))
                fatmp_0 = fatmpNode_0.read()
                fatmp_1 = fatmpNode_1.read()
                UbTmp_0 = UbTmpNode_0.read()
                UbTmp_1 = UbTmpNode_1.read()

                VaNewTmp_0 = z_step*np.dot(fatmp_0, np.conjugate(UbTmp_0))*w_step
                VaNewTmp_1 = z_step*np.dot(fatmp_1, np.conjugate(UbTmp_1))*w_step
                VaNewTmp += 0.5 * (VaNewTmp_0 + VaNewTmp_1)

                # Calculate difference between old and new Va
                VdiffNorm = np.linalg.norm(VaNewTmp-VaTmp) / np.linalg.norm(VaNewTmp)
                VdiffNormMean += VdiffNorm

                # Save new Va
                VaTmpNode[:] = VaNewTmp

        VdiffNormMean = VdiffNormMean/zRange.size
        print "-> VdiffNormMean =", VdiffNormMean 


        print "-> Computing Ub"
        UdiffNormMean = 0
        UcNewTmp = Uinitial.copy()
        for i in np.arange(zRange.size):
            if (i == 0):
                if i == 0:
                    # At the first data point we cannot perform the trapezoid routine
                    # and instead opt for a pseudo midstep method. Just to be clear 
                    # this isn't a midstep method but gives sufficient results for 
                    # the first step 
                    
                    UbTmpNode = h5file.get_node('/Ub', 'zPos'+str(i))
                    UbTmp = UbTmpNode.read()

                    fbtmpNode_0 = h5file.get_node('/fb', 'zPos'+str(i))
                    VaTmpNode_0 = h5file.get_node('/Va', 'zPos'+str(i))
                    fbtmp_0 = fbtmpNode_0.read()
                    VaTmp_0 = VaTmpNode_0.read()

                    UbNewTmp_0 = Uinitial + z_step*np.dot(fbtmp_0, np.conjugate(VaTmp_0))*w_step
                    UbNewTmp = UbNewTmp_0

                    # Calculate difference between old and new Ub
                    UdiffNorm = np.linalg.norm(UbNewTmp-UbTmp) / np.linalg.norm(UbNewTmp - Uinitial)
                    UdiffNormMean += UdiffNorm
                    # Save new Ub
                    UbTmpNode[:] = UbNewTmp

                if i == 1:
                    ## Trapezoid integration routine for the second data point
                    UbTmpNode = h5file.get_node('/Ub', 'zPos'+str(i))
                    UbTmp = UbTmpNode.read()

                    fbtmpNode_0 = h5file.get_node('/fb', 'zPos'+str(i-1))
                    fbtmpNode_1 = h5file.get_node('/fb', 'zPos'+str(i))
                    VaTmpNode_0 = h5file.get_node('/Va', 'zPos'+str(i-1))
                    VaTmpNode_1 = h5file.get_node('/Va', 'zPos'+str(i))
                    fbtmp_0 = fbtmpNode_0.read()
                    fbtmp_1 = fbtmpNode_1.read()
                    VaTmp_0 = VaTmpNode_0.read()
                    VaTmp_1 = VaTmpNode_1.read()

                    UbNewTmp_0 = z_step*np.dot(fbtmp_0, np.conjugate(VaTmp_0))*w_step
                    UbNewTmp_1 = z_step*np.dot(fbtmp_1, np.conjugate(VaTmp_1))*w_step
                    UbNewTmp = Uinitial + 0.5 * (UbNewTmp_0 + UbNewTmp_1)

                    # Calculate difference between old and new Ub
                    UdiffNorm = np.linalg.norm(UbNewTmp-UbTmp) / np.linalg.norm(UbNewTmp - Uinitial)
                    UdiffNormMean += UdiffNorm

                    # Save new Uc
                    UbTmpNode[:] = UbNewTmp

            else:
                ## Trapezoid integration routine 
                UbNewTmpNode = h5file.get_node('/Ub', 'zPos'+str(i-1))
                UbTmpNode = h5file.get_node('/Ub', 'zPos'+str(i))
                UbNewTmp = UbNewTmpNode.read()
                UbTmp = UbTmpNode.read()

                fbtmpNode_0 = h5file.get_node('/fb', 'zPos'+str(i-1))
                fbtmpNode_1 = h5file.get_node('/fb', 'zPos'+str(i))
                VaTmpNode_0 = h5file.get_node('/Va', 'zPos'+str(i-1))
                VaTmpNode_1 = h5file.get_node('/Va', 'zPos'+str(i))
                fbtmp_0 = fbtmpNode_0.read()
                fbtmp_1 = fbtmpNode_1.read()
                VaTmp_0 = VaTmpNode_0.read()
                VaTmp_1 = VaTmpNode_1.read()


                UbNewTmp_0 = z_step*np.dot(fbtmp_0, np.conjugate(VaTmp_0))*w_step
                UbNewTmp_1 = z_step*np.dot(fbtmp_1, np.conjugate(VaTmp_1))*w_step
                UbNewTmp = UbNewTmp + 0.5 * (UbNewTmp_0 + UbNewTmp_1)

                #Calculate Difference between old and new Ub
                UdiffNorm = np.linalg.norm(UbNewTmp-UbTmp) / np.linalg.norm(UbNewTmp - Uinitial)
                UdiffNormMean += UdiffNorm
                # Save new Ub
                UbTmpNode[:] = UbNewTmp


        UdiffNormMean = UdiffNormMean/zRange.size
        print "UdiffNormMean =", UdiffNormMean 


        if UdiffNormMean <= 1e-6 and VdiffNormMean <= 1e-6:
            print iteration, "iterations needed to converge."
            break 


    h5file.close()
