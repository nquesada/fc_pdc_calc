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
""" This file contains the function calc_FC with calculates
the analytic and rigorous solution of a given frequency conversion
process as specified in the corresponding paper. It saves the data
in a hdf5 file for further analysis.

Details on the math can be found in the corresponding paper."""

import numpy as np
import tables


############################
# Definitions and formulas #
############################

## Define FC functions

# x = omega
# y = omega'

# Phasematching function
# as defined for E_a: x = omega, y = omega', k_p = A, k_c = B, k_a = C
def dk(x, y, A, B, C):
    """ Return the k-vector mismatch between the three waves."""
    return (A*(y-x) - B*(y) + C*(x))


# Pump distribution
# (Note that the pump distribution is not normalized to replicate
# the formula in the paper.)
def alpha_no_vec(x, sigma_x):
    """ Return the amplitude of the pump distribution alpha center at 0
    with given width sigma and frequency x"""
    # Gaussian pump
    return np.exp(-x**2 / (2*sigma_x**2) )

    # First Hermite Function pump
    #return x * np.exp(-x**2 / (2*sigma_x**2) )

    # Rectangular pump
    #if x <= -sigma_x:
    #    return 0
    #elif x >= sigma_x:
    #    return 0
    #else:
    #    return 1

# Vectorize alpha in order to cope with the if for a rectangular pump
alpha = np.vectorize(alpha_no_vec)


# Analytical phase-matching function
def pm(x, y, z_start, z_stop, A, B, C):
    """ Analytic phase-matching function for a waveguide of length
    z_stop - z_start. It is however assumed that the waveguide
    starts at 0 and ends at z_stop - z_start. This enables us to
    write the solution as a sinc which circumvents various numerical
    issues. """

    return (z_stop - z_start) * np.sinc((z_stop - z_start)/2*dk(x, y, A, B, C)/np.pi) \
        * np.exp(-1j*dk(x, y, A, B, C)*(z_stop - z_start)/2)


# Analytical solution (A beam)
# (I cannot name it f like in the paper since this is occupied by the f
# function from the differential equation)
def ra_analytical(x, y, z_start, z_stop, coupling, A, B, C, pump_width):
    """ Analytical solution for the A beam of first-order frequency 
    conversion."""
    return -1j * np.conjugate(coupling) * np.conjugate(alpha(x-y, pump_width)) \
            * pm(x, y, z_start, z_stop, A, B, C)

# Analytical solution (C beam)
# (I cannot name it f like in the paper since this is occupied by the f
# function from the differential equation)
def rc_analytical(x, y, z_start, z_stop, coupling, A, B, C, pump_width):
    """ Analytical solution for the C beam of first-order SFG.
        Note: rc_analytical = complex conjugate and x-y switch of ra_analytical"""
    return 1j * coupling * alpha(y-x, pump_width) \
            * np.conjugate(pm(y, x, z_start, z_stop, A, B, C))

# Numerical phase-matching function
def grating(z):
    return np.exp(-0.1*z**2)

def phi_num(x, y, z, A, B, C):
    """ Return the value of the phi function in the differential equation"""
    return np.exp(-1j * dk(x, y, A, B, C)*z)

    # Additional hypergrating to get rid of sinc side peaks:
    # (Currently not used)
    #return grating(z) * np.exp(-1j * dk(x, y, A, B, C)*z)

# f function with coupling value 
def f_a(x, y, z, coupling, A, B, C, pump_width):
    """ Body of the differential equation for A 
    (on the right of the differential equation)"""
    return -1j * np.conjugate(coupling) * alpha(y-x, pump_width) \
            * phi_num(x, y, z, A, B, C)

def f_c(x, y, z, coupling, A, B, C, pump_width):
    """ Body of the differential equation for C 
    (on the right of the differential equation)"""
    # This function is just the conjugate and x-y switched version of f_a
    return np.conjugate(f_a(y, x, z, coupling, A, B, C, pump_width))



# The big function creation the analytic and rigorous solution for FC
def calc_FC(coupling, w_start, w_stop, w_steps, z_start, z_stop, z_steps,\
        A, B, C, pump_width, save_directory):
    """ Solve FC in three different ways:
    1) Analytical first-order solution
    2) Numerical first-order solution (identical to analytic solution, but 
       features a numerical z-integration)
    3) Rigorous solution via an iterative approach.
    
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


    ## Generate Initial Matrices to store the results for Ua, Uc, Va and Vc:
    #   In all generated matrices "dtype=complex" has to be explicitly stated,
    #   otherwise the applied matrix operations cannot handle complex numbers
    print "# Generating Ua, Uc, Va and Vc initial #"

    Vinitial = np.zeros( (wRange.size, wRange.size), dtype=complex)
    # This is the definition of a discretized delta function (note the /w_step)
    Uinitial = np.identity(wRange.size, dtype=complex) / w_step

    # Create and open hdf5 File
    ## Attention do not change the naming, it's exact nature is used in the 
    ## analysis scripts.
    directory_name = save_directory + "/FC_calc"
    evaluation_parameters = "---PAR--pow_" + "%.3f" % coupling \
            + "--w_" + "%.1f" % w_start + "_" + "%.1f" % w_stop + "_" \
            + "%.4d" % w_steps + \
            "--z_" +  "%.1f" % z_start + "_" + "%.1f" % z_stop + "_" \
            + "%.4d" % z_steps
    fc_parameters = "---SFG--A_" + "%.1f" % A + "--B_" + "%.1f" % B \
            + "--C_" + "%.1f" % C + "--pumpWidth_" + "%.5f" % pump_width

    filename = directory_name + evaluation_parameters + fc_parameters + ".h5"
    h5file = tables.open_file(filename, mode = "w", title = "FC-Bog-Prop")

    # Create subgroup for Va, Vc, Ua and Uc
    groupVa = h5file.create_group("/", "Va", "z-Position")
    groupUa = h5file.create_group("/", "Ua", "z-Position")
    groupVc = h5file.create_group("/", "Vc", "z-Position")
    groupUc = h5file.create_group("/", "Uc", "z-Position")

    # Generate initial Va and Ua Arrays

    # Pytables does not want identifiers with a "." consequently the zPos is
    # identified by  0,1,2,3 and the actual position has to be recalculated
    # with help of the given z_start and z_step 
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


    print "-> Generating Initial Vc"
    print "-> Generating Vc initial array at z =", 
    for i in np.arange(zRange.size):
        h5file.create_array(groupVc, "zPos" + str(i), Vinitial, "z-Position") 
        print i,
    print "finished."

    print "-> Generating Initial Uc"
    print "-> Generating Uc initial array at z =",
    for i in np.arange(zRange.size):
        h5file.create_array(groupUc, "zPos" + str(i), Uinitial, "z-Position") 
        print i,
    print "finished."

   
    ## Create f functions
    print "# Generating f functions #"

    # Note the flipped Y, X in order to cope with numpys  array definition
    # The matrix multiplication used in our solution will only work with this
    # exact notation
    Y, X = np.meshgrid(wRange,wRange)

    groupFa = h5file.create_group("/", "fa", "z Position")
    for i in np.arange(zRange.size):
        famatrix = f_a(X, Y, zRange[i], coupling, A, B, C, pump_width)
        h5file.create_array(groupFa, "zPos" + str(i), famatrix, "zPos-Set") 
        print "-> fa array z =", zRange[i], "finished"

    print ""

    # Actually fc is just a transposed fa with an additional minus sign
    groupFc = h5file.create_group("/", "fc", "z Position")
    for i in np.arange(zRange.size):
        fcmatrix = f_c(X, Y, zRange[i], coupling, A, B, C, pump_width)
        h5file.create_array(groupFc, "zPos" + str(i), fcmatrix, "zPos-Set") 
        print "-> fc array z =", zRange[i], "finished"



    ## Create numerical first-order solution
    print "# Generating Ra, Rc numerical first-order solution. #"

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



    groupRcfirstItNum = h5file.create_group("/", "RcfirstItNum", "z Position")
    RcfirstItTmp = Vinitial.copy()
    for i in np.arange(zRange.size):

        if i == 0 or i == 1:
            if i == 0:
                # At the first data point we cannot perform the trapezoid routine
                # and instead opt for a pseudo midstep method. Just to be clear 
                # this isn't a midstep method but gives sufficient results for 
                # the first step 
                fctmpNode_0 = h5file.get_node('/fc', 'zPos'+str(i))
                fctmp_0 = fctmpNode_0.read()
                RcfirstItTmp_0 = z_step*np.dot(fctmp_0, Uinitial)*w_step
                RcfirstItTmp = RcfirstItTmp_0
                h5file.create_array(groupRcfirstItNum, "zPos" + str(i), RcfirstItTmp, "zPos Set") 

            if i == 1:
                ## Trapezoid integration routine for the second data point
                fctmpNode_0 = h5file.get_node('/fc', 'zPos'+str(i-1))
                fctmpNode_1 = h5file.get_node('/fc', 'zPos'+str(i))
                fctmp_0 = fctmpNode_0.read()
                fctmp_1 = fctmpNode_1.read()
                RcfirstItTmp_0 = z_step*np.dot(fctmp_0, Uinitial)*w_step
                RcfirstItTmp_1 = z_step*np.dot(fctmp_1, Uinitial)*w_step
                RcfirstItTmp = 0.5 * (RcfirstItTmp_0 + RcfirstItTmp_1)
                h5file.create_array(groupRcfirstItNum, "zPos" + str(i), RcfirstItTmp, "zPos Set") 

        else:
            ## Trapezoid integration routine 
            RcfirstItTmp_Node = h5file.get_node('/RcfirstItNum', 'zPos'+str(i-1))
            RcfirstItTmp = RcfirstItTmp_Node.read()

            fctmpNode_0 = h5file.get_node('/fc', 'zPos'+str(i-1))
            fctmpNode_1 = h5file.get_node('/fc', 'zPos'+str(i))
            fctmp_0 = fctmpNode_0.read()
            fctmp_1 = fctmpNode_1.read()
            RcfirstItTmp_0 = z_step*np.dot(fctmp_0, Uinitial)*w_step
            RcfirstItTmp_1 = z_step*np.dot(fctmp_1, Uinitial)*w_step
            RcfirstItTmp += 0.5 * (RcfirstItTmp_0 + RcfirstItTmp_1)
            h5file.create_array(groupRcfirstItNum, "zPos" + str(i), RcfirstItTmp, "zPos Set") 
            print "-> RcfirstItNum z =", zRange[i], "finished"



    ## Create analytical first-order solutions Ra and Rc which will be used
    ## to reconstruct Va/Ua and Vc/Uc in the analysis
    ## (These correspond the f function in the exponent of the unitary
    ## transformation creating the analytical solution in the paper)
    print "# Generating Ra, Rc analytical solution. #"
    groupRafirstItAna= h5file.create_group("/", "RafirstItAna", "z Position")
    for i in np.arange(zRange.size):
        RafirstItAna = ra_analytical(X, Y, z_start, zRange[i], coupling, A, B, C, pump_width)
        h5file.create_array(groupRafirstItAna, "zPos" + str(i), RafirstItAna , "zPos Set") 
        print "-> RafirstItAna z =", zRange[i], "finished"

    groupRcfirstItAna= h5file.create_group("/", "RcfirstItAna", "z Position")
    for i in np.arange(zRange.size):
        RcfirstItAna = rc_analytical(X, Y, z_start, zRange[i], coupling, A, B, C, pump_width)
        h5file.create_array(groupRcfirstItAna, "zPos" + str(i), RcfirstItAna, "zPos Set") 
        print "-> RcfirstItAna z =", zRange[i], "finished"


    ## Calculate rigorous solution
    print "# Generating rigorous solution. #"
    print ""
    print "############################"
    print "# Start Iteration process. #"
    print "############################"


    print "## Iterating Vc and Ua ##"
    # This value should never be reached
    maxIterations = 100
    for iteration in np.arange(maxIterations):
        print ""
        print "# Iteration:", iteration
        print "-> Computing Vc"
        VdiffNormMean = 0

        for i in np.arange(zRange.size):
            if (i == 0 or i == 1):
                if i == 0:
                    # At the first data point we cannot perform the trapezoid routine
                    # and instead opt for a pseudo midstep method. Just to be clear 
                    # this isn't a midstep method but gives sufficient results for 
                    # the first step 
                    
                    # Load old Vc data
                    VcTmpNode = h5file.get_node('/Vc', 'zPos'+str(i))
                    VcTmp = VcTmpNode.read()

                    fctmpNode_0 = h5file.get_node('/fc', 'zPos'+str(i))
                    UaTmpNode_0 = h5file.get_node('/Ua', 'zPos'+str(i))
                    fctmp_0 = fctmpNode_0.read()
                    UaTmp_0 = UaTmpNode_0.read()

                    VcNewTmp_0 = z_step*np.dot(fctmp_0, UaTmp_0)*w_step
                    VcNewTmp = VcNewTmp_0

                    # Calculate difference between old and new Vc
                    VdiffNorm = np.linalg.norm(VcNewTmp-VcTmp) / np.linalg.norm(VcNewTmp)
                    VdiffNormMean += VdiffNorm
                    # Save new Vc
                    VcTmpNode[:] = VcNewTmp

                if i == 1:
                    ## Trapezoid integration routine for the second data point
                    VcTmpNode = h5file.get_node('/Vc', 'zPos'+str(i))
                    VcTmp = VcTmpNode.read()

                    fctmpNode_0 = h5file.get_node('/fc', 'zPos'+str(i-1))
                    fctmpNode_1 = h5file.get_node('/fc', 'zPos'+str(i))
                    UaTmpNode_0 = h5file.get_node('/Ua', 'zPos'+str(i-1))
                    UaTmpNode_1 = h5file.get_node('/Ua', 'zPos'+str(i))
                    fctmp_0 = fctmpNode_0.read()
                    fctmp_1 = fctmpNode_1.read()
                    UaTmp_0 = UaTmpNode_0.read()
                    UaTmp_1 = UaTmpNode_1.read()

                    VcNewTmp_0 = z_step*np.dot(fctmp_0, UaTmp_0)*w_step
                    VcNewTmp_1 = z_step*np.dot(fctmp_1, UaTmp_1)*w_step
                    VcNewTmp = 0.5 * (VcNewTmp_0 + VcNewTmp_1)

                    # Calculate difference between old and new Vc
                    VdiffNorm = np.linalg.norm(VcNewTmp-VcTmp) / np.linalg.norm(VcNewTmp)
                    VdiffNormMean += VdiffNorm

                    # Save new Vc
                    VcTmpNode[:] = VcNewTmp

            else:
                ## Trapezoid integration routine 
                VcNewTmpNode = h5file.get_node('/Vc', 'zPos'+str(i-1))
                VcTmpNode = h5file.get_node('/Vc', 'zPos'+str(i))
                VcNewTmp = VcNewTmpNode.read()
                VcTmp = VcTmpNode.read()

                fctmpNode_0 = h5file.get_node('/fc', 'zPos'+str(i-1))
                fctmpNode_1 = h5file.get_node('/fc', 'zPos'+str(i))
                UaTmpNode_0 = h5file.get_node('/Ua', 'zPos'+str(i-1))
                UaTmpNode_1 = h5file.get_node('/Ua', 'zPos'+str(i))
                fctmp_0 = fctmpNode_0.read()
                fctmp_1 = fctmpNode_1.read()
                UaTmp_0 = UaTmpNode_0.read()
                UaTmp_1 = UaTmpNode_1.read()

                VcNewTmp_0 = z_step*np.dot(fctmp_0, UaTmp_0)*w_step
                VcNewTmp_1 = z_step*np.dot(fctmp_1, UaTmp_1)*w_step
                VcNewTmp += 0.5 * (VcNewTmp_0 + VcNewTmp_1)

                # Calculate difference between old and new Vc
                VdiffNorm = np.linalg.norm(VcNewTmp-VcTmp) / np.linalg.norm(VcNewTmp)
                VdiffNormMean += VdiffNorm

                # Save new Vc
                VcTmpNode[:] = VcNewTmp


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
                    VcTmpNode_0 = h5file.get_node('/Vc', 'zPos'+str(i))
                    fatmp_0 = fatmpNode_0.read()
                    VcTmp_0 = VcTmpNode_0.read()

                    UaNewTmp_0 = Uinitial - z_step*np.dot(fatmp_0, VcTmp_0)*w_step
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
                    VcTmpNode_0 = h5file.get_node('/Vc', 'zPos'+str(i-1))
                    VcTmpNode_1 = h5file.get_node('/Vc', 'zPos'+str(i))
                    fatmp_0 = fatmpNode_0.read()
                    fatmp_1 = fatmpNode_1.read()
                    VcTmp_0 = VcTmpNode_0.read()
                    VcTmp_1 = VcTmpNode_1.read()

                    UaNewTmp_0 = z_step*np.dot(fatmp_0, VcTmp_0)*w_step
                    UaNewTmp_1 = z_step*np.dot(fatmp_1, VcTmp_1)*w_step
                    UaNewTmp = Uinitial - 0.5 * (UaNewTmp_0 + UaNewTmp_1)

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
                VcTmpNode_0 = h5file.get_node('/Vc', 'zPos'+str(i-1))
                VcTmpNode_1 = h5file.get_node('/Vc', 'zPos'+str(i))
                fatmp_0 = fatmpNode_0.read()
                fatmp_1 = fatmpNode_1.read()
                VcTmp_0 = VcTmpNode_0.read()
                VcTmp_1 = VcTmpNode_1.read()


                UaNewTmp_0 = z_step*np.dot(fatmp_0, VcTmp_0)*w_step
                UaNewTmp_1 = z_step*np.dot(fatmp_1, VcTmp_1)*w_step
                UaNewTmp = UaNewTmp - 0.5 * (UaNewTmp_0 + UaNewTmp_1)

                #Calculate Difference between old and new Ua
                UdiffNorm = np.linalg.norm(UaNewTmp-UaTmp) / np.linalg.norm(UaNewTmp - Uinitial)
                UdiffNormMean += UdiffNorm
                # Save new Ua
                UaTmpNode[:] = UaNewTmp


        UdiffNormMean = UdiffNormMean/zRange.size
        print "-> UdiffNormMean =", UdiffNormMean 


        if UdiffNormMean <= 1e-6 and VdiffNormMean <= 1e-6:
            print iteration, "iterations needed to converge."
            print ""
            print ""
            break 


    ## This part is mostly a copy of the previous one.
    print "## Iterating Va and Uc ##"
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
                    UcTmpNode_0 = h5file.get_node('/Uc', 'zPos'+str(i))
                    fatmp_0 = fatmpNode_0.read()
                    UcTmp_0 = UcTmpNode_0.read()

                    VaNewTmp_0 = z_step*np.dot(fatmp_0, UcTmp_0)*w_step
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
                    UcTmpNode_0 = h5file.get_node('/Uc', 'zPos'+str(i-1))
                    UcTmpNode_1 = h5file.get_node('/Uc', 'zPos'+str(i))
                    fatmp_0 = fatmpNode_0.read()
                    fatmp_1 = fatmpNode_1.read()
                    UcTmp_0 = UcTmpNode_0.read()
                    UcTmp_1 = UcTmpNode_1.read()

                    VaNewTmp_0 = z_step*np.dot(fatmp_0, UcTmp_0)*w_step
                    VaNewTmp_1 = z_step*np.dot(fatmp_1, UcTmp_1)*w_step
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
                UcTmpNode_0 = h5file.get_node('/Uc', 'zPos'+str(i-1))
                UcTmpNode_1 = h5file.get_node('/Uc', 'zPos'+str(i))
                fatmp_0 = fatmpNode_0.read()
                fatmp_1 = fatmpNode_1.read()
                UcTmp_0 = UcTmpNode_0.read()
                UcTmp_1 = UcTmpNode_1.read()

                VaNewTmp_0 = z_step*np.dot(fatmp_0, UcTmp_0)*w_step
                VaNewTmp_1 = z_step*np.dot(fatmp_1, UcTmp_1)*w_step
                VaNewTmp += 0.5 * (VaNewTmp_0 + VaNewTmp_1)

                # Calculate difference between old and new Va
                VdiffNorm = np.linalg.norm(VaNewTmp-VaTmp) / np.linalg.norm(VaNewTmp)
                VdiffNormMean += VdiffNorm

                # Save new Va
                VaTmpNode[:] = VaNewTmp

        VdiffNormMean = VdiffNormMean/zRange.size
        print "-> VdiffNormMean =", VdiffNormMean 


        print "-> Computing Uc"
        UdiffNormMean = 0
        UcNewTmp = Uinitial.copy()
        for i in np.arange(zRange.size):
            if (i == 0):
                if i == 0:
                    # At the first data point we cannot perform the trapezoid routine
                    # and instead opt for a pseudo midstep method. Just to be clear 
                    # this isn't a midstep method but gives sufficient results for 
                    # the first step 
                    
                    UcTmpNode = h5file.get_node('/Uc', 'zPos'+str(i))
                    UcTmp = UcTmpNode.read()

                    fctmpNode_0 = h5file.get_node('/fc', 'zPos'+str(i))
                    VaTmpNode_0 = h5file.get_node('/Va', 'zPos'+str(i))
                    fctmp_0 = fctmpNode_0.read()
                    VaTmp_0 = VaTmpNode_0.read()

                    UcNewTmp_0 = Uinitial - z_step*np.dot(fctmp_0, VaTmp_0)*w_step
                    UcNewTmp = UcNewTmp_0

                    # Calculate difference between old and new Uc
                    UdiffNorm = np.linalg.norm(UcNewTmp-UcTmp) / np.linalg.norm(UcNewTmp - Uinitial)
                    UdiffNormMean += UdiffNorm
                    # Save new Uc
                    UcTmpNode[:] = UcNewTmp

                if i == 1:
                    ## Trapezoid integration routine for the second data point
                    UcTmpNode = h5file.get_node('/Uc', 'zPos'+str(i))
                    UcTmp = UcTmpNode.read()

                    fctmpNode_0 = h5file.get_node('/fc', 'zPos'+str(i-1))
                    fctmpNode_1 = h5file.get_node('/fc', 'zPos'+str(i))
                    VaTmpNode_0 = h5file.get_node('/Va', 'zPos'+str(i-1))
                    VaTmpNode_1 = h5file.get_node('/Va', 'zPos'+str(i))
                    fctmp_0 = fctmpNode_0.read()
                    fctmp_1 = fctmpNode_1.read()
                    VaTmp_0 = VaTmpNode_0.read()
                    VaTmp_1 = VaTmpNode_1.read()

                    UcNewTmp_0 = z_step*np.dot(fctmp_0, VaTmp_0)*w_step
                    UcNewTmp_1 = z_step*np.dot(fctmp_1, VaTmp_1)*w_step
                    UcNewTmp = Uinitial - 0.5 * (UcNewTmp_0 + UcNewTmp_1)

                    # Calculate difference between old and new Uc
                    UdiffNorm = np.linalg.norm(UcNewTmp-UcTmp) / np.linalg.norm(UcNewTmp - Uinitial)
                    UdiffNormMean += UdiffNorm

                    # Save new Uc
                    UcTmpNode[:] = UcNewTmp

            else:
                ## Trapezoid integration routine 
                UcNewTmpNode = h5file.get_node('/Uc', 'zPos'+str(i-1))
                UcTmpNode = h5file.get_node('/Uc', 'zPos'+str(i))
                UcNewTmp = UcNewTmpNode.read()
                UcTmp = UcTmpNode.read()

                fctmpNode_0 = h5file.get_node('/fc', 'zPos'+str(i-1))
                fctmpNode_1 = h5file.get_node('/fc', 'zPos'+str(i))
                VaTmpNode_0 = h5file.get_node('/Va', 'zPos'+str(i-1))
                VaTmpNode_1 = h5file.get_node('/Va', 'zPos'+str(i))
                fctmp_0 = fctmpNode_0.read()
                fctmp_1 = fctmpNode_1.read()
                VaTmp_0 = VaTmpNode_0.read()
                VaTmp_1 = VaTmpNode_1.read()


                UcNewTmp_0 = z_step*np.dot(fctmp_0, VaTmp_0)*w_step
                UcNewTmp_1 = z_step*np.dot(fctmp_1, VaTmp_1)*w_step
                UcNewTmp = UcNewTmp - 0.5 * (UcNewTmp_0 + UcNewTmp_1)

                # Calculate Difference between old and new Uc
                UdiffNorm = np.linalg.norm(UcNewTmp-UcTmp) / np.linalg.norm(UcNewTmp - Uinitial)
                UdiffNormMean += UdiffNorm
                # Save new Uc
                UcTmpNode[:] = UcNewTmp


        UdiffNormMean = UdiffNormMean/zRange.size
        print "UdiffNormMean =", UdiffNormMean 


        if UdiffNormMean <= 1e-6 and VdiffNormMean <= 1e-6:
            print iteration, "iterations needed to converge."
            break 

    h5file.close()
