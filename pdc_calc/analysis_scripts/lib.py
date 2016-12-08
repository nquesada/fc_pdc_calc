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
""" This file collects functions which are used in various analysis scripts. """

import sys
import numpy as np
from scipy import linalg

def parse_PDC_file():
    """ This function parses the filename given via the command line, extracts
    all parameters for further analysis and gives the path to open the actual
    hdf5 file."""
    # Note: This parser depends on the exact nature of the filename do not
    # change calc_time_ord_FC.py without updating this function

    parameters = sys.argv[1:]

    path_to_hdf5_file = parameters[0]
    save_directory = parameters[1]


    ## Obtain the process parameters from the path_to_hdf5_file
    # title is just a dummy variable where the unimportant parts of the strings
    # are stored.
    (title, evaluation_paramters, pdc_parameters) = path_to_hdf5_file.split("---")

    (title, coupling, w_parameters, z_parameters) = evaluation_paramters.split("--")

    (title, A, B, C, pump_width) = pdc_parameters.split("--")


    # Get the frequency and z ranges 
    (title, w_start, w_stop, w_steps) = w_parameters.split("_")
    (title, z_start, z_stop, z_steps) = z_parameters.split("_")

    # Get the process parameters
    coupling = coupling.lstrip("pow_")
    A = A.lstrip("A_")
    B = B.lstrip("B_")
    C = C.lstrip("C_")

    pump_width = pump_width.lstrip("pumpWidth_")
    pump_width = pump_width.rstrip(".h5")


    # Print the obtained parameters 
    print "# PDC parameters" 
    print "coupling", coupling 

    print "w_start", w_start
    print "w_stop", w_stop
    print "w_steps", w_steps

    print "z_start", z_start
    print "z_stop", z_stop
    print "z_steps", z_steps

    print "A", A
    print "B", B
    print "C", C

    print "pump_width", pump_width

    # Convert the strings to float
    w_start = float(w_start)
    w_stop = float(w_stop)
    w_steps = float(w_steps)

    z_start = float(z_start)
    z_stop = float(z_stop)
    z_steps = float(z_steps)

    coupling = float(coupling)
    pump_width = float(pump_width)

    return (w_start, w_stop, w_steps, z_start, z_stop, z_steps, \
            coupling, pump_width, A, B, C, path_to_hdf5_file, save_directory)


def construct_u_and_v(r, w_step, wRange):
    """ Constructs the V and U matrices given the R matrix from the analytical
    or first-order numerical solution."""

    # Calculate necessary arrays
    delta_list = np.ones_like(wRange) / w_step

    Uinitial = np.identity(wRange.size, dtype=complex) / w_step

    # Perform SVD to get the modes
    rA, rD, rBh = linalg.svd(r)
    # Correct for w_step to get actual singular values
    rD = rD * w_step
    # Calculate vD and uDp singular values
    # (uDp = singular values of uD without identity)
    vD = np.sinh(rD)
    uD = np.cosh(rD)

    uDp = uD - 1

    # Transform into numerical basis again.
    vD = vD / w_step
    uDp = uDp / w_step

    # Construct U_a and Va
    # The trick here is to construct U while treating the delta function 
    # separately since the direct calculation leads to numerical issues.
    A = rA
    B = np.transpose(np.conjugate(rBh))

    u_a = Uinitial + np.dot(A, np.dot(np.diag(uDp), \
        np.conjugate(np.transpose(A))))

    v_a = np.dot(np.dot(A, np.diag(vD)), np.transpose(B))

    # Construct U_b und V_b solution
    # In contrast to the FC code the complete solution is created from a single
    # SVD since otherwise phase problems occur
    u_b = Uinitial + np.dot(B, np.dot(np.diag(uDp), \
        np.conjugate(np.transpose(B))))

    v_b = np.dot(np.dot(B, np.diag(vD)), np.transpose(A))

    return u_a, v_a, u_b, v_b
