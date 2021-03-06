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
""" This script analysis the differences between the different solutions at
different z-positions *inside* the waveguide. 

The directory in which the hdf5 file resides has to be supplied via command
line. A new directory diff_analysis storing the results is created in the
analysis directory.

It checks the differences between the analytical solution (ana) and the
first-order numerical solution(num). Since both should be identical this is a
very good test of the integration routine. Furthermore the rig solution (rig)
and the analytic solution (ana) are compared. For very low conversion
efficiencies they should be identical which enables a quick verification of the
algorithm calculating the rigorous solution. """

import numpy as np
import tables
import sys, os
import pickle
import matplotlib.pyplot as plt

import lib


print "##########################"
print "# Extended diff analysis #"
print "##########################"


# Path to the analysis directory
parameters = sys.argv[1:]

path_to_hdf5_file = parameters[0]
path = parameters[1]


def diff_Mat(matA, matB):
    diff = np.linalg.norm(matA - matB)

    norm_matA = np.linalg.norm(matA) 
    norm_matB = np.linalg.norm(matB) 

    relDiff = diff / (0.5*(norm_matA + norm_matB))
    return relDiff


# Load process properties via pickle
pkl_file = open(path + 'FC_properties.pkl', 'rb')
fc_properties = pickle.load(pkl_file)
pkl_file.close()


## Parse the string given via the command line
w_start, w_stop, w_steps, z_start, z_stop, z_steps, coupling \
        , pump_width, A, B, C, path_to_hdf5_file, save_directory = lib.parse_FC_file()

## Calculate the z-ranges to access the data at the end of the crystal 
zRange = np.linspace(z_start, z_stop , z_steps)

## Calculate the step sizes
wRange = np.linspace(w_start, w_stop , w_steps)
w_step = wRange[1] - wRange[0]
z_step = zRange[1] - zRange[0]

## Calculate Uinitial which also serves as the discretized delta function
## in the calculations
Uinitial = np.identity(wRange.size, dtype=complex) / fc_properties['w_step']


## Open the corresponding hdf5 data file
h5file = tables.openFile(path_to_hdf5_file, mode = "r", title = "FC-Bog-Prop")


# Calculate and plot the differences between the ana and num solutions 
# at different z-Positions:
diff_ra = []
diff_rc = []
for zPos in np.arange(zRange.size):
    ra_ana_node = h5file.getNode('/RafirstItAna', 'zPos'+str(zPos))
    ra_num_node = h5file.getNode('/RafirstItNum', 'zPos'+str(zPos))
    rc_ana_node = h5file.getNode('/RcfirstItAna', 'zPos'+str(zPos))
    rc_num_node = h5file.getNode('/RcfirstItNum', 'zPos'+str(zPos))

    ra_ana_zPos = ra_ana_node.read()
    ra_num_zPos = ra_num_node.read()
    rc_ana_zPos = rc_ana_node.read()
    rc_num_zPos = rc_num_node.read()

    diff_ra.append(diff_Mat(ra_ana_zPos, ra_num_zPos))
    diff_rc.append(diff_Mat(rc_ana_zPos, rc_num_zPos))



# Calculate and plot the differences between the ana and rig solutions 
# at different z-Positions:
# (This may take quite some time)
diff_ua = []
diff_uc = []
diff_va = []
diff_vc = []
for zPos in np.arange(zRange.size):
    ra_ana_node = h5file.getNode('/RafirstItAna', 'zPos'+str(zPos))
    rc_ana_node = h5file.getNode('/RcfirstItAna', 'zPos'+str(zPos))
    ua_rig_node = h5file.getNode('/Ua', 'zPos'+str(zPos))
    va_rig_node = h5file.getNode('/Va', 'zPos'+str(zPos))
    uc_rig_node = h5file.getNode('/Uc', 'zPos'+str(zPos))
    vc_rig_node = h5file.getNode('/Vc', 'zPos'+str(zPos))

    ra_ana_zPos = ra_ana_node.read()
    rc_ana_zPos = rc_ana_node.read()
    ua_rig_zPos = ua_rig_node.read()
    va_rig_zPos = va_rig_node.read()
    uc_rig_zPos = uc_rig_node.read()
    vc_rig_zPos = vc_rig_node.read()

    ua_ana_zPos, va_ana_zPos = lib.construct_u_and_v(ra_ana_zPos, w_step, wRange)
    uc_ana_zPos, vc_ana_zPos = lib.construct_u_and_v(rc_ana_zPos, w_step, wRange)

    diff_ua.append(diff_Mat(ua_ana_zPos - Uinitial, ua_rig_zPos - Uinitial))
    diff_uc.append(diff_Mat(uc_ana_zPos - Uinitial, uc_rig_zPos - Uinitial))

    diff_va.append(diff_Mat(va_ana_zPos, va_rig_zPos))
    diff_vc.append(diff_Mat(vc_ana_zPos, vc_rig_zPos))



# Plot the data
os.mkdir(path + "/extended_diff_analysis/")

plt.figure()

plt.subplot(1,1,1)
plt.plot(zRange, diff_ra, 'r', linewidth = 4)
plt.plot(zRange, diff_rc, 'b', linewidth = 4,  linestyle="dashed")
plt.xlabel('zPos', fontsize=20, labelpad=15)
plt.ylabel('Diff', fontsize=20)
plt.title("Diff ana num", fontsize=20)
plt.legend( ('A beam', 'C beam'), loc='upper right', shadow=True)

plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None)
plt.savefig(path + 'extended_diff_analysis//ana_num_diff.png')
plt.close()


plt.figure()

plt.subplot(1,1,1)
plt.plot(diff_ra[:10], 'r', linewidth = 4)
plt.plot(diff_rc[:10], 'b', linewidth = 4,  linestyle="dashed")
plt.xlabel('zPos', fontsize=20, labelpad=15)
plt.ylabel('Diff', fontsize=20)
plt.title("Diff ana num", fontsize=20)
plt.legend( ('A beam', 'C beam'), loc='upper right', shadow=True)

plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None)
plt.savefig(path + 'extended_diff_analysis/ana_num_diff_zoom.png')
plt.close()




# Calculate and plot the differences between the ana and rig solutions at
# different z-Positions:
# (This may take quite some time)
diff_ua = []
diff_uc = []
diff_va = []
diff_vc = []
for zPos in np.arange(zRange.size):
    ra_ana_node = h5file.getNode('/RafirstItAna', 'zPos'+str(zPos))
    rc_ana_node = h5file.getNode('/RcfirstItAna', 'zPos'+str(zPos))
    ua_rig_node = h5file.getNode('/Ua', 'zPos'+str(zPos))
    va_rig_node = h5file.getNode('/Va', 'zPos'+str(zPos))
    uc_rig_node = h5file.getNode('/Uc', 'zPos'+str(zPos))
    vc_rig_node = h5file.getNode('/Vc', 'zPos'+str(zPos))

    ra_ana_zPos = ra_ana_node.read()
    rc_ana_zPos = rc_ana_node.read()
    ua_rig_zPos = ua_rig_node.read()
    va_rig_zPos = va_rig_node.read()
    uc_rig_zPos = uc_rig_node.read()
    vc_rig_zPos = vc_rig_node.read()

    ua_ana_zPos, va_ana_zPos = lib.construct_u_and_v(ra_ana_zPos, w_step, wRange)
    uc_ana_zPos, vc_ana_zPos = lib.construct_u_and_v(rc_ana_zPos, w_step, wRange)

    diff_ua.append(diff_Mat(ua_ana_zPos - Uinitial, ua_rig_zPos - Uinitial))
    diff_uc.append(diff_Mat(uc_ana_zPos - Uinitial, uc_rig_zPos - Uinitial))

    diff_va.append(diff_Mat(va_ana_zPos, va_rig_zPos))
    diff_vc.append(diff_Mat(vc_ana_zPos, vc_rig_zPos))


# Plot the data
plt.figure(figsize=(15,7))

plt.subplot(1,2,1)
plt.plot(zRange, diff_ua, 'r', linewidth = 4)
plt.plot(zRange, diff_uc, 'b', linewidth = 4,  linestyle="dashed")
plt.xlabel('zPos', fontsize=20, labelpad=15)
plt.ylabel('Diff', fontsize=20)
plt.title("Diff U ana rig", fontsize=20)
plt.legend( ('A beam', 'C beam'), loc='upper right', shadow=True)
plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None)

plt.subplot(1,2,2)
plt.plot(zRange, diff_va, 'r', linewidth = 4)
plt.plot(zRange, diff_vc, 'b', linewidth = 4,  linestyle="dashed")
plt.xlabel('zPos', fontsize=20, labelpad=15)
plt.ylabel('Diff', fontsize=20)
plt.title("Diff V ana rig", fontsize=20)
plt.legend( ('A beam', 'C beam'), loc='upper right', shadow=True)
plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None)

plt.savefig(path + 'extended_diff_analysis/ana_rig_diff.png')
plt.close()


plt.figure(figsize=(15,7))

plt.subplot(1,2,1)
plt.plot(diff_ua[:10], 'r', linewidth = 4)
plt.plot(diff_uc[:10], 'b', linewidth = 4,  linestyle="dashed")
plt.xlabel('zPos', fontsize=20, labelpad=15)
plt.ylabel('Diff', fontsize=20)
plt.title("Diff U ana rig", fontsize=20)
plt.legend( ('A beam', 'C beam'), loc='upper right', shadow=True)
plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None)

plt.subplot(1,2,2)
plt.plot(diff_va[:10], 'r', linewidth = 4)
plt.plot(diff_vc[:10], 'b', linewidth = 4,  linestyle="dashed")
plt.xlabel('zPos', fontsize=20, labelpad=15)
plt.ylabel('Diff', fontsize=20)
plt.title("Diff V ana rig", fontsize=20)
plt.legend( ('A beam', 'C beam'), loc='upper right', shadow=True)
plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None)

plt.savefig(path + 'extended_diff_analysis/ana_rig_diff_zoom.png')
plt.close()
