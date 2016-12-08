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
""" This script parses the hdf5 files and creates a directory for the analysis
and saves the raw data, as well as the process properties to file.

The file one wants to analyze as well as the directory where *all* the results
should be stored (must already exist) has to be supplied via the command line.
"""

import numpy as np
import tables
import pickle
import os

import lib

print "######################"
print "# Exporting raw data #"
print "######################"


#########################
# Import and parse data #
#########################

## Parse the string given via the command line
w_start, w_stop, w_steps, z_start, z_stop, z_steps, coupling \
        , pump_width, A, B, C, path_to_hdf5_file, save_directory = lib.parse_PDC_file()

## Calculate the z-ranges to access the data at the end of the crystal 
zRange = np.linspace(z_start, z_stop , z_steps)

## Calculate the step sizes
wRange = np.linspace(w_start, w_stop , w_steps)
w_step = wRange[1] - wRange[0]
z_step = zRange[1] - zRange[0]

## Open the corresponding hdf5 data file
h5file = tables.openFile(path_to_hdf5_file, mode = "r", title = "PDC-Bog-Prop")


## Load Results
# Naming convention:
# ra_ana labels the analytical f_a(w,w') solution 
# rb_ana labels the analytical f_c(w,w') solution 
# ra_num labels the numerical f_a(w,w') solution  
# rb_num labels the numerical f_c(w,w') solution 
# va_rig labels the rigorous solution of V_a(w,w') 
# vb_rig labels the rigorous solution of V_c(w,w') 
# ua_rig labels the rigorous solution of U_a(w,w') 
# ub_rig labels the rigorous solution of U_c(w,w') 

ra_ana_node = h5file.getNode('/RafirstItAna', 'zPos'+str(zRange.size-1))
ra_num_node= h5file.getNode('/RafirstItNum', 'zPos'+str(zRange.size-1))
rb_ana_node = h5file.getNode('/RbfirstItAna', 'zPos'+str(zRange.size-1))
rb_num_node= h5file.getNode('/RbfirstItNum', 'zPos'+str(zRange.size-1))

va_rig_node = h5file.getNode('/Va', 'zPos'+str(zRange.size-1))
ua_rig_node = h5file.getNode('/Ua', 'zPos'+str(zRange.size-1))
vb_rig_node = h5file.getNode('/Vb', 'zPos'+str(zRange.size-1))
ub_rig_node = h5file.getNode('/Ub', 'zPos'+str(zRange.size-1))

ra_ana = ra_ana_node.read()
ra_num = ra_num_node.read()
rb_ana = rb_ana_node.read()
rb_num = rb_num_node.read()

va_rig = va_rig_node.read()
ua_rig = ua_rig_node.read()
vb_rig = vb_rig_node.read()
ub_rig = ub_rig_node.read()

## Close the corresponding hdf5 data file
h5file.close()


########################################
# Create folder structure for analysis #
########################################

## Create directories to save analysis:
(title, sep, filename) = path_to_hdf5_file.rpartition("/")

filename = filename.rstrip(".h5")

directoryname = save_directory + "/" + filename

os.makedirs(directoryname)
# This directory stores the data exported 
# from the hdf5 file
os.mkdir(directoryname + "/raw_data/")


#############################
## Save process properties ##
#############################

## Create text file to store the process properties in a human
## readable format:
f = open(directoryname + '/PDC_properties.txt', 'w')

f.write("####################\n")
f.write("## PDC properties ##\n")
f.write("####################\n")
f.write("\n")
f.write("# Evaluation parameters\n")
f.write("w_start, w_stop, w_steps: " + str(w_start) + ", " + str(w_stop) + ", " + str(w_steps) + "\n")
f.write("z_start, z_stop, z_steps: " + str(z_start) + ", " + str(z_stop) + ", " + str(z_steps) + "\n")

f.write("# Process parameters\n")
f.write("A , B, C: " + str(A) + ", " +  str(B) +  ", " + str(C) + "\n")
f.write("coupling, pump_width: " + str(coupling) + ", " + str(pump_width) + "\n")
f.write("\n \n")


## Create a dictionary to store the process properties in a machine
## readable format via pickle
pdc_properties = {'w_start': w_start, 'w_stop': w_stop, 'w_steps': w_steps, \
        'w_step': w_step, 'z_start': z_start, 'z_stop': z_stop, \
        'z_steps': z_steps, 'z_step': z_step, \
        'A': A, 'B': B, 'C': C, 'coupling': coupling, 'pump_width': pump_width}

f = open(directoryname + '/PDC_properties.pkl', 'wb')
pickle.dump(pdc_properties, f)
f.close()


#########################
# Save raw data to file #
#########################

# The view option enables us to save the complex numbers into a single file
# (http://stackoverflow.com/questions/6494102/\
#        how-to-save-and-load-an-array-of-complex-numbers-using-numpy-savetxt)
#
# Attention: This trick will fail when you try to save a transposed matrices
# via this method since transpose itself is also only a view. In this case the
# imaginary values are saved in the rows instead of the columns. This is a
# problem during the import since the import doesn't take this into account.
# One can however solve this by explicitly copying the transposed array in a
# new one which fixes the issue (this is performed in these scripts).
# Alternatively one would have to rewrite all save and load functions.

np.savetxt(directoryname + "/raw_data/ra_ana.txt.gz", ra_ana.view(float))
np.savetxt(directoryname + "/raw_data/ra_num.txt.gz", ra_num.view(float))
np.savetxt(directoryname + "/raw_data/rb_ana.txt.gz", rb_ana.view(float))
np.savetxt(directoryname + "/raw_data/rb_num.txt.gz", rb_num.view(float))

np.savetxt(directoryname + "/raw_data/va_rig.txt.gz", va_rig.view(float))
np.savetxt(directoryname + "/raw_data/ua_rig.txt.gz", ua_rig.view(float))
np.savetxt(directoryname + "/raw_data/vb_rig.txt.gz", vb_rig.view(float))
np.savetxt(directoryname + "/raw_data/ub_rig.txt.gz", ub_rig.view(float))
