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
""" This scripts creates the plots for the corresponding paper.  It generates
several plots which compare the analytical and the rigorous solution.

The directory in which the analysis resides has to be supplied via command
line. A new directory paper_plot is created next to it.
"""

import sys, os
import numpy as np
import pickle
import matplotlib.pyplot as plt

print "######################"
print "# Create Paper Plots #"
print "######################"


# Path the analysis directory
path = sys.argv[1]

# Load process properties via pickle
pkl_file = open(path + 'FC_properties.pkl', 'rb')
fc_properties = pickle.load(pkl_file)
pkl_file.close()

w_start = fc_properties['w_start']
w_stop = fc_properties['w_stop']

# Calculate necessary arrays:
wRange = np.linspace(fc_properties['w_start'], fc_properties['w_stop'], \
        fc_properties['w_steps'])


###############
# Import Data #
###############

# Import the Va matrices
va_ana = np.loadtxt(path + 'U_and_V_matrices/data/ana/va_ana.txt.gz').view(complex)
va_rig = np.loadtxt(path + 'U_and_V_matrices/data/rig/va_rig.txt.gz').view(complex)

# Import the Schmidt modes and values
vaA_rig  = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/vaA_rig.txt.gz').view(complex)
vaD_rig  = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/vaD_rig.txt.gz')
vaBh_rig = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/rig/vaBh_rig.txt.gz').view(complex)

vaA_ana  = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/vaA_ana.txt.gz').view(complex)
vaD_ana  = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/vaD_ana.txt.gz')
vaBh_ana = np.loadtxt(path + 'SVD_modes_and_amplitudes/data/ana/vaBh_ana.txt.gz').view(complex)


#################
# Plot the data #
#################
os.mkdir(path + "/paper_plot")


## Contour Plot of Va for the paper
params = {
'text.fontsize': 24,
'legend.fontsize': 20,
'xtick.labelsize': 16,
'ytick.labelsize': 16}
plt.rcParams.update(params)

# Cut va_ana and va_rig to zoom into the plot to show the interesting regions
# Here it is assumed that w_start and w_stop are symmetric about 0.
zoom_faktor = 1.8

va_ana_steps = va_ana.shape[0]
va_ana_center = va_ana_steps / 2
va_ana_zoom_steps = va_ana_steps / float(zoom_faktor)

va_ana_zoom = va_ana[va_ana_center - va_ana_zoom_steps/2: va_ana_center + va_ana_zoom_steps /2, \
    va_ana_center - va_ana_zoom_steps/2: va_ana_center + va_ana_zoom_steps /2]


va_rig_steps = va_rig.shape[0]
va_rig_center = va_rig_steps / 2
va_rig_zoom_steps = va_rig_steps / float(zoom_faktor)

va_rig_zoom = va_rig[va_rig_center - va_rig_zoom_steps/2: va_rig_center + va_rig_zoom_steps /2, \
    va_rig_center - va_rig_zoom_steps/2: va_rig_center + va_rig_zoom_steps /2]


w_range = w_stop - w_start
w_range_zoom = (va_ana_zoom_steps / va_ana_steps) * w_range
w_start_zoom = w_range_zoom / 2
w_stop_zoom = - w_range_zoom / 2


plt.figure(figsize=(5,10))

plt.subplot(2,1,1)
im = plt.imshow(np.abs(va_ana_zoom), origin="lower", cmap=plt.cm.hot, \
        extent=(w_start_zoom, w_stop_zoom, w_start_zoom, w_stop_zoom) )
t = plt.title(r"$|V_a(\nu, \nu')|$", fontsize=25)
t.set_y(1.05)
plt.xlabel(r"$\nu \, (a.u.)$", fontsize=25, labelpad = 15)
plt.ylabel(r"$\nu' \, (a.u.)$", fontsize=25, labelpad = -5)
plt.colorbar(im, use_gridspec=True)


plt.subplot(2,1,2)
im = plt.imshow(np.abs(va_rig_zoom), origin="lower", cmap=plt.cm.hot, \
        extent=(w_start_zoom, w_stop_zoom, w_start_zoom, w_stop_zoom) )
t = plt.title(r"$|V_a(\nu, \nu')|$", fontsize=25)
t.set_y(1.05)
plt.xlabel(r"$\nu \, (a.u.)$", fontsize=25, labelpad = 15)
plt.ylabel(r"$\nu' \, (a.u.)$", fontsize=25, labelpad = -5)
plt.colorbar(im, use_gridspec=True)

plt.tight_layout()

plt.savefig(path + "paper_plot/va_abs.png")
plt.close()




## Compare conversion *amplitudes* between rigorous and numerical solution
params = {
'text.fontsize': 25,
'legend.fontsize': 25,
'xtick.labelsize': 27,
'ytick.labelsize': 27
}
plt.rcParams.update(params)

plt.figure(figsize=(8,6))
N = 10
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

plt.subplot(111)
rects1 = plt.bar(ind, vaD_ana[:N], width, color='r')
rects2 = plt.bar(ind+width, vaD_rig[:N], width, color='b')

plt.xlabel('$k$', fontsize=37)
plt.ylabel('$\sin(r_k)$', fontsize=37, labelpad=10)
plt.axis([0, 10, 0, 1.00])
plt.title('FC amplitudes', fontsize=36)

# This line is for highly correlated frequency conversion only:
#plt.legend( (rects1[0], rects2[0]), ('Analytic model', 'Rigorous model'), \
#        shadow=True, loc ="lower right")
# This line is for uncorrelated frequency conversion only:
plt.legend( (rects1[0], rects2[0]), ('Analytic model', 'Rigorous model'), \
       shadow=True)

plt.tight_layout()

plt.savefig(path + "paper_plot/conversion_amplitudes_ana_rig.png")
plt.close()



## Compare conversion *efficiencies* between rigorous and analytical solution
plt.figure(figsize=(8,6))
N = 10
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

plt.subplot(111)
rects1 = plt.bar(ind, vaD_ana[:N]**2, width, color='r')
rects2 = plt.bar(ind+width, vaD_rig[:N]**2, width, color='b')

plt.xlabel('$k$', fontsize=37)
plt.ylabel('$\sin^2(r_k)$', fontsize=37, labelpad=10)
plt.axis([0, 10, 0, 1.00])
plt.title('FC efficiency', fontsize=36)

# This line is for highly correlated frequency conversion only.
#plt.legend( (rects1[0], rects2[0]), ('Analytic model', 'Rigorous model'), \
#        shadow=True, loc ="lower right")
# This line is for uncorrelated frequency conversion only:
plt.legend( (rects1[0], rects2[0]), ('Analytic model', 'Rigorous model'), \
        shadow=True)

plt.tight_layout()

plt.savefig(path + "paper_plot/conversion_efficiencies_ana_rig.png")
plt.close()


# Save text file with conversion amplitudes and efficiencies

f = open(path + "paper_plot/properties.txt", 'w')
f.write("###################\n")
f.write("## FC properties ##\n")
f.write("###################\n")
f.write("\n")
f.write("# Conversion amplitudes of the first ten modes (ana):\n")
f.write(str(vaD_ana[:10]))
f.write("\n")
f.write("# Conversion efficiencies of the first ten modes (ana):\n")
f.write(str(vaD_ana[:10]**2))
f.write("\n\n")
f.write("# Conversion amplitudes of the first ten modes (rig):\n")
f.write(str(vaD_rig[:10]))
f.write("\n")
f.write("# Conversion efficiencies of the first ten modes (rig):\n")
f.write(str(vaD_rig[:10]**2))
f.close()


# Compare the first Schmidt Modes of the A and C beam
params = {
'text.fontsize': 20,
'legend.fontsize': 16,
'xtick.labelsize': 20,
'ytick.labelsize': 20}
plt.rcParams.update(params)


fig = plt.figure(figsize=(13,5))

plt.subplot(1,2,1)
plt.plot(wRange, np.abs(vaA_ana[0]), 'r', linewidth = 6)
plt.plot(wRange, np.abs(vaA_rig[0]), linewidth = 6, linestyle="dashed")
plt.xlabel(r'$\nu \, (a.u.)$', fontsize=25, labelpad=15)
plt.ylabel('Amplitude', fontsize=20, labelpad=5)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8])
t = plt.title(r'$|\varphi_1(\nu)| / |\psi_1(\nu)|$', fontsize=25)
t = t.set_y(1.03)
plt.xlim([w_start_zoom, w_stop_zoom])


plt.subplot(1,2,2)
plt.plot(wRange, np.abs(vaBh_ana[0]), 'r', linewidth = 6)
plt.plot(wRange, np.abs(vaBh_rig[0]), linewidth = 6, linestyle="dashed")
plt.xlabel(r'$\nu \, (a.u.)$', fontsize=25, labelpad=15)
plt.ylabel('Amplitude', fontsize=20, labelpad=5)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8])
t = plt.title(r'$|\phi_1(\nu)| / |\xi_1(\nu)|$', fontsize=25)
t.set_y(1.03)
plt.xlim([w_start_zoom, w_stop_zoom])

plt.tight_layout()

plt.savefig(path + "paper_plot/compare_schmidt_modes_ana_rig.png")
plt.close()
