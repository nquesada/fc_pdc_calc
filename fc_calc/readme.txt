###########
# fc calc #
###########

This folder contains the code to calculate and analyse frequency conversion
processes including time-ordering effects as detailed in the corresponding
paper.

## Overview:
The whole calculation is split into two steps:

Step one calcultes the raw data for a given frequency conversion process,
specified by its 6 defining parameters. Example files are
"calc_correlated_FC.py" and "calc_uncorrelated_FC.py" which reproduce the cases
presented in the paper and can be easily adapted to any frequency conversion
process. The end results is a .hdf5 file which, depending on the grid size, can
get quite big.

In step two the solution is analysed using the "analyse_FC.py" script. From the
raw .hdf5 file it exports all important data, performs a variety of different
tests and visualizations on the end result. Everything is stored in a specified
folder.

Furthermore the scripts "analyse_all_FC.py" and "evaluate_data_FC.py" enable
the analysis of many frequency conversion processes simultaneously.


## The individual scripts:

# Calculation scripts:
- calc_correlated_FC.py
This script calculates the generic correlated frequency conversion process, as
presented in the paper, and puts the obtained data into a specified directory.

- calc_uncorrelated_FC.py
This script calculates the generic uncorrelated frequency conversion process,
as presented in the paper, and puts the obtained data into a specified
directory.

- calc_FC.py
This file contains the function 'calc_FC' which is used by the
"calc_correlated_FC.py" and "calc_uncorrelated_FC.py" scripts to evaluate the
frequency conversion process using the analytic and the rigorous model as given
in the corresponding paper.

# Analysis scripts:
- analyse_FC.py
This script analyses a given calculated frequency conversion process using the
scripts in the folder analysis_scripts. Its modular nature makes it very easy
to add additional scripts to perform further analysis.

- analyse_all_FC.py
This bash script automatically analyses all data files in a specified
directory.

- evaluate_data_FC.py
This script copies all files from a directory containing various analysed FC
states into a new folder structure where the individual files are grouped
together. This is very useful to examine how changes in the power or grid sizes
affect the solution.
