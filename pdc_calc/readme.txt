############
# pdc calc #
############

This folder contains the complete code to calculate and analyse parametric
down-conversion processes as detailed in the corresponding paper. It
especially enables the recalculation of the presented data.

# Calculation
- calc_PDC.py
This script contains the function 'calc_PDC' which evaluates a  parametric
down-conversion process using the analytic and the rigorous model as given in the corresponding paper and saves the obtained results in a hdf5 file.

- calc_correlated_PDC.py
This script calculates a generic correlated parametric down-conversion
process (the one from the paper)and puts the obtained data into a specified
directory.

- calc_uncorrelated_PDC.py
This script contains the parameters to calculate a generic uncorrelated
parametric down-conversion process (the one from the paper) and puts the
obtained data into a specific directory.


# Analysis
- analyse_PDC.py
This script analyses a given calculated paramteric down-conversion process
using the scripts in the folder analysis scripts.

- analyse_all_PDC.py
This bash script which analyses all data files in a specified directory

- evaluate_data_PDC.py
This script copies all files from a directory containing various analysed PDC states into a new folder structure where the individual files are grouped together. This is very useful to examine how changes in the power or grid sizes affect the solution.


# Note:
The code for PDC and FC only differs in small details, like specific signs in equations.
