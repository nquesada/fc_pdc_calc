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


#!/bin/bash

# This script automates the analysis of the hdf5 files. One only has to specify
# the input directory of the files to be evaluated and the output directory
# where the results should be stored.


input_file_directory=$1
output_file_directory=$2


for input_file in $1*; do
    echo "####################################"
    echo "####################################"
    echo "Processing: $input_file"
    echo "####################################"
    echo "####################################"
    python analyse_FC.py $input_file $output_file_directory
    echo "####################################"

done
