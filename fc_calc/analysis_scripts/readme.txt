This folder contains a collection of scripts to analyse the calculated hdf5 files:

- export_raw_data_from_hdf5.py
    Parse the data from the hdf5 file and save the process parameters and raw
    data. The original hdf5 file can afterwards be deleted.

- construct_U_and_V_matrices.py
    From the raw data construct the U and V matrices of the solution and plot
    the results.

- check_U_and_V_matrices.py
    Check if the U and V matrices from a canonical solution.

- SVD_of_U_and_V.py
    From the U and V matrices calculate the process amplitudes and modes via a
    SVD (Singular value decomposition).

- check_SVD.py
    Check if the modes and amplitudes obtained from U and V form a canonical
    transformation.

- paper_plot.py
    Create some nice plots of the results, which are presented in the
    corresponding paper.

- extended_diff_analysis.py
    Check the differences between the solutions inside the medium. This is
    very useful to test the accuracy of the numerical integration routine.
    (May take a lot of time.)
