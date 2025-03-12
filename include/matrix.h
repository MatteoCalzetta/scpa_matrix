#ifndef MATRIX_LIST_H
#define MATRIX_LIST_H

static const char *matrix_filenames[] = {
    "adder_dcop_32.mtx",
    "bcsstk17.mtx",
    "cage4.mtx",
    "cant.mtx",
    "cavity10_b.mtx",
    "cavity10_x.mtx",
    "cavity10.mtx",
    "Cube_Coup_dt0.mtx",
    "mac_econ_fwd500.mtx",
    "mcfe.mtx",
    "mhda416.mtx",
    "ML_Laplace.mtx",
    "olafu_b.mtx",
    "olafu.mtx",
    "olm1000.mtx",
    "rdist2.mtx",
    "west2021.mtx"
    //"test_matrix.mtx"
};

static const int num_matrices = sizeof(matrix_filenames) / sizeof(matrix_filenames[0]);

#endif