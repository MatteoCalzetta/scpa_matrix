#ifndef MATRIX_LIST_H
#define MATRIX_LIST_H



static const char *matrix_filenames[] = {
    "adder_dcop_32.mtx",
    "af_1_k101.mtx",
    "af23560.mtx",
    "amazon0302.mtx",
    "bcsstk17.mtx",
    "cage4.mtx",
    "cant.mtx",
    "cavity10.mtx",
    "cop20k_A.mtx",
    "Cube_Coup_dt0.mtx",
    "dc1.mtx",
    "FEM_3D_thermal1.mtx",
    "lung2.mtx",
    "mac_econ_fwd500.mtx",
    "mcfe.mtx",
    "mhd4800a.mtx",
    "mhda416.mtx",
    "ML_Laplace.mtx",
    "nlpkkt80.mtx",
    "olafu.mtx",
    "olm1000.mtx",
    "PR02R.mtx",
    "raefsky2.mtx",
    "rdist2.mtx",
    "roadNet-PA.mtx",
    "thermal1.mtx",
    "thermal2.mtx",
    "thermomech_TK.mtx",
    "webbase-1M.mtx",
    "west2021.mtx"
    //"test_matrix.mtx"
};

static const int num_matrices = sizeof(matrix_filenames) / sizeof(matrix_filenames[0]);

struct matrixPerformance {
    char nameMatrix[50];
    double seconds;
    double flops;
    double gigaFlops;
    double relativeError;
};

#endif
