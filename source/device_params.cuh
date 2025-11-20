// =========================
// device_params.cuh
// =========================
#pragma once
#include <cuda_runtime.h>

// ---------------- Device-visible simulation configuration ----------------
struct DeviceParams {

    // Lattice size and spacing
    int xlen;
    int ylen;
    double xsize;
    double ysize;
    double lsx;
    double lsy;
    int halo;
    int threads_per_block;

    // Field and grid sizes
    int number_magnetization_fields;
    int number_coordinates;
    int number_total_fields;
    int dim_grid;
    int dim_fields;
    double grid_volume;

    // Visualization mode
    int display_mode;

    // Initial configuration
    bool ansatz_bloch;
    bool ansatz_neel;
    bool ansatz_heusler;
    int soliton_id;
    double skyrmion_rotation;
    int vortex_type;

    // Field theory parameters
    double q;
    double alpha;
    double beta;
    double gamma;
    double skyrmion_number;
    double ha;
    double hb1;
    double hb2;
    double hc;
    double M0;
    double u1;
    double u2;
    double vortex1_number;
    double vortex2_number;
    double vortex_number;
    double ainf;

    // Arrested Newton flow parameters
    double time_step;
};

// One copy in constant memory, readable by all kernels
extern __device__ __constant__ DeviceParams dparams;

cudaError_t setDeviceParams(const DeviceParams& h);