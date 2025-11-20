// =========================
// params.hpp
// =========================
#pragma once
#include <cuda_runtime.h>
#include "device_params.cuh"
#include <math.h>

// ---------------- Parameters ----------------
struct Params {

    // Optimizer parameters
    double tolerance{1.0E-4};
    double courant{0.5};
    int loop_output{2000}; 
    int loop_display{100};
    int loops_max{100000};
    
    // Display options
    int iters_per_render{3};
    bool newtonflow{false};
    bool killkinen{true};
    bool output_results{false};
    int display_mode{0};
    bool real_time_visualization{true};

    // System size
    int xlen{512};
    int ylen{512};
    int wx{512};
    int wy{512};
    double xsize{80.0};
    double ysize{80.0};
    double lsx{xsize / (xlen - 1)};
    double lsy{ysize / (ylen - 1)};
    int halo{2};
    
    // Initial magnetization ansatz
    bool ansatz_bloch{true}; 
    bool ansatz_neel{false};
    bool ansatz_heusler{false}; 
    int soliton_id{2};
    double skyrmion_rotation{0.0};
    int vortex_type{0};
    double skyrmion_number{1.0};
    double vortex1_number{2.0};
    double vortex2_number{1.0};

    // Field theory parameters
    double q{1.0};
    double alpha{-1.0}; // -1.0 No SF, -1.5 SF (weak), -5.0 SF (strong) 
    double beta{1.0}; // 1.0 no SF, 1.0 SF
    double gamma{1.0};
    
    double ha{(-19.0 / 4.0)};
    double hb1{(-1.0 * ha)}; 
    double hb2{1.0};
    double hc{(-0.5)}; 

    // Ground state parameters
    double M0{sqrt(-1.0 * alpha / beta)}; 
    double u1{sqrt(-1.0 * (ha + 2.0 * hc) / (hb1 + 2.0 * hb2))};
    double u2{sqrt(-1.0 * (ha + 2.0 * hc) / (hb1 + 2.0 * hb2))};
    double vortex_number{(pow(u1, 2) * vortex1_number + pow(u2, 2) * vortex2_number) / (pow(u1, 2) + pow(u2, 2))};
    double ainf{(vortex_number / q)};

    // Length scales
    double coherence_length{(1.0 / sqrt(-2.0 * ha))};
    double magnetic_penetration_depth{(1.0 / (q * u1))};
    double coherence_magnetization{(1.0 / sqrt(q * q * u1 * u1 - 1.0))};

    // Fixed/initial optimizer parameters
    double time_step{courant * lsx};
    double error{tolerance + 1.0};
    int epochs{0};

    // Dimensions of the field theory
    int number_magnetization_fields{3};
    int number_coordinates{2};
    int number_total_fields{10};

    // Launch configuration (computed from width/height)
    unsigned int threads_per_block_x{32};
    unsigned int threads_per_block_y{16};
    unsigned int threads_per_block{1024};
    dim3 block{threads_per_block_x, threads_per_block_y, 1};
    dim3 grid{};               // computed in finalize()
    dim3 reduction_block{threads_per_block,1,1};
    dim3 reduction_grid{};

    // Byte sizes derived from dims
    size_t n_grid_bytes{};
    size_t n_vector_bytes{};
    size_t n_fields_bytes{};
    size_t n_coords_bytes{};
    size_t n_grid_sum_bytes{};

    // Physical/derived scalars
    double energy{}, skyrmion{}, vortex{};

    // Constants of the field layout (kept fixed here; make runtime if needed)
    int dim_grid{xlen * ylen};
    int dim_fields{number_total_fields * dim_grid};
    double grid_volume{lsx * lsy};

    void finalize() {
        grid = dim3((xlen  + block.x - 1) / block.x, (ylen + block.y - 1) / block.y, 1);
        reduction_grid = dim3((dim_grid + reduction_block.x - 1) / reduction_block.x, 1, 1);
        n_grid_bytes    = dim_grid * sizeof(double);
        n_vector_bytes  = dim_grid * number_magnetization_fields * sizeof(double);
        n_fields_bytes  = dim_fields * sizeof(double);
        n_coords_bytes  = dim_grid * number_coordinates * sizeof(double);
        n_grid_sum_bytes = size_t(reduction_grid.x) * sizeof(double);
    }

    // Converts host parameters to device parameters
    DeviceParams toDeviceParams(int display_mode=1) const {
        DeviceParams dp{};
        dp.threads_per_block = threads_per_block;
        dp.xlen  = xlen;
        dp.ylen = ylen;
        dp.lsx = lsx;
        dp.lsy = lsy;
        dp.xsize = xsize;
        dp.ysize = ysize;
        dp.halo = halo;
        dp.number_coordinates = number_coordinates;
        dp.number_total_fields = number_total_fields;
        dp.number_magnetization_fields = number_magnetization_fields;
        dp.dim_grid = dim_grid;
        dp.dim_fields = dim_fields;
        dp.grid_volume = grid_volume;
        dp.display_mode = display_mode;
        dp.time_step = time_step;
        dp.ansatz_bloch = ansatz_bloch;
        dp.ansatz_neel = ansatz_neel;
        dp.ansatz_heusler = ansatz_heusler;
        dp.soliton_id = soliton_id;
        dp.skyrmion_rotation = skyrmion_rotation;
        dp.vortex_type = vortex_type;
        dp.q = q;
        dp.alpha = alpha;
        dp.beta = beta;
        dp.gamma = gamma;
        dp.skyrmion_number = skyrmion_number;
        dp.ha = ha;
        dp.hb1 = hb1;
        dp.hb2 = hb2;
        dp.hc = hc;
        dp.M0 = M0;
        dp.u1 = u1;
        dp.u2 = u2;
        dp.vortex1_number = vortex1_number;
        dp.vortex2_number = vortex2_number;
        dp.vortex_number = vortex_number;
        dp.ainf = ainf;
        return dp;
    }
};