// =========================
// kernels.hpp
// =========================
#pragma once
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <math.h>
#include <thrust/complex.h>
#include "device_params.cuh"
#include "params.hpp"
#include "cuda_gl_interop.h"
#include "cuda_runtime.h"

// ---------------- Define pi ----------------
#define M_PI 3.14159265358979323846

// ---------------- CUDA error check ----------------
#define CHECK(call){const cudaError_t error = call; if(error!=cudaSuccess){printf("CUDAERROR :( code:%d, reason:%s\n",error,cudaGetErrorString(error));exit(1);}}

// ---------------- Dimensions ----------------
#define no_fields 10
#define s_threads_per_block 1024

// ---------------- Clear/arrest variable ----------------
void clear_variable_wrapper(double* d_var, const Params& p, size_t size, bool sync);
void arrest_velocity_wrapper(double* d_Field, const Params& p, bool sync);

// ---------------- Initial configuration ----------------
void create_grid_wrapper(double* d_grid, const Params& p);
void create_skyrmion_wrapper(int pxi, int pxj, double rotation_angle, double* d_Field, double* d_grid, const Params& p);
void create_vortex_wrapper(int pxi, int pxj, int vortex, double* d_Field, double* d_grid, const Params& p);
void create_initial_configuration_wrapper(double* d_Velocity, double* d_Field, double* d_grid, double* d_k1, double* d_k2, double* d_k3, double* d_k4, double* d_l1, double* d_l2, double* d_l3, double* d_l4, double* d_Temp, const Params& p);

// ---------------- Compute observables ----------------
double compute_sum_wrapper(double* d_var, double* d_gridsum, double* h_gridsum, const Params& p, size_t size);
double compute_energy_wrapper(double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p);
double compute_skyrmion_number_wrapper(double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p);
double compute_vortex_number_wrapper(int a, double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p);
void compute_magnetic_field_wrapper(double* d_Field, double* d1fd1x, double* d_MagneticFluxDensity, const Params& p);
void compute_supercurrent_wrapper(double* d_Field, double* d2fd2x, double* d_Supercurrent, const Params& p);

// ---------------- Arrested Newton flow ----------------
void do_gradient_step_wrapper(double* d_Velocity, double* d_Field, double* d1fd1x, double* d2fd2x, double* d_EnergyGradient, const Params& p);
void do_rk4_kstep_wrapper(double* d_k, double* d_Velocity, double factor, double* d_l, const Params& p);
void do_rk4_lstep_wrapper(double* d_Temp, double* d_Field, double factor, double* d_k, const Params& p);
void do_rk4_wrapper(double* d_Velocity, double* d_Field, double* d_k1, double* d_k2, double* d_k3, double* d_k4, double* d_l1, double* d_l2, double* d_l3, double* d_l4, double* d_grid, const Params& p);
double do_arrested_newton_flow(double* d_Velocity, double* d_Field, double* d1fd1x, double* d2fd2x, double* d_EnergyGradient, double* d_k1, double* d_k2, double* d_k3, double* d_k4, double* d_l1, double* d_l2, double* d_l3, double* d_l4, double* d_grid, double* d_Temp, double* d_en, double* d_entmp, double* d_gridsum, double* h_gridsum, double* d_maxeg, double* h_maxeg, Params& p);

// ---------------- Compute min/max ----------------
double compute_max_density(double* d_en, double* d_entmp, double* d_gridmax, double* h_gridmax, const Params& p);
double compute_min_density(double* d_en, double* d_entmp, double* d_gridmax, double* h_gridmax, const Params& p);
double compute_max_field(double* d_EnergyGradient, double* d_maxeg, double* h_maxeg, const Params& p);

// ---------------- Visualization ----------------
void show_density_wrapper(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p);
void show_magnetic_flux_wrapper(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p);
void show_higgs1_wrapper(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d_gridsum, double* h_gridsum, const Params& p);
void show_higgs2_wrapper(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d_gridsum, double* h_gridsum, const Params& p);
void show_magnetization_wrapper(uchar4* d_out, double* d_Field, const Params& p);

// ---------------- Compute normalization ----------------
double compute_norm_higgs1_wrapper(double* d_en, double* d_entmp, double* d_Field, double* d_gridsum, double* h_gridsum, const Params& p);
double compute_norm_higgs2_wrapper(double* d_en, double* d_entmp, double* d_Field, double* d_gridsum, double* h_gridsum, const Params& p);
void compute_norm_magnetization_wrapper(double* d_Field, const Params& p);

// ---------------- Output data ----------------
void Output_Data(double* h_Field, double* h_EnergyDensity_Higgs, double* h_MagneticFluxDensity, double* h_BaryonDensity, double* h_Supercurrent, double* h_grid, double* Lattice_Points, double* Lattice_Vectors, const Params& p);

// ---------------- Namespaces ----------------
namespace kernels {
    inline void uploadDeviceParams(const Params& p, int display_mode=1) {
        DeviceParams dp = p.toDeviceParams(display_mode);
        cudaError_t err = setDeviceParams(dp);
    }
    inline void set_field_zero(double* d_Field, const Params& p, bool sync=true) { arrest_velocity_wrapper(d_Field, p, sync); }
    inline void clear(double* d_var, const Params& p, size_t size, bool sync=true) { clear_variable_wrapper(d_var, p, size, sync); }
}

namespace initial_configuration {
    inline void create_grid(double* d_grid, const Params& p) { create_grid_wrapper(d_grid, p); }
    inline void create_initial_configuration(double* d_Velocity, double* d_Field, double* d_grid, double* d_k1, double* d_k2, double* d_k3, double* d_k4, double* d_l1, double* d_l2, double* d_l3, double* d_l4, double* d_Temp, const Params& p){
        create_initial_configuration_wrapper(d_Velocity, d_Field, d_grid, d_k1, d_k2, d_k3, d_k4, d_l1, d_l2, d_l3, d_l4, d_Temp, p); 
    }
    inline void create_skyrmion(int pxi, int pxj, double rotation_angle, double* d_Field, double* d_grid, const Params& p) {
        create_skyrmion_wrapper(pxi, pxj, rotation_angle, d_Field, d_grid, p);
    }
    inline void create_vortex(int pxi, int pxj, int vortex, double* d_Field, double* d_grid, const Params& p) {
        create_vortex_wrapper(pxi, pxj, vortex, d_Field, d_grid, p);
    }
}

namespace observables {
    inline double compute_energy(double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p) {
        return compute_energy_wrapper(d_en, d_entmp, d_Field, d1fd1x, d_gridsum, h_gridsum, p);
    }
    inline double compute_skyrmion_number(double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p) {
        return compute_skyrmion_number_wrapper(d_en, d_entmp, d_Field, d1fd1x, d_gridsum, h_gridsum, p);
    }
    inline double compute_vortex_number(double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p) {
        return compute_vortex_number_wrapper(2, d_en, d_entmp, d_Field, d1fd1x, d_gridsum, h_gridsum, p);
    }
    inline void compute_magnetic_field(double* d_Field, double* d1fd1x, double* d_MagneticFluxDensity, const Params& p) { compute_magnetic_field_wrapper(d_Field, d1fd1x, d_MagneticFluxDensity, p); }
    inline void compute_supercurrent(double* d_Field, double* d2fd2x, double* d_Supercurrent, const Params& p) { compute_supercurrent_wrapper(d_Field, d2fd2x, d_Supercurrent, p); }

}

namespace minimization {
    inline double accelerated_gradient_descent(double* d_Velocity, double* d_Field, double* d1fd1x, double* d2fd2x, double* d_EnergyGradient, double* d_k1, double* d_k2, double* d_k3, double* d_k4, double* d_l1, double* d_l2, double* d_l3, double* d_l4, double* d_grid, double* d_Temp, double* d_en, double* d_entmp, double* d_gridsum, double* h_gridsum, double* d_maxeg, double* h_maxeg, Params& p) {
        return do_arrested_newton_flow(d_Velocity, d_Field, d1fd1x, d2fd2x, d_EnergyGradient, d_k1, d_k2, d_k3, d_k4, d_l1, d_l2, d_l3, d_l4, d_grid, d_Temp, d_en, d_entmp, d_gridsum, h_gridsum, d_maxeg, h_maxeg, p);
    }
}

namespace visualization {
    inline void show_density(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p) {
        show_density_wrapper(d_out, d_en, d_entmp, d_Field, d1fd1x, d_gridsum, h_gridsum, p);
    }
    inline void show_magnetic_flux(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p) {
        show_magnetic_flux_wrapper(d_out, d_en, d_entmp, d_Field, d1fd1x, d_gridsum, h_gridsum, p);
    }
    inline void show_vortex1(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d_gridsum, double* h_gridsum, const Params& p){
        show_higgs1_wrapper(d_out, d_en, d_entmp, d_Field, d_gridsum, h_gridsum, p);
    }
    inline void show_vortex2(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d_gridsum, double* h_gridsum, const Params& p){
        show_higgs2_wrapper(d_out, d_en, d_entmp, d_Field, d_gridsum, h_gridsum, p);
    }
    inline void show_magnetization(uchar4* d_out, double* d_Field, const Params& p) { show_magnetization_wrapper(d_out, d_Field, p); }
}