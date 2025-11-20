// =========================
// buffers.hpp
// =========================
#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include "params.hpp"

// ---------------- Buffers for host ----------------
struct HostBuffers {
    double* gridsum = nullptr;
    double* maxeg   = nullptr;
    double* grid    = nullptr;
    double* Field   = nullptr;

    double* EnergyDensity_Higgs = nullptr;
    double* MagneticFluxDensity = nullptr; 
    double* BaryonDensity       = nullptr;
    double* Supercurrent        = nullptr; 

    double* Lattice_Vectors = nullptr; 
    double* Lattice_Points  = nullptr; 

    void allocate(const Params& p) {
        gridsum = (double*)std::malloc(p.n_grid_sum_bytes);
        maxeg   = (double*)std::malloc(p.n_fields_bytes);
        grid    = (double*)std::malloc(p.n_coords_bytes);
        Field   = (double*)std::malloc(p.n_fields_bytes);
        EnergyDensity_Higgs = (double*)std::malloc(p.n_grid_bytes);
        MagneticFluxDensity = (double*)std::malloc(p.n_vector_bytes);
        BaryonDensity       = (double*)std::malloc(p.n_grid_bytes);
        Supercurrent        = (double*)std::malloc(p.n_vector_bytes);
        Lattice_Vectors     = (double*)std::malloc(p.number_coordinates * p.number_coordinates * sizeof(double));
        Lattice_Points      = (double*)std::malloc(p.number_coordinates * sizeof(double));
    }

    void freeAll() {
        std::free(gridsum); gridsum=nullptr;
        std::free(maxeg);   maxeg=nullptr;
        std::free(grid);    grid=nullptr;
        std::free(Field);   Field=nullptr;
        std::free(EnergyDensity_Higgs); EnergyDensity_Higgs=nullptr;
        std::free(MagneticFluxDensity); MagneticFluxDensity=nullptr;
        std::free(BaryonDensity); BaryonDensity=nullptr;
        std::free(Supercurrent); Supercurrent=nullptr;
        std::free(Lattice_Vectors); Lattice_Vectors=nullptr;
        std::free(Lattice_Points);  Lattice_Points=nullptr;
    }
};

// ---------------- Buffers for device ----------------
struct DeviceBuffers {
    double* Field = nullptr;
    double* Velocity = nullptr;
    double* k1 = nullptr; double* k2 = nullptr; double* k3 = nullptr; double* k4 = nullptr;
    double* l1 = nullptr; double* l2 = nullptr; double* l3 = nullptr; double* l4 = nullptr;
    double* Temp = nullptr;

    double* grid = nullptr;

    double* EnergyGradient = nullptr; double* maxeg = nullptr;
    double* en = nullptr; double* entmp = nullptr; double* gridsum = nullptr;

    double* MagneticFluxDensity = nullptr; double* Supercurrent = nullptr;

    double* d1fd1x = nullptr;
    double* d2fd2x = nullptr;

    void allocate(const Params& p) {
        cudaMalloc((void**)&Field, p.n_fields_bytes);
        cudaMalloc((void**)&Velocity, p.n_fields_bytes);
        cudaMalloc((void**)&k1, p.n_fields_bytes);
        cudaMalloc((void**)&k2, p.n_fields_bytes);
        cudaMalloc((void**)&k3, p.n_fields_bytes);
        cudaMalloc((void**)&k4, p.n_fields_bytes);
        cudaMalloc((void**)&l1, p.n_fields_bytes);
        cudaMalloc((void**)&l2, p.n_fields_bytes);
        cudaMalloc((void**)&l3, p.n_fields_bytes);
        cudaMalloc((void**)&l4, p.n_fields_bytes);
        cudaMalloc((void**)&Temp, p.n_fields_bytes);
        cudaMalloc((void**)&grid, p.n_coords_bytes);
        cudaMalloc((void**)&EnergyGradient, p.n_fields_bytes);
        cudaMalloc((void**)&maxeg,          p.n_fields_bytes);
        cudaMalloc((void**)&en,             p.n_grid_bytes);
        cudaMalloc((void**)&entmp,          p.n_grid_bytes);
        cudaMalloc((void**)&gridsum,        p.n_grid_sum_bytes);
        cudaMalloc((void**)&MagneticFluxDensity, p.n_vector_bytes);
        cudaMalloc((void**)&Supercurrent,        p.n_vector_bytes);
        cudaMalloc((void**)&d1fd1x, sizeof(double) * p.number_coordinates * p.number_total_fields * p.dim_grid);
        cudaMalloc((void**)&d2fd2x, sizeof(double) * p.number_coordinates * p.number_coordinates * p.number_total_fields * p.dim_grid);
    }

    void freeAll() {
        cudaFree(Field); Field=nullptr;
        cudaFree(Velocity); Velocity=nullptr;
        cudaFree(k1); k1=nullptr; cudaFree(k2); k2=nullptr; cudaFree(k3); k3=nullptr; cudaFree(k4); k4=nullptr;
        cudaFree(l1); l1=nullptr; cudaFree(l2); l2=nullptr; cudaFree(l3); l3=nullptr; cudaFree(l4); l4=nullptr;
        cudaFree(Temp); Temp=nullptr;
        cudaFree(grid); grid=nullptr;
        cudaFree(EnergyGradient);   EnergyGradient=nullptr;
        cudaFree(maxeg);            maxeg=nullptr;
        cudaFree(en);               en=nullptr;
        cudaFree(entmp);            entmp=nullptr;
        cudaFree(gridsum);          gridsum=nullptr;
        cudaFree(MagneticFluxDensity); MagneticFluxDensity=nullptr;
        cudaFree(Supercurrent);        Supercurrent=nullptr;
        cudaFree(d1fd1x);        d1fd1x=nullptr;
        cudaFree(d2fd2x);        d2fd2x=nullptr;
    }
};
