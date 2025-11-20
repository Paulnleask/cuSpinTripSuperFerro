// =========================
// simulation.hpp
// =========================
#pragma once
#include "params.hpp"
#include "buffers.hpp"
#include "kernels.hpp"
#include "output.hpp"

// ---------------- Simulation structure ----------------
struct Simulation {
    
    // Define host and device buffers/parameters
    Params p{};
    HostBuffers h{};
    DeviceBuffers d{};

    // Initialise host and device buffers/parameters, then create an initial configuration
    void init() {
        p.finalize();
        h.allocate(p);
        d.allocate(p);
        kernels::uploadDeviceParams(p);
        kernels::set_field_zero(d.Field, p);
        kernels::set_field_zero(d.Velocity, p);
        initial_configuration::create_grid(d.grid, p);
        initial_configuration::create_initial_configuration(d.Velocity, d.Field, d.grid, d.k1, d.k2, d.k3, d.k4, d.l1, d.l2, d.l3, d.l4, d.Temp, p);
    }

    // Uninitalise host and device buffers
    void uninit() {
        d.freeAll();
        h.freeAll();
    }

    // Print the simulation details
    void details(){

        // Simulation title
        std::printf("\n---------------- Composite Topological Excitations in Spin Triplet Superconducting Ferromagnets ----------------\n\n");
        
        // Optimization method
        std::printf("Optimization method: Arrested Newton Flow.\n\n");

        // Grid properties
        std::printf("Grid points: %d*%d\n", p.xlen, p.ylen);
        std::printf("Grid spacing: %.3f*%.3f\n", p.lsx, p.lsy);
        std::printf("System size: %d\n\n", p.xlen * p.ylen * p.number_total_fields);

        // Display coupling constants used
        std::printf("Ground state: m=%.1f, u1=%.1f, u2=%.1f\n", p.M0, p.u1, p.u2);
        std::printf("Coherence lengths: xi_s=%.3f, xi_m=%.3f, lambda=%.3f\n", p.coherence_length, p.coherence_magnetization, p.magnetic_penetration_depth);

        // Skyrmion/vortex types
        if (p.ansatz_bloch) {
            std::printf("\nSkyrmion type: Bloch.\n");
        }
        else if (p.ansatz_neel) {
            std::printf("\nSkyrmion type: Neel.\n");
        }
        else if (p.ansatz_heusler) {
            std::printf("\nSkyrmion type: Antiskyrmion.\n");
        }
        std::printf("Vortex type: Nielsen-Olesen.\n\n");
    }

    // Do optimizer step
    double step() {
        return minimization::accelerated_gradient_descent(d.Velocity, d.Field, d.d1fd1x, d.d2fd2x, d.EnergyGradient, d.k1, d.k2, d.k3, d.k4, d.l1, d.l2, d.l3, d.l4, d.grid, d.Temp, d.en, d.entmp, d.gridsum, h.gridsum, d.maxeg, h.maxeg, p);
    }

    // Compute optimizer observables
    void computeObservables() {
        p.energy = observables::compute_energy(d.en, d.entmp, d.Field, d.d1fd1x, d.gridsum, h.gridsum, p);
        p.skyrmion  = observables::compute_skyrmion_number(d.en, d.entmp, d.Field, d.d1fd1x, d.gridsum, h.gridsum, p);
        p.vortex = observables::compute_vortex_number(d.en, d.entmp, d.Field, d.d1fd1x, d.gridsum, h.gridsum, p);
    }

    // Output optimizer data
    void output_data() {
        cudaMemcpy(h.Field, d.Field, p.n_fields_bytes, cudaMemcpyDeviceToHost);
        observables::compute_energy(d.en, d.entmp, d.Field, d.d1fd1x, d.gridsum, h.gridsum, p);
        cudaMemcpy(h.EnergyDensity_Higgs, d.en, p.n_grid_bytes, cudaMemcpyDeviceToHost);
        observables::compute_skyrmion_number(d.en, d.entmp, d.Field, d.d1fd1x, d.gridsum, h.gridsum, p);
        cudaMemcpy(h.BaryonDensity, d.en, p.n_grid_bytes, cudaMemcpyDeviceToHost);
        observables::compute_magnetic_field(d.Field, d.d1fd1x, d.MagneticFluxDensity, p);
        cudaMemcpy(h.MagneticFluxDensity, d.MagneticFluxDensity, p.n_vector_bytes, cudaMemcpyDeviceToHost);
        observables::compute_supercurrent(d.Field, d.d2fd2x, d.Supercurrent, p);
        cudaMemcpy(h.Supercurrent, d.Supercurrent, p.n_vector_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h.grid, d.grid, p.n_coords_bytes, cudaMemcpyDeviceToHost);
        Output_Data(h.Field, h.EnergyDensity_Higgs, h.MagneticFluxDensity, h.BaryonDensity, h.Supercurrent, h.grid, h.Lattice_Points, h.Lattice_Vectors, p);
    }
};