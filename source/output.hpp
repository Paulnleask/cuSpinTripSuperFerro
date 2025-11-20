// =========================
// output.hpp
// =========================
#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include "params.hpp"

// ---------------- Indexing on host ----------------
inline int flat(int a, int i, int j, const Params& p) {
	return  j + i * p.ylen + a * p.xlen * p.ylen;
}

// ---------------- Outputs the field data ----------------
void Output_Field_dat(double* field, std::string name, const Params& p) {
	std::ofstream file(name);
	for (int a = 0; a < p.number_total_fields; a++) {
		for (int x = 0; x < p.xlen; x++) {
			for (int y = 0; y < p.ylen; y++) {
				file << std::setprecision(32) << field[flat(a, x, y, p)];
				file << "\t";
			}
			file << "\n";
		}
		file << "\n";
	}
	file.close();
}

// ---------------- Output iteration data (energy, etc.) ----------------
void Output_IterationData_dat(double* iteration_data, std::string name, int Loop_Count) {
	std::ofstream file(name);
	for (int n = 0; n < Loop_Count; n++) {
		file << std::setprecision(32) << iteration_data[n];
		file << "\t";
	}
	file.close();
}

// ---------------- Output density data (energy density, etc.) ----------------
void Output_DensityData_dat(double* density, int a, std::string name, const Params& p) {
	std::ofstream file(name);
	for (int x = 0; x < p.xlen; x++) {
		for (int y = 0; y < p.ylen; y++) {
			file << std::setprecision(32) << density[flat(a, x, y, p)];
			file << "\t";
		}
		file << "\n";
	}
	file.close();
}

// ---------------- Output files for plotting in plotting.py ----------------
void Output_Data(double* h_Field, double* h_EnergyDensity_Higgs, double* h_MagneticFluxDensity, double* h_BaryonDensity, double* h_Supercurrent, double* h_grid, double* Lattice_Points, double* Lattice_Vectors, const Params& p) {
	Lattice_Points[0] = p.xlen; Lattice_Points[1] = p.ylen;
	Lattice_Vectors[0] = h_grid[flat(0, p.xlen - 1, p.ylen - 1, p)]; Lattice_Vectors[1] = 0.0;
	Lattice_Vectors[2] = 0.0; Lattice_Vectors[3] = h_grid[flat(1, p.xlen - 1, p.ylen - 1, p)];
    Output_Field_dat(h_Field, "output/d_Field.dat", p);
	Output_IterationData_dat(Lattice_Points, "output/Higgs_LatticePoints.dat", p.number_coordinates);
	Output_IterationData_dat(Lattice_Vectors, "output/Higgs_LatticeVectors.dat", p.number_coordinates * p.number_coordinates);
	Output_DensityData_dat(h_EnergyDensity_Higgs, 0, "output/Higgs_EnergyDensity.dat", p);
	Output_DensityData_dat(h_MagneticFluxDensity, 0, "output/Higgs_ChargeDensityX.dat", p);
	Output_DensityData_dat(h_MagneticFluxDensity, 1, "output/Higgs_ChargeDensityY.dat", p);
	Output_DensityData_dat(h_MagneticFluxDensity, 2, "output/Higgs_ChargeDensity.dat", p);
	Output_DensityData_dat(h_BaryonDensity, 0, "output/Magnet_ChargeDensity.dat", p);
	Output_DensityData_dat(h_grid, 0, "output/Higgs_xGrid.dat", p);
	Output_DensityData_dat(h_grid, 1, "output/Higgs_yGrid.dat", p);
	Output_DensityData_dat(h_Field, 0, "output/Magnet_Field1.dat", p);
	Output_DensityData_dat(h_Field, 1, "output/Magnet_Field2.dat", p);
	Output_DensityData_dat(h_Field, 2, "output/Magnet_Field3.dat", p);
	Output_DensityData_dat(h_Field, 3, "output/Gauge_Field1.dat", p);
	Output_DensityData_dat(h_Field, 4, "output/Gauge_Field2.dat", p);
	Output_DensityData_dat(h_Field, 5, "output/Gauge_Field3.dat", p);
	Output_DensityData_dat(h_Field, 6, "output/Higgs_Field1.dat", p);
	Output_DensityData_dat(h_Field, 7, "output/Higgs_Field2.dat", p);
	Output_DensityData_dat(h_Field, 8, "output/Higgs_Field3.dat", p);
	Output_DensityData_dat(h_Field, 9, "output/Higgs_Field4.dat", p);
	Output_DensityData_dat(h_Supercurrent, 0, "output/J_Supercurrent1.dat", p);
	Output_DensityData_dat(h_Supercurrent, 1, "output/J_Supercurrent2.dat", p);
	Output_DensityData_dat(h_Supercurrent, 2, "output/J_Supercurrent3.dat", p);
}
