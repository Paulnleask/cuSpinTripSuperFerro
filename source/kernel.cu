// =========================
// kernel.cu
// =========================
#include "kernels.hpp"
#include "kernels_common.cuh"

// ---------------- RGB color clipping ----------------
__device__ __forceinline__  int clip(int a) {
	return (a < 0) ? 0 : ((a > 255) ? 255 : a);
}

// ---------------- Arrest velocity ----------------
__global__ void arrest_velocity_kernel(double* d_Field) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		for (int a = 0; a < dparams.number_total_fields; a++) {
			d_Field[index(a, x, y)] = 0.0;
		}
	}
}

__host__ void arrest_velocity_wrapper(double* d_Field, const Params& p, bool sync) {
	kernels::uploadDeviceParams(p);
	arrest_velocity_kernel << <p.grid, p.block >> > (d_Field);
	if (sync) { CHECK(cudaDeviceSynchronize()); }
}

// ---------------- Clear variable ----------------
__global__ void clear_variable_kernel(double* d_var, size_t size) {
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) d_var[id] = 0;
}

__host__ void clear_variable_wrapper(double* d_var, const Params& p, size_t size, bool sync) {
	kernels::uploadDeviceParams(p);
	clear_variable_kernel << <p.grid, p.block >> > (d_var, size);
	if (sync) { CHECK(cudaDeviceSynchronize()); }
}

// ---------------- Initial configuration functions ----------------
__device__ __forceinline__ double position(double xcent, double ycent, int x, int y, double* grid) { // Creates haloial position with centre offset
	return sqrt(pow(grid[index(0, x, y)] - xcent, 2) + pow(grid[index(1, x, y)] - ycent, 2));
}

__device__ __forceinline__ double profile_function_superconductor(double r, double max_r, double value) {
	if (r > max_r) {
		return 1.0;
	}
	else {
		return tanh(value * r);
	}
}

__device__ __forceinline__ double profile_function_magnetization(double r, double max_r) {
	if (r > max_r) {
		return 0.0;
	}
	else {
		return (M_PI)*exp(-pow(2.0 * r / max_r, 2));
	}
}

__device__ __forceinline__ double angle(double r, double xcent, double ycent, double orientation, int x, int y, double* grid) { // Creates angular position with centre offset
	double theta = -orientation; // Initialize theta
	if (r == 0) {
		theta += 0.0;
	}
	if (grid[index(0, x, y)] - xcent == 0 && grid[index(1, x, y)] - ycent > 0) {
		theta += (M_PI) / 2;
	}
	if (grid[index(0, x, y)] - xcent == 0 && grid[index(1, x, y)] - ycent < 0) {
		theta += 3 * (M_PI) / 2;
	}
	if (grid[index(0, x, y)] - xcent > 0 && grid[index(1, x, y)] - ycent == 0) {
		theta += 0.0;
	}
	if (grid[index(0, x, y)] - xcent < 0 && grid[index(1, x, y)] - ycent == 0) {
		theta += (M_PI);
	}
	if (grid[index(0, x, y)] - xcent > 0 && grid[index(1, x, y)] - ycent > 0) {
		theta += acos((grid[index(0, x, y)] - xcent) / (r));
	}
	if (grid[index(0, x, y)] - xcent < 0 && grid[index(1, x, y)] - ycent > 0) {
		theta += acos((grid[index(0, x, y)] - xcent) / (r));
	}
	if (grid[index(0, x, y)] - xcent > 0 && grid[index(1, x, y)] - ycent < 0) {
		theta += -acos((grid[index(0, x, y)] - xcent) / (r));
	}
	if (grid[index(0, x, y)] - xcent < 0 && grid[index(1, x, y)] - ycent < 0) {
		theta += -acos((grid[index(0, x, y)] - xcent) / (r));
	}
	return theta;
}

// ---------------- Creat grid ----------------
__global__ void create_grid_kernel(double* d_grid) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		d_grid[index(0, x, y)] = dparams.lsx * (double)x; // Value of x at a lattice point
		d_grid[index(1, x, y)] = dparams.lsy * (double)y; // Value of y at a lattice point
	}
}

__host__ void create_grid_wrapper(double* d_grid, const Params& p) {
	kernels::uploadDeviceParams(p);
	create_grid_kernel << <p.grid, p.block >> > (d_grid);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Create Nielsen-Olsen vortex ----------------
__global__ void create_vortex_kernel(double* d_Field, double* d_grid, int pxi, int pxj, int vortex) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	if ((pxi >= dparams.xlen) || (pxj >= dparams.ylen) || (pxi < 0) || (pxj < 0)) {
		return;
	}
	else {
		double xcent = d_grid[index(0, pxi, pxj)];
		double ycent = d_grid[index(1, pxi, pxj)];
		double r1 = position(xcent, ycent, x, y, d_grid);
		double rmax = max(dparams.xsize, dparams.ysize);
		double theta1 = angle(r1, xcent, ycent, 0, x, y, d_grid);
		double fs = profile_function_superconductor(r1, rmax, 0.2);
		// Higgs fields 1
		thrust::complex<double> old_OP1(d_Field[index(6, x, y)], d_Field[index(7, x, y)]);
		thrust::complex<double> int_OP1(dparams.u1 * fs * cos(pow(-1.0, vortex) * theta1), dparams.u1 * fs * sin(pow(-1.0, vortex) * theta1));
		thrust::complex<double> new_OP1 = (old_OP1 * int_OP1) / dparams.u1;
		d_Field[index(6, x, y)] = new_OP1.real();
		d_Field[index(7, x, y)] = new_OP1.imag();
		// Higgs fields 1
		thrust::complex<double> old_OP2(d_Field[index(8, x, y)], d_Field[index(9, x, y)]);
		thrust::complex<double> int_OP2(dparams.u2 * fs * cos(pow(-1.0, vortex) * theta1), dparams.u2 * fs * sin(pow(-1.0, vortex) * theta1));
		thrust::complex<double> new_OP2 = (old_OP2 * int_OP2) / dparams.u2;
		d_Field[index(8, x, y)] = new_OP2.real();
		d_Field[index(9, x, y)] = new_OP2.imag();
	}
}

__host__ void create_vortex_wrapper(int pxi, int pxj, int vortex, double* d_Field, double* d_grid, const Params& p) {
	kernels::uploadDeviceParams(p);
	create_vortex_kernel << <p.grid, p.block >> > (d_Field, d_grid, pxi, pxj, vortex);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Create skyrmion ----------------
__global__ void create_skyrmion_kernel(double* d_Field, double* d_grid, int pxi, int pxj, double rotation_angle) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	if ((pxi >= dparams.xlen) || (pxj >= dparams.ylen) || (pxi < 0) || (pxj < 0)) {
		return;
	}
	else {
		double xcent = d_grid[index(0, pxi, pxj)];
		double ycent = d_grid[index(1, pxi, pxj)];
		double r1 = position(xcent, ycent, x, y, d_grid);
		double rmax = max(dparams.xsize, dparams.ysize);
		double theta1 = angle(r1, xcent, ycent, rotation_angle, x, y, d_grid);
		double fm = profile_function_magnetization(r1, rmax / 10.0);
		thrust::complex<double> old_W(d_Field[index(0, x, y)] / (1.0 + d_Field[index(2, x, y)]), d_Field[index(1, x, y)] / (1.0 + d_Field[index(2, x, y)]));
		thrust::complex<double> int_W(0.0,0.0);
		thrust::complex<double> I(0.0, 1.0);
		if (dparams.ansatz_bloch) {
			int_W = - tan(fm / 2) * sin(theta1) + I * tan(fm / 2) * cos(theta1);
		}
		else if (dparams.ansatz_neel) {
			int_W = tan(fm / 2) * cos(theta1) + I * tan(fm / 2) * sin(theta1);
		}
		else if (dparams.ansatz_heusler) {
			int_W = -tan(fm / 2) * sin(theta1) - I * tan(fm / 2) * cos(theta1);
		}
		else {
			int_W = 0.0 + I * 0.0;
		}
		thrust::complex<double> new_W = old_W + int_W;
		d_Field[index(0, x, y)] = 2.0 * new_W.real() / (1 + thrust::norm(new_W));
		d_Field[index(1, x, y)] = 2.0 * new_W.imag() / (1 + thrust::norm(new_W));
		d_Field[index(2, x, y)] = (1 - thrust::norm(new_W)) / (1 + thrust::norm(new_W));
		compute_norm_magnetization(d_Field, x, y);
	}
}

__host__ void create_skyrmion_wrapper(int pxi, int pxj, double rotation_angle, double* d_Field, double* d_grid, const Params& p) {
	kernels::uploadDeviceParams(p);
	create_skyrmion_kernel << <p.grid, p.block >> > (d_Field, d_grid, pxi, pxj, rotation_angle);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Create initial configuration ----------------
__global__ void create_initial_configuration_kernel(double* d_Velocity, double* d_Field, double* d_grid, double* d_k1, double* d_k2, double* d_k3, double* d_k4, double* d_l1, double* d_l2, double* d_l3, double* d_l4, double* d_Temp) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		double xcent = (d_grid[index(0, 0, 0)] + d_grid[index(0, dparams.xlen - 1, dparams.ylen - 1)]) / 2.0;
		double ycent = (d_grid[index(1, 0, 0)] + d_grid[index(1, dparams.xlen - 1, dparams.ylen - 1)]) / 2.0;
		double r1 = position(xcent, ycent, x, y, d_grid);
		double rmax = max(dparams.xsize, dparams.ysize);
		double theta1 = angle(r1, xcent, ycent, 0, x, y, d_grid);
		double fm = profile_function_magnetization(r1, rmax / 10.0);
		double fs = profile_function_superconductor(r1, rmax, 0.2);
		// Magnetization
		if (dparams.ansatz_bloch) {
			d_Field[index(0, x, y)] = -sin(fm) * sin(dparams.skyrmion_number * theta1);
			d_Field[index(1, x, y)] = sin(fm) * cos(dparams.skyrmion_number * theta1);
			d_Field[index(2, x, y)] = cos(fm);
			compute_norm_magnetization(d_Field, x, y);
		}
		else if (dparams.ansatz_neel) {
			d_Field[index(0, x, y)] = sin(fm) * cos(dparams.skyrmion_number * theta1);
			d_Field[index(1, x, y)] = sin(fm) * sin(dparams.skyrmion_number * theta1);
			d_Field[index(2, x, y)] = cos(fm);
			compute_norm_magnetization(d_Field, x, y);
		}
		else if (dparams.ansatz_heusler) {
			d_Field[index(0, x, y)] = -sin(fm) * sin(dparams.skyrmion_number * theta1);
			d_Field[index(1, x, y)] = -sin(fm) * cos(dparams.skyrmion_number * theta1);
			d_Field[index(2, x, y)] = cos(fm);
			compute_norm_magnetization(d_Field, x, y);
		}
		else {
			d_Field[index(0, x, y)] = 0.0;
			d_Field[index(1, x, y)] = 0.0;
			d_Field[index(2, x, y)] = 1.0;
		}
		// Gauge fields
		d_Field[index(3, x, y)] = -dparams.ainf * fs * sin(theta1) / r1;
		d_Field[index(4, x, y)] = dparams.ainf * fs * cos(theta1) / r1;
		d_Field[index(5, x, y)] = 0.0;
		// Higgs fields 1
		d_Field[index(6, x, y)] = dparams.u1 * fs * cos(dparams.vortex1_number * theta1);
		d_Field[index(7, x, y)] = dparams.u1 * fs * sin(dparams.vortex1_number * theta1);
		// Higgs fields 2
		d_Field[index(8, x, y)] = dparams.u2 * fs * cos(dparams.vortex2_number * theta1);
		d_Field[index(9, x, y)] = dparams.u2 * fs * sin(dparams.vortex2_number * theta1);
		// Initalize the other fields
		for (int a = 0; a < dparams.number_total_fields; a++) {
			d_Velocity[index(a, x, y)] = 0.0;
			d_k1[index(a, x, y)] = 0.0;
			d_k2[index(a, x, y)] = 0.0;
			d_k3[index(a, x, y)] = 0.0;
			d_k4[index(a, x, y)] = 0.0;
			d_l1[index(a, x, y)] = 0.0;
			d_l2[index(a, x, y)] = 0.0;
			d_l3[index(a, x, y)] = 0.0;
			d_l4[index(a, x, y)] = 0.0;
			d_Temp[index(a, x, y)] = 0.0;
		}
	}
}

__host__ void create_initial_configuration_wrapper(double* d_Velocity, double* d_Field, double* d_grid, double* d_k1, double* d_k2, double* d_k3, double* d_k4, double* d_l1, double* d_l2, double* d_l3, double* d_l4, double* d_Temp, const Params& p) {
	kernels::uploadDeviceParams(p);
	create_initial_configuration_kernel << <p.grid, p.block >> > (d_Velocity, d_Field, d_grid, d_k1, d_k2, d_k3, d_k4, d_l1, d_l2, d_l3, d_l4, d_Temp);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Compute derivatives ----------------
__device__ void compute_derivatve_first(double* d1fd1x, double* d_Field, const int a, const int x, const int y) {
	if (x > dparams.halo - 1 && x < dparams.xlen - dparams.halo) {
		d1fd1x[indexd1(0, a, x, y)] = (1.0 / 12.0 * d_Field[index(a, x - 2, y)] - 2.0 / 3.0 * d_Field[index(a, x - 1, y)] + 2.0 / 3.0 * d_Field[index(a, x + 1, y)] - 1.0 / 12.0 * d_Field[index(a, x + 2, y)]) / (dparams.lsx);;
	}
	else {
		d1fd1x[indexd1(0, a, x, y)] = 0.0;
	}
	if (y > dparams.halo - 1 && y < dparams.ylen - dparams.halo) {
		d1fd1x[indexd1(1, a, x, y)] = (1.0 / 12.0 * d_Field[index(a, x, y - 2)] - 2.0 / 3.0 * d_Field[index(a, x, y - 1)] + 2.0 / 3.0 * d_Field[index(a, x, y + 1)] - 1.0 / 12.0 * d_Field[index(a, x, y + 2)]) / (dparams.lsy);

	}
	else {
		d1fd1x[indexd1(1, a, x, y)] = 0.0;
	}
}

__device__ void compute_derivatve_second(double* d2fd2x, double* d_Field, const int a, const int x, const int y) {
	if (x > dparams.halo - 1 && x < dparams.xlen - dparams.halo) {
		d2fd2x[indexd2(0, 0, a, x, y)] = (-1.0 / 12.0 * d_Field[index(a, x - 2, y)] + 4.0 / 3.0 * d_Field[index(a, x - 1, y)] - 5.0 / 2.0 * d_Field[index(a, x, y)] + 4.0 / 3.0 * d_Field[index(a, x + 1, y)] - 1.0 / 12.0 * d_Field[index(a, x + 2, y)]) / (dparams.lsx * dparams.lsx);
	}
	else {
		d2fd2x[indexd2(0, 0, a, x, y)] = 0.0;
	}
	if (y > dparams.halo - 1 && y < dparams.ylen - dparams.halo) {
		d2fd2x[indexd2(1, 1, a, x, y)] = (-1.0 / 12.0 * d_Field[index(a, x, y - 2)] + 4.0 / 3.0 * d_Field[index(a, x, y - 1)] - 5.0 / 2.0 * d_Field[index(a, x, y)] + 4.0 / 3.0 * d_Field[index(a, x, y + 1)] - 1.0 / 12.0 * d_Field[index(a, x, y + 2)]) / (dparams.lsy * dparams.lsy);
	}
	else {
		d2fd2x[indexd2(1, 1, a, x, y)] = 0.0;
	}
	if (x > dparams.halo && y > dparams.halo && x < dparams.xlen - dparams.halo && y < dparams.ylen - dparams.halo) {
		d2fd2x[indexd2(1, 0, a, x, y)] = (-1.0 / 12.0 * d_Field[index(a, x - 2, y - 2)] + 4.0 / 3.0 * d_Field[index(a, x - 1, y - 1)] - 5.0 / 2.0 * d_Field[index(a, x, y)] + 4.0 / 3.0 * d_Field[index(a, x + 1, y + 1)] - 1.0 / 12.0 * d_Field[index(a, x + 2, y + 2)]) / (2.0 * dparams.grid_volume) - (-1.0 / 12.0 * d_Field[index(a, x - 2, y)] + 4.0 / 3.0 * d_Field[index(a, x - 1, y)] - 5.0 / 2.0 * d_Field[index(a, x, y)] + 4.0 / 3.0 * d_Field[index(a, x + 1, y)] - 1.0 / 12.0 * d_Field[index(a, x + 2, y)]) / (2.0 * dparams.lsx * dparams.lsx) - (-1.0 / 12.0 * d_Field[index(a, x, y - 2)] + 4.0 / 3.0 * d_Field[index(a, x, y - 1)] - 5.0 / 2.0 * d_Field[index(a, x, y)] + 4.0 / 3.0 * d_Field[index(a, x, y + 1)] - 1.0 / 12.0 * d_Field[index(a, x, y + 2)]) / (2.0 * dparams.lsy * dparams.lsy);
	}
	else {
		d2fd2x[indexd2(1, 0, a, x, y)] = 0.0;
	}
	d2fd2x[indexd2(0, 1, a, x, y)] = d2fd2x[indexd2(1, 0, a, x, y)];
}

// ---------------- Compute the energy ----------------
__device__ double compute_energy(double* d_Field, double* d1fd1x, const int x, const int y) {
	double energy = 0.0;
	energy += dparams.ha / 2.0 * (pow(d_Field[index(6, x, y)], 2) + pow(d_Field[index(7, x, y)], 2) + pow(d_Field[index(8, x, y)], 2) + pow(d_Field[index(9, x, y)], 2));
	energy += dparams.hb1 / 4.0 * pow(pow(d_Field[index(6, x, y)], 2) + pow(d_Field[index(7, x, y)], 2), 2) + dparams.hb1 / 4.0 * pow(pow(d_Field[index(8, x, y)], 2) + pow(d_Field[index(9, x, y)], 2), 2);
	energy += dparams.hb2 * (pow(d_Field[index(6, x, y)], 2) + pow(d_Field[index(7, x, y)], 2)) * (pow(d_Field[index(8, x, y)], 2) + pow(d_Field[index(9, x, y)], 2));
	energy += 2.0 * dparams.hc * (d_Field[index(6, x, y)] * d_Field[index(8, x, y)] + d_Field[index(7, x, y)] * d_Field[index(9, x, y)]);
	energy += 1.0 / 2.0 * pow(d1fd1x[indexd1(0, 6, x, y)], 2) + 1.0 / 2.0 * pow(d1fd1x[indexd1(1, 6, x, y)], 2) + 1.0 / 2.0 * pow(d1fd1x[indexd1(0, 7, x, y)], 2) + 1.0 / 2.0 * pow(d1fd1x[indexd1(1, 7, x, y)], 2) + 1.0 / 2.0 * pow(d1fd1x[indexd1(0, 8, x, y)], 2) + 1.0 / 2.0 * pow(d1fd1x[indexd1(1, 8, x, y)], 2) + 1.0 / 2.0 * pow(d1fd1x[indexd1(0, 9, x, y)], 2) + 1.0 / 2.0 * pow(d1fd1x[indexd1(1, 9, x, y)], 2);
	energy += 1.0 / 2.0 * pow(dparams.q, 2) * (pow(d_Field[index(3, x, y)], 2) + pow(d_Field[index(4, x, y)], 2) + pow(d_Field[index(5, x, y)], 2)) * (pow(d_Field[index(6, x, y)], 2) + pow(d_Field[index(7, x, y)], 2) + pow(d_Field[index(8, x, y)], 2) + pow(d_Field[index(9, x, y)], 2));
	energy += dparams.q * d_Field[index(3, x, y)] * (d_Field[index(6, x, y)] * d1fd1x[indexd1(0, 7, x, y)] - d_Field[index(7, x, y)] * d1fd1x[indexd1(0, 6, x, y)] + d_Field[index(8, x, y)] * d1fd1x[indexd1(0, 9, x, y)] - d_Field[index(9, x, y)] * d1fd1x[indexd1(0, 8, x, y)]) + dparams.q * d_Field[index(4, x, y)] * (d_Field[index(6, x, y)] * d1fd1x[indexd1(1, 7, x, y)] - d_Field[index(7, x, y)] * d1fd1x[indexd1(1, 6, x, y)] + d_Field[index(8, x, y)] * d1fd1x[indexd1(1, 9, x, y)] - d_Field[index(9, x, y)] * d1fd1x[indexd1(1, 8, x, y)]);
	energy += 1.0 / 2.0 * pow(d1fd1x[indexd1(0, 4, x, y)], 2) + 1.0 / 2.0 * pow(d1fd1x[indexd1(1, 3, x, y)], 2) - d1fd1x[indexd1(0, 4, x, y)] * d1fd1x[indexd1(1, 3, x, y)] + 1.0 / 2.0 * pow(d1fd1x[indexd1(0, 5, x, y)], 2) + 1.0 / 2.0 * pow(d1fd1x[indexd1(1, 5, x, y)], 2);
	energy += dparams.alpha / 2.0 * (pow(d_Field[index(0, x, y)], 2) + pow(d_Field[index(1, x, y)], 2) + pow(d_Field[index(2, x, y)], 2)) + dparams.beta / 4.0 * pow(pow(d_Field[index(0, x, y)], 2) + pow(d_Field[index(1, x, y)], 2) + pow(d_Field[index(2, x, y)], 2), 2);
	energy += pow(dparams.gamma, 2) / 2.0 * (pow(d1fd1x[indexd1(0, 0, x, y)], 2) + pow(d1fd1x[indexd1(0, 1, x, y)], 2) + pow(d1fd1x[indexd1(0, 2, x, y)], 2) + pow(d1fd1x[indexd1(1, 0, x, y)], 2) + pow(d1fd1x[indexd1(1, 1, x, y)], 2) + pow(d1fd1x[indexd1(1, 2, x, y)], 2));
	energy += d_Field[index(1, x, y)] * d1fd1x[indexd1(0, 5, x, y)] - d_Field[index(0, x, y)] * d1fd1x[indexd1(1, 5, x, y)] - d_Field[index(2, x, y)] * d1fd1x[indexd1(0, 4, x, y)] + d_Field[index(2, x, y)] * d1fd1x[indexd1(1, 3, x, y)];
	energy -= dparams.ha / 2.0 * (pow(dparams.u1, 2) + pow(dparams.u2,2)) + dparams.hb1 / 4.0 * (pow(dparams.u1, 4) + pow(dparams.u2,4)) + dparams.hb2 * pow(dparams.u1,2) * pow(dparams.u2,2) + 2.0 * dparams.hc * dparams.u1 * dparams.u2 + dparams.alpha / 2.0 * pow(dparams.M0, 2) + dparams.beta / 4.0 * pow(dparams.M0, 4);
	energy *= dparams.grid_volume;
	return energy;
}

__global__ void compute_energy_kernel(double* d_en, double* d_Field, double* d1fd1x) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		for (int a = 0; a < dparams.number_total_fields; a++) {
			compute_derivatve_first(d1fd1x, d_Field, a, x, y);
		}
		d_en[index(0, x, y)] = compute_energy(d_Field, d1fd1x, x, y);
	}
}

double compute_energy_wrapper(double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_energy_kernel << < p.grid, p.block >> > (d_en, d_Field, d1fd1x);
	CHECK(cudaMemcpy(d_entmp, d_en, p.n_grid_bytes, cudaMemcpyDeviceToDevice));
	CHECK(cudaDeviceSynchronize());
	double energy = compute_sum_wrapper(d_entmp, d_gridsum, h_gridsum, p, p.dim_grid);
	clear_variable_wrapper(d_entmp, p, p.dim_grid, false);
	clear_variable_wrapper(d_en, p, p.dim_grid, false);
	return energy;
}

// ---------------- Compute the skyrmion number ----------------
__device__ double compute_skyrmion_number(double* d_Field, double* d1fd1x, const int x, const int y) {
	double Charge = 0.0;
	int levicivita3D[3][3][3] = { {{0,0,0},{0,0,1},{0,-1,0}},{{0,0,-1},{0,0,0},{1,0,0}},{{0,1,0},{-1,0,0},{0,0,0}} }; // Levi-Civita symbol
	for (int a = 0; a < dparams.number_magnetization_fields; a++) {
		for (int b = 0; b < dparams.number_magnetization_fields; b++) {
			for (int c = 0; c < dparams.number_magnetization_fields; c++) {
				Charge += levicivita3D[a][b][c] * d_Field[index(a, x, y)] * d1fd1x[indexd1(0, b, x, y)] * d1fd1x[indexd1(1, c, x, y)];
			}
		}
	}
	Charge *= dparams.grid_volume / (4 * M_PI);
	return Charge;
}

__global__ void compute_skyrmion_number_kernel(double* d_en, double* d_Field, double* d1fd1x) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		for (int a = 0; a < dparams.number_total_fields; a++) {
			compute_derivatve_first(d1fd1x, d_Field, a, x, y);
		}
		d_en[index(0, x, y)] = compute_skyrmion_number(d_Field, d1fd1x, x, y);
	}
}

double compute_skyrmion_number_wrapper(double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_skyrmion_number_kernel << < p.grid, p.block >> > (d_en, d_Field, d1fd1x);
	CHECK(cudaMemcpy(d_entmp, d_en, p.n_grid_bytes, cudaMemcpyDeviceToDevice));
	CHECK(cudaDeviceSynchronize());
	double charge = compute_sum_wrapper(d_entmp, d_gridsum, h_gridsum, p, p.dim_grid);
	clear_variable_wrapper(d_entmp, p, p.dim_grid, false);
	clear_variable_wrapper(d_en, p, p.dim_grid, false);
	return charge;
}

// ---------------- Compute the vortex number ----------------
__device__ double compute_vortex_number(double* d1fd1x, const int a, int x, int y) {
	double magneticField = 0.0;
	// B1 - B in yz-plane
	if (a == 0) {
		magneticField += d1fd1x[indexd1(1, 5, x, y)];
	}
	// B2 - B in xz-plane
	else if (a == 1) {
		magneticField -= d1fd1x[indexd1(0, 5, x, y)];
	}
	// B3 - B in xy-plane
	else if (a == 2) {
		magneticField += d1fd1x[indexd1(0, 4, x, y)] - d1fd1x[indexd1(1, 3, x, y)];
	}
	magneticField *= dparams.grid_volume / (2.0 * M_PI * dparams.q);
	return magneticField;
}

__global__ void compute_vortex_number_kernel(double* d_en, double* d_Field, double* d1fd1x, const int a) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		for (int i = 0; i < dparams.number_total_fields; i++) {
			compute_derivatve_first(d1fd1x, d_Field, i, x, y);
		}
		d_en[index(0, x, y)] = compute_vortex_number(d1fd1x, a, x, y);
	}
}

double compute_vortex_number_wrapper(const int a, double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_vortex_number_kernel << < p.grid, p.block >> > (d_en, d_Field, d1fd1x, a);
	CHECK(cudaMemcpy(d_entmp, d_en, p.n_grid_bytes, cudaMemcpyDeviceToDevice));
	CHECK(cudaDeviceSynchronize());
	double charge = compute_sum_wrapper(d_entmp, d_gridsum, h_gridsum, p, p.dim_grid);
	clear_variable_wrapper(d_entmp, p, p.dim_grid, false);
	clear_variable_wrapper(d_en, p, p.dim_grid, false);
	return charge;
}

// ---------------- Compute the magnetic field ----------------
__global__ void compute_magnetic_field_kernel(double* d_Field, double* d1fd1x, double* d_MagneticFluxDensity) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		for (int a = 0; a < dparams.number_total_fields; a++) {
			compute_derivatve_first(d1fd1x, d_Field, a, x, y);
		}
		for (int a = 0; a < dparams.number_magnetization_fields; a++) {
			d_MagneticFluxDensity[index(a, x, y)] = compute_vortex_number(d1fd1x, a, x, y);
		}
	}
}

__host__ void compute_magnetic_field_wrapper(double* d_Field, double* d1fd1x, double* d_MagneticFluxDensity, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_magnetic_field_kernel << <p.grid, p.block >> > (d_Field, d1fd1x, d_MagneticFluxDensity);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Compute norm of the superconducting OP1 ----------------
__device__ double compute_norm_higgs1(double* d_Field, const int x, const int y) {
	return pow(d_Field[index(6, x, y)], 2) + pow(d_Field[index(7, x, y)], 2);
}

__global__ void compute_norm_higgs1_kernel(double* d_en, double* d_Field) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		d_en[index(0, x, y)] = compute_norm_higgs1(d_Field, x, y);
	}
}

double compute_norm_higgs1_wrapper(double* d_en, double* d_entmp, double* d_Field, double* d_gridsum, double* h_gridsum, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_norm_higgs1_kernel << < p.grid, p.block >> > (d_en, d_Field);
	CHECK(cudaMemcpy(d_entmp, d_en, p.n_grid_bytes, cudaMemcpyDeviceToDevice));
	CHECK(cudaDeviceSynchronize());
	double charge = compute_sum_wrapper(d_entmp, d_gridsum, h_gridsum, p, p.dim_grid);
	clear_variable_wrapper(d_entmp, p, p.dim_grid, false);
	clear_variable_wrapper(d_en, p, p.dim_grid, false);
	return charge;
}

// ---------------- Compute norm of the superconducting OP2 ----------------
__device__ double compute_norm_higgs2(double* d_Field, const int x, const int y) {
	return pow(d_Field[index(8, x, y)], 2) + pow(d_Field[index(9, x, y)], 2);
}

__global__ void compute_norm_higgs2_kernel(double* d_en, double* d_Field) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		d_en[index(0, x, y)] = compute_norm_higgs2(d_Field, x, y);
	}
}

double compute_norm_higgs2_wrapper(double* d_en, double* d_entmp, double* d_Field, double* d_gridsum, double* h_gridsum, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_norm_higgs2_kernel << < p.grid, p.block >> > (d_en, d_Field);
	CHECK(cudaMemcpy(d_entmp, d_en, p.n_grid_bytes, cudaMemcpyDeviceToDevice));
	CHECK(cudaDeviceSynchronize());
	double charge = compute_sum_wrapper(d_entmp, d_gridsum, h_gridsum, p, p.dim_grid);
	clear_variable_wrapper(d_entmp, p, p.dim_grid, false);
	clear_variable_wrapper(d_en, p, p.dim_grid, false);
	return charge;
}

// ---------------- Compute the supercurrent ----------------
__device__ void compute_supercurrent(double* d_Supercurrent, double* d2fd2x, const int x, const int y) { // Calculates the Energy variation of the Baby Skyrme model and modifies the d_Velocity
	for (int a = 0; a < dparams.number_magnetization_fields; a++) {
		d_Supercurrent[index(a, x, y)] = 0.0;
	}
	// Compute the supercurrent
	d_Supercurrent[index(0, x, y)] += d2fd2x[indexd2(1, 0, 4, x, y)] - d2fd2x[indexd2(1, 1, 3, x, y)];
	d_Supercurrent[index(1, x, y)] += d2fd2x[indexd2(0, 1, 3, x, y)] - d2fd2x[indexd2(0, 0, 4, x, y)];
	d_Supercurrent[index(2, x, y)] -= d2fd2x[indexd2(0, 0, 5, x, y)] + d2fd2x[indexd2(1, 1, 5, x, y)];
}

__global__ void compute_supercurrent_kernel(double* d_Field, double* d2fd2x, double* d_Supercurrent) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		for (int a = 0; a < dparams.number_total_fields; a++) {
			compute_derivatve_second(d2fd2x, d_Field, a, x, y);
		}
		compute_supercurrent(d_Supercurrent, d2fd2x, x, y);
	}
}

__host__ void compute_supercurrent_wrapper(double* d_Field, double* d2fd2x, double* d_Supercurrent, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_supercurrent_kernel << <p.grid, p.block >> > (d_Field, d2fd2x, d_Supercurrent);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Compute the energy gradient ----------------
__device__ void do_gradient_step(double* d_Velocity, double* d_Field, double* d_EnergyGradient, double* d1fd1x, double* d2fd2x, const int x, const int y) { // Calculates the Energy variation of the Baby Skyrme model and modifies the d_Velocity

	// Local energy gradient (increases speed)
	double EnergyGradient[no_fields];
	for(int a = 0; a < dparams.number_total_fields; a++){
		EnergyGradient[a] = 0.0;
	}
	// Magnetization
	EnergyGradient[0] += dparams.alpha * d_Field[index(0, x, y)] + dparams.beta * d_Field[index(0, x, y)] * (pow(d_Field[index(0, x, y)], 2) + pow(d_Field[index(1, x, y)], 2) + pow(d_Field[index(2, x, y)], 2)) - pow(dparams.gamma, 2) * d2fd2x[indexd2(0, 0, 0, x, y)] - pow(dparams.gamma, 2) * d2fd2x[indexd2(1, 1, 0, x, y)] - d1fd1x[indexd1(1, 5, x, y)];
	EnergyGradient[1] += dparams.alpha * d_Field[index(1, x, y)] + dparams.beta * d_Field[index(1, x, y)] * (pow(d_Field[index(0, x, y)], 2) + pow(d_Field[index(1, x, y)], 2) + pow(d_Field[index(2, x, y)], 2)) - pow(dparams.gamma, 2) * d2fd2x[indexd2(0, 0, 1, x, y)] - pow(dparams.gamma, 2) * d2fd2x[indexd2(1, 1, 1, x, y)] + d1fd1x[indexd1(0, 5, x, y)];
	EnergyGradient[2] += dparams.alpha * d_Field[index(2, x, y)] + dparams.beta * d_Field[index(2, x, y)] * (pow(d_Field[index(0, x, y)], 2) + pow(d_Field[index(1, x, y)], 2) + pow(d_Field[index(2, x, y)], 2)) - pow(dparams.gamma, 2) * d2fd2x[indexd2(0, 0, 2, x, y)] - pow(dparams.gamma, 2) * d2fd2x[indexd2(1, 1, 2, x, y)] - d1fd1x[indexd1(0, 4, x, y)] + d1fd1x[indexd1(1, 3, x, y)];
	// Gauge field
	EnergyGradient[3] += pow(dparams.q, 2) * d_Field[index(3, x, y)] * (pow(d_Field[index(6, x, y)], 2) + pow(d_Field[index(7, x, y)], 2) + pow(d_Field[index(8, x, y)], 2) + pow(d_Field[index(9, x, y)], 2)) + dparams.q * (d_Field[index(6, x, y)] * d1fd1x[indexd1(0, 7, x, y)] - d_Field[index(7, x, y)] * d1fd1x[indexd1(0, 6, x, y)] + d_Field[index(8, x, y)] * d1fd1x[indexd1(0, 9, x, y)] - d_Field[index(9, x, y)] * d1fd1x[indexd1(0, 8, x, y)]) + d2fd2x[indexd2(1, 0, 4, x, y)] - d2fd2x[indexd2(1, 1, 3, x, y)] - d1fd1x[indexd1(1, 2, x, y)];
	EnergyGradient[4] += pow(dparams.q, 2) * d_Field[index(4, x, y)] * (pow(d_Field[index(6, x, y)], 2) + pow(d_Field[index(7, x, y)], 2) + pow(d_Field[index(8, x, y)], 2) + pow(d_Field[index(9, x, y)], 2)) + dparams.q * (d_Field[index(6, x, y)] * d1fd1x[indexd1(1, 7, x, y)] - d_Field[index(7, x, y)] * d1fd1x[indexd1(1, 6, x, y)] + d_Field[index(8, x, y)] * d1fd1x[indexd1(1, 9, x, y)] - d_Field[index(9, x, y)] * d1fd1x[indexd1(1, 8, x, y)]) + d2fd2x[indexd2(0, 1, 3, x, y)] - d2fd2x[indexd2(0, 0, 4, x, y)] + d1fd1x[indexd1(0, 2, x, y)];
	EnergyGradient[5] += pow(dparams.q, 2) * d_Field[index(5, x, y)] * (pow(d_Field[index(6, x, y)], 2) + pow(d_Field[index(7, x, y)], 2) + pow(d_Field[index(8, x, y)], 2) + pow(d_Field[index(9, x, y)], 2)) - d2fd2x[indexd2(0, 0, 5, x, y)] - d2fd2x[indexd2(1, 1, 5, x, y)] - d1fd1x[indexd1(0, 1, x, y)] + d1fd1x[indexd1(1, 0, x, y)];
	// Higgs field
	EnergyGradient[6] += dparams.ha * d_Field[index(6, x, y)] + dparams.hb1 * d_Field[index(6, x, y)] * (pow(d_Field[index(6, x, y)], 2) + pow(d_Field[index(7, x, y)], 2)) - d2fd2x[indexd2(0, 0, 6, x, y)] - d2fd2x[indexd2(1, 1, 6, x, y)] + pow(dparams.q, 2) * (pow(d_Field[index(3, x, y)], 2) + pow(d_Field[index(4, x, y)], 2) + pow(d_Field[index(5, x, y)], 2)) * d_Field[index(6, x, y)] + 2.0 * dparams.q * d_Field[index(3, x, y)] * d1fd1x[indexd1(0, 7, x, y)] + 2.0 * dparams.q * d_Field[index(4, x, y)] * d1fd1x[indexd1(1, 7, x, y)] + dparams.q * d_Field[index(7, x, y)] * (d1fd1x[indexd1(0, 3, x, y)] + d1fd1x[indexd1(1, 4, x, y)]) + 2.0 * dparams.hb2 * d_Field[index(6, x, y)] * (pow(d_Field[index(8, x, y)], 2) + pow(d_Field[index(9, x, y)], 2)) + 2.0 * dparams.hc * d_Field[index(8, x, y)];
	EnergyGradient[7] += dparams.ha * d_Field[index(7, x, y)] + dparams.hb1 * d_Field[index(7, x, y)] * (pow(d_Field[index(6, x, y)], 2) + pow(d_Field[index(7, x, y)], 2)) - d2fd2x[indexd2(0, 0, 7, x, y)] - d2fd2x[indexd2(1, 1, 7, x, y)] + pow(dparams.q, 2) * (pow(d_Field[index(3, x, y)], 2) + pow(d_Field[index(4, x, y)], 2) + pow(d_Field[index(5, x, y)], 2)) * d_Field[index(7, x, y)] - 2.0 * dparams.q * d_Field[index(3, x, y)] * d1fd1x[indexd1(0, 6, x, y)] - 2.0 * dparams.q * d_Field[index(4, x, y)] * d1fd1x[indexd1(1, 6, x, y)] - dparams.q * d_Field[index(6, x, y)] * (d1fd1x[indexd1(0, 3, x, y)] + d1fd1x[indexd1(1, 4, x, y)]) + 2.0 * dparams.hb2 * d_Field[index(7, x, y)] * (pow(d_Field[index(8, x, y)], 2) + pow(d_Field[index(9, x, y)], 2)) + 2.0 * dparams.hc * d_Field[index(9, x, y)];
	EnergyGradient[8] += dparams.ha * d_Field[index(8, x, y)] + dparams.hb1 * d_Field[index(8, x, y)] * (pow(d_Field[index(8, x, y)], 2) + pow(d_Field[index(9, x, y)], 2)) - d2fd2x[indexd2(0, 0, 8, x, y)] - d2fd2x[indexd2(1, 1, 8, x, y)] + pow(dparams.q, 2) * (pow(d_Field[index(3, x, y)], 2) + pow(d_Field[index(4, x, y)], 2) + pow(d_Field[index(5, x, y)], 2)) * d_Field[index(8, x, y)] + 2.0 * dparams.q * d_Field[index(3, x, y)] * d1fd1x[indexd1(0, 9, x, y)] + 2.0 * dparams.q * d_Field[index(4, x, y)] * d1fd1x[indexd1(1, 9, x, y)] + dparams.q * d_Field[index(9, x, y)] * (d1fd1x[indexd1(0, 3, x, y)] + d1fd1x[indexd1(1, 4, x, y)]) + 2.0 * dparams.hb2 * d_Field[index(8, x, y)] * (pow(d_Field[index(6, x, y)], 2) + pow(d_Field[index(7, x, y)], 2)) + 2.0 * dparams.hc * d_Field[index(6, x, y)];
	EnergyGradient[9] += dparams.ha * d_Field[index(9, x, y)] + dparams.hb1 * d_Field[index(9, x, y)] * (pow(d_Field[index(8, x, y)], 2) + pow(d_Field[index(9, x, y)], 2)) - d2fd2x[indexd2(0, 0, 9, x, y)] - d2fd2x[indexd2(1, 1, 9, x, y)] + pow(dparams.q, 2) * (pow(d_Field[index(3, x, y)], 2) + pow(d_Field[index(4, x, y)], 2) + pow(d_Field[index(5, x, y)], 2)) * d_Field[index(9, x, y)] - 2.0 * dparams.q * d_Field[index(3, x, y)] * d1fd1x[indexd1(0, 8, x, y)] - 2.0 * dparams.q * d_Field[index(4, x, y)] * d1fd1x[indexd1(1, 8, x, y)] - dparams.q * d_Field[index(8, x, y)] * (d1fd1x[indexd1(0, 3, x, y)] + d1fd1x[indexd1(1, 4, x, y)]) + 2.0 * dparams.hb2 * d_Field[index(9, x, y)] * (pow(d_Field[index(6, x, y)], 2) + pow(d_Field[index(7, x, y)], 2)) + 2.0 * dparams.hc * d_Field[index(7, x, y)];
	// Computes the gradient
	for (int a = 0; a < dparams.number_total_fields; a++) {
		d_EnergyGradient[index(a, x, y)] = EnergyGradient[a];
	}
	// Project the Skyrme field
	project_orthogonal_magnetization(d_EnergyGradient, d_Field, x, y);
	// Calculates the velocity at each lattice point
	for (int a = 0; a < dparams.number_total_fields; a++) {
		d_Velocity[index(a, x, y)] = -dparams.time_step * d_EnergyGradient[index(a, x, y)];
	}
}

__global__ void do_gradient_step_kernel(double* d_Velocity, double* d_Field, double* d1fd1x, double* d2fd2x, double* d_EnergyGradient) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		for (int a = 0; a < dparams.number_total_fields; a++) {
			compute_derivatve_first(d1fd1x, d_Field, a, x, y);
			compute_derivatve_second(d2fd2x, d_Field, a, x, y);
		}
		do_gradient_step(d_Velocity, d_Field, d_EnergyGradient, d1fd1x, d2fd2x, x, y);
	}
}

__host__ void do_gradient_step_wrapper(double* d_Velocity, double* d_Field, double* d1fd1x, double* d2fd2x, double* d_EnergyGradient, const Params& p) {
	kernels::uploadDeviceParams(p);
	do_gradient_step_kernel << <p.grid, p.block >> > (d_Velocity, d_Field, d1fd1x, d2fd2x, d_EnergyGradient);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Do RK4 k step ----------------
__device__ void do_rk4_kstep(double* d_k, double* d_Velocity, const double factor, double* d_l, const int x, const int y) { // Calculates the Energy variation of the Baby Skyrme model and modifies the d_Velocity
	for (int a = 0; a < dparams.number_total_fields; a++) {
		d_k[index(a, x, y)] = dparams.time_step * (d_Velocity[index(a, x, y)] + factor * d_l[index(a, x, y)]);
	}
}

__global__ void do_rk4_kstep_kernel(double* d_k, double* d_Velocity, const double factor, double* d_l) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		do_rk4_kstep(d_k, d_Velocity, factor, d_l, x, y);
	}
}

__host__ void do_rk4_kstep_wrapper(double* d_k, double* d_Velocity, const double factor, double* d_l, const Params& p) {
	kernels::uploadDeviceParams(p);
	do_rk4_kstep_kernel << <p.grid, p.block >> > (d_k, d_Velocity, factor, d_l);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Do RK4 l step ----------------
__device__ void do_rk4_lstep(double* d_Temp, double* d_Field, const double factor, double* d_k, const int x, const int y) { // Calculates the Energy variation of the Baby Skyrme model and modifies the d_Velocity
	for (int a = 0; a < dparams.number_total_fields; a++) {
		d_Temp[index(a, x, y)] = d_Field[index(a, x, y)] + factor * d_k[index(a, x, y)];
	}
}

__global__ void do_rk4_lstep_kernel(double* d_Temp, double* d_Field, const double factor, double* d_k) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		do_rk4_lstep(d_Temp, d_Field, factor, d_k, x, y);
	}
}

__host__ void do_rk4_lstep_wrapper(double* d_Temp, double* d_Field, const double factor, double* d_k, const Params& p) {
	kernels::uploadDeviceParams(p);
	do_rk4_lstep_kernel << <p.grid, p.block >> > (d_Temp, d_Field, factor, d_k);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Do RK4 ----------------
__device__ void do_rk4(double* d_Velocity, double* d_Field, double* d_k1, double* d_k2, double* d_k3, double* d_k4, double* d_l1, double* d_l2, double* d_l3, double* d_l4, double* d_grid, const int x, const int y) { // Calculates the Energy variation of the Baby Skyrme model and modifies the d_Velocity
	for (int a = 0; a < dparams.number_total_fields; a++) {
		d_Velocity[index(a, x, y)] += (1.0 / 6.0) * (d_l1[index(a, x, y)] + 2.0 * d_l2[index(a, x, y)] + 2.0 * d_l3[index(a, x, y)] + d_l4[index(a, x, y)]);
		d_Field[index(a, x, y)] += (1.0 / 6.0) * (d_k1[index(a, x, y)] + 2.0 * d_k2[index(a, x, y)] + 2.0 * d_k3[index(a, x, y)] + d_k4[index(a, x, y)]);
	}
	compute_norm_magnetization(d_Field, x, y);
	project_orthogonal_magnetization(d_Velocity, d_Field, x, y);
}

__global__ void do_rk4_kernel(double* d_Velocity, double* d_Field, double* d_k1, double* d_k2, double* d_k3, double* d_k4, double* d_l1, double* d_l2, double* d_l3, double* d_l4, double* d_grid) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		do_rk4(d_Velocity, d_Field, d_k1, d_k2, d_k3, d_k4, d_l1, d_l2, d_l3, d_l4, d_grid, x, y);
	}
}

__host__ void do_rk4_wrapper(double* d_Velocity, double* d_Field, double* d_k1, double* d_k2, double* d_k3, double* d_k4, double* d_l1, double* d_l2, double* d_l3, double* d_l4, double* d_grid, const Params& p) {
	kernels::uploadDeviceParams(p);
	do_rk4_kernel << <p.grid, p.block >> > (d_Velocity, d_Field, d_k1, d_k2, d_k3, d_k4, d_l1, d_l2, d_l3, d_l4, d_grid);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Do arrested Newton flow ----------------
double do_arrested_newton_flow(double* d_Velocity, double* d_Field, double* d1fd1x, double* d2fd2x, double* d_EnergyGradient, double* d_k1, double* d_k2, double* d_k3, double* d_k4, double* d_l1, double* d_l2, double* d_l3, double* d_l4, double* d_grid, double* d_Temp, double* d_en, double* d_entmp, double* d_gridsum, double* h_gridsum, double* d_maxeg, double* h_maxeg, Params& p) {

	// Computes initial energy
	double prev_energy = observables::compute_energy(d_en, d_entmp, d_Field, d1fd1x, d_gridsum, h_gridsum, p);

	// Compute the Runge-Kutta slopes
	do_rk4_kstep_wrapper(d_k1, d_Velocity, 0.0, d_Velocity, p);
	do_rk4_lstep_wrapper(d_Temp, d_Field, 0.0, d_Field, p);
	do_gradient_step_wrapper(d_l1, d_Temp, d1fd1x, d2fd2x, d_EnergyGradient, p);

	do_rk4_kstep_wrapper(d_k2, d_Velocity, 0.5, d_l1, p);
	do_rk4_lstep_wrapper(d_Temp, d_Field, 0.5, d_k1, p);
	do_gradient_step_wrapper(d_l2, d_Temp, d1fd1x, d2fd2x, d_EnergyGradient, p);

	do_rk4_kstep_wrapper(d_k3, d_Velocity, 0.5, d_l2, p);
	do_rk4_lstep_wrapper(d_Temp, d_Field, 0.5, d_k2, p);
	do_gradient_step_wrapper(d_l3, d_Temp, d1fd1x, d2fd2x, d_EnergyGradient, p);

	do_rk4_kstep_wrapper(d_k4, d_Velocity, 1.0, d_l3, p);
	do_rk4_lstep_wrapper(d_Temp, d_Field, 1.0, d_k3, p);
	do_gradient_step_wrapper(d_l4, d_Temp, d1fd1x, d2fd2x, d_EnergyGradient, p);

	// Calculates the field and velocity using a 4th order runge-kutta method
	do_rk4_wrapper(d_Velocity, d_Field, d_k1, d_k2, d_k3, d_k4, d_l1, d_l2, d_l3, d_l4, d_grid, p);

	// Calculates the new energy and arrests the flow if necessary
	double new_energy = observables::compute_energy(d_en, d_entmp, d_Field, d1fd1x, d_gridsum, h_gridsum, p);
	if (new_energy > prev_energy && p.killkinen == true) {
		arrest_velocity_wrapper(d_Velocity, p, true);
	}

	// Calculates the error value
	return compute_max_field(d_EnergyGradient, d_maxeg, h_maxeg, p);
}

double compute_max_field(double* d_EnergyGradient, double* d_maxeg, double* h_maxeg, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_max_kernel << <p.reduction_grid, p.reduction_block >> > (d_EnergyGradient, d_maxeg, p.dim_fields);
	CHECK(cudaMemcpy(h_maxeg, d_maxeg, p.n_fields_bytes, cudaMemcpyDeviceToHost));
	double max = 0;
	for (unsigned int i = 0; i < p.reduction_grid.x; i++) {
		if (h_maxeg[i] > max) {
			max = h_maxeg[i];
		}
	}
	return max;
}

double compute_max_density(double* d_en, double* d_entmp, double* d_gridmax, double* h_gridmax, const Params& p) {
	kernels::uploadDeviceParams(p);
	CHECK(cudaMemcpy(d_entmp, d_en, p.n_grid_bytes, cudaMemcpyDeviceToDevice));
	compute_max_kernel << <p.reduction_grid, p.reduction_block >> > (d_entmp, d_gridmax, p.dim_grid);
	CHECK(cudaMemcpy(h_gridmax, d_gridmax, p.n_grid_sum_bytes, cudaMemcpyDeviceToHost));
	double max = 0;
	for (unsigned int i = 0; i < p.reduction_grid.x; i++) {
		if (h_gridmax[i] > max) {
			max = h_gridmax[i];
		}
	}
	return max;
}

double compute_min_density(double* d_en, double* d_entmp, double* d_gridmax, double* h_gridmax, const Params& p) {
	kernels::uploadDeviceParams(p);
	CHECK(cudaMemcpy(d_entmp, d_en, p.n_grid_bytes, cudaMemcpyDeviceToDevice));
	compute_min_kernel << <p.reduction_grid, p.reduction_block >> > (d_entmp, d_gridmax, p.dim_grid);
	CHECK(cudaMemcpy(h_gridmax, d_gridmax, p.n_grid_sum_bytes, cudaMemcpyDeviceToHost));
	double min = 0.0;
	for (unsigned int i = 0; i < p.reduction_grid.x; i++) {
		if (h_gridmax[i] < min) {
			min = h_gridmax[i];
		}
	}
	return min;
}

// ---------------- Plot the energy density ----------------
__global__ void show_density_kernel(uchar4* d_out, double* d_en, double minEn, double maxEn) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		// MATLAB Jet RGB values
		const int idx = index(0, x, y);
		double hue = (d_en[idx] - minEn) / (maxEn - minEn);
		if (hue <= 1.0 / 8.0) {
			d_out[idx].x = 0.0; // red
			d_out[idx].y = 0.0; // green
			d_out[idx].z = clip((4.0 * hue + 0.5) * 255.0); // blue
		}
		if (hue > 1.0 / 8.0 && hue <= 3.0 / 8.0) {
			d_out[idx].x = 0.0; // red
			d_out[idx].y = clip((4.0 * hue - 0.5) * 255.0); // green
			d_out[idx].z = clip(1.0 * 255.0); // blue
		}
		if (hue > 3.0 / 8.0 && hue <= 5.0 / 8.0) {
			d_out[idx].x = clip((4.0 * hue - 3.0 / 2.0) * 255.0); // red
			d_out[idx].y = clip(1.0 * 255.0); // green
			d_out[idx].z = clip((-4.0 * hue + 5.0 / 2.0) * 255.0); // blue
		}
		if (hue > 5.0 / 8.0 && hue <= 7.0 / 8.0) {
			d_out[idx].x = clip(1.0 * 255.0); // red
			d_out[idx].y = clip((-4.0 * hue + 7.0 / 2.0) * 255.0); // green
			d_out[idx].z = 0.0; // blue
		}
		if (hue > 7.0 / 8.0) {
			d_out[idx].x = clip((-4.0 * hue + 9.0 / 2.0) * 255.0); // red
			d_out[idx].y = 0.0; // green
			d_out[idx].z = 0.0; // blue
		}
	}
}

void show_density_wrapper(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_energy_wrapper(d_en, d_entmp, d_Field, d1fd1x, d_gridsum, h_gridsum, p);
	double density_max = compute_max_density(d_en, d_entmp, d_gridsum, h_gridsum, p);
	double density_min = compute_min_density(d_en, d_entmp, d_gridsum, h_gridsum, p);
	show_density_kernel << <p.grid, p.block >> > (d_out, d_en, density_min, density_max);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Plot the superconducting OP1 density ----------------
void show_higgs1_wrapper(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d_gridsum, double* h_gridsum, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_norm_higgs1_wrapper(d_en, d_entmp, d_Field, d_gridsum, h_gridsum, p);
	double maxvalue = compute_max_density(d_en, d_entmp, d_gridsum, h_gridsum, p);
	double minvalue = compute_min_density(d_en, d_entmp, d_gridsum, h_gridsum, p);
	show_density_kernel << <p.grid, p.block >> > (d_out, d_en, minvalue, maxvalue);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Plot the superconducting OP2 density ----------------
void show_higgs2_wrapper(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d_gridsum, double* h_gridsum, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_norm_higgs2_wrapper(d_en, d_entmp, d_Field, d_gridsum, h_gridsum, p);
	double maxvalue = compute_max_density(d_en, d_entmp, d_gridsum, h_gridsum, p);
	double minvalue = compute_min_density(d_en, d_entmp, d_gridsum, h_gridsum, p);
	show_density_kernel << <p.grid, p.block >> > (d_out, d_en, minvalue, maxvalue);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Plot the magnetic flux density ----------------
void show_magnetic_flux_wrapper(uchar4* d_out, double* d_en, double* d_entmp, double* d_Field, double* d1fd1x, double* d_gridsum, double* h_gridsum, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_vortex_number_wrapper(2, d_en, d_entmp, d_Field, d1fd1x, d_gridsum, h_gridsum, p);
	double maxvalue = compute_max_density(d_en, d_entmp, d_gridsum, h_gridsum, p);
	double minvalue = compute_min_density(d_en, d_entmp, d_gridsum, h_gridsum, p);
	show_density_kernel << <p.grid, p.block >> > (d_out, d_en, minvalue, maxvalue);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Plot the magnetization ----------------
__device__ void show_magnetization(uchar4* d_out, double* d_Field, int x, int y) {
	// HSV values
	double hue = (1.0 / 2.0 + 1.0 / (2.0 * M_PI) * atan2(d_Field[index(0, x, y)], d_Field[index(1, x, y)])) * 360.0;
	double saturation = 1.0 / 2.0 - 1.0 / 2.0 * tanh(3.0 * (d_Field[index(2, x, y)] - 0.5));
	double value = d_Field[index(2, x, y)] + 1.0;
	// HSV2RGB
	double p, r, t, ff;
	if (hue >= 360.0) {
		hue = 0.0;
	}
	hue /= 60.0;
	long i = (long)hue;
	ff = hue - i;
	p = value * (1.0 - saturation);
	r = value * (1.0 - (saturation * ff));
	t = value * (1.0 - (saturation * (1.0 - ff)));
	double R, G, B;
	switch (i) {
	case 0:
		R = value;
		G = t;
		B = p;
		break;
	case 1:
		R = r;
		G = value;
		B = p;
		break;
	case 2:
		R = p;
		G = value;
		B = t;
		break;
	case 3:
		R = p;
		G = r;
		B = value;
		break;
	case 4:
		R = t;
		G = p;
		B = value;
		break;
	case 5:
	default:
		R = value;
		G = p;
		B = r;
		break;
	}
	const int idx = index(0, x, y);
	d_out[idx].x = clip(R * 255.0); // red
	d_out[idx].y = clip(G * 255.0); // green
	d_out[idx].z = clip(B * 255.0); // blue
}

__global__ void show_magnetization_kernel(uchar4* d_out, double* d_Field) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		show_magnetization(d_out, d_Field, x, y);
	}
}

void show_magnetization_wrapper(uchar4* d_out, double* d_Field, const Params& p) {
	kernels::uploadDeviceParams(p);
	show_magnetization_kernel << <p.grid, p.block >> > (d_out, d_Field);
	CHECK(cudaDeviceSynchronize());
}