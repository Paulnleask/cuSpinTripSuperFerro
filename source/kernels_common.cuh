// =========================
// kernels_common.cuh
// =========================
#pragma once
#include "device_params.cuh"

// ---------------- Thread coordinates ----------------
__device__ __forceinline__ int gx() { return blockIdx.x * blockDim.x + threadIdx.x; }
__device__ __forceinline__ int gy() { return blockIdx.y * blockDim.y + threadIdx.y; }

// ---------------- Bounds check ----------------
__device__ __forceinline__ bool inBounds(int x, int y) {
    return (unsigned)x < (unsigned)dparams.xlen && (unsigned)y < (unsigned)dparams.ylen;
}

// ---------------- Indexing by array unrolling - index(f, x, y) ----------------
__device__ __forceinline__ int index(int a, int i, int j) {
    return  j + i * dparams.ylen + a * dparams.xlen * dparams.ylen;
}

// ---------------- First derivative indexing - indexd1(c, f, x, y) ----------------
__device__ __forceinline__ int indexd1(int a, int b, int i, int j) {
    return  j + i * dparams.ylen + a * dparams.xlen * dparams.ylen + b * dparams.xlen * dparams.ylen * dparams.number_coordinates;
}

// ---------------- Second derivative indexing - indexd2(c, c, f, x, y) ----------------
__device__ __forceinline__ int indexd2(int a, int b, int c, int i, int j) {
    return  j + i * dparams.ylen + a * dparams.xlen * dparams.ylen + b * dparams.xlen * dparams.ylen * dparams.number_coordinates + c * dparams.xlen * dparams.ylen * dparams.number_coordinates * dparams.number_coordinates;
}

// ---------------- Normalize the magnetization field ----------------
__device__ void compute_norm_magnetization(double* d_Field, int x, int y) {
	double normalization_constant = 0.0;
	for (int a = 0; a < dparams.number_magnetization_fields; a++) {
		normalization_constant += pow(d_Field[index(a, x, y)], 2);
	}
	normalization_constant = sqrt(normalization_constant);
	for (int a = 0; a < dparams.number_magnetization_fields; a++) {
		d_Field[index(a, x, y)] *= dparams.M0 / normalization_constant;
	}
}

__global__ void compute_norm_magnetization_kernel(double* d_Field) {
	int x = gx();
	int y = gy();
	if (!inBounds(x,y)) {
		return;
	}
	else {
		compute_norm_magnetization(d_Field, x, y);
	}
}

__host__ void compute_norm_magnetization_wrapper(double* d_Field, const Params& p) {
	kernels::uploadDeviceParams(p);
	compute_norm_magnetization_kernel << <p.grid, p.block >> > (d_Field);
	CHECK(cudaDeviceSynchronize());
}

// ---------------- Project orthogonal to the magnetization field ----------------
__device__ void project_orthogonal_magnetization(double* function, double* d_Field, int x, int y) {
	double LagrangeMultiplier = 0.0;
	for (int a = 0; a < dparams.number_magnetization_fields; a++) {
		LagrangeMultiplier += function[index(a, x, y)] * d_Field[index(a, x, y)]; // Lagrange multiplier for the function
	}
	for (int a = 0; a < dparams.number_magnetization_fields; a++) {
		function[index(a, x, y)] -= LagrangeMultiplier * d_Field[index(a, x, y)]; // Ensures the function and magnetization are perpendicular
	}
}

// ---------------- Parallel block-wise reduction using shared memory ----------------
__global__ void compute_sum_kernel(double* d_var, double* d_gridsum, size_t size) {
	extern __shared__ double sdata[s_threads_per_block];
	// Thread ID
	const int tid = threadIdx.x;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x; // column
	sdata[tid] = d_var[idx];
	__syncthreads();
	// boundary check
	if (idx >= size) {
		return;
	}
	// in-place reduction in global memory
	int strideMax = (blockIdx.x < gridDim.x - 1 ? blockDim.x / 2 : (size - blockIdx.x * blockDim.x) / 2);
	for (int stride = strideMax; stride > 0; stride >>= 1) {
		if (tid < stride) {
			sdata[tid] += sdata[tid + stride];
		}
		__syncthreads();
	}
	// write result to global memory
	if (tid == 0) {
		d_gridsum[blockIdx.x] = sdata[0];
	}
}

__host__ double compute_sum_wrapper(double* d_var, double* d_gridsum, double* h_gridsum, const Params& p, size_t size) {
	kernels::uploadDeviceParams(p);
	compute_sum_kernel << <p.reduction_grid, p.reduction_block >> > (d_var, d_gridsum, size);
	CHECK(cudaMemcpy(h_gridsum, d_gridsum, p.n_grid_sum_bytes, cudaMemcpyDeviceToHost));
	double sum = 0;
	for (unsigned int i = 0; i < p.reduction_grid.x; i++) {
		sum += h_gridsum[i];
	}
	return sum;
}

// ---------------- Computes the maximum using shared memory ----------------
__global__ void compute_max_kernel(double* d_var, double* d_gridmax, size_t size) {
	extern __shared__ double sdata[s_threads_per_block];
	// Thread ID
	const int tid = threadIdx.x;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x; // column
	sdata[tid] = d_var[idx];
	__syncthreads();
	// boundary check
	if (idx >= size) {
		return;
	}
	// in-place reduction in global memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s && sdata[tid] < sdata[tid + s]) {
			sdata[tid] = sdata[tid + s];
		}
		__syncthreads();
	}
	// write result to global memory
	if (tid == 0) {
		d_gridmax[blockIdx.x] = sdata[0];
	}
}

// ---------------- Computes the minimum using shared memory ----------------
__global__ void compute_min_kernel(double* d_var, double* d_gridmax, size_t size) {
	extern __shared__ double sdata[s_threads_per_block];
	// Thread ID
	const int tid = threadIdx.x;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = d_var[idx];
	__syncthreads();
	// boundary check
	if (idx >= size) {
		return;
	}
	// in-place reduction in global memory
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s && sdata[tid] > sdata[tid + s]) {
			sdata[tid] = sdata[tid + s];
		}
		__syncthreads();
	}
	// write result to global memory
	if (tid == 0) {
		d_gridmax[blockIdx.x] = sdata[0];
	}
}