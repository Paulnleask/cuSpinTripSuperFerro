// =========================
// device_params.cu
// =========================
#include "device_params.cuh"

__device__ __constant__ DeviceParams dparams;

// ---------------- Copies parameters to device ----------------
cudaError_t setDeviceParams(const DeviceParams& h) {
    // This must be compiled by NVCC so the symbol form is accepted
    return cudaMemcpyToSymbol(dparams, &h, sizeof(DeviceParams), 0, cudaMemcpyHostToDevice);
}
