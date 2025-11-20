# 🚀 Simulating Composite Topological Excitations in Superconducting Ferromagnetic Metals with Triplet Pairing

A high-performance **CUDA + C++20** simulation project built with **CMake**, featuring GPU acceleration and OpenGL visualization via **GLEW** and **GLUT**.
Used to simulate **magnetic skyrmion-superconducting vortices** in two-band ferromagnetic metals with equal spin triplet pairing.
Can be used to describe ferromagnetic and superconducting states in materials such as **URhGe, UCoGe and UGe<sub>2</sub>**.

---

## 🧩 Features
- CUDA and modern C++ integration  
- Automatic source discovery (`*.cpp`, `*.cu`)  
- GLEW + GLUT for OpenGL rendering
- Click, drop and isorotate skyrmion-vortex-vortex pairs
- Include bilinear Josephson effects
- Configurable GPU architectures  
- Debug build support with GPU debug symbols  

---

## 📦 Requirements
| Dependency | Minimum Version | Notes |
|-------------|----------------|--------|
| **CMake** | 3.21 | or newer |
| **CUDA Toolkit** | 11.0+ | includes `nvcc` and runtime libraries |
| **GLEW** | — | OpenGL extensions |
| **GLUT** | — | Windowing and context management |
| **C++ Compiler** | C++20 capable | GCC 11+, Clang 13+, MSVC 2019+ |

---

## 🛠️ Building the Project

```bash
# 1️⃣ Clone the repository
git clone https://github.com/Paulnleask/cuSpinTripSuperFerro.git
cd cuSpinTripSuperFerro

# 2️⃣ Create a build directory
mkdir build

# 3️⃣ Configure with CMake
cmake --preset=release

# 4️⃣ Build
cmake --build --preset=release

# 5️⃣ Run
build\release\simulation.exe
