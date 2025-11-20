import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import zoom

# Font setting
import matplotlib.font_manager as font_manager
matplotlib.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=matplotlib.get_data_path() + '/fonts/ttf/cmr10.ttf')
matplotlib.rcParams['font.serif']=cmfont.get_name()
matplotlib.rcParams['mathtext.fontset']='cm'
matplotlib.rcParams['axes.unicode_minus']=False

# Data loader
def load_dat(path):
    return np.loadtxt(path)

# Bicubic interpolation using scipy
def bicubic_upscale(arr, factor=2):
    return zoom(arr, zoom=factor, order=4)

# Creates top-down surface plot panel using pcolormesh 
def draw_panel(ax, X, Y, Z, title, xlen, ylen, xlim=None, ylim=None, cmap="jet"):
    # (works with nonuniform grids X,Y shaped like Z)
    pc = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap)
    ax.set_aspect(xlen / ylen)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_facecolor("none")
    plt.colorbar(pc, ax=ax)
    ax.grid(False)
    plt.axes(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return pc

# Load lattice
latticePoints = load_dat("output/Higgs_LatticePoints.dat")
latticeVectors = load_dat("output/Higgs_LatticeVectors.dat")

xSize = int(latticePoints[0])
ySize = int(latticePoints[1])
X1 = np.array([latticeVectors[0], latticeVectors[1]])
X2 = np.array([latticeVectors[2], latticeVectors[3]])

# Load grid
xGrid = load_dat("output/Higgs_yGrid.dat")
yGrid = load_dat("output/Higgs_xGrid.dat")

xMin, xMax = np.min(xGrid), np.max(xGrid)
yMin, yMax = np.min(yGrid), np.max(yGrid)
xLength = xMax - xMin
yLength = yMax - yMin

# Spacing estimates (not strictly needed after interpolation)
xSpacing = np.abs(xGrid[0, 0] - xGrid[1, 0])
ySpacing = np.abs(yGrid[0, 0] - yGrid[0, 1])

# Load fields
Field1 = load_dat("output/Higgs_Field1.dat")
Field2 = load_dat("output/Higgs_Field2.dat")
Field3 = load_dat("output/Higgs_Field3.dat")
Field4 = load_dat("output/Higgs_Field4.dat")
MagnetField1 = load_dat("output/Magnet_Field1.dat")
MagnetField2 = load_dat("output/Magnet_Field2.dat")
MagnetField3 = load_dat("output/Magnet_Field3.dat")
Supercurrent1 = load_dat("output/J_Supercurrent1.dat")
Supercurrent2 = load_dat("output/J_Supercurrent2.dat")
Supercurrent3 = load_dat("output/J_Supercurrent3.dat")
GaugeField1 = load_dat("output/Gauge_Field1.dat")
GaugeField2 = load_dat("output/Gauge_Field2.dat")
GaugeField3 = load_dat("output/Gauge_Field3.dat")
EnergyDensity = load_dat("output/Higgs_EnergyDensity.dat")
ChargeDensity = load_dat("output/Magnet_ChargeDensity.dat")
MagneticFluxDensityXY = load_dat("output/Higgs_ChargeDensity.dat")

# ---------- totals ----------
E_total = np.sum(EnergyDensity)
Phi_total = -np.sum(MagneticFluxDensityXY)
B_total=np.sum(ChargeDensity)
print(f"Energy={E_total:0.4f}")
print(f"Vortex={Phi_total:0.4f}")
print(f"Skyrmion={B_total:0.4f}")

# 2D bicubic interpolations
xGrid = bicubic_upscale(xGrid, 2)
yGrid = bicubic_upscale(yGrid, 2)
Field1 = bicubic_upscale(Field1, 2)
Field2 = bicubic_upscale(Field2, 2)
Field3 = bicubic_upscale(Field3, 2)
Field4 = bicubic_upscale(Field4, 2)
MagnetField1 = bicubic_upscale(MagnetField1, 2)
MagnetField2 = bicubic_upscale(MagnetField2, 2)
MagnetField3 = bicubic_upscale(MagnetField3, 2)
Supercurrent1 = bicubic_upscale(Supercurrent1, 2)
Supercurrent2 = bicubic_upscale(Supercurrent2, 2)
Supercurrent3 = bicubic_upscale(Supercurrent3, 2)
GaugeField1 = bicubic_upscale(GaugeField1, 2)
GaugeField2 = bicubic_upscale(GaugeField2, 2)
GaugeField3 = bicubic_upscale(GaugeField3, 2)
EnergyDensity = bicubic_upscale(EnergyDensity, 2)
ChargeDensity = bicubic_upscale(ChargeDensity, 2)
MagneticFluxDensityXY = bicubic_upscale(MagneticFluxDensityXY, 2)

# Define modulus of Higgs field
psi1 = Field1 * Field1 + Field2 * Field2
psi2 = Field3 * Field3 + Field4 * Field4

# Runge color construction in HSV
H = 0.5 + (1.0 / (2.0 * np.pi)) * np.arctan2(MagnetField1, MagnetField2) # hue
S = 0.5 - 0.5 * np.tanh(3.0 * (MagnetField3 - 0.5))                      # saturation
V = MagnetField3 + 1.0                                                   # value
# Stack HSV -> RGB
rgbOutput = np.clip(hsv_to_rgb(np.stack([H, S, V], axis=-1)), 0, 1)

# ---------- Figure 1: densities ----------
fig1 = plt.figure(figsize=(10, 5), dpi=500)
gs = fig1.add_gridspec(2, 3, wspace=0.25, hspace=0.35)

# Energy density
ax11 = fig1.add_subplot(gs[0, 0])
draw_panel(ax11, xGrid, yGrid, EnergyDensity, r"$\mathcal{E}(\vec{x})$", xLength, yLength, xlim=(xMin,xMax), ylim=(yMin,yMax))

# Topological charge density
ax12 = fig1.add_subplot(gs[0, 1])
draw_panel(ax12, xGrid, yGrid, ChargeDensity, r"$n(\vec{x})$", xLength, yLength, xlim=(xMin,xMax), ylim=(yMin,yMax))

# Runge coloring (use the RGB image directly)
ax13 = fig1.add_subplot(gs[0, 2])
im = ax13.imshow(rgbOutput, origin="lower", extent=[xMin, xMax, yMin, yMax], interpolation="nearest")
ax13.set_aspect(xLength / yLength)
ax13.set_xlim(xMin, xMax)
ax13.set_ylim(yMin, yMax)
ax13.set_title(r"$\vec{m}(\vec{x})$")
ax13.set_facecolor("none")
for spine in ax13.spines.values():
    spine.set_visible(False)

# Plots the colorbar for the Runge sphere coloring
inset = fig1.add_axes([0.89, 0.59, 0.025, 0.32])
ps_n = 1000
ps_theta = np.linspace(-np.pi, np.pi, ps_n)
ps_sigma = np.linspace(-1.0, 1.0, ps_n // 2)
ps_h = 0.5 + (1.0 / (2.0 * np.pi)) * ps_theta
ps_s = 0.5 - 0.5 * np.tanh(3.0 * (ps_sigma - 0.5))
ps_v = ps_sigma + 1.0
ps_S, ps_H = np.meshgrid(ps_s, ps_h)
ps_V, _ = np.meshgrid(ps_v, ps_h)
ps_rgb = np.clip(hsv_to_rgb(np.stack([ps_H, ps_S, ps_V], axis=-1)), 0, 1)
# Display with axes
inset.imshow(ps_rgb, origin="lower", extent=[ps_sigma.min(), ps_sigma.max(), ps_theta.min(), ps_theta.max()], interpolation="nearest")
inset.set_xlabel(r"$m_z$")
inset.set_ylabel(r"$\arg(m_x+im_y)$")
inset.yaxis.set_label_position("right")
inset.yaxis.tick_right()
inset.set_yticks([-np.pi, np.pi])
inset.set_yticklabels(["0", r"$2\pi$"])
inset.set_xticks([-1, 1])
for spine in inset.spines.values():
    spine.set_visible(True)

# |psi1|^2 density
ax21 = fig1.add_subplot(gs[1, 0])
draw_panel(ax21, xGrid, yGrid, psi1, r"$|\psi_1(\vec{x})|^2$", xLength, yLength, xlim=(xMin,xMax), ylim=(yMin,yMax))

# |psi1|^2 density
ax22 = fig1.add_subplot(gs[1, 1])
draw_panel(ax22, xGrid, yGrid, psi2, r"$|\psi_2(\vec{x})|^2$", xLength, yLength, xlim=(xMin,xMax), ylim=(yMin,yMax))

# Magnetic flux density (with -/(2π))
ax23 = fig1.add_subplot(gs[1, 2])
draw_panel(ax23, xGrid, yGrid, MagneticFluxDensityXY, r"$\Phi(\vec{x})$", xLength, yLength, xlim=(xMin,xMax), ylim=(yMin,yMax), cmap="bone")

# Supercurrent[3]
# ax23 = fig1.add_subplot(gs[1, 2])
# draw_panel(ax23, xGrid, yGrid, Supercurrent3, r"$J_3(\vec{x})$", xLength, yLength, xlim=(xMin,xMax), ylim=(yMin,yMax))

# Plots and labels the figure
fig1.suptitle(f"Energy={E_total:0.4f}, Vortex={Phi_total:0.4f}, Skyrmion={B_total:0.4f}")
fig1.savefig("Plot_Densities.png", dpi=500, bbox_inches="tight")

# ---------- Figure 2: Magnetization components ----------
fig2, axs2 = plt.subplots(1, 3, figsize=(8, 2), dpi=100)
draw_panel(axs2[0], xGrid, yGrid, MagnetField1, r"$m_1(\vec{x})$", xLength, yLength)
draw_panel(axs2[1], xGrid, yGrid, MagnetField2, r"$m_2(\vec{x})$", xLength, yLength)
draw_panel(axs2[2], xGrid, yGrid, MagnetField3, r"$m_3(\vec{x})$", xLength, yLength)
fig2.tight_layout()
fig2.savefig("Plot_MagnetField.png", dpi=500, bbox_inches="tight")

# ---------- Figure 3: Gauge field components ----------
fig3, axs3 = plt.subplots(1, 3, figsize=(8, 2), dpi=100)
draw_panel(axs3[0], xGrid, yGrid, GaugeField1, r"$A_1(\vec{x})$", xLength, yLength)
draw_panel(axs3[1], xGrid, yGrid, GaugeField2, r"$A_2(\vec{x})$", xLength, yLength)
draw_panel(axs3[2], xGrid, yGrid, GaugeField3, r"$A_3(\vec{x})$", xLength, yLength)
fig3.tight_layout()
fig3.savefig("Plot_GaugeField.png", dpi=500, bbox_inches="tight")