#### This python code generates the three Fermi surfaces of Strontium Ruthenate
#### using a tight binding model based on Eq. 2 of https://arxiv.org/pdf/1208.6344 and Eq. 1 of https://arxiv.org/pdf/1305.2317


import numpy as np

def idx(x, y, orb, Nx, Ny):
    """Map (x,y,orbital) → Hamiltonian index"""
    site = x + Nx * y
    return 3 * site + orb


def build_tb_hamiltonian(Nx, Ny,
                          t1=0.145, t2=0.016,
                          t3=0.081, t4=0.039,
                          t5=0.000, t6=0.005,
                          mu_xz=0.122, mu_yz=0.122, mu_xy=0.122,
                          periodic=True):
    """
    Real-space tight-binding Hamiltonian for Sr2RuO4
    Spinless, 3 orbitals per site
    """

    Nsite = Nx * Ny
    Norb = 3
    dim = Nsite * Norb
    H = np.zeros((dim, dim), dtype=np.complex128)

    def inside(x, y):
        return 0 <= x < Nx and 0 <= y < Ny

    def hop(i, j, val):
        H[i, j] += val
        H[j, i] += np.conj(val)

    # Loop over lattice sites
    for x in range(Nx):
        for y in range(Ny):

            # On-site chemical potentials
            H[idx(x,y,0,Nx,Ny), idx(x,y,0,Nx,Ny)] -= mu_xz
            H[idx(x,y,1,Nx,Ny), idx(x,y,1,Nx,Ny)] -= mu_yz
            H[idx(x,y,2,Nx,Ny), idx(x,y,2,Nx,Ny)] -= mu_xy

            # Nearest neighbors
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                xx, yy = x + dx, y + dy
                if periodic:
                    xx %= Nx
                    yy %= Ny

                if inside(xx, yy):
                    # xz orbital
                    if dx != 0:
                        hop(idx(x,y,0,Nx,Ny), idx(xx,yy,0,Nx,Ny), -t1)
                    else:
                        hop(idx(x,y,0,Nx,Ny), idx(xx,yy,0,Nx,Ny), -t2)

                    # yz orbital
                    if dy != 0:
                        hop(idx(x,y,1,Nx,Ny), idx(xx,yy,1,Nx,Ny), -t1)
                    else:
                        hop(idx(x,y,1,Nx,Ny), idx(xx,yy,1,Nx,Ny), -t2)

                    # xy orbital
                    hop(idx(x,y,2,Nx,Ny), idx(xx,yy,2,Nx,Ny), -t3)

            # Next-nearest neighbors for xy
            for dx, dy in [(1,1), (1,-1), (-1,1), (-1,-1)]:
                xx, yy = x + dx, y + dy
                if periodic:
                    xx %= Nx
                    yy %= Ny

                if inside(xx, yy):
                    hop(idx(x,y,2,Nx,Ny), idx(xx,yy,2,Nx,Ny), -t4)

            # xz–yz hybridization (diagonal, sign structure)
            for dx, dy, sgn in [
                ( 1, 1,  1),
                ( 1,-1, -1),
                (-1, 1, -1),
                (-1,-1,  1),
            ]:
                xx, yy = x + dx, y + dy
                if periodic:
                    xx %= Nx
                    yy %= Ny

                if inside(xx, yy):
                    i = idx(xx,yy,0,Nx,Ny)  # xz
                    j = idx(x,y,1,Nx,Ny)    # yz
                    H[i,j] += -t6 * sgn
                    H[j,i] += -t6 * sgn

    return H

Nx, Ny = 10, 10
H = build_tb_hamiltonian(Nx, Ny)

evals = np.linalg.eigvalsh(H)
print("Lowest energies:", evals[:10])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Publication-style plotting defaults (LaTeX text rendering + larger fonts).
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "axes.linewidth": 1.3,
    "lines.linewidth": 2.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "savefig.dpi": 300,
})


### Parameters are based on Eq. 2 of https://arxiv.org/pdf/1208.6344 and Eq. 1 of https://arxiv.org/pdf/1305.2317
t1, t2, t3, t4, t5, t6 = 1.0, 0.1, 0.8, 0.35, 0.0, 0.1
### t1 is about 0.145 eV according to https://arxiv.org/pdf/1212.3994 (but it also uses spin-orbit coupling, which we neglect here)
# mumu = 0.122
mumu = 1.1
delta = -0.2 ## crystal field splitting of xy orbital
mu_xz, mu_yz, mu_xy = mumu, mumu, mumu

# ----------------------------
# k-grid
# ----------------------------
Nk = 400
kx = np.linspace(-np.pi, np.pi, Nk)
ky = np.linspace(-np.pi, np.pi, Nk)
KX, KY = np.meshgrid(kx, ky)

# ----------------------------
# Band energies
# ----------------------------
def bands(kx, ky):
    exz = -2*t1*np.cos(kx) - 2*t2*np.cos(ky) - mu_xz
    eyz = -2*t1*np.cos(ky) - 2*t2*np.cos(kx) - mu_yz
    exy = (-2*t3*(np.cos(kx)+np.cos(ky))
           -4*t4*np.cos(kx)*np.cos(ky) - 2*t5*(np.cos(2*kx)+np.cos(2*ky))
           + delta
           -mu_xy)

    g = -4*t6*np.sin(kx)*np.sin(ky)

    # Diagonalize 2x2 xz/yz block analytically
    e_plus  = 0.5*(exz + eyz) + np.sqrt((0.5*(exz - eyz))**2 + g**2)
    e_minus = 0.5*(exz + eyz) - np.sqrt((0.5*(exz - eyz))**2 + g**2)

    return e_minus, e_plus, exy

E1, E2, E3 = bands(KX, KY)

# ----------------------------
# Plot Fermi surface
# ----------------------------
fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)

ax.contour(KX, KY, E1, levels=[0], colors="#1f77b4", linewidths=2.6)
ax.contour(KX, KY, E2, levels=[0], colors="#d62728", linewidths=2.6, linestyles="dashdot")
ax.contour(KX, KY, E3, levels=[0], colors="#2ca02c", linewidths=2.6, linestyles="dotted")

ax.set_xlabel(r"$k_x a$", labelpad=6)
ax.set_ylabel(r"$k_y a$", labelpad=6)
# ax.set_title(r"Fermi Surface of Sr$_2$RuO$_4$", pad=10)
ax.set_aspect("equal")
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)

ticks = [-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi]
tick_labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
ax.set_xticks(ticks, tick_labels)
ax.set_yticks(ticks, tick_labels)
ax.tick_params(which="both", top=True, right=True)
ax.minorticks_on()
ax.grid(False)

legend_handles = [
    Line2D([0], [0], color="#1f77b4", lw=2.6, label=r"$\alpha$"),
    Line2D([0], [0], color="#d62728", lw=2.6, ls="-.", label=r"$\beta$"),
    Line2D([0], [0], color="#2ca02c", lw=2.6, ls=":", label=r"$\gamma$"),
]
ax.legend(handles=legend_handles, frameon=True, facecolor="white", edgecolor="black", framealpha=1.0, loc="center")

plt.savefig("strontium_ruthenate_fermi_surface.pdf", bbox_inches="tight", pad_inches=0.02)
plt.show()

