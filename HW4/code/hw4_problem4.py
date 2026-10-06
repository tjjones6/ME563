"""
ME 563: Intermediate Fluid Dynamics
hw4_problem4.py

Author: Tyler Jones
Institution: Univeristy of Wisconsin-Madison
Last Edit: 10.06.2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
A_H = 1e-3      # amplitude ratio A/H
kH = 10.0       # nondimensional wavenumber
N_PERIODS = 10  # number of wave periods to integrate

x0 = 0.5                              # x0/H
depths = [0.0, -0.05, -0.10, -0.20]   # y0/H (first one is the required pathline)

def velocity(t, z):
    x, y = z
    phase = kH * x - t
    u = -A_H * np.sin(phase) * np.cosh(kH * (y + 1.0)) / np.sinh(kH)
    v = A_H * np.cos(phase) * np.sinh(kH * (y + 1.0)) / np.sinh(kH)
    return [u, v]

t_end = 2 * np.pi * N_PERIODS
t_eval = np.linspace(0.0, t_end, 4000)

paths = {}
for y0 in depths:
    sol = solve_ivp(velocity, (0.0, t_end), [x0, y0], method="DOP853",
                    t_eval=t_eval, rtol=1e-11, atol=1e-14)
    paths[y0] = sol.y

# Compare net drift per period with the second-order Stokes drift
# u_s/(omega H) = (A/H)^2 kH cosh(2kH(y*+1)) / (2 sinh^2 kH)
for y0, (x, y) in paths.items():
    drift_num = (x[-1] - x[0]) / N_PERIODS / A_H
    drift_th = (2 * np.pi * A_H * kH * np.cosh(2 * kH * (y0 + 1.0))
                / (2 * np.sinh(kH) ** 2))
    print(f"y0/H = {y0:+.2f}: drift per period / A = {drift_num:.4f} "
          f"(Stokes theory {drift_th:.4f})")

# Plot
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

# (a) Required pathline, starting at (x0, y0) = (0.5, 0), on true axes
x, y = paths[0.0]
ax1.plot(x, y, color=colors[0], lw=1.5)
ax1.plot(x[0], y[0], "o", color="k", ms=6, label="start, $t^*=0$")
ax1.set_xlabel(r"$x/H$")
ax1.set_ylabel(r"$y/H$")
ax1.set_title(rf"(a) Pathline from $(x_0, y_0)/H = (0.5, 0)$, {N_PERIODS} periods")
ax1.set_aspect("equal", adjustable="datalim")
ax1.ticklabel_format(useOffset=False)
ax1.legend(loc="upper right")
ax1.grid(True, color="0.9")

# (b) Pathlines at several depths, displacement scaled by A
for c, y0 in zip(colors, depths):
    x, y = paths[y0]
    ax2.plot((x - x0) / A_H, (y - y0) / A_H, color=c, lw=1.5,
             label=rf"$y_0/H = {y0:g}$")
ax2.set_xlabel(r"$(x - x_0)/A$")
ax2.set_ylabel(r"$(y - y_0)/A$")
ax2.set_title("(b) Particle displacement at several depths")
ax2.set_aspect("equal", adjustable="datalim")
ax2.legend(loc="upper right")
ax2.grid(True, color="0.9")

fig.suptitle(rf"Water-wave pathlines, $A/H = 10^{{-3}}$, $kH = {kH:g}$")
fig.tight_layout()
fig.savefig("../figures/hw4_p4_pathlines.png", dpi=600)
plt.show()
