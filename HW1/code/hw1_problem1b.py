"""
ME 563: Intermediate Fluid Dynamics
hw1_problem1b.py

Author: Tyler Jones
Institution: Univeristy of Wisconsin-Madison
Last Edit: 09.13.2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("pump_data.xlsx")
print(data.head())

g = 9.81

# Unit conversions to SI
GPM_TO_M3S = 3.785411784e-3 / 60.0 
FT_TO_M = 0.3048
RPM_TO_RADS = 2 * np.pi / 60.0

Q = data["Flowrate_gpm"].to_numpy() * GPM_TO_M3S     # m^3/s
H = data["Head_ft"].to_numpy() * FT_TO_M             # m
Omega = data["Speed_rpm"].to_numpy() * RPM_TO_RADS   # rad/s

d = 1.0
flow_coeff = Q / (Omega * d**3)
head_coeff = g * H / (Omega**2 * d**2)

plt.figure(figsize=(12, 8))
for speed, group in data.groupby("Speed_rpm"):
    mask = (data["Speed_rpm"] == speed).to_numpy()
    order = np.argsort(flow_coeff[mask])
    plt.plot(flow_coeff[mask][order], head_coeff[mask][order],
              "o-", label=f"{speed} rpm")

plt.xlabel(r"Flow coefficient, $\phi = \frac{Q}{\Omega d^3}$")
plt.ylabel(r"Head coefficient, $\psi = \frac{gH}{\Omega^2 d^2}$")
plt.title("Nondimensional Pump Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/pump_nondim.png", dpi=600)
plt.show()