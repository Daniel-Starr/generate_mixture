# -*- coding: utf-8 -*-
"""
NO2 → Voigt absorbance from HITRAN HDF5
- Reads HITRAN HDF5 (lbl) at: E:\generate_mixture\gas_hdf5\no2.hdf5
- Computes Voigt absorbance in 400.0-4000.0 cm^-1 with 0.2 cm^-1 grid
- Conditions: [10.0] ppm, L=500.0 cm, T=296.0 K, p=1.0 atm, instr_FWHM=1.0 cm^-1
- Saves Excel + PNG to: E:\generate_mixture\output_hdf5

Requires: numpy, pandas, h5py, scipy, matplotlib
Install:   pip install numpy pandas h5py scipy matplotlib
"""

import os, sys, math
import numpy as np
import pandas as pd
import h5py
from scipy.special import wofz
import matplotlib.pyplot as plt

# ------------------ USER SETTINGS ------------------
H5_PATH = r"E:\generate_mixture\gas_hdf5\NO2.hdf5"
OUT_DIR = r"../output_hdf5_test"
MOLEC_ID = None     # 10=NO2. If None -> auto-detect (must be unique in file)
MOLAR_MASS = 46.0055 # g/mol (used for Doppler broadening)

# Spectral range and grid
NU_MIN, NU_MAX, DNU = 400.0, 4000.0, 0.2
INSTR_FWHM = 1.0     # cm^-1 instrument resolution (Gaussian FWHM)
S_MIN = 1e-27               # intensity cutoff to speed up (None to disable)

# Conditions
T_K = 296.0
P_ATM = 1.0
PPM_LIST = [10.0]
L_CM = 500.0                 # path length in cm
# ---------------------------------------------------

def number_density_cm3(T_K=296.0, p_atm=1.0):
    k_B = 1.380649e-23; p_Pa = p_atm*101325.0
    return (p_Pa/(k_B*T_K))/1e6

def gaussian_sigma_from_FWHM(FWHM):
    return FWHM/(2.0*np.sqrt(2.0*np.log(2.0)))

def doppler_sigma_wavenumber(nu0_cm, T_K, molar_mass_g_mol):
    Na = 6.02214076e23
    m_kg = (molar_mass_g_mol/1000.0)/Na
    k_B = 1.380649e-23; c = 299792458.0
    sigma_m_inv = (nu0_cm*100.0) * np.sqrt(k_B*T_K/(m_kg*c*c))
    return sigma_m_inv/100.0

def voigt_profile(nu, nu0, sigma_g, gamma_l):
    z = ((nu - nu0) + 1j*gamma_l) / (sigma_g*np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma_g*np.sqrt(2.0*np.pi))

def load_df(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "lbl" not in f:
            raise RuntimeError("'lbl' dataset not found in HDF5")
        arr = f["lbl"][:]
    cols = arr.dtype.names
    return pd.DataFrame({c: arr[c] for c in cols})

def select_molecule(df, molec_id):
    present = sorted(df["molec_id"].unique().tolist())
    if molec_id is None:
        if len(present) != 1:
            raise RuntimeError(f"HDF5 contains multiple molecule IDs {present}. "
                               f"Please set MOLEC_ID explicitly.")
        molec_id = int(present[0])
    df2 = df[(df["molec_id"] == molec_id) & (df["nu"] >= NU_MIN-5) & (df["nu"] <= NU_MAX+5)].copy()
    if S_MIN is not None:
        df2 = df2[df2["sw"] >= S_MIN].copy()
    df2.sort_values("nu", inplace=True)
    return molec_id, present, df2

def compute_sigma_voigt(df2, molar_mass):
    nu0 = df2["nu"].values
    S = df2["sw"].values
    gamma_air = df2["gamma_air"].values
    n_air = df2["n_air"].values
    delta_air = df2["delta_air"].values

    Tref=296.0; pref=1.0
    gamma_l = gamma_air * (P_ATM/pref) * (Tref/T_K)**(n_air)
    nu0_eff = nu0 + delta_air * (P_ATM/pref)

    sigma_D = doppler_sigma_wavenumber(nu0_eff, T_K, molar_mass)
    sigma_instr = gaussian_sigma_from_FWHM(INSTR_FWHM)
    sigma_eff = np.sqrt(sigma_D**2 + sigma_instr**2)

    nu = np.arange(NU_MIN, NU_MAX + DNU, DNU)
    sigma_total = np.zeros_like(nu)

    for j in range(len(nu0_eff)):
        center = nu0_eff[j]; gL = gamma_l[j]; sG = sigma_eff[j]
        half = max(1.0, 10.0*gL + 6.0*sG)
        i1 = max(0, int(np.floor((center - half - NU_MIN)/DNU)))
        i2 = min(len(nu)-1, int(np.ceil((center + half - NU_MIN)/DNU)))
        if i2 <= i1: 
            continue
        seg = nu[i1:i2+1]
        sigma_total[i1:i2+1] += S[j] * voigt_profile(seg, center, sG, gL)

    return nu, sigma_total

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_df(H5_PATH)
    molec_id, present, df2 = select_molecule(df, MOLEC_ID)
    print(f"Present molecule IDs: {present}  |  Using molec_id={molec_id}")
    print(f"Lines used: {len(df2)}")
    nu, sigma_nu = compute_sigma_voigt(df2, MOLAR_MASS)

    n_tot = number_density_cm3(T_K, P_ATM)
    out = pd.DataFrame({ "wavenumber_cm^-1": nu })

    for ppm in PPM_LIST:
        n_species = n_tot * (ppm*1e-6)
        A = sigma_nu * n_species * L_CM / math.log(10.0)
        out[f"A_{ppm:g}ppm_{int(L_CM)}cm_VOIGT"] = A

    # Save
    base = os.path.splitext(os.path.basename(H5_PATH))[0]
    xlsx = os.path.join(OUT_DIR, f"{base}_VOIGT_{int(NU_MIN)}-{int(NU_MAX)}cm-1_{INSTR_FWHM:g}cm-1.xlsx")
    out.to_excel(xlsx, index=False)
    print("Saved Excel:", xlsx)

    # Plot first concentration
    first = [c for c in out.columns if c.startswith("A_")][0]
    plt.figure(figsize=(10,5))
    plt.plot(out["wavenumber_cm^-1"], out[first], linewidth=1)
    plt.xlabel("Wavenumber (cm^-1)"); plt.ylabel(f"Absorbance ({first.split('_')[1]})")
    plt.title(f"{base.upper()} (Voigt, {INSTR_FWHM} cm^-1 instr.)")
    plt.xlim(NU_MIN, NU_MAX); plt.tight_layout()
    png = xlsx.replace(".xlsx", ".png")
    plt.savefig(png, dpi=150); plt.close()
    print("Saved Plot:", png)

if __name__ == "__main__":
    main()
