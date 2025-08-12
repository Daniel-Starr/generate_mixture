
# -*- coding: utf-8 -*-
"""
Batch-generate absorbance spectra (Voigt profile) from HITRAN HDF5 (lbl) files.
Folder layout assumed (relative to this script):
    ./gas_hdf5/   -> contains *.hdf5 (e.g., no.hdf5, no2.hdf5, so2.hdf5, cs2.hdf5)
    ./output/     -> results will be written here

What it does per file:
    - Auto-detect molecule_id from the HDF5 "lbl" dataset (uses the most frequent ID)
    - Compute line-broadening (Lorentz from gamma_air & n_air; Doppler from molar mass)
    - Combine Doppler with instrument Gaussian (FWHM) in quadrature -> Voigt
    - Accumulate sigma(ν) on a grid using local windows (fast) with optional S-intensity cutoff
    - Convert to absorbance with Beer–Lambert for each concentration in PPM_LIST
    - Save one Excel per gas (multiple concentration columns) + PNG plot

Requires: numpy, pandas, h5py, scipy, matplotlib
Run:  python batch_voigt_from_hitran.py
"""

import os, sys, math, glob
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy.special import wofz
import matplotlib.pyplot as plt

# --------------------------- USER CONFIG ---------------------------
# Relative to this script's directory (adapt if you run elsewhere)
BASE_DIR = Path(__file__).resolve().parent
IN_DIR   = BASE_DIR / "E:\generate_mixture\gas_hdf5"
OUT_DIR  = BASE_DIR / "output_hdf5"

# Spectral settings
NU_MIN, NU_MAX, DNU = 400.0, 4000.0, 0.2       # cm^-1 grid (use 0.1 for denser grid)
INSTR_FWHM = 1.0                                # cm^-1 instrument resolution (Gaussian FWHM)数值越大，谱线越“胖”，细节被抹平；越小，谱线更尖锐
S_MIN = 1e-27                                   # line intensity cutoff to speed up (set None to disable)阈值越大，保留的线越少，速度越快，但极弱线/远翼会被丢掉；阈值越小或设 None，更准更全，但更慢

# Conditions
T_K = 296.0      #室温 296 K
P_ATM = 1.0      # 1 个大气压
PPM_LIST = [10.0]                               # concentrations to generate (ppm)
PATH_LEN_CM = 500.0                             # 5 m = 500 cm 气体池长度

# Optional: override molar mass (g/mol) if you know your molecule
# Keys can be either molec_id (int) or lowercase filename stem ("no","no2","so2","cs2")
MOLAR_MASS = {
    8: 30.0061,    # NO (14N16O)
    10: 46.0055,   # NO2
    9: 64.066,     # SO2
    "no": 30.0061,
    "no2": 46.0055,
    "so2": 64.066,
    "cs2": 76.14,  # CS2
}

# ------------------------------------------------------------------

def number_density_cm3(T_K=296.0, p_atm=1.0):
    k_B = 1.380649e-23; p_Pa = p_atm*101325.0
    return (p_Pa/(k_B*T_K))/1e6

def gaussian_sigma_from_FWHM(FWHM):
    return FWHM/(2.0*np.sqrt(2.0*np.log(2.0)))

def doppler_sigma_wavenumber(nu0_cm, T_K, molar_mass_g_mol):
    Na = 6.02214076e23
    m_kg = (molar_mass_g_mol/1000.0)/Na
    k_B = 1.380649e-23; c = 299792458.0
    sigma_m_inv = (nu0_cm*100.0) * np.sqrt(k_B*T_K/(m_kg*c*c))  # m^-1
    return sigma_m_inv/100.0  # cm^-1

def voigt_profile(nu, nu0, sigma_g, gamma_l):
    z = ((nu - nu0) + 1j*gamma_l) / (sigma_g*np.sqrt(2.0))
    return np.real(wofz(z)) / (sigma_g*np.sqrt(2.0*np.pi))

def most_frequent(arr):
    vals, counts = np.unique(arr, return_counts=True)
    return vals[counts.argmax()]

def load_lbl_to_df(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "lbl" not in f:
            raise RuntimeError(f"'lbl' dataset not found in {h5_path}")
        arr = f["lbl"][:]
    cols = arr.dtype.names
    df = pd.DataFrame({c: arr[c] for c in cols})
    return df

def infer_molar_mass(molec_id, file_stem):
    # Prioritize explicit mapping by id, then by filename stem, else default 44.0 g/mol
    if molec_id in MOLAR_MASS:
        return MOLAR_MASS[molec_id]
    key = file_stem.lower()
    if key in MOLAR_MASS:
        return MOLAR_MASS[key]
    return 44.0

def compute_sigma_voigt(df, molec_id, molar_mass, nu_min, nu_max, dnu, T_K, p_atm, instr_fwhm, s_min):
    # Filter by molecule and window
    if "molec_id" in df.columns:
        df = df[df["molec_id"] == molec_id].copy()
    df = df[(df["nu"] >= nu_min-5) & (df["nu"] <= nu_max+5)].copy()
    if s_min is not None and "sw" in df.columns:
        df = df[df["sw"] >= s_min].copy()
    df.sort_values("nu", inplace=True)

    # Extract columns with fallbacks
    nu0 = df["nu"].values
    S = df["sw"].values
    gamma_air = df.get("gamma_air", pd.Series(np.full(len(df), 0.05))).values  # reasonable default
    n_air = df.get("n_air", pd.Series(np.zeros(len(df)))).values
    delta_air = df.get("delta_air", pd.Series(np.zeros(len(df)))).values

    # Line shape params
    Tref=296.0; pref=1.0
    gamma_l = gamma_air * (p_atm/pref) * (Tref/T_K)**(n_air)
    nu0_eff = nu0 + delta_air * (p_atm/pref)

    sigma_D = doppler_sigma_wavenumber(nu0_eff, T_K, molar_mass)
    sigma_instr = gaussian_sigma_from_FWHM(instr_fwhm)
    sigma_eff = np.sqrt(sigma_D**2 + sigma_instr**2)

    # Grid
    nu = np.arange(nu_min, nu_max + dnu, dnu)
    sigma_total = np.zeros_like(nu)

    # Local window accumulation
    for j in range(len(nu0_eff)):
        center = nu0_eff[j]; gL = gamma_l[j]; sG = sigma_eff[j]
        half = max(1.0, 10.0*gL + 6.0*sG)
        i1 = max(0, int(np.floor((center - half - nu_min)/dnu)))
        i2 = min(len(nu)-1, int(np.ceil((center + half - nu_min)/dnu)))
        if i2 <= i1: 
            continue
        seg = nu[i1:i2+1]
        sigma_total[i1:i2+1] += S[j] * voigt_profile(seg, center, sG, gL)

    return nu, sigma_total, len(df)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(IN_DIR.glob("*.hdf5"))
    if not files:
        print(f"No HDF5 found in {IN_DIR}")
        return

    n_tot = number_density_cm3(T_K, P_ATM)
    print(f"Total number density @ {T_K} K, {P_ATM} atm: {n_tot:.3e} molecule/cm^3")
    print(f"Files: {[f.name for f in files]}")

    for fp in files:
        print(f"\n=== Processing {fp.name} ===")
        df = load_lbl_to_df(fp)
        # Auto-detect molecule id
        if "molec_id" in df.columns:
            mol_id = int(most_frequent(df["molec_id"].values))
        else:
            mol_id = -1
        molar_mass = infer_molar_mass(mol_id, fp.stem)
        print(f"Detected molecule_id={mol_id}, using molar mass={molar_mass} g/mol")

        nu, sigma_nu, used = compute_sigma_voigt(df, mol_id, molar_mass,
                                                 NU_MIN, NU_MAX, DNU, T_K, P_ATM,
                                                 INSTR_FWHM, S_MIN)
        # Prepare DataFrame with ν column
        out_df = pd.DataFrame({"wavenumber_cm^-1": nu})
        # Add absorbance columns for each ppm
        for ppm in PPM_LIST:
            n_species = n_tot * (ppm * 1e-6)
            A = sigma_nu * n_species * PATH_LEN_CM / math.log(10.0)
            col = f"A_{ppm:g}ppm_{int(PATH_LEN_CM)}cm_VOIGT"
            out_df[col] = A

        # Save Excel
        excel_name = f"{fp.stem}_VOIGT_{int(NU_MIN)}-{int(NU_MAX)}cm-1_{int(INSTR_FWHM)}cm-1.xlsx"
        excel_path = OUT_DIR / excel_name
        out_df.to_excel(excel_path, index=False)
        print(f"Saved Excel: {excel_path}  (lines used: {used})")

        # Save PNG (first concentration only)
        first_col = [c for c in out_df.columns if c.startswith("A_")][0]
        plt.figure(figsize=(10,5))
        plt.plot(out_df["wavenumber_cm^-1"], out_df[first_col], linewidth=1)
        plt.xlabel("Wavenumber (cm$^{-1}$)")
        plt.ylabel(f"Absorbance ({first_col.split('_')[1]})")
        plt.title(f"{fp.stem.upper()} (Voigt, {INSTR_FWHM} cm$^{{-1}}$ instr.)")
        plt.xlim(NU_MIN, NU_MAX)
        plt.tight_layout()
        png_path = OUT_DIR / (excel_name.replace(".xlsx", ".png"))
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved Plot:  {png_path}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
