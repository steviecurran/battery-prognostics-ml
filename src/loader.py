import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter, find_peaks
import requests
from pathlib import Path

#infile = "../data/Oxford_Battery_Degradation_Dataset_1.mat"; mat_data = loadmat(infile)

DATA_URL = (
    "https://ora.ox.ac.uk/objects/"
    "uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac/"
    "files/mc247ds58f"
)

DATA_FILE = Path(__file__).parent.parent / "data" / "Oxford_Battery_Degradation_Dataset_1.mat"


def download_dataset():
    if DATA_FILE.exists():
        return

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Oxford Battery Degradation Dataset...")

    with requests.get(DATA_URL, stream=True) as r:
        r.raise_for_status()

        with open(DATA_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print("Download complete.")

def load_oxford_dataset(test):
    download_dataset()
    mat_data = loadmat(DATA_FILE)
    
    dfs = [] 
    for cell_no in range(1, 9):
        cell = mat_data[f"Cell{cell_no}"]

        for cycle_name in cell.dtype.names:
            cycle_number = int(cycle_name.replace("cyc", ""))
            cycle = cell[cycle_name][0, 0]
            measurement = cycle[test][0, 0]
            
            if test not in cycle.dtype.names:
                continue
                
            t = measurement["t"][0, 0].flatten()
            v = measurement["v"][0, 0].flatten()
            q = measurement["q"][0, 0].flatten()
            T = measurement["T"][0, 0].flatten()

            if len(t) == 0:
                continue

            time_min = (t - t[0]) * 24 * 60
            
            cycle_dict = {"Cell": cell_no,"Cycle": cycle_number,"Test": test, 
                "Time_min": time_min, "Voltage_V": v,"Charge_mAh": q,"Temperature_C": T
            }
            dfs.append(pd.DataFrame(cycle_dict))
          
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["Cell","Cycle", "Test","Time_min", "Voltage_V", "Charge_mAh", "Temperature_C"])
        
    return df    


# =====================================================================
# THE UNIFIED RAW SUB-GROUP FEATURE EXTRACTOR
# =====================================================================

def extract_advanced_features(group):
    """Calculates all advanced time-series, derivative, and threshold metrics
    by iterating through the raw cell/cycle arrays exactly once.
    """
    # Ensure raw measurements are strictly sequential
    group = group.sort_values("Time_min")

    # Determine direction dynamically per group to prevent global reference bugs
    is_discharge = str(group["Test"].iloc[0]).endswith("dc")

    # -------------------------------------------------------------
    # A. INITIAL TEMPERATURE RATE (dT/dt over early minutes)
    # -------------------------------------------------------------
    early_window = group[group["Time_min"] <= 5.0]
    if len(early_window) >= 3:
        x, y = early_window["Time_min"].values, early_window["Temperature_C"].values
        x_m, y_m = np.mean(x), np.mean(y)
        denom = np.sum((x - x_m) ** 2)
        dT_dt = np.sum((x - x_m) * (y - y_m)) / denom if denom != 0 else np.nan
    else:
        dT_dt = np.nan

    # -------------------------------------------------------------
    # B. THRESHOLD CROSSING TIMES (t_to_V) - FIXED FOR INDEX SAFETY
    # -------------------------------------------------------------
    cond_40 = group["Voltage_V"] <= 4.0 if is_discharge else group["Voltage_V"] >= 4.0
    cond_38 = group["Voltage_V"] <= 3.8 if is_discharge else group["Voltage_V"] >= 3.8
    cond_35 = group["Voltage_V"] <= 3.5 if is_discharge else group["Voltage_V"] >= 3.5

    t_40_val = group.loc[cond_40, "Time_min"].values[0] if cond_40.any() else np.nan
    t_38_val = group.loc[cond_38, "Time_min"].values[0] if cond_38.any() else np.nan
    t_35_val = group.loc[cond_35, "Time_min"].values[0] if cond_35.any() else np.nan

    # -------------------------------------------------------------
    # C. TIME-DOMAIN GRADIENTS
    # -------------------------------------------------------------
    if len(group) >= 5:
        dv_init = group["Voltage_V"].iloc[4] - group["Voltage_V"].iloc[0]
        dt_init = group["Time_min"].iloc[4] - group["Time_min"].iloc[0]
        init_gradient = dv_init / dt_init if dt_init != 0 else np.nan
    else:
        init_gradient = np.nan

    idx_38 = (group["Voltage_V"] - 3.8).abs().idxmin()
    idx_35 = (group["Voltage_V"] - 3.5).abs().idxmin()
    dt_window = group.loc[idx_35, "Time_min"] - group.loc[idx_38, "Time_min"]
    dv_window = group.loc[idx_35, "Voltage_V"] - group.loc[idx_38, "Voltage_V"]
    window_gradient = (
        dv_window / dt_window if abs(dv_window) > 0.2 and dt_window != 0 else np.nan
    )

    # -------------------------------------------------------------
    # D. NEW ADVANCED METRIC: SOLID-STATE dQ/dV PEAK MARKERS
    # -------------------------------------------------------------
    window = 15
    if len(group) <= window:
        window = len(group) if len(group) % 2 != 0 else len(group) - 1

    if window >= 5:
        smoothed_v = savgol_filter(group["Voltage_V"], window, polyorder=2)
    else:
        smoothed_v = group["Voltage_V"].values

    dV = np.diff(smoothed_v)
    dq = np.diff(group["Charge_mAh"])
    dq_dV = np.where(dV != 0, dq / dV, 0)

    # Find dominant electrochemical phase transformation peak
    peaks, _ = find_peaks(np.abs(dq_dV), prominence=5.0, distance=10)
    if len(peaks) > 0:
        highest_idx = peaks[np.argmax(np.abs(dq_dV[peaks]))]
        dq_dV_peak_height = dq_dV[highest_idx]
        dq_dV_peak_voltage = group["Voltage_V"].iloc[highest_idx]
    else:
        # FIXED: Resolved naming case-mismatch typo
        dq_dV_peak_height, dq_dV_peak_voltage = np.nan, np.nan

    # -------------------------------------------------------------
    # E. CYCLE ENERGY EXPORT (Wh) via Numerical Integration
    # -------------------------------------------------------------
    time_hours = group["Time_min"].values / 60.0
    voltage_vals = group["Voltage_V"].values
    current_amps = 0.2
    energy_wh = np.trapz(voltage_vals * current_amps, x=time_hours)

    # -------------------------------------------------------------
    # F. VOLTAGE AT FIXED CAPACITY (Overpotential Fingerprint)
    # -------------------------------------------------------------
    target_capacity = 300.0  
    idx_at_capacity = (group["Charge_mAh"].abs() - target_capacity).abs().idxmin()
    V_at_checkpoint = group.loc[idx_at_capacity, "Voltage_V"]

    # -------------------------------------------------------------
    # G. VOLTAGE-TIME INTEGRAL (VTI)
    # -------------------------------------------------------------
    time_vals_min = group["Time_min"].values
    v_time_integral = np.trapz(voltage_vals, x=time_vals_min)

    # Return everything neatly mapped inside a structured Pandas Series
    # FIXED: Unified all variable names to prevent case mismatches
    return pd.Series(
        {
            "dT_by_dt": dT_dt,
            "t_to_4.0_V": t_40_val,
            "t_to_3.8_V": t_38_val,
            "t_to_3.5_V": t_35_val,
            "Initial_V_slope": init_gradient,
            "3.8-3.5_V_slope": window_gradient,
            "dq_dV_peak_height": dq_dV_peak_height,
            "dq_dV_peak_voltage": dq_dV_peak_voltage,
            "cycle_energy_wh": energy_wh,
            "V_at_300mAh": V_at_checkpoint,
            "VTI": v_time_integral,
        }
    )

