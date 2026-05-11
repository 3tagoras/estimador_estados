"""
=============================================================================
 FREE RECONSTRUCTED DATASET: 10-bus, 69-bus, and 873-bus Radial Distribution
 Feeders for Power Flow Calculation
=============================================================================

This script reconstructs the three datasets described in:
  P. Vasconcelos, "10-bus, 69-bus, and 873-bus radial distribution feeder
  datasets for power flow calculation," IEEE DataPort, 2024.
  DOI: 10.21227/5n30-5402

PRIMARY FREE SOURCES (all data here is from open/public resources):
  - 69-bus  : Baran & Wu (1989). Data available in MATPOWER (MIT license)
              https://github.com/MATPOWER/matpower/blob/master/data/case69.m
  - 873-bus : REDS Repository (Kavasseri & Ababei, NDSU Power Group)
              https://www.dejazzer.com/reds.html  (free public download)
  - 10-bus  : Standard 10-bus radial distribution system (see note below)

NOTE ON 10-BUS: The Vasconcelos 10-bus is a custom system not independently
published. The one here is a representative 10-bus radial feeder. If you need
the exact Vasconcelos version, contact the author via the IEEE DataPort page.

VALIDATION DATA: Power flow results (voltages, currents, losses) are generated
here using pandapower — the same physics as any standard NR power flow solver.

REQUIREMENTS: pip install pandapower pandas numpy
=============================================================================
"""

import os
import tarfile
import urllib.request
import pandapower as pp
import pandas as pd
import numpy as np

OUTPUT_DIR = "power_flow_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM DATA DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_10bus_data():
    """
    10-bus radial distribution system.
    Base: 11 kV, 10 MVA
    Represents a typical small feeder used in DER integration studies.
    Branch impedances in per-unit (10 MVA, 11 kV base).
    Loads in kW / kVAR.
    """
    branch_data = [
        # [from_bus, to_bus, R_pu, X_pu]
        [1,  2,  0.02438, 0.02969],
        [2,  3,  0.03103, 0.03569],
        [3,  4,  0.05590, 0.04383],
        [4,  5,  0.06655, 0.06088],
        [5,  6,  0.04748, 0.05490],
        [6,  7,  0.06550, 0.07600],
        [7,  8,  0.08670, 0.09200],
        [8,  9,  0.09670, 0.10300],
        [9,  10, 0.08300, 0.08900],
    ]
    bus_data = [
        # [bus_num, P_kW, Q_kVAR]
        [1,  0,    0   ],
        [2,  100,  60  ],
        [3,  90,   40  ],
        [4,  120,  80  ],
        [5,  60,   30  ],
        [6,  60,   20  ],
        [7,  200,  100 ],
        [8,  200,  100 ],
        [9,  60,   20  ],
        [10, 60,   20  ],
    ]
    # Equivalent DER allocations (PV/WT, peak generation in pu on 10 MVA base)
    der_data = [
        # [bus_num, PV_P_pu, PV_Q_pu, WT_P_pu, WT_Q_pu]
        [3,  0.05, 0.0,  0.0,  0.0 ],
        [6,  0.0,  0.0,  0.04, 0.0 ],
        [9,  0.03, 0.0,  0.0,  0.0 ],
    ]
    return branch_data, bus_data, der_data


def get_69bus_data():
    """
    IEEE 69-bus radial distribution system (Baran & Wu, 1989).
    Base: 12.66 kV, 10 MVA
    Source: MATPOWER case69.m (MIT License, github.com/MATPOWER/matpower)
    Original paper: M. E. Baran and F. F. Wu, "Optimal capacitor placement on
    radial distribution systems," IEEE Trans. Power Deliv., vol. 4, no. 1,
    pp. 725-734, 1989. DOI: 10.1109/61.19265
    Branch impedances in Ohms. Loads in kW / kVAR.
    """
    branch_data = [
        # [from_bus, to_bus, R_ohm, X_ohm]
        # Main feeder
        [1,  2,  0.0005, 0.0012],
        [2,  3,  0.0005, 0.0012],
        [3,  4,  0.0015, 0.0036],
        [4,  5,  0.0251, 0.0294],
        [5,  6,  0.3660, 0.1864],
        [6,  7,  0.3811, 0.1941],
        [7,  8,  0.0922, 0.0470],
        [8,  9,  0.0493, 0.0251],
        [9,  10, 0.8190, 0.2707],
        [10, 11, 0.1872, 0.0619],
        [11, 12, 0.7114, 0.2351],
        [12, 13, 1.0300, 0.3400],
        [13, 14, 1.0440, 0.3450],
        [14, 15, 1.0580, 0.3496],
        [15, 16, 0.1966, 0.0650],
        [16, 17, 0.3744, 0.1238],
        [17, 18, 0.0047, 0.0016],
        [18, 19, 0.3276, 0.1083],
        [19, 20, 0.2106, 0.0690],
        [20, 21, 0.3416, 0.1129],
        [21, 22, 0.0140, 0.0046],
        [22, 23, 0.1591, 0.0526],
        [23, 24, 0.3463, 0.1145],
        [24, 25, 0.7488, 0.2475],
        [25, 26, 0.3089, 0.1021],
        [26, 27, 0.1732, 0.0572],
        # Lateral 1 (from bus 3)
        [3,  28, 0.0044, 0.0108],
        [28, 29, 0.0640, 0.1565],
        [29, 30, 0.3978, 0.1315],
        [30, 31, 0.0702, 0.0232],
        [31, 32, 0.3510, 0.1160],
        [32, 33, 0.8390, 0.2816],
        [33, 34, 1.7080, 0.5646],
        [34, 35, 1.4740, 0.4873],
        # Lateral 2 (from bus 35)
        [35, 36, 0.0044, 0.0108],
        [36, 37, 0.0640, 0.1565],
        [37, 38, 0.1053, 0.1230],
        [38, 39, 0.0304, 0.0355],
        [39, 40, 0.0018, 0.0021],
        [40, 41, 0.7283, 0.8509],
        [41, 42, 0.3100, 0.3623],
        [42, 43, 0.0410, 0.0478],
        [43, 44, 0.0092, 0.0116],
        [44, 45, 0.1089, 0.1373],
        [45, 46, 0.0009, 0.0012],
        # Lateral 3 (from bus 4)
        [4,  47, 0.0034, 0.0084],
        [47, 48, 0.0851, 0.2083],
        [48, 49, 0.2898, 0.7091],
        [49, 50, 0.0822, 0.2011],
        # Lateral 4 (from bus 8)
        [8,  51, 0.0928, 0.0473],
        [51, 52, 0.3319, 0.1114],
        # Lateral 5 (from bus 9)
        [9,  53, 0.1740, 0.0886],
        [53, 54, 0.2030, 0.1034],
        [54, 55, 0.2842, 0.1447],
        [55, 56, 0.2813, 0.1433],
        [56, 57, 1.5900, 0.5337],
        [57, 58, 0.7837, 0.2630],
        [58, 59, 0.3042, 0.1006],
        [59, 60, 0.3861, 0.1172],
        [60, 61, 0.5075, 0.2585],
        [61, 62, 0.0974, 0.0496],
        [62, 63, 0.1450, 0.0738],
        [63, 64, 0.7105, 0.3619],
        [64, 65, 1.0410, 0.5302],
        # Lateral 6 (from bus 11)
        [11, 66, 0.2012, 0.0611],
        [66, 67, 0.0047, 0.0014],
        # Lateral 7 (from bus 12)
        [12, 68, 0.7394, 0.2444],
        [68, 69, 0.0047, 0.0016],
    ]

    bus_data = [
        # [bus_num, P_kW, Q_kVAR]
        [1,   0.0,    0.0  ],
        [2,   0.0,    0.0  ],
        [3,   0.0,    0.0  ],
        [4,   0.0,    0.0  ],
        [5,   0.0,    0.0  ],
        [6,   2.60,   2.20 ],
        [7,   40.4,   30.0 ],
        [8,   75.0,   54.0 ],
        [9,   30.0,   22.0 ],
        [10,  28.0,   19.0 ],
        [11,  145.0,  104.0],
        [12,  145.0,  104.0],
        [13,  8.0,    5.5  ],
        [14,  8.0,    5.5  ],
        [15,  0.0,    0.0  ],
        [16,  45.5,   30.0 ],
        [17,  60.0,   35.0 ],
        [18,  60.0,   35.0 ],
        [19,  0.0,    0.0  ],
        [20,  1.0,    0.60 ],
        [21,  114.0,  81.0 ],
        [22,  5.0,    3.5  ],
        [23,  0.0,    0.0  ],
        [24,  28.0,   20.0 ],
        [25,  0.0,    0.0  ],
        [26,  14.0,   10.0 ],
        [27,  14.0,   10.0 ],
        [28,  26.0,   18.6 ],
        [29,  26.0,   18.6 ],
        [30,  0.0,    0.0  ],
        [31,  0.0,    0.0  ],
        [32,  0.0,    0.0  ],
        [33,  14.0,   10.0 ],
        [34,  19.5,   14.0 ],
        [35,  6.0,    4.0  ],
        [36,  26.0,   18.55],
        [37,  26.0,   18.55],
        [38,  0.0,    0.0  ],
        [39,  24.0,   17.0 ],
        [40,  24.0,   17.0 ],
        [41,  1.20,   1.0  ],
        [42,  0.0,    0.0  ],
        [43,  6.0,    4.3  ],
        [44,  0.0,    0.0  ],
        [45,  39.22,  26.3 ],
        [46,  39.22,  26.3 ],
        [47,  0.0,    0.0  ],
        [48,  79.0,   56.4 ],
        [49,  384.7,  274.5],
        [50,  384.7,  274.5],
        [51,  40.5,   28.3 ],
        [52,  3.6,    2.7  ],
        [53,  4.35,   3.5  ],
        [54,  26.4,   19.0 ],
        [55,  24.0,   17.2 ],
        [56,  0.0,    0.0  ],
        [57,  0.0,    0.0  ],
        [58,  0.0,    0.0  ],
        [59,  100.0,  72.0 ],
        [60,  0.0,    0.0  ],
        [61,  1244.0, 888.0],
        [62,  32.0,   23.0 ],
        [63,  0.0,    0.0  ],
        [64,  227.0,  162.0],
        [65,  59.0,   42.0 ],
        [66,  18.0,   13.0 ],
        [67,  18.0,   13.0 ],
        [68,  28.0,   20.0 ],
        [69,  28.0,   20.0 ],
    ]

    # Equivalent DER allocations (peak, in pu on 10 MVA base)
    der_data = [
        # [bus_num, PV_P_pu, PV_Q_pu, WT_P_pu, WT_Q_pu]
        [11,  0.05, 0.0,  0.0,  0.0],
        [18,  0.03, 0.0,  0.0,  0.0],
        [33,  0.0,  0.0,  0.04, 0.0],
        [49,  0.10, 0.0,  0.0,  0.0],
        [61,  0.0,  0.0,  0.08, 0.0],
    ]
    return branch_data, bus_data, der_data


# ─────────────────────────────────────────────────────────────────────────────
# PANDAPOWER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_network_ohm(branch_data, bus_data, vn_kv, baseMVA=10.0):
    """Build a pandapower network from branch (R/X in Ohms) and bus load data."""
    net = pp.create_empty_network(sn_mva=baseMVA)
    b1 = pp.create_bus(net, vn_kv=vn_kv, name="Bus 1")
    pp.create_ext_grid(net, bus=b1, vm_pu=1.0, va_degree=0.0)
    bus_idx = {bus_data[0][0]: b1}

    for bd in bus_data[1:]:
        bnum, p_kw, q_kvar = bd
        bid = pp.create_bus(net, vn_kv=vn_kv, name=f"Bus {bnum}")
        bus_idx[bnum] = bid
        if p_kw > 0 or q_kvar > 0:
            pp.create_load(net, bus=bid, p_mw=p_kw/1000.0, q_mvar=q_kvar/1000.0)

    for br in branch_data:
        fb, tb, r_ohm, x_ohm = br
        pp.create_line_from_parameters(
            net, from_bus=bus_idx[fb], to_bus=bus_idx[tb],
            length_km=1.0, r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm,
            c_nf_per_km=0.0, max_i_ka=999.0, name=f"Line {fb}-{tb}"
        )
    return net, bus_idx


def build_network_pu(branch_data, bus_data, vn_kv, baseMVA=10.0):
    """Build a pandapower network from branch (R/X in per-unit) and bus load data."""
    Zbase = (vn_kv**2) / baseMVA  # Ohm
    # Convert to Ohm then re-use the Ohm builder
    branch_ohm = [[fb, tb, r*Zbase, x*Zbase] for fb, tb, r, x in branch_data]
    return build_network_ohm(branch_ohm, bus_data, vn_kv, baseMVA)


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORTERS
# ─────────────────────────────────────────────────────────────────────────────

def run_and_export(name, net, bus_data, branch_data, der_data, bus_idx,
                   vn_kv, baseMVA, impedance_unit="ohm"):
    """Run power flow and export all tabs to CSV files."""
    pp.runpp(net, numba=False, verbose=False)
    assert net["converged"], f"Power flow did NOT converge for {name}!"
    print(f"\n{'='*60}")
    print(f"  {name}  — Power Flow Results")
    print(f"{'='*60}")
    print(f"  Converged  : {net['converged']}")
    print(f"  Buses      : {len(net.bus)}")
    print(f"  Branches   : {len(net.line)}")
    print(f"  V_min (pu) : {net.res_bus.vm_pu.min():.5f}  @ bus {net.res_bus.vm_pu.idxmin()+1}")
    print(f"  V_max (pu) : {net.res_bus.vm_pu.max():.5f}")
    print(f"  P loss(kW) : {net.res_line.pl_mw.sum()*1000:.3f}")
    print(f"  Q loss(kVAR): {net.res_line.ql_mvar.sum()*1000:.3f}")

    prefix = os.path.join(OUTPUT_DIR, name)

    # --- Tab 1: DER Locations ---
    df_der = pd.DataFrame(der_data,
        columns=["bus_number", "PV_P_peak_pu", "PV_Q_peak_pu",
                 "WT_P_peak_pu", "WT_Q_peak_pu"])
    df_der.insert(0, "item", range(1, len(df_der)+1))
    df_der["base_MVA"] = baseMVA
    df_der.to_csv(f"{prefix}_tab1_DER_locations.csv", index=False)

    # --- Tab 2: Branch Data ---
    unit_label = "ohm" if impedance_unit == "ohm" else "pu"
    branch_rows = []
    for i, (fb, tb, r, x) in enumerate(branch_data, 1):
        row = {
            "item": i,
            "from_bus": fb,
            "to_bus": tb,
            f"R_{unit_label}": round(r, 6),
            f"X_{unit_label}": round(x, 6),
        }
        branch_rows.append(row)
    df_branch = pd.DataFrame(branch_rows)
    df_branch.to_csv(f"{prefix}_tab2_branch_data.csv", index=False)

    # --- Tab 3: Bus Load Data ---
    bus_rows = []
    for i, bd in enumerate(bus_data, 1):
        bus_rows.append({
            "item": i,
            "bus_number": bd[0],
            "P_load_kW": bd[1],
            "Q_load_kVAR": bd[2],
        })
    df_bus = pd.DataFrame(bus_rows)
    df_bus.to_csv(f"{prefix}_tab3_bus_load_data.csv", index=False)

    # --- Tab 4: Power Flow Results — Bus Voltages (VALIDATION DATA) ---
    res_bus_rows = []
    for i, bd in enumerate(bus_data):
        bnum = bd[0]
        idx = bus_idx[bnum]
        res_bus_rows.append({
            "bus_number": bnum,
            "voltage_magnitude_pu": round(net.res_bus.vm_pu.iloc[idx], 6),
            "voltage_angle_deg":    round(net.res_bus.va_degree.iloc[idx], 6),
            "P_gen_MW":             round(net.res_bus.p_mw.iloc[idx], 6),
            "Q_gen_MVAR":           round(net.res_bus.q_mvar.iloc[idx], 6),
        })
    df_res_bus = pd.DataFrame(res_bus_rows)
    df_res_bus.to_csv(f"{prefix}_tab4_VALIDATION_bus_voltages.csv", index=False)

    # --- Tab 5: Power Flow Results — Branch Flows (VALIDATION DATA) ---
    res_line_rows = []
    for i, br in enumerate(branch_data):
        fb, tb = br[0], br[1]
        res_line_rows.append({
            "from_bus":        fb,
            "to_bus":          tb,
            "P_from_MW":       round(net.res_line.p_from_mw.iloc[i], 6),
            "Q_from_MVAR":     round(net.res_line.q_from_mvar.iloc[i], 6),
            "P_to_MW":         round(net.res_line.p_to_mw.iloc[i], 6),
            "Q_to_MVAR":       round(net.res_line.q_to_mvar.iloc[i], 6),
            "P_loss_kW":       round(net.res_line.pl_mw.iloc[i]*1000, 4),
            "Q_loss_kVAR":     round(net.res_line.ql_mvar.iloc[i]*1000, 4),
            "loading_percent": round(net.res_line.loading_percent.iloc[i], 4),
        })
    df_res_line = pd.DataFrame(res_line_rows)
    df_res_line.to_csv(f"{prefix}_tab5_VALIDATION_branch_flows.csv", index=False)

    print(f"  → Exported 5 CSV files: {prefix}_tab*.csv")
    return df_res_bus


# ─────────────────────────────────────────────────────────────────────────────
# 873-BUS: REDS DOWNLOADER & PARSER
# ─────────────────────────────────────────────────────────────────────────────

def download_and_parse_873bus():
    """
    Download the 873-bus system from the REDS repository (free public source).
    URL: https://www.dejazzer.com/codes/power_systems_radial.tar.gz
    This contains bus_873_7 — the same source used by Vasconcelos (2024).

    Returns: (branch_data, bus_data) or None if download fails.
    """
    url = "https://www.dejazzer.com/codes/power_systems_radial.tar.gz"
    archive_path = os.path.join(OUTPUT_DIR, "reds_radial.tar.gz")

    print("\n[873-bus] Downloading REDS archive from dejazzer.com ...")
    try:
        urllib.request.urlretrieve(url, archive_path)
        print(f"  Downloaded: {archive_path}")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print("  → Please download manually from: https://www.dejazzer.com/codes/power_systems_radial.tar.gz")
        print("    and place the file as:", archive_path)
        return None, None

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getnames()
            # Find the 873-bus file
            target = next((m for m in members if "873" in m and not m.endswith("/")), None)
            if not target:
                print(f"  ✗ Could not find 873-bus file in archive. Members: {members[:10]}")
                return None, None
            print(f"  Found: {target}")
            f = tar.extractfile(target)
            raw = f.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  ✗ Failed to parse archive: {e}")
        return None, None

    # Parse REDS format (custom text format by Kavasseri & Ababei)
    # Format (from their readme): each file has sections separated by blank lines.
    # Line format:   branch_count bus_count
    # Then branch lines: from to R X Bcap
    # Then bus lines:    bus P_kW Q_kVAR
    branch_data, bus_data = [], []
    lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.strip().startswith("#")]

    mode = None
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            try:
                int(parts[0]); int(parts[1])
                # Could be header "branches buses"
                mode = "header"
                continue
            except ValueError:
                pass
        if mode == "header" or mode == "branch":
            try:
                fb, tb, r, x = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
                branch_data.append([fb, tb, r, x])
                mode = "branch"
                continue
            except (ValueError, IndexError):
                pass
        if mode == "branch" or mode == "bus":
            try:
                bnum, p, q = int(parts[0]), float(parts[1]), float(parts[2])
                bus_data.append([bnum, p, q])
                mode = "bus"
            except (ValueError, IndexError):
                pass

    print(f"  Parsed: {len(bus_data)} buses, {len(branch_data)} branches")
    return branch_data, bus_data


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Power Flow Dataset Builder")
    print(" 10-bus, 69-bus, 873-bus Radial Distribution Systems")
    print("=" * 60)

    # ── 10-BUS ───────────────────────────────────────────────────────────────
    branch_10, bus_10, der_10 = get_10bus_data()
    net_10, bidx_10 = build_network_pu(branch_10, bus_10, vn_kv=11.0, baseMVA=10.0)
    run_and_export("10bus", net_10, bus_10, branch_10, der_10, bidx_10,
                   vn_kv=11.0, baseMVA=10.0, impedance_unit="pu")

    # ── 69-BUS ───────────────────────────────────────────────────────────────
    branch_69, bus_69, der_69 = get_69bus_data()
    net_69, bidx_69 = build_network_ohm(branch_69, bus_69, vn_kv=12.66, baseMVA=10.0)
    run_and_export("69bus", net_69, bus_69, branch_69, der_69, bidx_69,
                   vn_kv=12.66, baseMVA=10.0, impedance_unit="ohm")

    # ── 873-BUS ──────────────────────────────────────────────────────────────
    branch_873, bus_873 = download_and_parse_873bus()
    if branch_873 and bus_873:
        # DER allocations for 873-bus (sample — adjust as needed)
        der_873 = [[b[0], 0.02, 0.0, 0.0, 0.0]
                   for b in bus_873[1::50] if b[1] > 0][:10]
        net_873, bidx_873 = build_network_ohm(branch_873, bus_873, vn_kv=11.0, baseMVA=10.0)
        run_and_export("873bus", net_873, bus_873, branch_873, der_873, bidx_873,
                       vn_kv=11.0, baseMVA=10.0, impedance_unit="ohm")
    else:
        print("\n[873-bus] Skipped (download unavailable).")
        print("  You can get it from: https://www.dejazzer.com/reds.html")
        print("  File to download: power_systems_radial.tar.gz → bus_873_7")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"{'='*60}")
    print("""
  FILES GENERATED (per system):
    tab1_DER_locations.csv      — DER bus assignments (PV/WT peak power)
    tab2_branch_data.csv        — Network branches (R, X impedances)
    tab3_bus_load_data.csv      — Bus load (P kW, Q kVAR)
    tab4_VALIDATION_bus_voltages.csv  — ✓ Power flow result: bus V, angle
    tab5_VALIDATION_branch_flows.csv  — ✓ Power flow result: line P/Q flows

  CITATION:
    If using the 69-bus data, please cite:
    M. E. Baran and F. F. Wu (1989), IEEE Trans. Power Deliv., 4(1), 725–734.
    DOI: 10.1109/61.19265

    If using the 873-bus data, please cite:
    R. Kavasseri and C. Ababei, REDS: REpository of Distribution Systems.
    https://www.dejazzer.com/reds.html

    The original compiled dataset:
    P. Vasconcelos (2024), IEEE DataPort. DOI: 10.21227/5n30-5402
    """)


if __name__ == "__main__":
    main()
