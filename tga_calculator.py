import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import io

# ─── Password gate ────────────────────────────────────────────────────────────
def check_password():
    if st.session_state.get("authenticated"):
        return True
    pwd = st.text_input("Password", type="password", key="pwd_input")
    if pwd:
        if pwd == st.secrets.get("APP_PASSWORD", ""):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False

if not check_password():
    st.stop()

MW_CaOH2 = 74.093
MW_H2O = 18.015
MW_CaCO3 = 100.087
MW_CO2 = 44.010


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_tga_excel(file_bytes):
    """Load TGA data from Excel, auto-detecting the data header row.
    Uses positional column mapping to handle duplicate DTG column names.
    Expected column order: Time, Temp, DTA, TG, DTG (ug/min), ...
    """
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))

    header_row = None
    initial_weight_mg = None

    for i, row in enumerate(rows):
        if row[0] is not None and isinstance(row[1], (int, float)) and row[2] == "mg":
            if initial_weight_mg is None:
                initial_weight_mg = float(row[1])

        row_str = [str(c) if c is not None else "" for c in row]
        if any("Time" in s for s in row_str) and any("Temp" in s for s in row_str):
            header_row = i
            break

    if header_row is None:
        return None, None, "Header row not found (need columns: Time, Temp, TG, DTG)"

    hdr = rows[header_row]
    pos = {}
    for j, cell in enumerate(hdr):
        s = str(cell).strip().lower() if cell is not None else ""
        if "time" in s and "Time" not in pos:
            pos["Time"] = j
        elif "temp" in s and "Temp" not in pos:
            pos["Temp"] = j
        elif s == "dta" and "DTA" not in pos:
            pos["DTA"] = j
        elif s == "tg" and "TG" not in pos:
            pos["TG"] = j
        elif "dtg" in s and "DTG" not in pos:
            pos["DTG"] = j

    fallback = {0: "Time", 1: "Temp", 2: "DTA", 3: "TG", 4: "DTG"}
    for name, col_idx in fallback.items():
        if col_idx not in pos.values():
            pos[col_idx] = name

    col_indices = [pos["Time"], pos["Temp"], pos.get("DTA", 2), pos["TG"], pos["DTG"]]

    data_rows = []
    for row in rows[header_row + 2:]:
        if len(row) <= max(col_indices):
            continue
        val0 = row[col_indices[0]]
        if val0 is None:
            continue
        try:
            float(val0)
        except (TypeError, ValueError):
            break
        data_rows.append([row[c] for c in col_indices])

    df = pd.DataFrame(data_rows, columns=["Time", "Temp", "DTA", "TG", "DTG"])
    for col in ["Time", "Temp", "TG", "DTG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Time", "Temp", "TG", "DTG"])
    return df, initial_weight_mg, None


def load_tga_csv(file_bytes, encoding="utf-8"):
    """Load TGA data from CSV, auto-detecting header row."""
    text = file_bytes.decode(encoding, errors="replace")
    lines = text.splitlines()
    header_row = None
    for i, line in enumerate(lines):
        if "Time" in line and ("Temp" in line or "TG" in line):
            header_row = i
            break
    if header_row is None:
        return None, None, "Header row not found (need columns: Time, Temp, TG, DTG)"

    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines[header_row:])), skiprows=[1], on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=df.columns[:2].tolist())

    col_map = {}
    for h in df.columns:
        hl = h.lower()
        if "time" in hl:
            col_map[h] = "Time"
        elif "temp" in hl:
            col_map[h] = "Temp"
        elif "dta" in hl:
            col_map[h] = "DTA"
        elif "dtg" in hl:
            col_map[h] = "DTG"
        elif h.upper() == "TG":
            col_map[h] = "TG"
    df = df.rename(columns=col_map)
    return df, None, None


# ─── Analysis helpers ─────────────────────────────────────────────────────────

def get_tg_at_temp(temps, tgs, target_temp):
    idx = np.argmin(np.abs(temps - target_temp))
    return tgs[idx]


def get_dtg_at_temp(temps, dtgs, target_temp):
    idx = np.argmin(np.abs(temps - target_temp))
    return dtgs[idx]


def smooth_dtg(dtgs):
    window = min(51, len(dtgs) // 10 * 2 + 1)
    if window < 5:
        window = 5
    if window % 2 == 0:
        window += 1
    return savgol_filter(dtgs, window_length=window, polyorder=3)


def find_valleys_auto(temps, dtgs):
    """Auto-detect T1, T2, T3 from DTG valley minima."""
    smoothed = smooth_dtg(dtgs)

    def valley_in_range(lo, hi):
        mask = (temps >= lo) & (temps <= hi)
        if not mask.any():
            return None
        return float(temps[mask][np.argmin(smoothed[mask])])

    T1 = valley_in_range(300, 410) or 380.0
    T2 = valley_in_range(430, 600) or 460.0
    T3 = valley_in_range(750, 870) or 800.0
    return T1, T2, T3


# ─── Calculation methods ──────────────────────────────────────────────────────

def _m_ref(temps, tgs, T_ref, initial_weight_mg):
    tg_ref = get_tg_at_temp(temps, tgs, T_ref)
    if initial_weight_mg is not None:
        return initial_weight_mg + tg_ref / 1000.0
    return abs(tg_ref) / 1000.0


def calculate_stepwise(df, T_ref, T1, T2, T3, initial_weight_mg):
    """
    Stepwise method: mass loss = TG difference at boundary temperatures.
    """
    temps = df["Temp"].values
    tgs = df["TG"].values

    m_ref_mg = _m_ref(temps, tgs, T_ref, initial_weight_mg)
    tg_T1 = get_tg_at_temp(temps, tgs, T1)
    tg_T2 = get_tg_at_temp(temps, tgs, T2)
    tg_T3 = get_tg_at_temp(temps, tgs, T3)

    H2O_mg = (tg_T1 - tg_T2) / 1000.0
    CO2_mg = (tg_T2 - tg_T3) / 1000.0

    CaOH2_mg = H2O_mg * (MW_CaOH2 / MW_H2O)
    CaCO3_mg = CO2_mg * (MW_CaCO3 / MW_CO2)

    return {
        "m_ref_mg": m_ref_mg,
        "H2O_mg": H2O_mg, "CO2_mg": CO2_mg,
        "CaOH2_mg": CaOH2_mg, "CaCO3_mg": CaCO3_mg,
        "CaOH2_pct": CaOH2_mg / m_ref_mg * 100,
        "CO2_pct": CO2_mg / m_ref_mg * 100,
        "CaCO3_pct": CaCO3_mg / m_ref_mg * 100,
    }


def calculate_tangential(df, T_ref, T1, T2, T3, initial_weight_mg):
    """
    Tangential method: integrates (DTG − linear baseline) × dt over each peak.
    The linear baseline connects the DTG values at the valley boundaries,
    removing background drift from the mass loss calculation.

    Ca(OH)₂:  baseline from (T1, DTG(T1)) → (T2, DTG(T2))
    CaCO₃:    baseline from (T2, DTG(T2)) → (T3, DTG(T3))
    """
    temps = df["Temp"].values
    tgs = df["TG"].values
    dtgs = df["DTG"].values
    times = df["Time"].values

    m_ref_mg = _m_ref(temps, tgs, T_ref, initial_weight_mg)

    dtg_T1 = get_dtg_at_temp(temps, dtgs, T1)
    dtg_T2 = get_dtg_at_temp(temps, dtgs, T2)
    dtg_T3 = get_dtg_at_temp(temps, dtgs, T3)

    def peak_integral(T_start, T_end, dtg_start, dtg_end):
        mask = (temps >= T_start) & (temps <= T_end)
        t_seg = times[mask]
        dtg_seg = dtgs[mask]
        temp_seg = temps[mask]
        if len(t_seg) < 2:
            return 0.0
        # Linear baseline in temperature domain, sampled at each data point
        baseline = dtg_start + (dtg_end - dtg_start) * (temp_seg - T_start) / (T_end - T_start)
        net = dtg_seg - baseline  # positive where peak rises above baseline
        return float(np.trapz(net, t_seg)) / 1000.0  # ug -> mg

    H2O_mg = peak_integral(T1, T2, dtg_T1, dtg_T2)
    CO2_mg = peak_integral(T2, T3, dtg_T2, dtg_T3)

    CaOH2_mg = H2O_mg * (MW_CaOH2 / MW_H2O)
    CaCO3_mg = CO2_mg * (MW_CaCO3 / MW_CO2)

    return {
        "m_ref_mg": m_ref_mg,
        "H2O_mg": H2O_mg, "CO2_mg": CO2_mg,
        "CaOH2_mg": CaOH2_mg, "CaCO3_mg": CaCO3_mg,
        "CaOH2_pct": CaOH2_mg / m_ref_mg * 100,
        "CO2_pct": CO2_mg / m_ref_mg * 100,
        "CaCO3_pct": CaCO3_mg / m_ref_mg * 100,
        "dtg_T1": dtg_T1, "dtg_T2": dtg_T2, "dtg_T3": dtg_T3,
    }


MW_CaO = 56.077


def calculate_doc(df, r_tan, initial_weight_mg, M950_cem, mcao_cem_pct,
                  mch0_norm=None):
    """
    Degree of Carbonation (DoC) using tangential method results.

    Requires cement chemistry inputs:
      M950_cem   : residue fraction at 950°C for pure cement (e.g. 98.1 %)
      mcao_cem   : CaO content of the cement (e.g. 65.01 %)
      mch0_norm  : Ca(OH)₂ content of uncarbonated reference sample,
                   normalized to cement content (%) — optional, for DoCch

    Formulas (verified against NETZSCH reference sheet):
      m_cement   = m_ref × (M950_sample / M950_cem)
      mcc_norm   = mcc_mg_tan / m_cement × 100
      mch_norm   = mch_mg_tan / m_cement × 100
      DoChcp     = mcc_norm / (mcao_cem × MW_CaCO3/MW_CaO) × 100
      DoCch      = (mch0_norm − mch_norm) / mch0_norm × 100   [needs mch0_norm]
      DoCch_molar= n(CaCO₃) / [n(CaCO₃) + n(Ca(OH)₂)] × 100 [always available]
    """
    temps = df["Temp"].values
    tgs = df["TG"].values

    m_ref_mg = r_tan["m_ref_mg"]
    CaOH2_mg = r_tan["CaOH2_mg"]
    CaCO3_mg = r_tan["CaCO3_mg"]

    # M950: residue fraction relative to m_ref (NOT initial weight)
    # Formula: M950 = mass_at_950 / m_ref × 100
    # Then: m_cement = mass_at_950 / (M950_cem/100)
    T_max = temps.max()
    tg_at_max = get_tg_at_temp(temps, tgs, T_max)
    mass_at_max = initial_weight_mg + tg_at_max / 1000.0
    M950_sample = mass_at_max / m_ref_mg * 100.0  # % — denominator is m_ref, not initial_weight

    # Cement mass in sample (residue / M950_cem fraction)
    m_cement_mg = mass_at_max / (M950_cem / 100.0)

    # Normalized contents (% per gram cement)
    mch_norm = CaOH2_mg / m_cement_mg * 100.0
    mcc_norm = CaCO3_mg / m_cement_mg * 100.0

    # DoChcp
    DoChcp = mcc_norm / (mcao_cem_pct * (MW_CaCO3 / MW_CaO)) * 100.0

    # DoCch — molar ratio (always)
    n_CaCO3 = CaCO3_mg / MW_CaCO3
    n_CaOH2 = CaOH2_mg / MW_CaOH2
    DoCch_molar = n_CaCO3 / (n_CaCO3 + n_CaOH2) * 100.0 if (n_CaCO3 + n_CaOH2) > 0 else 0.0

    # DoCch — reference sample method (needs mch0_norm)
    DoCch_ref = None
    if mch0_norm is not None and mch0_norm > 0:
        DoCch_ref = (mch0_norm - mch_norm) / mch0_norm * 100.0

    return {
        "M950_sample": M950_sample,
        "m_cement_mg": m_cement_mg,
        "mch_norm": mch_norm,
        "mcc_norm": mcc_norm,
        "DoChcp": DoChcp,
        "DoCch_molar": DoCch_molar,
        "DoCch_ref": DoCch_ref,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_tga(df, T1, T2, T3, T_ref=105.0, show_tangential=True):
    temps = df["Temp"].values
    tgs = df["TG"].values
    dtgs = df["DTG"].values

    dtg_smooth = smooth_dtg(dtgs)

    tg_ref_val = get_tg_at_temp(temps, tgs, T_ref)
    tg_norm = (tgs - tg_ref_val) / abs(tg_ref_val) * -100  # % mass loss

    dtg_T1 = get_dtg_at_temp(temps, dtgs, T1)
    dtg_T2 = get_dtg_at_temp(temps, dtgs, T2)
    dtg_T3 = get_dtg_at_temp(temps, dtgs, T3)

    fig = go.Figure()

    # Raw DTG (faint)
    fig.add_trace(go.Scatter(
        x=temps, y=dtgs, name="DTG (raw)", mode="lines",
        line=dict(color="lightgray", width=0.8)
    ))
    # Smoothed DTG
    fig.add_trace(go.Scatter(
        x=temps, y=dtg_smooth, name="DTG (smoothed)", mode="lines",
        line=dict(color="black", width=1.8), yaxis="y1"
    ))
    # TG curve (right axis)
    fig.add_trace(go.Scatter(
        x=temps, y=tg_norm, name="TG (mass loss %)", mode="lines",
        line=dict(color="royalblue", width=2), yaxis="y2"
    ))

    mask_1 = (temps >= T1) & (temps <= T2)
    mask_2 = (temps >= T2) & (temps <= T3)

    if show_tangential:
        # ── Tangential baselines ──────────────────────────────────────────────
        # Ca(OH)₂ baseline: straight line T1→T2
        bl_CaOH2 = dtg_T1 + (dtg_T2 - dtg_T1) * (temps[mask_1] - T1) / (T2 - T1)
        fig.add_trace(go.Scatter(
            x=temps[mask_1], y=bl_CaOH2,
            name="Tangential baseline (Ca(OH)₂)", mode="lines",
            line=dict(color="darkorange", width=1.5, dash="dot"), yaxis="y1"
        ))
        # CaCO₃ baseline: straight line T2→T3
        bl_CaCO3 = dtg_T2 + (dtg_T3 - dtg_T2) * (temps[mask_2] - T2) / (T3 - T2)
        fig.add_trace(go.Scatter(
            x=temps[mask_2], y=bl_CaCO3,
            name="Tangential baseline (CaCO₃)", mode="lines",
            line=dict(color="darkgreen", width=1.5, dash="dot"), yaxis="y1"
        ))

        # ── Shaded: area between DTG curve and tangential baseline ────────────
        # Ca(OH)₂ — fill between smoothed DTG and baseline
        fig.add_trace(go.Scatter(
            x=np.concatenate([temps[mask_1], temps[mask_1][::-1]]),
            y=np.concatenate([dtg_smooth[mask_1], bl_CaOH2[::-1]]),
            fill="toself", fillcolor="rgba(255,140,0,0.35)",
            line=dict(width=0), name="Ca(OH)₂ (tangential)", yaxis="y1"
        ))
        # CaCO₃
        fig.add_trace(go.Scatter(
            x=np.concatenate([temps[mask_2], temps[mask_2][::-1]]),
            y=np.concatenate([dtg_smooth[mask_2], bl_CaCO3[::-1]]),
            fill="toself", fillcolor="rgba(34,139,34,0.35)",
            line=dict(width=0), name="CaCO₃ (tangential)", yaxis="y1"
        ))

        # ── Faint stepwise shading (for comparison) ───────────────────────────
        fig.add_trace(go.Scatter(
            x=np.concatenate([temps[mask_1], temps[mask_1][::-1]]),
            y=np.concatenate([dtg_smooth[mask_1], np.zeros(mask_1.sum())]),
            fill="toself", fillcolor="rgba(255,165,0,0.1)",
            line=dict(width=0), name="Ca(OH)₂ (stepwise, faint)", yaxis="y1",
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([temps[mask_2], temps[mask_2][::-1]]),
            y=np.concatenate([dtg_smooth[mask_2], np.zeros(mask_2.sum())]),
            fill="toself", fillcolor="rgba(50,205,50,0.1)",
            line=dict(width=0), name="CaCO₃ (stepwise, faint)", yaxis="y1",
            showlegend=False
        ))
    else:
        # Stepwise only shading
        fig.add_trace(go.Scatter(
            x=np.concatenate([temps[mask_1], temps[mask_1][::-1]]),
            y=np.concatenate([dtg_smooth[mask_1], np.zeros(mask_1.sum())]),
            fill="toself", fillcolor="rgba(255,165,0,0.35)",
            line=dict(width=0), name="Ca(OH)₂ (stepwise)", yaxis="y1"
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([temps[mask_2], temps[mask_2][::-1]]),
            y=np.concatenate([dtg_smooth[mask_2], np.zeros(mask_2.sum())]),
            fill="toself", fillcolor="rgba(50,205,50,0.35)",
            line=dict(width=0), name="CaCO₃ (stepwise)", yaxis="y1"
        ))

    # Boundary vertical lines
    for T, label, color in [(T1, f"T1={T1:.1f}°C", "orange"),
                             (T2, f"T2={T2:.1f}°C", "green"),
                             (T3, f"T3={T3:.1f}°C", "green")]:
        fig.add_vline(x=T, line=dict(color=color, dash="dash", width=1.5),
                      annotation_text=label, annotation_position="top right")

    fig.update_layout(
        title="TGA / DTG Curve",
        xaxis=dict(title="Temperature (°C)", range=[100, 950]),
        yaxis=dict(title="DTG (μg/min)", side="left"),
        yaxis2=dict(title="Mass loss (%)", side="right", overlaying="y",
                    autorange="reversed"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        height=520,
        hovermode="x unified"
    )
    return fig


# ─── Results display helper ───────────────────────────────────────────────────

def show_results(r, T1, T2, T3, method_name):
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Reference mass (m_ref)", f"{r['m_ref_mg']:.4f} mg")
        st.metric("H₂O from Ca(OH)₂", f"{r['H2O_mg']:.4f} mg")
        st.metric("CO₂ from CaCO₃", f"{r['CO2_mg']:.4f} mg")
    with col_r2:
        st.metric("Ca(OH)₂ content", f"{r['CaOH2_mg']:.4f} mg")
        st.metric("CaCO₃ content", f"{r['CaCO3_mg']:.4f} mg")
    with col_r3:
        st.metric("Ca(OH)₂ rate", f"{r['CaOH2_pct']:.3f} %")
        st.metric("CO₂ absorption rate", f"{r['CO2_pct']:.3f} %")
        st.metric("CaCO₃ rate", f"{r['CaCO3_pct']:.3f} %")

    detail = pd.DataFrame([
        {"Component": "Ca(OH)₂", "T_start (°C)": f"{T1:.2f}", "T_end (°C)": f"{T2:.2f}",
         "Mass loss (mg)": f"{r['H2O_mg']:.4f}", "Species": "H₂O",
         "Content (mg)": f"{r['CaOH2_mg']:.4f}", "Content (%)": f"{r['CaOH2_pct']:.3f}"},
        {"Component": "CaCO₃", "T_start (°C)": f"{T2:.2f}", "T_end (°C)": f"{T3:.2f}",
         "Mass loss (mg)": f"{r['CO2_mg']:.4f}", "Species": "CO₂",
         "Content (mg)": f"{r['CaCO3_mg']:.4f}", "Content (%)": f"{r['CaCO3_pct']:.3f}"},
    ])
    st.dataframe(detail, use_container_width=True, hide_index=True)


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="TGA Calculator", page_icon="🔬", layout="wide")
st.title("🔬 TGA Calculator — Ca(OH)₂ & CaCO₃")

with st.expander("ℹ️ Methods overview", expanded=False):
    st.markdown("""
| Method | How mass loss is calculated |
|--------|----------------------------|
| **Stepwise** | ΔTG = TG(T_start) − TG(T_end) — direct TG signal difference |
| **Tangential** | ∫(DTG − baseline) dt — integrates DTG peak area above a linear baseline connecting the valley points, removing background drift |

Both share the same temperature boundaries T1, T2, T3 (valley points in the DTG curve).

| Range | Reaction |
|-------|----------|
| T1 → T2 | Ca(OH)₂ → CaO + H₂O &nbsp; ⟹ &nbsp; Ca(OH)₂ = H₂O × (74.09/18.02) |
| T2 → T3 | CaCO₃ → CaO + CO₂ &nbsp;&nbsp; ⟹ &nbsp; CaCO₃ = CO₂ × (100.09/44.01) |
""")

# ─── File upload ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "TGA データファイルをアップロード / Upload TGA data file",
    type=["xlsx", "xls", "csv"],
    help="Excel (.xlsx/.xls) or CSV with columns: Time, Temp, TG, DTG"
)

if uploaded is None:
    st.info("ファイルをアップロードしてください / Please upload a TGA data file")
    st.stop()

file_bytes = uploaded.read()
error = None

if uploaded.name.lower().endswith((".xlsx", ".xls")):
    df, initial_weight_mg, error = load_tga_excel(file_bytes)
else:
    for enc in ["utf-8", "shift-jis", "cp932", "latin-1"]:
        df, initial_weight_mg, error = load_tga_csv(file_bytes, enc)
        if df is not None:
            break

if error or df is None:
    st.error(f"Failed to load file: {error}")
    st.stop()

st.success(f"✅ {len(df):,} rows loaded")

# ─── Settings ─────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)
with col_a:
    iw_default = float(initial_weight_mg) if initial_weight_mg else 15.0
    iw_label = "Initial sample weight (mg)" + ("" if initial_weight_mg else " — enter manually")
    initial_weight_mg = st.number_input(iw_label, value=iw_default,
                                        min_value=0.1, max_value=1000.0, step=0.1)
with col_b:
    T_ref = st.number_input("Reference temperature (°C)", value=105.0,
                             min_value=50.0, max_value=200.0, step=1.0,
                             help="End of isothermal hold — used as mass reference")

# ─── Filter to heating ramp ───────────────────────────────────────────────────
df_ramp = df[df["Temp"] >= T_ref].copy().reset_index(drop=True)
if df_ramp.empty:
    st.error("No data above reference temperature. Check file format.")
    st.stop()

temps = df_ramp["Temp"].values
dtgs = df_ramp["DTG"].values

# ─── Temperature boundaries ───────────────────────────────────────────────────
T1_auto, T2_auto, T3_auto = find_valleys_auto(temps, dtgs)

st.markdown("---")
st.subheader("Temperature Boundaries")
st.caption("Auto-detected from DTG valley minima — adjust with sliders if needed.")

col1, col2, col3 = st.columns(3)
with col1:
    T1 = st.slider("T1 — Ca(OH)₂ start (°C)", 250.0, 430.0,
                   round(T1_auto, 1), 0.5,
                   help="Valley between free-water evaporation and Ca(OH)₂ peak")
with col2:
    T2 = st.slider("T2 — Ca(OH)₂ / CaCO₃ boundary (°C)", 420.0, 620.0,
                   round(T2_auto, 1), 0.5,
                   help="Valley between Ca(OH)₂ peak and CaCO₃ peak")
with col3:
    T3 = st.slider("T3 — CaCO₃ end (°C)", 720.0, 900.0,
                   round(T3_auto, 1), 0.5,
                   help="Valley after CaCO₃ peak")

# ─── Calculate both methods ───────────────────────────────────────────────────
r_sw = calculate_stepwise(df_ramp, T_ref, T1, T2, T3, initial_weight_mg)
r_tan = calculate_tangential(df_ramp, T_ref, T1, T2, T3, initial_weight_mg)

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig = plot_tga(df_ramp, T1, T2, T3, T_ref, show_tangential=True)
st.plotly_chart(fig, use_container_width=True)

# ─── Results tabs ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Results")

tab_sw, tab_tan, tab_cmp = st.tabs(["📐 Stepwise", "📈 Tangential", "🔄 Comparison"])

with tab_sw:
    st.markdown("**Stepwise method** — mass loss = TG(T_start) − TG(T_end)")
    show_results(r_sw, T1, T2, T3, "Stepwise")
    with st.expander("Formulas"):
        r = r_sw
        st.markdown(f"""
m_ref = {initial_weight_mg:.3f} + TG({T_ref:.0f}°C)/1000 = **{r['m_ref_mg']:.4f} mg**

**Ca(OH)₂**
H₂O = TG({T1:.1f}°C) − TG({T2:.1f}°C) = **{r['H2O_mg']:.4f} mg**
Ca(OH)₂ = {r['H2O_mg']:.4f} × (74.093/18.015) = **{r['CaOH2_mg']:.4f} mg** ({r['CaOH2_pct']:.3f}%)

**CaCO₃**
CO₂ = TG({T2:.1f}°C) − TG({T3:.1f}°C) = **{r['CO2_mg']:.4f} mg**
CaCO₃ = {r['CO2_mg']:.4f} × (100.087/44.010) = **{r['CaCO3_mg']:.4f} mg** ({r['CaCO3_pct']:.3f}%)
""")

with tab_tan:
    st.markdown("**Tangential method** — ∫(DTG − linear baseline) dt over each peak")
    show_results(r_tan, T1, T2, T3, "Tangential")
    with st.expander("Formulas"):
        r = r_tan
        st.markdown(f"""
m_ref = **{r['m_ref_mg']:.4f} mg** (same as stepwise)

**Ca(OH)₂** — baseline: DTG({T1:.1f}°C)={r['dtg_T1']:.3f} → DTG({T2:.1f}°C)={r['dtg_T2']:.3f} μg/min
H₂O = ∫[T1→T2] (DTG − baseline) dt = **{r['H2O_mg']:.4f} mg**
Ca(OH)₂ = {r['H2O_mg']:.4f} × (74.093/18.015) = **{r['CaOH2_mg']:.4f} mg** ({r['CaOH2_pct']:.3f}%)

**CaCO₃** — baseline: DTG({T2:.1f}°C)={r['dtg_T2']:.3f} → DTG({T3:.1f}°C)={r['dtg_T3']:.3f} μg/min
CO₂ = ∫[T2→T3] (DTG − baseline) dt = **{r['CO2_mg']:.4f} mg**
CaCO₃ = {r['CO2_mg']:.4f} × (100.087/44.010) = **{r['CaCO3_mg']:.4f} mg** ({r['CaCO3_pct']:.3f}%)
""")

with tab_cmp:
    st.markdown("**Method comparison**")
    cmp = pd.DataFrame([
        {
            "Component": "Ca(OH)₂",
            "Stepwise — Content (mg)": f"{r_sw['CaOH2_mg']:.4f}",
            "Stepwise — Content (%)": f"{r_sw['CaOH2_pct']:.3f}",
            "Tangential — Content (mg)": f"{r_tan['CaOH2_mg']:.4f}",
            "Tangential — Content (%)": f"{r_tan['CaOH2_pct']:.3f}",
            "Diff (%)": f"{r_sw['CaOH2_pct'] - r_tan['CaOH2_pct']:+.3f}",
        },
        {
            "Component": "CaCO₃",
            "Stepwise — Content (mg)": f"{r_sw['CaCO3_mg']:.4f}",
            "Stepwise — Content (%)": f"{r_sw['CaCO3_pct']:.3f}",
            "Tangential — Content (mg)": f"{r_tan['CaCO3_mg']:.4f}",
            "Tangential — Content (%)": f"{r_tan['CaCO3_pct']:.3f}",
            "Diff (%)": f"{r_sw['CaCO3_pct'] - r_tan['CaCO3_pct']:+.3f}",
        },
        {
            "Component": "CO₂ absorption",
            "Stepwise — Content (mg)": f"{r_sw['CO2_mg']:.4f}",
            "Stepwise — Content (%)": f"{r_sw['CO2_pct']:.3f}",
            "Tangential — Content (mg)": f"{r_tan['CO2_mg']:.4f}",
            "Tangential — Content (%)": f"{r_tan['CO2_pct']:.3f}",
            "Diff (%)": f"{r_sw['CO2_pct'] - r_tan['CO2_pct']:+.3f}",
        },
    ])
    st.dataframe(cmp, use_container_width=True, hide_index=True)
    st.caption("Diff = Stepwise − Tangential. Tangential is smaller because it subtracts the DTG background drift.")

# ─── DoC section ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Degree of Carbonation (DoC)")

with st.expander("ℹ️ DoC formulas", expanded=False):
    st.markdown("""
Tangential method values are used for DoC (consistent with NETZSCH reference).

| Symbol | Formula |
|--------|---------|
| M950_sample | mass@950°C / initial_weight × 100 &nbsp; (from TGA) |
| m_cement | m_ref × (M950_sample / M950_cem) |
| mcc_norm | CaCO₃_tan / m_cement × 100 &nbsp; (% per g cement) |
| mch_norm | Ca(OH)₂_tan / m_cement × 100 &nbsp; (% per g cement) |
| **DoChcp** | mcc_norm / (mcao_cem × MW_CaCO₃/MW_CaO) × 100 |
| **DoCch (molar)** | n(CaCO₃) / [n(CaCO₃) + n(Ca(OH)₂)] × 100 &nbsp; *(no reference needed)* |
| **DoCch (ref)** | (mch0_norm − mch_norm) / mch0_norm × 100 &nbsp; *(needs uncarbonated reference)* |
""")

doc_col1, doc_col2, doc_col3 = st.columns(3)
with doc_col1:
    M950_cem = st.number_input(
        "M950_cem — residue at 950°C for pure cement (%)",
        value=98.1, min_value=80.0, max_value=100.0, step=0.1,
        help="Typical OPC: ~98.1%. From LOI test or cement data sheet."
    )
with doc_col2:
    mcao_cem = st.number_input(
        "CaO content of cement (%)",
        value=65.0, min_value=30.0, max_value=80.0, step=0.1,
        help="From X-ray fluorescence or Bogue calculation. Typical OPC: ~64–66%."
    )
with doc_col3:
    mch0_norm = st.number_input(
        "mch0_norm — Ca(OH)₂ of uncarbonated reference (% per g cement)",
        value=0.0, min_value=0.0, max_value=100.0, step=0.1,
        help="From TGA of an uncarbonated reference sample, normalized to cement content. "
             "Leave 0 to skip DoCch(ref)."
    )
    mch0_norm_val = mch0_norm if mch0_norm > 0 else None

doc = calculate_doc(df_ramp, r_tan, initial_weight_mg, M950_cem, mcao_cem, mch0_norm_val)

doc_r1, doc_r2, doc_r3 = st.columns(3)
with doc_r1:
    st.metric("M950 sample", f"{doc['M950_sample']:.3f} %",
              help="Residue fraction at max temperature")
    st.metric("m_cement (estimated)", f"{doc['m_cement_mg']:.4f} mg")
with doc_r2:
    st.metric("mch_norm (Ca(OH)₂ / g cement)", f"{doc['mch_norm']:.3f} %")
    st.metric("mcc_norm (CaCO₃ / g cement)", f"{doc['mcc_norm']:.3f} %")
with doc_r3:
    st.metric("DoChcp", f"{doc['DoChcp']:.2f} %",
              help="Degree of carbonation relative to total CaO in cement")
    st.metric("DoCch (molar)", f"{doc['DoCch_molar']:.2f} %",
              help="n(CaCO₃) / [n(CaCO₃) + n(Ca(OH)₂)] — no reference needed")
    if doc["DoCch_ref"] is not None:
        st.metric("DoCch (reference)", f"{doc['DoCch_ref']:.2f} %",
                  help="(mch0_norm − mch_norm) / mch0_norm — uses uncarbonated reference")
    else:
        st.metric("DoCch (reference)", "— (enter mch0_norm above)")

with st.expander("DoC calculation details"):
    r = r_tan
    st.markdown(f"""
**Cement mass estimation**
M950_sample = {doc['M950_sample']:.3f}%
m_cement = {r['m_ref_mg']:.4f} × ({doc['M950_sample']:.3f} / {M950_cem}) = **{doc['m_cement_mg']:.4f} mg**

**Normalized contents**
mch_norm = {r['CaOH2_mg']:.4f} / {doc['m_cement_mg']:.4f} × 100 = **{doc['mch_norm']:.3f}%**
mcc_norm = {r['CaCO3_mg']:.4f} / {doc['m_cement_mg']:.4f} × 100 = **{doc['mcc_norm']:.3f}%**

**DoChcp**
= mcc_norm / (mcao_cem × MW_CaCO₃/MW_CaO) × 100
= {doc['mcc_norm']:.3f} / ({mcao_cem} × {MW_CaCO3/MW_CaO:.4f}) × 100
= {doc['mcc_norm']:.3f} / {mcao_cem * MW_CaCO3/MW_CaO:.3f} × 100 = **{doc['DoChcp']:.2f}%**

**DoCch (molar)**
n(CaCO₃) = {r['CaCO3_mg']:.4f} / {MW_CaCO3} = {r['CaCO3_mg']/MW_CaCO3:.5f} mmol
n(Ca(OH)₂) = {r['CaOH2_mg']:.4f} / {MW_CaOH2} = {r['CaOH2_mg']/MW_CaOH2:.5f} mmol
DoCch = {r['CaCO3_mg']/MW_CaCO3:.5f} / ({r['CaCO3_mg']/MW_CaCO3:.5f} + {r['CaOH2_mg']/MW_CaOH2:.5f}) × 100 = **{doc['DoCch_molar']:.2f}%**
""" + (f"""
**DoCch (reference)**
= ({mch0_norm:.3f} − {doc['mch_norm']:.3f}) / {mch0_norm:.3f} × 100 = **{doc['DoCch_ref']:.2f}%**
""" if doc["DoCch_ref"] is not None else ""))
