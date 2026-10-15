"""
RoadSense AI - Sensor Waveform Physics & Kinematics Visualizer
==============================================================
Provides high-fidelity graphical representations and analytical explanations
of how smartphone IMU sensor readings (Accelerometer & Gyroscope) physically
differentiate between a BUMP (elevation) and a POTHOLE (depression / cavity).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def generate_theoretical_kinematics(num_samples: int = 100, fs: float = 100.0) -> Dict[str, np.ndarray]:
    """
    Generates synthetic but physically calibrated 1-second IMU signal waveforms
    contrasting a Speed Bump traversal against a Pothole impact at 40 km/h.
    
    Physics parameters:
    - Baseline Gravity: 9.81 m/s^2 (vertical Z axis)
    - Bump: Positive vertical impulse (+a_z compression), smooth crest unloading,
            moderate pitch climb followed by pitch drop, low-to-moderate jerk.
    - Pothole: Sudden freefall drop (-a_z dip), followed by violent rim strike shock
               (+a_z spike up to 26+ m/s^2), high-frequency damping oscillations,
               extreme jerk transient (da/dt > 150 m/s^3), and sharp pitch dive.
    """
    t = np.linspace(0.0, 1.0, num_samples)
    dt = 1.0 / fs
    baseline_g = 9.81

    # 1. BUMP KINEMATICS (Speed Breaker / Ramp)
    # Smooth upward compression as front tire mounts ramp (t ~ 0.25 to 0.55s)
    bump_az = baseline_g + 7.5 * np.exp(-((t - 0.40) ** 2) / (2 * 0.06 ** 2)) \
                         - 2.8 * np.exp(-((t - 0.60) ** 2) / (2 * 0.05 ** 2)) \
                         + 1.4 * np.exp(-((t - 0.75) ** 2) / (2 * 0.04 ** 2))
    
    # Forward-backward pitch: front axle lifts (pitch > 0), then drops as rear mounts (pitch < 0)
    bump_pitch = 0.85 * np.sin(2 * np.pi * 1.8 * (t - 0.25)) * np.exp(-((t - 0.45) ** 2) / (2 * 0.12 ** 2))
    bump_pitch = np.where((t >= 0.2) & (t <= 0.8), bump_pitch, 0.0)

    # Lateral roll: minimal if both wheels mount symmetrically
    bump_roll = 0.12 * np.sin(2 * np.pi * 3.5 * t) * np.exp(-((t - 0.45) ** 2) / 0.05)

    # Jerk = d(az)/dt
    bump_jerk = np.gradient(bump_az, dt)

    # 2. POTHOLE KINEMATICS (Cavity / Rim Bottom-Out Strike)
    # Stage A: Freefall dip as wheel falls into hole (t ~ 0.30s, az drops to ~3.5 m/s^2)
    # Stage B: Violent rim collision / suspension bottom-out spike (t ~ 0.38s, az spikes to ~27 m/s^2)
    # Stage C: High-frequency damped harmonic ringing (shock rebound)
    pothole_az = baseline_g - 6.5 * np.exp(-((t - 0.28) ** 2) / (2 * 0.025 ** 2)) \
                            + 18.2 * np.exp(-((t - 0.36) ** 2) / (2 * 0.018 ** 2)) \
                            - 5.5 * np.exp(-((t - 0.44) ** 2) / (2 * 0.022 ** 2)) \
                            + 3.2 * np.exp(-((t - 0.52) ** 2) / (2 * 0.025 ** 2)) \
                            - 1.8 * np.exp(-((t - 0.60) ** 2) / (2 * 0.030 ** 2))
    
    # Gyro Pitch Rate: Sudden nose dive into hole (-pitch), followed by violent upward recoil
    pothole_pitch = -1.65 * np.exp(-((t - 0.29) ** 2) / (2 * 0.03 ** 2)) \
                    + 2.40 * np.exp(-((t - 0.37) ** 2) / (2 * 0.025 ** 2)) \
                    - 0.90 * np.exp(-((t - 0.46) ** 2) / (2 * 0.04 ** 2))

    # Potholes often hit one wheel asymmetric -> sharp roll oscillation
    pothole_roll = 0.85 * np.sin(2 * np.pi * 8.0 * (t - 0.32)) * np.exp(-((t - 0.45) ** 2) / (2 * 0.08 ** 2))
    pothole_roll = np.where((t >= 0.25) & (t <= 0.75), pothole_roll, 0.0)

    # Jerk = d(az)/dt
    pothole_jerk = np.gradient(pothole_az, dt)

    return {
        "time": t,
        "bump_az": bump_az,
        "bump_pitch": bump_pitch,
        "bump_roll": bump_roll,
        "bump_jerk": bump_jerk,
        "pothole_az": pothole_az,
        "pothole_pitch": pothole_pitch,
        "pothole_roll": pothole_roll,
        "pothole_jerk": pothole_jerk,
        "baseline_g": baseline_g,
    }


def create_bump_vs_pothole_comparison_figure() -> go.Figure:
    """
    Constructs a dual-row synchronized Plotly figure displaying:
    Row 1: Vertical Acceleration a_z (m/s^2) for Bump vs Pothole with annotated physical phases.
    Row 2: Jerk Transient (da/dt in m/s^3) & Gyroscope Pitch Rate (rad/s).
    """
    data = generate_theoretical_kinematics(num_samples=150, fs=150.0)
    t = data["time"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "🟡 Speed Bump: Vertical Accel <i>a<sub>z</sub></i> (Elevation Crest)",
            "🔴 Pothole: Vertical Accel <i>a<sub>z</sub></i> (Drop + Impact Shock)",
            "🟡 Speed Bump: Jerk Transient & Pitch Rate",
            "🔴 Pothole: Jerk Transient & Pitch Rate",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    # ---- PLOT 1: BUMP a_z ----
    fig.add_trace(
        go.Scatter(
            x=t, y=[data["baseline_g"]] * len(t),
            name="1G Gravity (9.81 m/s²)",
            line=dict(color="#78716C", dash="dash", width=1.2),
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=data["bump_az"],
            name="Bump a_z",
            line=dict(color="#D97706", width=2.5),
            fill="tonexty",
            fillcolor="rgba(217, 119, 6, 0.12)",
            hovertemplate="Time: %{x:.2f}s<br>Accel Z: %{y:.2f} m/s²<extra>Bump Waveform</extra>",
        ),
        row=1, col=1,
    )
    # Bump Annotations
    fig.add_annotation(
        x=0.40, y=float(np.max(data["bump_az"])),
        text="<b>Ramp Compression</b><br>+7.5 m/s² above 1G",
        showarrow=True, arrowhead=2, arrowcolor="#B45309", arrowsize=1,
        font=dict(size=10, color="#92400E"),
        bgcolor="#FEF3C7", bordercolor="#F59E0B", borderwidth=1,
        row=1, col=1,
    )
    fig.add_annotation(
        x=0.60, y=float(np.min(data["bump_az"][t > 0.45])),
        text="<b>Crest Unload</b><br>Smooth Descent",
        showarrow=True, arrowhead=2, arrowcolor="#B45309",
        font=dict(size=9, color="#92400E"),
        bgcolor="#FEF3C7",
        row=1, col=1,
    )

    # ---- PLOT 2: POTHOLE a_z ----
    fig.add_trace(
        go.Scatter(
            x=t, y=[data["baseline_g"]] * len(t),
            name="1G Gravity (9.81 m/s²)",
            line=dict(color="#78716C", dash="dash", width=1.2),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=data["pothole_az"],
            name="Pothole a_z",
            line=dict(color="#DC2626", width=2.5),
            fill="tonexty",
            fillcolor="rgba(220, 38, 38, 0.12)",
            hovertemplate="Time: %{x:.2f}s<br>Accel Z: %{y:.2f} m/s²<extra>Pothole Waveform</extra>",
        ),
        row=1, col=2,
    )
    # Pothole Annotations
    fig.add_annotation(
        x=0.28, y=float(np.min(data["pothole_az"])),
        text="<b>Freefall Drop</b><br>Wheel loses ground (3.3 m/s²)",
        showarrow=True, arrowhead=2, arrowcolor="#DC2626",
        font=dict(size=10, color="#991B1B"),
        bgcolor="#FEE2E2", bordercolor="#DC2626", borderwidth=1,
        row=1, col=2,
    )
    fig.add_annotation(
        x=0.36, y=float(np.max(data["pothole_az"])),
        text="<b>Violent Rim Strike Shock</b><br>Bottom-out peak (+28.0 m/s²)",
        showarrow=True, arrowhead=2, arrowcolor="#991B1B",
        font=dict(size=10, color="#991B1B"),
        bgcolor="#FEE2E2", bordercolor="#991B1B", borderwidth=1,
        row=1, col=2,
    )

    # ---- PLOT 3: BUMP JERK & PITCH ----
    fig.add_trace(
        go.Scatter(
            x=t, y=data["bump_jerk"],
            name="Bump Jerk da/dt (m/s³)",
            line=dict(color="#2563EB", width=1.6),
            hovertemplate="Jerk: %{y:.1f} m/s³<extra>Bump Jerk</extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=data["bump_pitch"] * 30.0,  # Scaled for visual comparison
            name="Bump Gyro Pitch ×30",
            line=dict(color="#059669", width=1.4, dash="dot"),
            hovertemplate="Pitch: %{y:.2f} (scaled)<extra>Bump Pitch</extra>",
        ),
        row=2, col=1,
    )

    # ---- PLOT 4: POTHOLE JERK & PITCH ----
    fig.add_trace(
        go.Scatter(
            x=t, y=data["pothole_jerk"],
            name="Pothole Jerk da/dt (m/s³)",
            line=dict(color="#991B1B", width=2.0),
            hovertemplate="Jerk: %{y:.1f} m/s³<extra>Pothole Jerk</extra>",
        ),
        row=2, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=data["pothole_pitch"] * 30.0,
            name="Pothole Gyro Pitch ×30",
            line=dict(color="#7C3AED", width=1.4, dash="dot"),
            hovertemplate="Pitch: %{y:.2f} (scaled)<extra>Pothole Pitch</extra>",
        ),
        row=2, col=2,
    )

    # Layout styling matching White, Beige, and Red design system
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAF7F2",
        font=dict(family="sans-serif", color="#1C1917"),
        height=560,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#EADBCC",
            borderwidth=1,
        ),
    )

    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(
                gridcolor="#EADBCC",
                zerolinecolor="#D6D3D1",
                title_text="Time (seconds)" if i == 2 else "",
                row=i, col=j,
            )
            fig.update_yaxes(
                gridcolor="#EADBCC",
                zerolinecolor="#D6D3D1",
                row=i, col=j,
            )

    fig.update_yaxes(title_text="Accel Z (m/s²)", row=1, col=1)
    fig.update_yaxes(title_text="Accel Z (m/s²)", row=1, col=2)
    fig.update_yaxes(title_text="Jerk (m/s³) / Pitch", row=2, col=1)
    fig.update_yaxes(title_text="Jerk (m/s³) / Pitch", row=2, col=2)

    return fig


def render_kinematics_explainer():
    """Renders the comprehensive visual & mathematical breakdown of Bump vs Pothole."""
    st.markdown("""
        <div style="background: #FFFFFF; border: 1px solid #EADBCC; border-radius: 12px; padding: 1.4rem; margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 1rem;">
                <div>
                    <h3 style="margin: 0; font-size: 1.4rem; font-weight: 800; color: #1C1917;">
                        📈 Kinematics Science: How Sensors Distinguish Bumps vs Potholes
                    </h3>
                    <p style="margin: 3px 0 0 0; font-size: 0.92rem; color: #78716C;">
                        The exact mathematical and physical IMU dynamics utilized by our 30-feature Random Forest ML model.
                    </p>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span style="background: #FFFBEB; color: #B45309; border: 1px solid #F59E0B; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;">
                        🟡 BUMP = ELEVATION IMPULSE
                    </span>
                    <span style="background: #FEF2F2; color: #DC2626; border: 1px solid #FCA5A5; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.78rem;">
                        🔴 POTHOLE = DROP + SHOCK STRIKE
                    </span>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 16px; margin-bottom: 1.2rem;">
                <!-- Bump Card -->
                <div style="background: #FAF7F2; border: 1px solid #EADBCC; border-left: 4px solid #D97706; border-radius: 8px; padding: 1rem;">
                    <div style="font-weight: 800; color: #B45309; font-size: 1.05rem; margin-bottom: 0.4rem;">
                        🟡 Speed Bump / Breaker Signature
                    </div>
                    <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.85rem; color: #44403C; line-height: 1.6;">
                        <li><strong>No Freefall Drop:</strong> The vehicle encounters an upward incline first. Vertical acceleration rises smoothly without an initial negative dip.</li>
                        <li><strong>Single Dominant Half-Sine Peak:</strong> Suspension springs compress smoothly over 0.25–0.45s, creating a rounded positive peak (14–18 m/s²).</li>
                        <li><strong>Moderate Jerk Gradient:</strong> Rate of change of acceleration (<em>da<sub>z</sub>/dt</em>) remains moderate (&lt;60 m/s³).</li>
                        <li><strong>Smooth Angular Pitch Swing:</strong> Front axle climbs (pitch &gt; 0), followed by rear axle climbing (pitch &lt; 0).</li>
                    </ul>
                </div>
                
                <!-- Pothole Card -->
                <div style="background: #FEF2F2; border: 1px solid #FCA5A5; border-left: 4px solid #DC2626; border-radius: 8px; padding: 1rem;">
                    <div style="font-weight: 800; color: #991B1B; font-size: 1.05rem; margin-bottom: 0.4rem;">
                        🔴 Pothole / Road Cavity Signature
                    </div>
                    <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.85rem; color: #44403C; line-height: 1.6;">
                        <li><strong>Initial Freefall Dip:</strong> The wheel suddenly loses pavement support, entering brief near-zero-g freefall (<em>a<sub>z</sub></em> drops to 2–5 m/s²).</li>
                        <li><strong>Violent Bottom-Out Strike:</strong> Wheel strikes the vertical asphalt lip. Suspension bottoms out against bump stops, creating an extreme impact spike (22–35+ m/s²).</li>
                        <li><strong>Extreme Jerk Transient:</strong> Near-vertical slope creates massive shock gradient (<em>da<sub>z</sub>/dt</em> &gt; 150–300+ m/s³).</li>
                        <li><strong>High-Frequency Ringing:</strong> Chassis resonates with multiple sharp harmonic vibration peaks (<em>peak_count</em> ≥ 5).</li>
                    </ul>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Render synchronized Plotly chart
    fig_comp = create_bump_vs_pothole_comparison_figure()
    st.plotly_chart(fig_comp, use_container_width=True)

    # Parameter Comparison Table
    st.markdown("<h4 style='color: #1C1917; margin-top: 1rem; margin-bottom: 0.5rem;'>🔬 Quantitative Physics &amp; ML Feature Comparison</h4>", unsafe_allow_html=True)

    param_table = pd.DataFrame([
        {
            "Kinematic Parameter": "Primary Waveform Morphology",
            "Speed Bump / Crest": "Smooth upward dome (compression) → gentle cresting unload",
            "Pothole / Depression": "Sharp negative dip (freefall) ➔ instantaneous violent positive spike",
            "Discriminative Power": "Very High (Shape & Symmetry)",
        },
        {
            "Kinematic Parameter": "Vertical Acceleration Range (acc_z_range)",
            "Speed Bump / Crest": "6.0 – 11.0 m/s²",
            "Pothole / Depression": "18.0 – 35.0+ m/s² (Severe metal shock)",
            "Discriminative Power": "High (Amplitude scale)",
        },
        {
            "Kinematic Parameter": "Maximum Jerk Transient (jerk_max_abs)",
            "Speed Bump / Crest": "25.0 – 65.0 m/s³ (Damped suspension travel)",
            "Pothole / Depression": "120.0 – 380.0+ m/s³ (Near-instantaneous strike)",
            "Discriminative Power": "Critical (Top RF Split Feature)",
        },
        {
            "Kinematic Parameter": "Harmonic Vibration Peaks (peak_count)",
            "Speed Bump / Crest": "1 to 2 smooth peaks (axle 1, axle 2)",
            "Pothole / Depression": "4 to 8 ringing oscillation peaks (chassis resonance)",
            "Discriminative Power": "High (Vibration count)",
        },
        {
            "Kinematic Parameter": "Gyroscope Pitch Rate (gyroscope_x_range)",
            "Speed Bump / Crest": "Moderate, progressive pitch rock (0.5 – 1.2 rad/s)",
            "Pothole / Depression": "Sudden violent nose-dive followed by recoil (1.5 – 3.5 rad/s)",
            "Discriminative Power": "High (Vehicle pitch angle)",
        },
        {
            "Kinematic Parameter": "Chassis Roll Asymmetry (gyroscope_y_std)",
            "Speed Bump / Crest": "Low (both wheels usually cross bump together)",
            "Pothole / Depression": "Very High (often hit by only left or right tire)",
            "Discriminative Power": "Medium-High (Lateral tilt)",
        },
    ])

    st.dataframe(param_table, use_container_width=True, hide_index=True)


def render_window_kinematic_deepdive(window_rec: Dict[str, Any], raw_df: Optional[pd.DataFrame] = None):
    """
    Renders an in-depth kinematic waveform analysis of an actual recorded 1-second window
    from the uploaded sensor dataset.
    """
    if not window_rec:
        return

    features = window_rec.get("features", {})
    pred_class = window_rec.get("predicted_class", "Unknown")
    conf = window_rec.get("confidence_percent", 0.0)
    w_start = window_rec.get("start_seconds", 0.0)
    w_end = window_rec.get("end_seconds", w_start + 1.0)
    w_idx = window_rec.get("window_index", 0)

    is_bump = pred_class == "Bump"
    badge_bg = "#FFFBEB" if is_bump else "#FEF2F2"
    badge_color = "#B45309" if is_bump else "#DC2626"
    badge_border = "#F59E0B" if is_bump else "#DC2626"

    st.markdown(f"""
        <div style="background: {badge_bg}; border: 1.5px solid {badge_border}; border-radius: 10px; padding: 1rem 1.2rem; margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-size: 1.2rem; font-weight: 800; color: {badge_color};">
                        {'⚠️ ' if is_bump else '🚨 '}Window #{w_idx}: {pred_class.upper()} ({conf}% Confidence)
                    </span>
                    <div style="font-size: 0.85rem; color: #57534E; margin-top: 2px;">
                        Time Interval: <strong>{w_start:.2f}s – {w_end:.2f}s</strong> • Provenance: <code>{window_rec.get('coordinate_provenance', 'unknown')}</code>
                    </div>
                </div>
                <div style="text-align: right; background: #FFFFFF; padding: 6px 14px; border-radius: 6px; border: 1px solid #EADBCC;">
                    <div style="font-size: 0.72rem; color: #78716C; text-transform: uppercase; font-weight: 700;">Max Shock (a_z)</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: {badge_color};">
                        {features.get('accelerometer_z_max_abs', 0.0):.2f} <span style="font-size: 0.8rem;">m/s²</span>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Plot actual cropped waveform if raw_df is available
    if raw_df is not None and not raw_df.empty:
        t_col = "seconds_elapsed" if "seconds_elapsed" in raw_df.columns else None
        if t_col:
            crop = raw_df[(raw_df[t_col] >= w_start) & (raw_df[t_col] <= w_end)]
            if not crop.empty:
                fig_crop = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=[
                        f"Actual Accelerometer Signals (Window {w_idx})",
                        f"Actual Gyroscope Angular Velocity (Window {w_idx})"
                    ],
                )

                t_crop = crop[t_col]
                # Acc traces
                if "accelerometer_x" in crop.columns:
                    fig_crop.add_trace(go.Scatter(x=t_crop, y=crop["accelerometer_x"], name="Acc X (Lateral)", line=dict(color="#2563EB", width=1.2)), row=1, col=1)
                if "accelerometer_y" in crop.columns:
                    fig_crop.add_trace(go.Scatter(x=t_crop, y=crop["accelerometer_y"], name="Acc Y (Forward)", line=dict(color="#D97706", width=1.2)), row=1, col=1)
                if "accelerometer_z" in crop.columns:
                    fig_crop.add_trace(go.Scatter(x=t_crop, y=crop["accelerometer_z"], name="Acc Z (Vertical Shock)", line=dict(color=badge_color, width=2.0)), row=1, col=1)

                # Gyro traces
                if "gyroscope_x" in crop.columns:
                    fig_crop.add_trace(go.Scatter(x=t_crop, y=crop["gyroscope_x"], name="Gyro X (Pitch)", line=dict(color="#059669", width=1.2)), row=1, col=2)
                if "gyroscope_y" in crop.columns:
                    fig_crop.add_trace(go.Scatter(x=t_crop, y=crop["gyroscope_y"], name="Gyro Y (Roll)", line=dict(color="#7C3AED", width=1.2)), row=1, col=2)
                if "gyroscope_z" in crop.columns:
                    fig_crop.add_trace(go.Scatter(x=t_crop, y=crop["gyroscope_z"], name="Gyro Z (Yaw)", line=dict(color="#DC2626", width=1.2)), row=1, col=2)

                fig_crop.update_layout(
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FAF7F2",
                    height=260,
                    margin=dict(l=30, r=30, t=35, b=30),
                    font=dict(color="#1C1917"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                )
                fig_crop.update_xaxes(gridcolor="#EADBCC", title_text="Seconds Elapsed")
                fig_crop.update_yaxes(gridcolor="#EADBCC")
                st.plotly_chart(fig_crop, use_container_width=True)

    # 30-feature grid for this window
    if features:
        with st.expander(f"🔍 View all 30 Kinematic Features for Window #{w_idx}", expanded=False):
            feature_items = [{"Feature": k, "Extracted Value": round(float(v), 4) if isinstance(v, (int, float)) else str(v)} for k, v in features.items()]
            df_feat = pd.DataFrame(feature_items)
            c_f1, c_f2 = st.columns(2)
            half = len(df_feat) // 2
            with c_f1:
                st.dataframe(df_feat.iloc[:half], use_container_width=True, hide_index=True)
            with c_f2:
                st.dataframe(df_feat.iloc[half:], use_container_width=True, hide_index=True)
