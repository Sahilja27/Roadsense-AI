"""
RoadSense AI - Municipal Road Authority & PWD Operations Portal
==============================================================
Chhatrapati Sambhajinagar Municipal Corporation (CSMC / ASCDCL / PWD Maharashtra)
Dedicated operations portal for municipal engineers and public works departments:
1. Pavement Network Quality Index & Urgent P1 Emergency Triage Matrix.
2. Ward-based Incident Work Queue with SLA Compliance Tracking.
3. Asphalt Material & Repair Cost Estimator in ₹ INR (Maharashtra PWD DSR).
4. Contractor Maintenance Work Order Assignment & 6-Stage Repair Lifecycle.
5. Chronic Road Memory Hotspot GIS Intelligence & Printable CSMC Work Orders.
"""

import streamlit as st
import pandas as pd
import time
import math
from typing import Dict, Any, List, Optional

from utils.api_client import (
    list_incidents,
    assign_incident,
    resolve_incident,
    update_incident,
    get_map_hotspots,
    get_dashboard_summary,
    is_backend_online,
)
from utils.street_view_dataset import (
    AURANGABAD_MUNICIPAL_WARDS,
    get_google_maps_url,
    get_google_street_view_url,
    find_nearest_aurangabad_landmark,
    render_live_footage_inspector,
)
from utils.ui_theme import apply_theme

# Maharashtra PWD Standard DSR (District Schedule of Rates) Unit Costs
PWD_DSR_RATES = {
    "asphalt_hot_mix_per_tonne": 6800.0,    # ₹ per metric tonne of Bituminous Concrete (BC / DBM)
    "tack_coat_per_liter": 95.0,            # ₹ per liter of SS-1 Bitumen Emulsion
    "gsb_aggregate_per_m3": 1450.0,         # ₹ per m^3 of Granular Sub-Base
    "labor_crew_per_day": 5200.0,           # ₹ per day (1 Supervisor + 4 Skilled Road Workers)
    "machinery_roller_per_day": 7500.0,     # ₹ per day (3-Ton Mini Vibratory Roller + Cutter)
    "base_pothole_avg_sqm": 0.65,           # Average pothole surface area in m^2
    "base_pothole_avg_depth_m": 0.065,      # Average pothole depth in meters (65mm)
    "asphalt_density_t_per_m3": 2.4,        # Tonnes per m^3 compacted hot-mix
}


def calculate_pothole_repair_materials(
    num_potholes: int,
    avg_area_sqm: float = 0.65,
    avg_depth_mm: float = 65.0,
) -> Dict[str, Any]:
    """
    Computes precise material tonnage, tack coat volume, labor hours, and
    estimated repair cost in ₹ INR based on Maharashtra PWD DSR rates.
    """
    depth_m = avg_depth_mm / 1000.0
    total_volume_m3 = num_potholes * avg_area_sqm * depth_m
    total_tonnage = total_volume_m3 * PWD_DSR_RATES["asphalt_density_t_per_m3"]
    tack_coat_liters = num_potholes * avg_area_sqm * 0.45  # 0.45 kg/m^2 approx 0.45 L
    gsb_volume_m3 = total_volume_m3 * 0.4  # Sub-base levelling course

    # Cost Calculation
    asphalt_cost = total_tonnage * PWD_DSR_RATES["asphalt_hot_mix_per_tonne"]
    tack_cost = tack_coat_liters * PWD_DSR_RATES["tack_coat_per_liter"]
    gsb_cost = gsb_volume_m3 * PWD_DSR_RATES["gsb_aggregate_per_m3"]
    
    # Days of crew work (approx 12-15 potholes patched per day per crew)
    crew_days = max(1.0, math.ceil(num_potholes / 12.0)) if num_potholes > 0 else 0.0
    labor_cost = crew_days * PWD_DSR_RATES["labor_crew_per_day"]
    machinery_cost = crew_days * PWD_DSR_RATES["machinery_roller_per_day"]
    
    subtotal = asphalt_cost + tack_cost + gsb_cost + labor_cost + machinery_cost
    contingency = subtotal * 0.08  # 8% departmental contingency
    total_cost_inr = subtotal + contingency

    return {
        "num_potholes": num_potholes,
        "total_volume_m3": round(total_volume_m3, 2),
        "asphalt_tonnage": round(total_tonnage, 2),
        "tack_coat_liters": round(tack_coat_liters, 1),
        "gsb_volume_m3": round(gsb_volume_m3, 2),
        "crew_days": crew_days,
        "asphalt_cost_inr": round(asphalt_cost, 0),
        "labor_cost_inr": round(labor_cost, 0),
        "machinery_cost_inr": round(machinery_cost, 0),
        "total_cost_inr": round(total_cost_inr, 0),
        "cost_per_pothole_inr": round(total_cost_inr / max(1, num_potholes), 0),
    }


def matches_csmc_ward(inc: Dict[str, Any], target_ward_name: str) -> bool:
    """Accurately checks if an incident falls within a designated CSMC municipal ward."""
    if not target_ward_name or target_ward_name == "All CSMC Wards":
        return True
    addr = str(inc.get("address_text", "")).lower()
    t_low = target_ward_name.lower()
    keywords = [
        "kranti chowk", "jalna road", "cidco", "chikalthana", 
        "beed bypass", "nirala bazar", "seven hills", "cannaught", "airport"
    ]
    for kw in keywords:
        if kw in t_low and kw in addr:
            return True
    lat = inc.get("latitude")
    lon = inc.get("longitude")
    if lat and lon:
        try:
            w_info = find_nearest_aurangabad_landmark(float(lat), float(lon))
            if w_info["ward_name"].strip().lower() == target_ward_name.strip().lower():
                return True
            if w_info["ward_id"] in target_ward_name:
                return True
        except Exception:
            pass
    return False


def render_authority_view():
    apply_theme()
    backend_status = is_backend_online()

    st.markdown(f"""
        <div style="padding-bottom: 0.8rem; border-bottom: 1px solid #EADBCC; margin-bottom: 1.2rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #1C1917;">
                        🏛️ Municipal Road Authority &amp; Triage Interface
                    </h1>
                    <p style="margin: 4px 0 0 0; font-size: 1rem; color: #78716C;">
                        Chhatrapati Sambhajinagar Municipal Corporation (CSMC) • Pavement Asset Management &amp; Work Orders
                    </p>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span style="background: #FEF2F2; color: #DC2626; border: 1px solid #FCA5A5; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                        ● CSMC / ASCDCL OPERATIONS
                    </span>
                    <span style="background: {'#ECFDF5' if backend_status else '#FEF2F2'}; 
                                 color: {'#059669' if backend_status else '#DC2626'}; 
                                 border: 1px solid {'#10B981' if backend_status else '#FCA5A5'}; 
                                 padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                        ● {'DATABASE SYNCED' if backend_status else 'OFFLINE FALLBACK'}
                    </span>
                    <span style="background: #FAF7F2; color: #78716C; border: 1px solid #EADBCC; padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                        MAHARASHTRA PWD DSR
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    incidents = list_incidents()
    hotspots = get_map_hotspots()
    summary = get_dashboard_summary()

    total_inc = len(incidents)
    urgent_p1 = len([i for i in incidents if i.get("severity") in ["High", "Critical"] and i.get("status") != "resolved"])
    assigned_count = len([i for i in incidents if i.get("status") == "assigned"])
    resolved_count = len([i for i in incidents if i.get("status") == "resolved"])
    pothole_count = len([i for i in incidents if "pothole" in str(i.get("distress_type", "")).lower()])
    bump_count = len([i for i in incidents if "bump" in str(i.get("distress_type", "")).lower()])

    # Material cost estimate for all active defects
    cost_calc = calculate_pothole_repair_materials(max(urgent_p1 + assigned_count, pothole_count if pothole_count > 0 else 14))

    # 1. Authority Top KPI Metric Cards (5 Columns)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #DC2626;">
                <div class="kpi-title">Urgent P1 Hazards</div>
                <div class="kpi-value" style="color: #DC2626;">{urgent_p1}</div>
                <div class="kpi-sub">SLA &lt;24h Emergency</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #D97706;">
                <div class="kpi-title">Field Work Orders</div>
                <div class="kpi-value" style="color: #B45309;">{assigned_count}</div>
                <div class="kpi-sub">Contractor Dispatched</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #059669;">
                <div class="kpi-title">Verified Repairs</div>
                <div class="kpi-value" style="color: #059669;">{resolved_count}</div>
                <div class="kpi-sub">QA Passed / Sealed</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #78716C;">
                <div class="kpi-title">Chronic Clusters</div>
                <div class="kpi-value" style="color: #1C1917;">{len(hotspots)}</div>
                <div class="kpi-sub">Road Memory Hotspots</div>
            </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
            <div class="kpi-card" style="border-top: 3px solid #991B1B;">
                <div class="kpi-title">Est. Repair Budget</div>
                <div class="kpi-value" style="color: #991B1B;">₹{cost_calc['total_cost_inr']:,.0f}</div>
                <div class="kpi-sub">{cost_calc['asphalt_tonnage']}t Hot-Mix Asphalt</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    tab_incidents, tab_cost, tab_hotspots, tab_actions = st.tabs([
        "📋 Incident Triage & Work Orders",
        "💰 Asphalt Material & Cost Estimator (₹ INR)",
        "🧠 Chronic Road Memory Hotspots",
        "📜 Audit Trail & Compliance",
    ])

    with tab_incidents:
        # Quick 1-click Filter Buttons for Potholes and Bumps
        st.markdown("<div style='margin-bottom: 0.5rem; font-weight: 700; color: #1C1917; font-size: 0.92rem;'>⚡ Quick Hazard Filter:</div>", unsafe_allow_html=True)
        b_c1, b_c2, b_c3 = st.columns(3)
        if "authority_hazard_filter" not in st.session_state:
            st.session_state["authority_hazard_filter"] = "All"

        with b_c1:
            if st.button(f"🔘 All Road Defects ({total_inc})", use_container_width=True, type="primary" if st.session_state["authority_hazard_filter"] == "All" else "secondary"):
                st.session_state["authority_hazard_filter"] = "All"
                st.rerun()
        with b_c2:
            if st.button(f"🚨 Potholes Only ({pothole_count})", use_container_width=True, type="primary" if st.session_state["authority_hazard_filter"] == "Pothole" else "secondary"):
                st.session_state["authority_hazard_filter"] = "Pothole"
                st.rerun()
        with b_c3:
            if st.button(f"⚠️ Speed Bumps Only ({bump_count})", use_container_width=True, type="primary" if st.session_state["authority_hazard_filter"] == "Bump" else "secondary"):
                st.session_state["authority_hazard_filter"] = "Bump"
                st.rerun()

        st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)

        # Detailed Ward & Attribute Filters
        col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns([1.6, 1.2, 1.4, 1.1, 1.1])
        with col_f1:
            ward_choices = ["All CSMC Wards"] + [w["ward_name"] for w in AURANGABAD_MUNICIPAL_WARDS]
            selected_ward_filter = st.selectbox("📍 Municipal Ward Filter", ward_choices, index=0)
        with col_f2:
            hazard_dropdown_options = ["All Classes", "Pothole", "Bump"]
            current_choice_idx = 0 if st.session_state["authority_hazard_filter"] == "All" else (1 if st.session_state["authority_hazard_filter"] == "Pothole" else 2)
            hazard_dropdown = st.selectbox("Hazard Class Filter", hazard_dropdown_options, index=current_choice_idx)
            if hazard_dropdown != "All Classes" and st.session_state["authority_hazard_filter"] != hazard_dropdown:
                st.session_state["authority_hazard_filter"] = hazard_dropdown
            elif hazard_dropdown == "All Classes" and st.session_state["authority_hazard_filter"] != "All":
                st.session_state["authority_hazard_filter"] = "All"
        with col_f3:
            search_query = st.text_input("🔍 Search Defects / Roads", placeholder="e.g. Pothole, Jalna Road...")
        with col_f4:
            status_filter = st.selectbox("Status Filter", ["All", "reported", "under_review", "assigned", "scheduled", "resolved"])
        with col_f5:
            sev_filter = st.selectbox("Severity Filter", ["All", "Critical", "High", "Medium", "Low"])

        active_hazard_class = st.session_state.get("authority_hazard_filter", "All")
        filtered_inc = []

        for i in incidents:
            # 1. Hazard class filter
            if active_hazard_class != "All" and i.get("distress_type") != active_hazard_class:
                continue
            
            # 2. Ward filter
            if not matches_csmc_ward(i, selected_ward_filter):
                continue
            
            # 3. Status filter
            if status_filter != "All" and i.get("status") != status_filter:
                continue
            
            # 4. Severity filter
            if sev_filter != "All" and i.get("severity") != sev_filter:
                continue
            
            # 5. Search query with intelligent typo-tolerance (handles 'potehole', 'bumop', etc.)
            if search_query:
                sq = search_query.strip().lower()
                is_pothole_search = any(term in sq for term in ["potehole", "pothole", "pothol", "potholes", "pit", "crater", "cavity"])
                is_bump_search = any(term in sq for term in ["bumop", "bump", "bumps", "speed breaker", "speedbreaker", "breaker", "hump", "ramp"])
                
                title_text = str(i.get("title", "")).lower()
                notes_text = str(i.get("notes", "")).lower()
                addr_text = str(i.get("address_text", "")).lower()
                d_type = str(i.get("distress_type", "")).lower()

                if is_pothole_search:
                    if d_type != "pothole" and "pothole" not in title_text and "pothole" not in notes_text and "pothole" not in addr_text:
                        continue
                elif is_bump_search:
                    if d_type != "bump" and "bump" not in title_text and "breaker" not in title_text and "bump" not in notes_text:
                        continue
                else:
                    if sq not in title_text and sq not in d_type and sq not in notes_text and sq not in addr_text:
                        continue

            filtered_inc.append(i)

        if filtered_inc:
            table_data = []
            for i in filtered_inc:
                lat = i.get("latitude")
                lon = i.get("longitude")
                ward_tag = "Jalna Road Division"
                if lat and lon:
                    try:
                        w_info = find_nearest_aurangabad_landmark(float(lat), float(lon))
                        ward_tag = w_info["ward_name"].split(" ")[0] + "..."
                    except Exception:
                        pass

                # Cost estimate per defect
                sev = i.get("severity", "Medium")
                est_cost = 4200 if sev == "Critical" else (2800 if sev == "High" else 1500)

                table_data.append({
                    "ID": i.get("id"),
                    "Title": i.get("title"),
                    "Distress Type": i.get("distress_type"),
                    "Severity": i.get("severity"),
                    "Status": str(i.get("status", "")).upper(),
                    "Municipal Ward": ward_tag,
                    "Est. Cost (₹)": f"₹{est_cost:,}",
                    "Assigned To": i.get("assigned_to") or "Unassigned",
                    "Reported": str(i.get("created_at", ""))[:16],
                })
            
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            # Defect Selection for Detection & Inspection
            inc_options = [
                f"Incident #{i['id']} | {i['distress_type']} | {i['title']}"
                for i in filtered_inc
            ]
            
            st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
            col_sel1, col_sel2 = st.columns([2.5, 1])
            with col_sel1:
                selected_inc_str = st.selectbox("🎯 Select Defect to Detect & Inspect Live on Map", inc_options, index=0)
                selected_id = int(selected_inc_str.split("|")[0].replace("Incident #", "").strip())
            with col_sel2:
                st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
                st.caption(f"Showing 1 of {len(filtered_inc)} filtered incidents")

            sel_inc = next((i for i in filtered_inc if i.get("id") == selected_id), filtered_inc[0])
            s_lat = float(sel_inc.get("latitude") or 19.87698)
            s_lon = float(sel_inc.get("longitude") or 75.35380)
            s_class = sel_inc.get("distress_type", "Pothole")
            s_conf = float(sel_inc.get("confidence") or 0.92)
            s_title = sel_inc.get("title", "Road Distress")
            s_addr = sel_inc.get("address_text") or f"Jalna Road corridor near {s_lat:.4f}, {s_lon:.4f}"

            # 🎯 Defect Detection & Live Inspection Station
            st.markdown("""
                <div style="background: #FFFFFF; border: 1.5px solid #EADBCC; border-radius: 12px; padding: 1.2rem; margin: 1rem 0; box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #EADBCC; padding-bottom: 8px; margin-bottom: 12px;">
                        <h4 style="margin: 0; color: #1C1917; font-size: 1.15rem;">
                            🎯 Live AI Dashcam Verification &amp; Sensor Telemetry (Defect #{})
                        </h4>
                        <span style="background: {}; color: {}; border: 1px solid {}; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                            {}
                        </span>
                    </div>
                </div>
            """.format(
                selected_id,
                '#FEF2F2' if s_class == 'Pothole' else '#FFFBEB',
                '#DC2626' if s_class == 'Pothole' else '#B45309',
                '#FCA5A5' if s_class == 'Pothole' else '#F59E0B',
                f"🚨 {s_class.upper()} DETECTED" if s_class == 'Pothole' else f"⚠️ {s_class.upper()} DETECTED"
            ), unsafe_allow_html=True)

            render_live_footage_inspector(
                distress_type=s_class,
                confidence=s_conf,
                location_name=s_addr,
                latitude=s_lat,
                longitude=s_lon,
                speed_kmh=42.0,
            )

            # Sensor Shock & Kinematic Details Badge
            st.markdown(f"""
                <div style="background: #FAF7F2; border: 1px solid #EADBCC; border-radius: 8px; padding: 12px 16px; margin: 10px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <strong style="color: #1C1917; font-size: 0.95rem;">🔬 Ingested Sensor Shock Telemetry:</strong>
                        <div style="color: #57534E; font-size: 0.85rem; margin-top: 3px;">
                            {sel_inc.get('notes', 'Kinematic sensor shock detected via smartphone accelerometer.')}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: #ECFDF5; color: #059669; border: 1px solid #10B981; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.82rem;">
                            VERIFICATION: {str(sel_inc.get('verification_status', 'SENSOR_CONFIRMED')).upper()}
                        </span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Incident Action & Work Order Dispatch Panel
            st.markdown("<h4 style='color: #1C1917; margin-top: 1.2rem; margin-bottom: 0.4rem;'>⚙️ CSMC Work Order Dispatch &amp; Status Pipeline</h4>", unsafe_allow_html=True)
            
            col_act1, col_act2, col_act3, col_act4 = st.columns([1, 1.2, 1.5, 1])

            with col_act1:
                action_inc_id = st.selectbox("Dispatch Incident ID", options=[i.get("id") for i in filtered_inc], index=[i.get("id") for i in filtered_inc].index(selected_id) if selected_id in [i.get("id") for i in filtered_inc] else 0)
            with col_act2:
                action_choice = st.selectbox(
                    "Lifecycle Action",
                    [
                        "Assign to Field Contractor",
                        "Mark Under Review",
                        "Mark Scheduled",
                        "Mark Asphalt Laying Active",
                        "Resolve & QA Pass",
                    ]
                )
            with col_act3:
                contractor_list = [
                    "CSMC Rapid Response Squad 1 (Kranti Chowk Depot)",
                    "PWD Aurangabad Division Team Alpha",
                    "ASCDCL Smart City Highway Squad",
                    "Contractor: Shri Ganesh Infra Roadways",
                    "Contractor: Marathwada Paving & Infrastructure",
                ]
                assignee_input = st.selectbox("Contractor / Crew", contractor_list)
            with col_act4:
                st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
                if st.button("Apply Action", type="primary", use_container_width=True):
                    try:
                        if "Assign" in action_choice:
                            assign_incident(action_inc_id, assigned_to=assignee_input, notes=f"Work Order Dispatched to {assignee_input}")
                            st.success(f"✓ Incident #{action_inc_id} assigned to {assignee_input}!")
                        elif "Resolve" in action_choice:
                            resolve_incident(action_inc_id, resolution_notes="Hot-mix asphalt laid, compacted with mini-roller, QA verified.")
                            st.success(f"✓ Incident #{action_inc_id} marked as RESOLVED & QA VERIFIED!")
                        elif "Asphalt Laying" in action_choice:
                            update_incident(action_inc_id, {"status": "assigned", "notes": "Asphalt hot-mix laying in progress."})
                            st.info(f"✓ Incident #{action_inc_id} updated: Asphalt Laying Active.")
                        elif "Scheduled" in action_choice:
                            update_incident(action_inc_id, {"status": "scheduled"})
                            st.success(f"✓ Incident #{action_inc_id} scheduled for maintenance.")
                        elif "Review" in action_choice:
                            update_incident(action_inc_id, {"status": "under_review"})
                            st.success(f"✓ Incident #{action_inc_id} placed under engineering review.")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Action failed: {e}")

            # Official CSMC Work Order Slip Generator
            sel_inc = next((i for i in filtered_inc if i.get("id") == selected_id), filtered_inc[0])
            sel_lat = sel_inc.get("latitude") or 19.8769
            sel_lon = sel_inc.get("longitude") or 75.3538
            ward_info = find_nearest_aurangabad_landmark(sel_lat, sel_lon)

            with st.expander(f"📄 Official CSMC Work Order Dispatch Slip (WO-{selected_id:04d})", expanded=False):
                st.markdown(f"""
                    <div style="background: #FFFFFF; border: 2px solid #1C1917; border-radius: 8px; padding: 1.5rem; font-family: monospace; color: #1C1917;">
                        <div style="text-align: center; border-bottom: 2px solid #1C1917; padding-bottom: 8px; margin-bottom: 12px;">
                            <div style="font-size: 1.25rem; font-weight: 800; letter-spacing: 1px;">CHHATRAPATI SAMBHAJINAGAR MUNICIPAL CORPORATION</div>
                            <div style="font-size: 0.9rem; color: #57534E;">Public Works Department (PWD) • Road Maintenance Work Order</div>
                            <div style="font-size: 0.8rem; color: #78716C; margin-top: 2px;">Issued under Maharashtra Municipal Corporations Act • Form PWD-R14</div>
                        </div>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.88rem; margin-bottom: 12px;">
                            <div><strong>WORK ORDER NO:</strong> CSMC-WO-{selected_id:04d}</div>
                            <div><strong>DATE ISSUED:</strong> {str(sel_inc.get('created_at', ''))[:10]}</div>
                            <div><strong>MUNICIPAL WARD:</strong> {ward_info['ward_name']}</div>
                            <div><strong>ZONE / DIVISION:</strong> {ward_info['zone']}</div>
                            <div><strong>GPS LOCATION:</strong> {sel_lat:.5f}° N, {sel_lon:.5f}° E</div>
                            <div><strong>ROAD CORRIDOR:</strong> Jalna Road (NH-753F) Arterial</div>
                            <div><strong>DEFECT TYPE:</strong> {sel_inc.get('distress_type', 'Pothole').upper()}</div>
                            <div><strong>SEVERITY LEVEL:</strong> {sel_inc.get('severity', 'High').upper()} (SLA: &lt;{ward_info['sla_target_hours']}h)</div>
                            <div><strong>ASSIGNED CONTRACTOR:</strong> {sel_inc.get('assigned_to') or 'CSMC Rapid Response Unit 1'}</div>
                            <div><strong>SUPERVISING ENGINEER:</strong> {ward_info['engineer']}</div>
                        </div>

                        <div style="border-top: 1px dashed #78716C; padding-top: 8px; margin-top: 8px; font-size: 0.82rem; line-height: 1.5;">
                            <strong>REQUIRED MATERIALS &amp; SPECIFICATIONS (PWD DSR 2025-26):</strong><br>
                            • Dense Bituminous Macadam (DBM) / Bituminous Concrete (BC): ~0.85 MT compacted<br>
                            • Bitumen Emulsion Tack Coat SS-1: ~1.2 Liters @ 0.45 kg/m²<br>
                            • Compaction Standard: Vibratory Mini Roller (3-Pass minimum until non-yielding)<br>
                            • Estimated Job Cost: ₹2,800 – ₹4,200 (Chargeable to Ward Road Maintenance Head)
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Export Work Orders
            st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
            df_export = pd.DataFrame(table_data)
            csv_export = df_export.to_csv(index=False)
            st.download_button(
                "📥 Export Filtered Work Orders (CSV)",
                data=csv_export,
                file_name="roadsense_csmc_work_orders.csv",
                mime="text/csv",
            )
        else:
            st.info("No road incidents found matching criteria.")

    with tab_cost:
        st.markdown("<h4 style='color: #1C1917; margin-bottom: 0.4rem;'>💰 Maharashtra PWD DSR Asphalt Material &amp; Cost Calculator</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color: #78716C; font-size: 0.9rem;'>Standardized algorithmic cost estimation for Bituminous Concrete (BC), Bitumen Emulsion, GSB sub-base, and labor crew deployment.</p>", unsafe_allow_html=True)

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            inp_potholes = st.number_input("Number of Potholes to Patch", min_value=1, max_value=500, value=urgent_p1 if urgent_p1 > 0 else 12)
        with col_in2:
            inp_area = st.slider("Average Pothole Surface Area (m²)", min_value=0.2, max_value=2.5, value=0.65, step=0.05)
        with col_in3:
            inp_depth = st.slider("Average Pothole Depth (mm)", min_value=30, max_value=120, value=65, step=5)

        calc = calculate_pothole_repair_materials(inp_potholes, avg_area_sqm=inp_area, avg_depth_mm=inp_depth)

        # Output Cards
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 3px solid #DC2626;">
                    <div class="kpi-title">Total Estimated Budget</div>
                    <div class="kpi-value" style="color: #DC2626;">₹{calc['total_cost_inr']:,.0f}</div>
                    <div class="kpi-sub">₹{calc['cost_per_pothole_inr']:,.0f} per pothole patch</div>
                </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 3px solid #D97706;">
                    <div class="kpi-title">Asphalt Hot-Mix (BC/DBM)</div>
                    <div class="kpi-value" style="color: #B45309;">{calc['asphalt_tonnage']} <span style="font-size: 1.1rem;">MT</span></div>
                    <div class="kpi-sub">Compact Volume: {calc['total_volume_m3']} m³</div>
                </div>
            """, unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 3px solid #059669;">
                    <div class="kpi-title">Tack Coat Emulsion</div>
                    <div class="kpi-value" style="color: #059669;">{calc['tack_coat_liters']} <span style="font-size: 1.1rem;">L</span></div>
                    <div class="kpi-sub">SS-1 Bitumen Emulsion</div>
                </div>
            """, unsafe_allow_html=True)
        with r4:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 3px solid #78716C;">
                    <div class="kpi-title">Crew Deployment</div>
                    <div class="kpi-value" style="color: #1C1917;">{calc['crew_days']} <span style="font-size: 1.1rem;">Days</span></div>
                    <div class="kpi-sub">1 Team + 3t Mini-Roller</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # Cost Breakdown Itemized Table
        cost_breakdown_data = [
            {"Expenditure Head": "Bituminous Concrete (BC / DBM Hot-Mix)", "Quantity": f"{calc['asphalt_tonnage']} MT", "PWD DSR Rate": "₹6,800 / MT", "Total Amount (₹)": f"₹{calc['asphalt_cost_inr']:,.0f}"},
            {"Expenditure Head": "Tack Coat (SS-1 Bitumen Emulsion)", "Quantity": f"{calc['tack_coat_liters']} Liters", "PWD DSR Rate": "₹95 / Liter", "Total Amount (₹)": f"₹{calc['tack_coat_liters'] * 95:,.0f}"},
            {"Expenditure Head": "Granular Sub-Base (GSB Crushed Stone)", "Quantity": f"{calc['gsb_volume_m3']} m³", "PWD DSR Rate": "₹1,450 / m³", "Total Amount (₹)": f"₹{calc['gsb_volume_m3'] * 1450:,.0f}"},
            {"Expenditure Head": "Skilled PWD Road Repair Crew", "Quantity": f"{calc['crew_days']} Shift Days", "PWD DSR Rate": "₹5,200 / Day", "Total Amount (₹)": f"₹{calc['labor_cost_inr']:,.0f}"},
            {"Expenditure Head": "3-Ton Mini Vibratory Roller & Asphalt Cutter", "Quantity": f"{calc['crew_days']} Machine Days", "PWD DSR Rate": "₹7,500 / Day", "Total Amount (₹)": f"₹{calc['machinery_cost_inr']:,.0f}"},
            {"Expenditure Head": "Departmental Contingency & Testing (8%)", "Quantity": "Standard 8%", "PWD DSR Rate": "8% of subtotal", "Total Amount (₹)": f"₹{calc['total_cost_inr'] * 0.08:,.0f}"},
            {"Expenditure Head": "TOTAL ESTIMATED SANCTION BUDGET", "Quantity": f"{inp_potholes} Potholes", "PWD DSR Rate": "-", "Total Amount (₹)": f"₹{calc['total_cost_inr']:,.0f}"},
        ]
        st.dataframe(pd.DataFrame(cost_breakdown_data), use_container_width=True, hide_index=True)

    with tab_hotspots:
        st.markdown("<h4 style='color: #1C1917;'>📍 Chronic Road Memory Hotspot Clusters — Chhatrapati Sambhajinagar</h4>", unsafe_allow_html=True)
        if hotspots:
            hotspot_table = []
            for h in hotspots:
                clat = h.get("center_lat", 19.8762)
                clon = h.get("center_lon", 75.3433)
                gmaps_link = get_google_maps_url(clat, clon)
                hotspot_table.append({
                    "Cluster Name": h.get("cluster_name"),
                    "Primary Distress": h.get("primary_hazard"),
                    "Severity": h.get("avg_severity"),
                    "Impact Recurrence": h.get("recurrence_count"),
                    "Priority Score": h.get("priority_score"),
                    "Status": str(h.get("status", "")).upper(),
                    "Coordinates": f"{clat:.4f}, {clon:.4f}",
                    "Last Detected": str(h.get("last_detected_at", ""))[:16],
                })
            st.dataframe(pd.DataFrame(hotspot_table), use_container_width=True)
        else:
            st.info("No chronic hotspot clusters detected yet. Process sensor recordings along Jalna Road to build spatial memory.")

    with tab_actions:
        st.markdown("<h4 style='color: #1C1917;'>📜 Municipal Authority Activity &amp; Audit Trail</h4>", unsafe_allow_html=True)
        activity = summary.get("recent_activity", [])
        if activity:
            st.dataframe(pd.DataFrame(activity), use_container_width=True)
        else:
            st.caption("No authority actions logged yet.")


# Execute when loaded as a page
render_authority_view()
