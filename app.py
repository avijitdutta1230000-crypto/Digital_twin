"""Live Smart City Digital Twin dashboard.

    streamlit run app.py

Replays the city hour by hour, shows the live map (road congestion, zone PM2.5 /
flood / power tiers), runs the cascade engine on 'what-if' rain, and routes an
ambulance to the nearest hospital using forecast conditions.
"""
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from sctwin.city import City
from sctwin import datasets as D
from sctwin.twin import CascadeEngine, emergency_route, tier

st.set_page_config(page_title="Smart City Digital Twin", layout="wide")
COL = {"LOW": "#2e7d32", "MEDIUM": "#f9a825", "HIGH": "#c62828"}


@st.cache_resource
def boot():
    city = City()
    return city, D.simulate_city(city, days=240, seed=42)


city, sim = boot()
free = np.array([d["free_speed"] for *_, d in city.G.edges(data=True)])
T = len(sim["time"])

st.sidebar.title("Twin controls")
t = st.sidebar.slider("Simulation hour", 0, T - 1, int(np.argmax((sim["flood"].max(1) > 35) & (sim["flood"].max(1) < 70))))
extra_rain = st.sidebar.slider("What-if: extra flood depth (cm) in low zones", 0, 60, 0)
src = st.sidebar.selectbox("Emergency call at intersection", list(city.G.nodes), index=8)
st.sidebar.caption(f"Data: {sim['source']} · {city.n_sensors} road sensors · {city.n_zones} zones")

flood = sim["flood"][t] + extra_rain * sim["zone_vuln"]
cong0 = np.clip(1 - sim["speed"][t] / free, 0, .95)
eng = CascadeEngine(city)
cong, pm, load, closed, events = eng.apply(flood, cong0, sim["pm25"][t], np.ones(city.n_zones))
route = emergency_route(city, cong, src)

st.title("Smart City Digital Twin — live view")
st.caption(str(sim["time"][t]))
c = st.columns(5)
c[0].metric("Avg speed, open roads", f"{np.delete(free * (1 - cong), closed).mean():.1f}")
c[1].metric("Max PM2.5 (µg/m³)", f"{pm.max():.0f}", tier("pm25", pm.max()))
c[2].metric("Max flood (cm)", f"{flood.max():.0f}", tier("flood", flood.max()))
c[3].metric("Roads closed", len(closed))
c[4].metric("Ambulance ETA", f"{route[0]:.1f} min" if route else "NO ROUTE")

m = folium.Map(location=[city.G.nodes[24]["lat"], city.G.nodes[24]["lon"]], zoom_start=14, tiles="OpenStreetMap")
for z, nodes in city.zones.items():
    lat = [city.G.nodes[n]["lat"] for n in nodes]; lon = [city.G.nodes[n]["lon"] for n in nodes]
    tr = max(tier("flood", flood[z]), tier("pm25", pm[z]), key=["LOW", "MEDIUM", "HIGH"].index)
    folium.Rectangle([[min(lat) - .002, min(lon) - .002], [max(lat) + .002, max(lon) + .002]],
                     color=COL[tr], fill=True, fill_opacity=.12, weight=1,
                     tooltip=f"Zone {z}: PM2.5 {pm[z]:.0f}, flood {flood[z]:.0f} cm, risk {tr}").add_to(m)
for u, v, d in city.G.edges(data=True):
    pts = [[city.G.nodes[n]["lat"], city.G.nodes[n]["lon"]] for n in (u, v)]
    k = cong[d["sensor"]]
    color = "#000" if d["closed"] else ("#c62828" if k > .6 else "#f9a825" if k > .35 else "#2e7d32")
    folium.PolyLine(pts, color=color, weight=6, dash_array="6" if d["closed"] else None,
                    tooltip=f"sensor {d['sensor']}: {free[d['sensor']] * (1 - k):.0f} km/h").add_to(m)
if route:
    folium.PolyLine([[city.G.nodes[n]["lat"], city.G.nodes[n]["lon"]] for n in route[1]],
                    color="#1565c0", weight=10, opacity=.5, tooltip="ambulance route").add_to(m)
for h in city.hospitals:
    folium.Marker([city.G.nodes[h]["lat"], city.G.nodes[h]["lon"]], icon=folium.Icon(color="red", icon="plus"), tooltip="Hospital").add_to(m)
folium.Marker([city.G.nodes[src]["lat"], city.G.nodes[src]["lon"]], icon=folium.Icon(color="blue"), tooltip="Emergency").add_to(m)

left, right = st.columns([3, 2])
with left:
    st_folium(m, height=560, use_container_width=True)
with right:
    st.subheader("Cascade alerts")
    for e in events or ["No cascading events this hour."]:
        st.write("• " + e)
    st.subheader("Last 48 h")
    w = slice(max(0, t - 48), t + 1)
    st.line_chart(pd.DataFrame({"rain (mm)": sim["rain"][w], "max flood (cm)": sim["flood"][w].max(1),
                                "mean PM2.5": sim["pm25"][w].mean(1)}, index=sim["time"][w]))
