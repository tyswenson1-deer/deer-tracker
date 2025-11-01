import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deer Tracker", layout="centered")

st.title("🦌 Deer Recovery Probability Map")
st.write("Answer the following questions to estimate where your deer might have gone.")

# Sidebar questions
st.sidebar.header("Shot Info")

shot_type = st.sidebar.selectbox(
    "Where was the deer hit?",
    ["Lung", "Liver", "Gut", "Shoulder", "Unknown"]
)

reaction = st.sidebar.selectbox(
    "How did the deer react?",
    ["Mule kick", "Hunched", "Ran hard", "Walked away", "Stood then walked"]
)

blood = st.sidebar.selectbox(
    "What did the blood look like?",
    ["Bright red w/ bubbles", "Dark red", "Watery", "Green/yellow", "None found"]
)

terrain = st.sidebar.selectbox(
    "Terrain type",
    ["Flat", "Rolling", "Steep ridge", "Creek bottom", "Thick brush", "Open field"]
)

wind = st.sidebar.selectbox(
    "Wind direction",
    ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
)

time_since = st.sidebar.slider("Hours since shot", 0.0, 24.0, 4.0, 0.5)

# Base grid setup
size_yards = 500
res = 10
grid_size = int(size_yards / res)
x = np.linspace(0, size_yards, grid_size)
y = np.linspace(0, size_yards, grid_size)
X, Y = np.meshgrid(x, y)

# Shot location (south edge center)
shot_x, shot_y = size_yards / 2, 0

# Base probability kernel
dist = np.sqrt((X - shot_x)**2 + (Y - shot_y)**2)
base_sigma = 80 if shot_type in ["Liver", "Gut"] else 60
K = np.exp(-0.5 * (dist / base_sigma)**2)

# Directional bias (north)
dir_bias = np.exp(-((np.degrees(np.arctan2(Y - shot_y, X - shot_x)) - 90) ** 2) / (2 * 50 ** 2))

# Blood factor
blood_factor = {"Bright red w/ bubbles": 0.6, "Dark red": 1.0, "Watery": 1.2, "Green/yellow": 1.5, "None found": 1.4}[blood]

# Terrain factor
terrain_factor = {"Flat": 1.0, "Rolling": 1.1, "Steep ridge": 1.2, "Creek bottom": 0.8, "Thick brush": 0.9, "Open field": 1.0}[terrain]

# Time decay
time_factor = np.exp(-time_since / 12)

# Combine
P = K * dir_bias * blood_factor * terrain_factor * time_factor
P /= P.max()  # Normalize

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
heatmap = ax.imshow(P, extent=[0, size_yards, 0, size_yards], origin="lower", cmap="YlOrRd")
ax.scatter(shot_x, shot_y, color="blue", marker="x", label="Shot Point")
ax.set_xlabel("East-West (yards)")
ax.set_ylabel("South-North (yards)")
ax.set_title("Probability Map")
plt.colorbar(heatmap, ax=ax, label="Relative Probability")
st.pyplot(fig)

st.markdown("### 🔎 Recommended starting area:")
max_idx = np.unravel_index(np.argmax(P), P.shape)
top_x = x[max_idx[1]]
top_y = y[max_idx[0]]
st.write(f"Start your search around **{int(top_x)} yds East, {int(top_y)} yds North** from your shot point.")
st.caption("This tool gives a data-based estimate. Always combine with your own sign and experience.")
