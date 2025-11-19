#!/usr/bin/env python3
"""
Animation of Earth and Moon positions relative to satellite
using spherical coordinates (elevation, azimuth, distance)
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import sys
from collections import defaultdict

# Load the pickle file
print("Loading data...")
with open('/pdo/users/djtufto/data/data_tess_4096_raw/angles/tess_angles_O11-54_data_dic.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Total number of observations: {len(data)}")

# Examine first entry
first_key = list(data.keys())[0]
first_entry = data[first_key]
print(f"\nSample observation {first_key}:")
print(f"Keys: {list(first_entry.keys())}")
print(f"Orbit: {first_entry['orbit']}")

# Group observations by orbit number
print("\nGrouping observations by orbit...")
orbits = defaultdict(list)
for obs_id, obs_data in data.items():
    orbit_num = obs_data['orbit']
    orbits[orbit_num].append((obs_id, obs_data))

# Sort observations within each orbit by observation ID
for orbit_num in orbits:
    orbits[orbit_num].sort(key=lambda x: x[0])

orbit_nums = sorted(orbits.keys())
print(f"Number of unique orbits: {len(orbit_nums)}")
print(f"Orbits: {orbit_nums}")

# Show orbit sizes
orbit_sizes = {orbit: len(obs) for orbit, obs in orbits.items()}
print(f"\nObservations per orbit:")
for orbit in orbit_nums:
    print(f"  Orbit {orbit}: {orbit_sizes[orbit]} observations")

# Choose orbit to animate
if len(sys.argv) > 1:
    orbit_to_animate = sys.argv[1]
else:
    # Use the orbit with the most observations
    orbit_to_animate = max(orbit_sizes, key=orbit_sizes.get)

print(f"\nAnimating orbit: {orbit_to_animate}")

if orbit_to_animate not in orbits:
    print(f"Error: Orbit {orbit_to_animate} not found!")
    print(f"Available orbits: {orbit_nums}")
    sys.exit(1)

orbit_observations = orbits[orbit_to_animate]
n_obs = len(orbit_observations)
print(f"Number of observations in orbit: {n_obs}")

# Extract time series data for this orbit
Eel = np.array([obs['Eel'] for _, obs in orbit_observations])
Eaz = np.array([obs['Eaz'] for _, obs in orbit_observations])
ED = np.array([1.0 / obs['1/ED'] for _, obs in orbit_observations])  # Convert from 1/ED to ED

Mel = np.array([obs['Mel'] for _, obs in orbit_observations])
Maz = np.array([obs['Maz'] for _, obs in orbit_observations])
MD = np.array([1.0 / obs['1/MD'] for _, obs in orbit_observations])  # Convert from 1/MD to MD

# Extract below_sunshade status
below_sunshade = np.array([obs['below_sunshade'] for _, obs in orbit_observations])

print(f"\nData ranges:")
print(f"  Earth elevation: [{Eel.min():.4f}, {Eel.max():.4f}] rad = [{np.degrees(Eel.min()):.2f}, {np.degrees(Eel.max()):.2f}] deg")
print(f"  Earth azimuth: [{Eaz.min():.4f}, {Eaz.max():.4f}] rad = [{np.degrees(Eaz.min()):.2f}, {np.degrees(Eaz.max()):.2f}] deg")
print(f"  Earth distance: [{ED.min():.4e}, {ED.max():.4e}]")
print(f"  Moon elevation: [{Mel.min():.4f}, {Mel.max():.4f}] rad = [{np.degrees(Mel.min()):.2f}, {np.degrees(Mel.max()):.2f}] deg")
print(f"  Moon azimuth: [{Maz.min():.4f}, {Maz.max():.4f}] rad = [{np.degrees(Maz.min()):.2f}, {np.degrees(Maz.max()):.2f}] deg")
print(f"  Moon distance: [{MD.min():.4e}, {MD.max():.4e}]")
print(f"  Below sunshade: {below_sunshade.sum()} / {len(below_sunshade)} observations ({100*below_sunshade.sum()/len(below_sunshade):.1f}%)")

# Convert spherical to Cartesian coordinates
# Spherical: (r, elevation, azimuth) where elevation is from XY plane
# x = r * cos(el) * cos(az)
# y = r * cos(el) * sin(az)
# z = r * sin(el)

def spherical_to_cartesian(r, elevation, azimuth):
    """Convert spherical coordinates to Cartesian"""
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z

# Convert Earth and Moon positions
Ex, Ey, Ez = spherical_to_cartesian(ED, Eel, Eaz)
Mx, My, Mz = spherical_to_cartesian(MD, Mel, Maz)

print(f"\nCartesian coordinates:")
print(f"  Earth X: [{Ex.min():.4e}, {Ex.max():.4e}]")
print(f"  Earth Y: [{Ey.min():.4e}, {Ey.max():.4e}]")
print(f"  Earth Z: [{Ez.min():.4e}, {Ez.max():.4e}]")
print(f"  Moon X: [{Mx.min():.4e}, {Mx.max():.4e}]")
print(f"  Moon Y: [{My.min():.4e}, {My.max():.4e}]")
print(f"  Moon Z: [{Mz.min():.4e}, {Mz.max():.4e}]")

print("\nCreating animation...")

# Helper function to create a cone for the satellite
def create_cone(height=0.3, radius=0.15, resolution=20):
    """Create a cone with flat end at top (pointing up in +Z)"""
    # Cone points down from z=height to z=0 (apex)
    z = np.linspace(0, height, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)

    theta_grid, z_grid = np.meshgrid(theta, z)
    # Radius increases linearly from apex (0) to base (radius)
    r_grid = radius * z_grid / height

    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    return x_grid, y_grid, z_grid

# Create 3D animation with black background (like space)
fig = plt.figure(figsize=(14, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

# Set pane colors to black
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('white')
ax.yaxis.pane.set_edgecolor('white')
ax.zaxis.pane.set_edgecolor('white')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

# Determine axis limits based on max distance
max_dist = max(ED.max(), MD.max()) * 0.55  # Balanced zoom level
ax.set_xlim([-max_dist, max_dist])
ax.set_ylim([-max_dist, max_dist])
ax.set_zlim([-max_dist, max_dist])

# Labels with white text
ax.set_xlabel('X (satellite frame)', fontsize=10, color='white')
ax.set_ylabel('Y (satellite frame)', fontsize=10, color='white')
ax.set_zlabel('Z (satellite frame) - UP ↑', fontsize=10, color='white')
ax.set_title(f'Earth and Moon Positions Relative to Satellite\nOrbit {orbit_to_animate} ({n_obs} observations)\n(Cone top = satellite top)',
             fontsize=12, color='white')

# Set tick colors to white
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')

# Draw satellite as cone (flat end at top)
cone_x, cone_y, cone_z = create_cone(height=0.3, radius=0.15)
ax.plot_surface(cone_x, cone_y, cone_z, color='red', alpha=0.7,
                edgecolors='darkred', linewidth=0.3)

# Add a horizontal plane at z=0 to show above/below boundary (dimmer for space)
xx, yy = np.meshgrid(np.linspace(-max_dist*0.3, max_dist*0.3, 10),
                      np.linspace(-max_dist*0.3, max_dist*0.3, 10))
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, alpha=0.15, color='cyan', edgecolors='cyan', linewidth=0.5)

# Add text labels for above/below with white text
ax.text(0, 0, max_dist*0.8, 'ABOVE\nSatellite', fontsize=12, ha='center', color='white',
        bbox=dict(boxstyle='round', facecolor='green', alpha=0.3, edgecolor='white'))
ax.text(0, 0, -max_dist*0.8, 'BELOW\nSatellite', fontsize=12, ha='center', color='white',
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3, edgecolor='white'))

# Initialize Earth and Moon points (will change color based on elevation)
earth_point = ax.scatter([], [], [], s=150, label='Earth', edgecolors='darkblue', linewidth=2)
moon_point = ax.scatter([], [], [], s=80, label='Moon', edgecolors='gray', linewidth=1.5)

# Add trajectory lines
earth_trail, = ax.plot([], [], [], alpha=0.4, linewidth=1.5)
moon_trail, = ax.plot([], [], [], alpha=0.4, linewidth=1)

# Time text with white color for visibility on black background
time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                      family='monospace', color='white',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white'))
distance_text = ax.text2D(0.02, 0.88, '', transform=ax.transAxes, fontsize=8,
                          family='monospace', color='white',
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white'))

# Create legend with custom markers (white text for dark background)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
           markeredgecolor='blue', markersize=10, linewidth=2,
           label='Earth ABOVE satellite (solid blue)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='blue', markersize=10, linewidth=2,
           label='Earth BELOW satellite (blue ring)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markeredgecolor='gray', markersize=8, linewidth=2,
           label='Moon ABOVE satellite (solid grey)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='gray', markersize=8, linewidth=2,
           label='Moon BELOW satellite (grey ring)'),
]
legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                   framealpha=0.8, edgecolor='white', facecolor='black')
# Set legend text to white
for text in legend.get_texts():
    text.set_color('white')

ax.grid(True, alpha=0.2, color='white')

# Animation functions
def init():
    earth_point._offsets3d = ([], [], [])
    moon_point._offsets3d = ([], [], [])
    earth_trail.set_data([], [])
    earth_trail.set_3d_properties([])
    moon_trail.set_data([], [])
    moon_trail.set_3d_properties([])
    time_text.set_text('')
    distance_text.set_text('')
    return earth_point, moon_point, earth_trail, moon_trail, time_text, distance_text

def animate(i):
    # Determine if Earth and Moon are above (positive elevation) or below (negative elevation)
    earth_above = Eel[i] > 0
    moon_above = Mel[i] > 0

    # Check if below sunshade
    is_below_sunshade = below_sunshade[i]

    # Update Earth position
    # Solid blue if above satellite, open blue ring if below
    earth_edge_color = 'blue'
    earth_face_color = 'blue' if earth_above else 'none'

    earth_point._offsets3d = ([Ex[i]], [Ey[i]], [Ez[i]])
    earth_point.set_facecolor(earth_face_color)
    earth_point.set_edgecolor(earth_edge_color)

    # Update Moon position
    # Solid grey if above satellite, open grey ring if below
    moon_edge_color = 'gray'
    moon_face_color = 'gray' if moon_above else 'none'

    moon_point._offsets3d = ([Mx[i]], [My[i]], [Mz[i]])
    moon_point.set_facecolor(moon_face_color)
    moon_point.set_edgecolor(moon_edge_color)

    # Update trails (show last 100 frames or all if less)
    trail_length = min(100, i + 1)
    trail_start = max(0, i + 1 - trail_length)

    # Earth trail - always blue
    earth_trail.set_data(Ex[trail_start:i+1], Ey[trail_start:i+1])
    earth_trail.set_3d_properties(Ez[trail_start:i+1])
    earth_trail.set_color('blue')

    # Moon trail - always grey
    moon_trail.set_data(Mx[trail_start:i+1], My[trail_start:i+1])
    moon_trail.set_3d_properties(Mz[trail_start:i+1])
    moon_trail.set_color('gray')

    # Update text
    obs_id, obs_data = orbit_observations[i]
    earth_status = "ABOVE ↑" if earth_above else "BELOW ↓"
    moon_status = "ABOVE ↑" if moon_above else "BELOW ↓"
    sunshade_status = "BELOW SUNSHADE (open)" if is_below_sunshade else "ABOVE SUNSHADE (filled)"

    time_text.set_text(f'Observation: {i+1}/{n_obs} (FFI: {obs_id}) | {sunshade_status}')
    distance_text.set_text(
        f'Earth: {earth_status} (el={np.degrees(Eel[i]):+.1f}°, dist={ED[i]:.4e}) | '
        f'Moon: {moon_status} (el={np.degrees(Mel[i]):+.1f}°, dist={MD[i]:.4e})'
    )

    # Rotate view for better visualization
    ax.view_init(elev=25, azim=30 + i * 0.3)

    return earth_point, moon_point, earth_trail, moon_trail, time_text, distance_text

# Create animation with all frames
n_frames = n_obs
step = max(1, n_frames // 300)  # Limit to ~300 frames for file size
frames = range(0, n_frames, step)

print(f"Total observations: {n_frames}")
print(f"Animation frames: {len(list(frames))} (step={step})")

anim = FuncAnimation(fig, animate, init_func=init, frames=frames,
                     interval=50, blit=False, repeat=True)

# Save animation
output_file = f'scripts/satellite_orbit_{orbit_to_animate}.gif'
print(f"\nSaving animation to {output_file}...")
try:
    writer = PillowWriter(fps=20)
    anim.save(output_file, writer=writer)
    print(f"Animation saved successfully!")
    print(f"\nOutput file: {output_file}")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Trying to save with different settings...")
    output_file = f'scripts/satellite_orbit_{orbit_to_animate}_low.gif'
    writer = PillowWriter(fps=10)
    # Use even fewer frames
    frames_low = range(0, n_frames, max(1, n_frames // 100))
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames_low,
                         interval=100, blit=False, repeat=True)
    anim.save(output_file, writer=writer)
    print(f"Animation saved to {output_file}")

print(f"\nTo animate a different orbit, run:")
print(f"  python3 scripts/animate_satellite_positions.py <orbit_number>")
print(f"  Example: python3 scripts/animate_satellite_positions.py {orbit_nums[0]}")
