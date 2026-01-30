import vispy.scene
from vispy.scene import visuals
import numpy as np
import matplotlib.pyplot as plt
from utils.mocap_parser import Take
from utils.config import config

# Used in the prepare mocap
def plot_3d_animation(pitch_marker_1, pitch_marker_2, pitch_marker_3,
                            volume_marker_1, volume_marker_2, volume_marker_3,
                            all_markers):

    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')  # Ensure a visible background
    view = canvas.central_widget.add_view()

    # Create scatter plot
    scatter = visuals.Markers(parent=view.scene)

    # Initialize with dummy data to avoid NoneType errors
    dummy_point = np.array([[0, 0, 0]])
    scatter.set_data(dummy_point, edge_color=None, face_color=[1, 1, 1, 0], size=10)  # Transparent

    # Set camera AFTER initializing data
    view.camera = 'turntable'
    
    # Explicitly set camera range to match expected coordinate bounds
    view.camera.set_range(x=(-1000, 1000), y=(-1000, 1000), z=(0, 2000))

    num_frames = len(pitch_marker_1)
    frame_counter = {'current': 0}  # Use a mutable object to track frame index

    def update(event):
        frame = frame_counter['current']

        # Extract new positions for this frame
        pitch_points = np.array([
            pitch_marker_1[frame], pitch_marker_2[frame], pitch_marker_3[frame]
        ])
        volume_points = np.array([
            volume_marker_1[frame], volume_marker_2[frame], volume_marker_3[frame]
        ])
        all_points = np.vstack((pitch_points, volume_points, all_markers[frame]))  # Include new markers
        
        # Colors: Red for pitch, Green for volume, White for additional markers
        colors = np.array([[1, 0, 0, 1]] * 3 + [[0, 1, 0, 1]] * 3 + [[1, 1, 1, 1]] * 9)

        # Update scatter plot
        scatter.set_data(all_points, edge_color=None, face_color=colors, size=10)

        # Explicitly request a redraw (ensures updates are visible)
        canvas.update()

        # Increment frame counter safely
        frame_counter['current'] = (frame_counter['current'] + 1) % num_frames

    # Start timer (fixed to ensure updates)
    timer = vispy.app.Timer(interval=1 / 441)  # ~441 FPS
    timer.connect(update)
    timer.start()

    vispy.app.run()

# Used in the data loader for visualization
def plot_hands(dataset):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Hand Markers (Centered & Normalized)', fontsize=16, fontweight='bold')
    
    ax_x = plt.subplot(3, 1, 1)
    ax_y = plt.subplot(3, 1, 2)
    ax_z = plt.subplot(3, 1, 3)
    
    frames = np.arange(len(dataset.mocap_feats))
    num_markers = dataset.mocap_feats.shape[1] // 3

    left_hand_count = 2
    
    for marker_idx in range(num_markers):
        x_col = marker_idx * 3
        y_col = marker_idx * 3 + 1
        z_col = marker_idx * 3 + 2
        
        x_data = dataset.mocap_feats[:, x_col]
        y_data = dataset.mocap_feats[:, y_col]
        z_data = dataset.mocap_feats[:, z_col]
        
        if marker_idx < left_hand_count:
            color = 'blue'
            label = f'LeftHand_{marker_idx + 1:03d}'
        else:
            color = 'red'
            label = f'RightHand_{marker_idx - left_hand_count + 1:03d}'
        
        ax_x.plot(frames, x_data, linewidth=0.8, alpha=0.5, color=color, label=label)
        ax_y.plot(frames, y_data, linewidth=0.8, alpha=0.5, color=color)
        ax_z.plot(frames, z_data, linewidth=0.8, alpha=0.5, color=color)
    
    ax_x.set_ylabel('X Position', fontsize=12)
    ax_x.grid(True, alpha=0.3)
    ax_x.set_title('X Axis')
    ax_x.legend(loc='upper right', fontsize=8)
    
    ax_y.set_ylabel('Y Position', fontsize=12)
    ax_y.grid(True, alpha=0.3)
    ax_y.set_title('Y Axis')
    
    ax_z.set_ylabel('Z Position', fontsize=12)
    ax_z.set_xlabel('Frame Number', fontsize=12)
    ax_z.grid(True, alpha=0.3)
    ax_z.set_title('Z Axis')
    
    plt.tight_layout()
    plt.show()

# Used in the data loader for visualization
def plot_audio_correlation(dataset):
    frames = np.arange(len(dataset.audio_feats))
    
    audio_volume = dataset.audio_feats[:, 1] if dataset.audio_feats.shape[1] > 0 else np.zeros(len(frames))
    audio_pitch = dataset.audio_feats[:, 0] if dataset.audio_feats.shape[1] > 1 else np.zeros(len(frames))
    
    left_hand_data = dataset.mocap_feats[:, :6]
    right_hand_data = dataset.mocap_feats[:, 6:27]
    
    # Figure 1: Volume + Left Hand
    fig1 = plt.figure(figsize=(14, 10))
    fig1.suptitle('Audio Volume vs Left Hand Movement', fontsize=16, fontweight='bold')
    
    ax1_x = plt.subplot(3, 1, 1)
    ax1_y = plt.subplot(3, 1, 2)
    ax1_z = plt.subplot(3, 1, 3)
    
    for marker_idx in range(2):
        x_data = left_hand_data[:, marker_idx * 3]
        y_data = left_hand_data[:, marker_idx * 3 + 1]
        z_data = left_hand_data[:, marker_idx * 3 + 2]
        
        ax1_x.plot(frames, x_data, linewidth=0.8, alpha=0.7, color='blue', label=f'LeftHand_{marker_idx + 1:03d}')
        ax1_y.plot(frames, y_data, linewidth=0.8, alpha=0.7, color='blue')
        ax1_z.plot(frames, z_data, linewidth=0.8, alpha=0.7, color='blue')
    
    ax1_x_vol = ax1_x.twinx()
    ax1_y_vol = ax1_y.twinx()
    ax1_z_vol = ax1_z.twinx()
    
    ax1_x_vol.plot(frames, audio_volume, linewidth=1.5, alpha=0.8, color='purple', label='Volume', linestyle='--')
    ax1_y_vol.plot(frames, audio_volume, linewidth=1.5, alpha=0.8, color='purple', linestyle='--')
    ax1_z_vol.plot(frames, audio_volume, linewidth=1.5, alpha=0.8, color='purple', linestyle='--')
    
    ax1_x.set_ylabel('X Position', fontsize=12, color='blue')
    ax1_x_vol.set_ylabel('Volume', fontsize=12, color='purple')
    ax1_x.grid(True, alpha=0.3)
    ax1_x.set_title('X Axis + Volume')
    ax1_x.legend(loc='upper left', fontsize=8)
    ax1_x_vol.legend(loc='upper right', fontsize=8)
    
    ax1_y.set_ylabel('Y Position', fontsize=12, color='blue')
    ax1_y_vol.set_ylabel('Volume', fontsize=12, color='purple')
    ax1_y.grid(True, alpha=0.3)
    ax1_y.set_title('Y Axis + Volume')
    
    ax1_z.set_ylabel('Z Position', fontsize=12, color='blue')
    ax1_z_vol.set_ylabel('Volume', fontsize=12, color='purple')
    ax1_z.set_xlabel('Frame Number', fontsize=12)
    ax1_z.grid(True, alpha=0.3)
    ax1_z.set_title('Z Axis + Volume')
    
    plt.tight_layout()
    
    # Figure 2: Pitch + Right Hand
    fig2 = plt.figure(figsize=(14, 10))
    fig2.suptitle('Audio Pitch vs Right Hand Movement', fontsize=16, fontweight='bold')
    
    ax2_x = plt.subplot(3, 1, 1)
    ax2_y = plt.subplot(3, 1, 2)
    ax2_z = plt.subplot(3, 1, 3)
    
    for marker_idx in range(7):
        x_data = right_hand_data[:, marker_idx * 3]
        y_data = right_hand_data[:, marker_idx * 3 + 1]
        z_data = right_hand_data[:, marker_idx * 3 + 2]
        
        ax2_x.plot(frames, x_data, linewidth=0.8, alpha=0.7, color='red', label=f'RightHand_{marker_idx + 1:03d}')
        ax2_y.plot(frames, y_data, linewidth=0.8, alpha=0.7, color='red')
        ax2_z.plot(frames, z_data, linewidth=0.8, alpha=0.7, color='red')
    
    ax2_x_pitch = ax2_x.twinx()
    ax2_y_pitch = ax2_y.twinx()
    ax2_z_pitch = ax2_z.twinx()
    
    ax2_x_pitch.plot(frames, audio_pitch, linewidth=1.5, alpha=0.8, color='green', label='Pitch', linestyle='--')
    ax2_y_pitch.plot(frames, audio_pitch, linewidth=1.5, alpha=0.8, color='green', linestyle='--')
    ax2_z_pitch.plot(frames, audio_pitch, linewidth=1.5, alpha=0.8, color='green', linestyle='--')
    
    ax2_x.set_ylabel('X Position', fontsize=12, color='red')
    ax2_x_pitch.set_ylabel('Pitch', fontsize=12, color='green')
    ax2_x.grid(True, alpha=0.3)
    ax2_x.set_title('X Axis + Pitch')
    ax2_x.legend(loc='upper left', fontsize=8)
    ax2_x_pitch.legend(loc='upper right', fontsize=8)
    
    ax2_y.set_ylabel('Y Position', fontsize=12, color='red')
    ax2_y_pitch.set_ylabel('Pitch', fontsize=12, color='green')
    ax2_y.grid(True, alpha=0.3)
    ax2_y.set_title('Y Axis + Pitch')
    
    ax2_z.set_ylabel('Z Position', fontsize=12, color='red')
    ax2_z_pitch.set_ylabel('Pitch', fontsize=12, color='green')
    ax2_z.set_xlabel('Frame Number', fontsize=12)
    ax2_z.grid(True, alpha=0.3)
    ax2_z.set_title('Z Axis + Pitch')
    
    plt.tight_layout()
    plt.show()

# Used to verify correct cleaning of the data
def plot_all_markers():
    
    take_name = config.take_name
    take = Take()
    take.readCSV(f"data/dataframes/MOCAP_{take_name}_CLEAN.csv")
    
    mocap_data = np.load(f"data/dataframes/{take_name}.npy")
    marker_names = list(take.markers.keys())
    num_features = mocap_data.shape[1]
    num_markers = num_features // 3
    
    # Create combined plot for all hand markers
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('All Markers - Combined View', fontsize=16, fontweight='bold')
    
    ax_x = plt.subplot(3, 1, 1)
    ax_y = plt.subplot(3, 1, 2)
    ax_z = plt.subplot(3, 1, 3)
    
    # Color palettes for each category
    right_hand_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
    left_hand_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
    
    right_hand_idx = 0
    left_hand_idx = 0
    pitch_idx = 0
    volume_idx = 0
    
    # Plot all markers
    for marker_idx in range(num_markers):
        marker_name = marker_names[marker_idx] if marker_idx < len(marker_names) else f"Marker_{marker_idx + 1}"
        
        # Determine category and color - only plot hand markers
        if config.right_hand_name in marker_name:
            color = right_hand_colors[right_hand_idx % len(right_hand_colors)]
            right_hand_idx += 1
        elif config.left_hand_name in marker_name:
            color = left_hand_colors[left_hand_idx % len(left_hand_colors)]
            left_hand_idx += 1
        elif config.pitch_antenna_name in marker_name.lower():
            pitch_idx += 1
            continue  # Skip pitch markers in figure 1
        elif config.volume_antenna_name in marker_name.lower():
            volume_idx += 1
            continue  # Skip volume markers in figure 1
        else:
            continue  # Skip markers not in any category
        
        # Extract X, Y, Z for this marker
        x_col = marker_idx * 3
        y_col = marker_idx * 3 + 1
        z_col = marker_idx * 3 + 2
        
        x_data = mocap_data[:, x_col].copy()
        y_data = mocap_data[:, y_col].copy()
        z_data = mocap_data[:, z_col].copy()
        
        # Replace 0 values with NaN so they don't show up in the plot
        x_data[x_data == 0] = np.nan
        y_data[y_data == 0] = np.nan
        z_data[z_data == 0] = np.nan
        
        # Plot on combined axes with alpha (only add label to first axis)
        ax_x.plot(x_data, linewidth=0.8, alpha=0.7, label=marker_name, color=color)
        ax_y.plot(y_data, linewidth=0.8, alpha=0.7, color=color)
        ax_z.plot(z_data, linewidth=0.8, alpha=0.7, color=color)
    
    ax_x.set_ylabel('X Position', fontsize=12)
    ax_x.grid(True, alpha=0.3)
    ax_x.set_title('X Axis')
    
    ax_y.set_ylabel('Y Position', fontsize=12)
    ax_y.grid(True, alpha=0.3)
    ax_y.set_title('Y Axis')
    
    ax_z.set_ylabel('Z Position', fontsize=12)
    ax_z.set_xlabel('Frame Number', fontsize=12)
    ax_z.grid(True, alpha=0.3)
    ax_z.set_title('Z Axis')
    
    # Single legend outside the subplots
    fig.legend(loc='center right', fontsize=8, bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend on the right
    plt.show()
