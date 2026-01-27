import vispy.scene
from vispy.scene import visuals
import numpy as np

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
