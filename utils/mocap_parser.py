import csv
import collections
import json
import numpy as np

ColumnMapping = collections.namedtuple('ColumnMapping', ['setter', 'axis', 'column'])

class RigidBody:
    def __init__(self, label):
        self.label = label
        self.positions = []
        self.times = []

    def _add_frame(self, t):
        self.times.append(t)
        self.positions.append(None)

    def _set_position(self, frame, axis, value):
        if value:
            if self.positions[frame] is None:
                self.positions[frame] = [0.0, 0.0, 0.0]
            self.positions[frame][axis] = float(value)

    def num_total_frames(self):
        return len(self.times)

    def num_valid_frames(self):
        return sum(1 for pt in self.positions if pt is not None)

class Marker:
    def __init__(self, label):
        self.label = label
        self.positions = []
        self.times = []

    def _add_frame(self, t):
        self.times.append(t)
        self.positions.append(None)

    def _set_position(self, frame, axis, value):
        if value:
            if self.positions[frame] is None:
                self.positions[frame] = [0.0, 0.0, 0.0]
            self.positions[frame][axis] = float(value)

class Take:
    def __init__(self, frame_rate=120.0):
        self.frame_rate = frame_rate
        self.rigid_bodies = {}
        self.markers = {}
        self._column_map = []

    def readCSV(self, path, verbose=False, config=None):
        self.rigid_bodies = {}
        self.markers = {}
        self._column_map = []
        self._config = config

        with open(path, newline='', encoding='utf-8') as file_handle:
            csv_reader = csv.reader(file_handle)
            self._read_header(csv_reader, verbose)
            self._read_data(csv_reader, verbose)

        print(f"\n=== Take Loaded ===")
        print(f"Rigid Bodies: {list(self.rigid_bodies.keys())}")
        print(f"Markers: {list(self.markers.keys())}")
        print(f"Total Frames: {len(next(iter(self.rigid_bodies.values())).times) if self.rigid_bodies else len(next(iter(self.markers.values())).times) if self.markers else 0}")
        print(f"Column Mappings: {len(self._column_map)}")
        print(f"===================\n")

        return self

    def get_all_marker_data_as_array(self):

        if not self.markers:
            return np.array([])
        
        # Get number of frames from first marker
        num_frames = len(next(iter(self.markers.values())).positions)
        marker_names = sorted(self.markers.keys())  # Sort for consistent ordering
        num_markers = len(marker_names)
        
        # Initialize array with zeros
        data = np.zeros((num_frames, num_markers * 3))
        
        # Fill array with marker data
        for marker_idx, marker_name in enumerate(marker_names):
            marker = self.markers[marker_name]
            for frame_idx, pos in enumerate(marker.positions):
                if pos is not None:
                    data[frame_idx, marker_idx * 3:(marker_idx * 3) + 3] = pos
        
        return data

    def _read_header(self, stream, verbose=False):
        # Read column headers from first row
        # Expected format: Frame,name_X,name_Y,name_Z,name_001_X,name_001_Y,name_001_Z,...
        column_headers = next(stream)
        
        # Get config names if provided
        config_names = []
        if self._config:
            config_names = [
                self._config.get('left_hand_name'),
                self._config.get('right_hand_name'),
                self._config.get('pitch_antenna_name'),
                self._config.get('volume_antenna_name')
            ]
            config_names = [name for name in config_names if name]  # Remove None values
        
        # Parse column headers
        for col_idx, col_name in enumerate(column_headers):
            if col_name == 'Frame':
                continue
            
            # Extract base name and axis from column name (e.g., "LeftHand_001_X" -> "LeftHand", "X")
            if '_' in col_name:
                parts = col_name.rsplit('_', 1)  # Split from right to get axis
                if len(parts) == 2 and parts[1] in ['X', 'Y', 'Z']:
                    base_name_with_marker = parts[0]
                    axis = parts[1]
                    
                    # Determine the object name (check against config names)
                    object_name = None
                    for config_name in config_names:
                        if config_name in base_name_with_marker:
                            object_name = config_name
                            break
                    
                    if not object_name:
                        # If no config match, use the base name before any numbers
                        object_name = base_name_with_marker.split('_')[0]
                    
                    # Check if this is a marker (has numbers after the name) or rigid body (no numbers)
                    if '_' in base_name_with_marker:
                        # This is likely a marker
                        marker = self.markers.setdefault(base_name_with_marker, Marker(base_name_with_marker))
                        axis_index = {'X': 0, 'Y': 1, 'Z': 2}[axis]
                        self._column_map.append(ColumnMapping(marker._set_position, axis_index, col_idx - 1))
                    else:
                        # This is a rigid body position
                        body = self.rigid_bodies.setdefault(base_name_with_marker, RigidBody(base_name_with_marker))
                        axis_index = {'X': 0, 'Y': 1, 'Z': 2}[axis]
                        self._column_map.append(ColumnMapping(body._set_position, axis_index, col_idx - 1))
        
        if verbose:
            print(f"Found {len(self.rigid_bodies)} rigid bodies: {list(self.rigid_bodies.keys())}")
            print(f"Found {len(self.markers)} markers: {list(self.markers.keys())}")

    def _read_data(self, stream, verbose=False):
        frame_idx = 0
        for row in stream:
            if not row or not row[0]:  # Skip empty rows
                continue
                
            frame_t = float(row[0])  # Use frame number as time
            values = row[1:]

            for body in self.rigid_bodies.values():
                body._add_frame(frame_t)
            for marker in self.markers.values():
                marker._add_frame(frame_t)

            for mapping in self._column_map:
                if mapping.column < len(values):
                    value = values[mapping.column]
                    # Handle empty values or '0.0' strings
                    if value and value.strip() and value != '0.0':
                        mapping.setter(frame_idx, mapping.axis, value)
            
            frame_idx += 1
