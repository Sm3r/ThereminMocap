import csv
import collections

ColumnMapping = collections.namedtuple('ColumnMapping', ['setter', 'axis', 'column'])

class RigidBody:
    def __init__(self, label, ID):
        self.label = label
        self.ID = ID
        self.positions = []
        self.rotations = []
        self.times = []

    def _add_frame(self, t):
        self.times.append(t)
        self.positions.append(None)
        self.rotations.append(None)

    def _set_position(self, frame, axis, value):
        if value:
            if self.positions[frame] is None:
                self.positions[frame] = [0.0, 0.0, 0.0]
            self.positions[frame][axis] = float(value)

    def _set_rotation(self, frame, axis, value):
        if value:
            if self.rotations[frame] is None:
                self.rotations[frame] = [0.0, 0.0, 0.0, 0.0]
            self.rotations[frame][axis] = float(value)

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
    def __init__(self, frame_rate=120.0, rotation_type='Quaternion', units='Meters'):
        self.frame_rate = frame_rate
        self.rotation_type = rotation_type
        self.units = units
        self.rigid_bodies = {}
        self.markers = {}
        self._raw_info = {}
        self._raw_types = []
        self._raw_labels = []
        self._raw_fields = []
        self._raw_axes = []
        self._ignored_labels = set()
        self._column_map = []

    def readCSV(self, path, verbose=False):
        self.rigid_bodies = {}
        self.markers = {}
        self._raw_info = {}
        self._ignored_labels = set()
        self._column_map = []

        with open(path, newline='', encoding='utf-8') as file_handle:
            csv_reader = csv.reader(file_handle)
            self._read_header(csv_reader, verbose)
            self._read_data(csv_reader, verbose)

        return self

    def _read_header(self, stream, verbose=False):
        line1 = next(stream)
        assert line1[0] == 'Format Version', f"Unrecognized header cell: {line1[0]}"
        format_version = line1[1]
        assert format_version in ['1.24', '1.21', '1.2'], f"Unsupported format version: {format_version}"

        self._raw_info = {line1[i]: line1[i + 1] for i in range(0, len(line1) - 1, 2)}

        self.rotation_type = self._raw_info.get('Rotation Type', 'Quaternion')
        assert self.rotation_type == 'Quaternion', f"Only Quaternion rotation is supported, found: {self.rotation_type}"
        self.frame_rate = float(self._raw_info.get('Export Frame Rate', 120))
        self.units = self._raw_info.get('Length Units', 'Meters')

        assert not next(stream), "Expected blank second header line."

        self._raw_types = next(stream)[2:]
        self._raw_labels = next(stream)[2:]
        line5 = next(stream)
        self._raw_fields = next(stream)[2:]
        self._raw_axes = next(stream)[2:]

        supported_types = {'Rigid Body', 'Rigid Body Marker', 'Marker'}
        assert set(self._raw_types).issubset(supported_types), "Unsupported object type found in header."

        for col, asset_type, label, ID, field, axis in zip(
                range(len(self._raw_types)), self._raw_types, self._raw_labels, line5[2:], self._raw_fields, self._raw_axes):

            if asset_type == 'Rigid Body':
                body = self.rigid_bodies.setdefault(label, RigidBody(label, ID))

                if field == 'Rotation':
                    axis_index = {'X': 0, 'Y': 1, 'Z': 2, 'W': 3}[axis]
                    self._column_map.append(ColumnMapping(body._set_rotation, axis_index, col))
                elif field == 'Position':
                    axis_index = {'X': 0, 'Y': 1, 'Z': 2}[axis]
                    self._column_map.append(ColumnMapping(body._set_position, axis_index, col))

            elif asset_type == 'Marker':
                marker = self.markers.setdefault(label, Marker(label))
                axis_index = {'X': 0, 'Y': 1, 'Z': 2}[axis]
                self._column_map.append(ColumnMapping(marker._set_position, axis_index, col))

            else:
                if label not in self._ignored_labels:
                    if verbose:
                        print(f"Ignoring object {label} of type {asset_type}.")
                    self._ignored_labels.add(label)

    def _read_data(self, stream, verbose=False):
        for row_num, row in enumerate(stream):
            frame_num = int(row[0])
            frame_t = float(row[1])
            values = row[2:]

            for body in self.rigid_bodies.values():
                body._add_frame(frame_t)
            for marker in self.markers.values():
                marker._add_frame(frame_t)

            for mapping in self._column_map:
                mapping.setter(row_num, mapping.axis, values[mapping.column])
