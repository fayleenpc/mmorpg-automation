import struct
import glob 
for pyile in glob.glob('*.map'):
    print(pyile)

with open('t2001_s1.map', 'rb') as file:
    # Example: Read the first 12 bytes as 3 float values (x, y, z)
    data = file.read(12)
    x, y, z = struct.unpack('fff', data)
    print(f"Coordinates: x={x}, y={y}, z={z}")

