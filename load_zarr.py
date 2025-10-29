import numpy as np
from tifffile import imwrite
import json
import os
import zstandard as zstd

def read_zarr_v3(path):
    """Manual Zarr v3 reader"""
    with open(f'{path}/zarr.json', 'r') as f:
        meta = json.load(f)
    
    shape = tuple(meta['shape'])
    dtype = np.dtype(meta['data_type'])
    chunk_shape = tuple(meta['chunk_grid']['configuration']['chunk_shape'])
    
    print(f"Shape: {shape}, dtype: {dtype}, chunks: {chunk_shape}")
    
    data = np.zeros(shape, dtype=dtype)
    
    # Walk through chunk directory tree
    chunk_dir = f'{path}/c'
    for root, dirs, files in os.walk(chunk_dir):
        for file in files:
            # Get chunk path relative to 'c' dir
            chunk_path = os.path.join(root, file)
            rel_path = os.path.relpath(chunk_path, chunk_dir)
            
            # Parse indices from path (e.g., "0/1/2")
            indices = [int(x) for x in rel_path.split(os.sep)]
            
            # Read and decompress chunk
            with open(chunk_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                chunk_data = dctx.decompress(f.read())
                chunk_arr = np.frombuffer(chunk_data, dtype=dtype).reshape(chunk_shape)
            
            # Place in array
            z, y, x = indices
            z_slice = slice(z*chunk_shape[0], min((z+1)*chunk_shape[0], shape[0]))
            y_slice = slice(y*chunk_shape[1], min((y+1)*chunk_shape[1], shape[1]))
            x_slice = slice(x*chunk_shape[2], min((x+1)*chunk_shape[2], shape[2]))
            
            data[z_slice, y_slice, x_slice] = chunk_arr[:z_slice.stop-z_slice.start,
                                                          :y_slice.stop-y_slice.start,
                                                          :x_slice.stop-x_slice.start]
    
    return data

cryo_arr = read_zarr_v3('./data/cryolite_binary.zarr')
meas_arr = read_zarr_v3('./data/cryolite_measurement.zarr')

fluo = (1.0 - cryo_arr)# Scale fluorophore concentration
dn = fluo * 0.0015  # Refractive index contrast

print(f"Original shape: {cryo_arr.shape}")

# Crop to 128 x 512 x 512 (z, y, x)
z_crop, y_crop, x_crop = 128, 512, 512

z_start = (cryo_arr.shape[0] - z_crop) // 2
y_start = (cryo_arr.shape[1] - y_crop) // 2
x_start = (cryo_arr.shape[2] - x_crop) // 2

cryo_crop = cryo_arr[z_start:z_start+z_crop, y_start:y_start+y_crop, x_start:x_start+x_crop]
meas_crop = meas_arr[z_start:z_start+z_crop, y_start:y_start+y_crop, x_start:x_start+x_crop]

print(f"Cropped shape: {cryo_crop.shape}")

# Scale
fluo = cryo_crop * 175
dn = fluo * 0.0

# Save
imwrite('./data/cryolite_fluorescence.tif', fluo.astype(np.float32))
imwrite('./data/cryolite_dn.tif', dn.astype(np.float32))
imwrite('./data/cryolite_measurement.tif', meas_crop.astype(np.float32))

print(f"Volume: {x_crop*2.412:.1f} × {y_crop*2.412:.1f} × {z_crop*1.989:.1f} μm³")


# Parameters
dx = 2.412  # μm
dz = 1.989  # μm
lbda = 0.637  # μm

print(f"\nParameters:")
print(f"dx = {dx} μm")
print(f"dz = {dz} μm")
print(f"λ = {lbda} μm")
print(f"Volume: {cryo_arr.shape[2]*dx:.1f} × {cryo_arr.shape[1]*dx:.1f} × {cryo_arr.shape[0]*dz:.1f} μm³")