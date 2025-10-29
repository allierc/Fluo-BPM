"""
Example: Run fluorescence simulation
"""
from fluorescence_bpm import run_simulation

# Cryolite dataset (512x512x128)
run_simulation(
    refractive_index_path='./data/cryolite_dn.tif',
    fluorescence_path='./data/cryolite_fluorescence.tif',
    dx=2.412,
    dz=1.989,
    lbda=0.637,
    z_min=0.0,
    z_max=128 * 1.989,  # 254.6 μm
    nm=1.33,
    na=0.5,
    device='cuda:0',
    stochastic=True,     # STORM mode
    sparsity=0.01,      # 0.5% active per frame
    n_iterations=1000,
    output_path='./output'
)

# Kidney dataset (512x512x30) - standard mode
# run_simulation(
#     refractive_index_path='./data/KidneyL_30.tif',
#     fluorescence_path='./data/Fluo_2_30.tif',
#     dx=0.1154,
#     dz=1.0,
#     lbda=0.5320,
#     z_min=-15.0,
#     z_max=15.0,
#     nm=1.33,
#     na=0.5,
#     device='cuda:0',
#     stochastic=False,
#     n_iterations=1000
# )

