import json
import pmcx
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import scipy.ndimage as nd

class VoxelBasis:
    def __init__(self, vol_shape, basis_size = (100,100,100)):
        """
        Initialize the voxel-based basis mapping.
        
        Parameters:
            vol_shape (tuple): shape of the original full-resolution volume (e.g., (100,100,100)).
            basis_size (tuple): desired shape of the lower-resolution basis grid (e.g., (50,50,50)).
        """
        self.vol_shape = vol_shape
        self.basis_size = basis_size
        # Compute zoom factors for mapping volume -> basis grid
        self.zoom_factors = [b / v for b, v in zip(basis_size, vol_shape)]
        # And the inverse zoom factors for mapping back
        self.inv_zoom_factors = [v / b for v, b in zip(vol_shape, basis_size)]
        
    def Map(self, direction, data, flatten=False):
        """
        Map data between the full volume (M) and the basis grid (S).
        
        Parameters:
            direction (str): 'M->S' maps from the full volume (M) to the basis grid (S);
                             'S->M' maps from the basis grid (S) back to the full volume (M).
            data (ndarray): data to be mapped. It can be a flattened 1D array or already reshaped.
            flatten (bool): if True, return the result as a flattened 1D array.
            
        Returns:
            ndarray: the mapped data.
        """
        if direction == 'M->S':
            # Reshape data to volume shape if needed
            if data.ndim == 1:
                data_vol = data.reshape(self.vol_shape)
            else:
                data_vol = data
            # Resample the volume to the basis grid dimensions using linear interpolation (order=1)
            mapped = nd.zoom(data_vol, self.zoom_factors, order=1)
        elif direction == 'S->M':
            # Reshape data to basis grid shape if needed
            if data.ndim == 1:
                data_basis = data.reshape(self.basis_size)
            else:
                data_basis = data
            # Resample from basis grid back to the full volume dimensions
            mapped = nd.zoom(data_basis, self.inv_zoom_factors, order=1)
        else:
            raise ValueError("Unknown mapping direction: choose 'M->S' or 'S->M'")
            
        if flatten:
            return mapped.flatten()
        return mapped

if __name__ == '__main__':
    use_basis = True # reduce jacobian size with basis
    basis_size = (100,100,100)
    grey_matter_index = 2 # 2 for MNI
    # Load the NIfTI file
    nifti_img = nib.load('MNI152_halved.nii.gz')
    # Extract voxel data as NumPy array
    vol = np.array(nifti_img.get_fdata())
    vol = np.squeeze(vol)
    print(np.unique(vol))
    
    srcpos = [
        [123,202,84, 1],
        [131,199,66, 1],
        [116,206,70, 1],
        [100,210,72, 1],
        [82,211,69, 1],
        [89,209,87, 1],
        [55,202,86, 1],
        [64,207,70, 1],
        [47,198,70, 1],
    ]
    srcparam1, srcparam2 = [], []
    for src in srcpos:
        srcparam1.append([0,0,0,0]) # cfg.{srcparam1,srcparam2}: 1x4 vectors, see cfg.srctype for details
        srcparam2.append([0,0,0,0]) #https://github.com/fangq/mcx/blob/5332f081fe5501281c7197a1feb09870ea252ad5/mcxlab/mcxlab.m#L188

    src_dir = [
        [-0.44619334560544055, -0.8921637092442303, -0.07039470324534593], 
        [-0.5636409158743776, -0.8094144915036114, -0.16479411062365973],
        [-0.35721068650099036, -0.9303785048327382, -0.08244005819193562], 
        [-0.06675513222043321, -0.9972821933085935, -0.03117658146483632], 
        [0.0813904317637634, -0.9546520346834588, -0.28638276884624236], 
        [-0.013912524239269104, -0.9619471251979644, -0.2728812378905213], 
        [0.48505360424442817, -0.8688942681043205, -0.09872057467897478], 
        [0.2737805190807639, -0.9567145284290323, -0.09869923234089521], 
        [0.5445422295683279, -0.8333496922719489, -0.09487913683736007]]
    
    detpos = [
        [123, 205,  65,   1],
        [130, 199,  79,   1],
        [124, 203,  73,   1],
        [115, 207,  80,   1],
        [ 81, 211,  82,   1],
        [ 90, 212,  67,   1],
        [ 90, 211,  78,   1],
        [ 99, 210,  82,   1],
        [ 56, 205,  66,   1],
        [ 63, 207,  80,   1],
        [ 54, 203,  76,   1],
        [ 48, 199,  79,   1],
    ]
    base_cfg = {
        'nphoton': int(1e7),
        'vol': vol,
        # Optical properties: each row is [mua, mus, g, n], need to double check values, g = anistropy
        'prop':  [
                [0.0,   0.0,  1.0, 1.0],      # Free space / Background
                [1.04, 60.8, 7, 1.37],  # White Matter
                [0.35, 7.1, 0.90, 1.37],  # Gray Matter
                [0.043, 0, 0.02, 1.33],  # CSF (Using NaN for N/A)
                [0.28, 16, 0.94, 1.43],  # Bone (General)
                [0.38, 18, 0.81, 1.37],  # Scalp (Skin)
                [0.075, 0.2, 0.90, 1.336],  # Eyeball (using midpoints of ranges)
                [0.125, 11.5, 0.90, 1.54],  # Compact Bone (using midpoints of ranges)
                [0.115, 5.5, 0.90, 1.50],  # Spongy Bone (using midpoints of ranges)
                [6.5, 3, 0.99, 1.40],  # Blood (oxy) (using midpoints of ranges)
                [4.5, 3, 0.99, 1.40],  # Blood (deoxy) (using midpoints of ranges)
                [0.30, 6.6, 0.90, 1.40]  # Muscle
            ],
        'srcpos': srcpos,
        'srcparam1': srcparam1,
        'srcparam2':srcparam2,
        'srcdir': src_dir,
        'srctype': 'isotropic', #https://github.com/fangq/mcx/blob/5332f081fe5501281c7197a1feb09870ea252ad5/mcxlab/mcxlab.m#L188
        #'srcpattern':      # see cfg.srctype for details
        #'omega': source modulation frequency (rad/s) for RF replay, 2*pi*f
        'tstart': 0,
        'tend': 2e-9, # end time of simulation
        'tstep': 5e-11, # time step through simulation
        'srcid': 0, # -1 mean each source separately simulated, 0 means altogether, number specifies single
        # Default detectors (this will be replaced per setup below).
        'detpos': detpos,
        'issrcfrom0': 1, # 1-first voxel is [0 0 0], [0]- first voxel is [1 1 1]
        'lambda': 735
    }

    # Common simulation settings
    base_cfg['autopilot'] = 1
    base_cfg['issrcfrom0'] = 1
    base_cfg['issavedet'] = 1
    base_cfg['savedetflag'] = "dspmxvwi"
    base_cfg['issaveseed'] = 1

    # Get number of sources and detectors
    n_src = len(srcpos)
    n_det = len(detpos)
    print(f"Number of sources: {n_src}")
    print(f"Number of detectors: {n_det}")
    
    # Create a matrix to hold all Jacobians 
    jacobian_matrix = []
    # Create a list to track the source-detector pairs
    measurement_order = []
    
    # Directory to save individual Jacobians
    output_dir = "individual_sd_jacobian_results"
    os.makedirs(output_dir, exist_ok=True)
    none_detected = []
    
    # Loop through each source-detector pair
    for src_idx in range(n_src):
        print(f"Processing source {src_idx+1}/{n_src}")
        
        # Create configuration with only current source
        src_cfg = base_cfg.copy()
        if n_src > 1:
            src_cfg['srcpos'] = base_cfg['srcpos'][src_idx:src_idx+1]
        
        for det_idx in range(n_det):
            print(f"  Processing detector {det_idx+1}/{n_det}")
            
            # Add this source-detector pair to our tracking list
            measurement_order.append((src_idx, det_idx))
            
            # Create configuration with only current detector
            sd_cfg = src_cfg.copy()
            if n_det > 1:
                sd_cfg['detpos'] = base_cfg['detpos'][det_idx:det_idx+1]
            
            # Run the forward simulation for this source-detector pair
            res = pmcx.mcxlab(sd_cfg)
            
            # Check if photons were detected
            if 'detp' in res and 'seeds' in res:
                detp = res['detp']
                seeds = res['seeds']
                
                # Set up Jacobian calculation
                jac_cfg = sd_cfg.copy()
                jac_cfg['seed'] = seeds
                jac_cfg['outputtype'] = "jacobian"
                jac_cfg['detphotons'] = detp["data"]
                
                # Run the Jacobian calculation
                jac_results = pmcx.mcxlab(jac_cfg)
                
                if 'flux' in jac_results:
                    # Get the Jacobian for this SD pair
                    jac_volume = np.sum(jac_results['flux'], axis=3)
                    
                    # Save the individual Jacobian volume
                    filename = f"{output_dir}/jacobian_src{src_idx}_det{det_idx}.npy"
                    np.save(filename, jac_volume)
                    
                    # Reshape to a row vector
                    jac_row = jac_volume.flatten()
                    
                    # Add to the Jacobian matrix
                    jacobian_matrix.append(jac_row)
                    
                    print(f"    Added Jacobian row, shape: {jac_row.shape}")
                else:
                    print(f"    No flux in Jacobian results for source {src_idx+1}, detector {det_idx+1}")
            else:
                none_detected.append((src_idx, det_idx))
                print(f"------No photons detected for source {src_idx+1}, detector {det_idx+1}  =( -------)")
                jacobian_matrix.append(np.zeros(vol.size)) # maintain consisten size
    
    print(f"None detected: {none_detected}")
    print(f"Number of none detected: {len(none_detected)}")
    # Convert to numpy array
    jacobian_matrix = np.array(jacobian_matrix)
    #measurement_order = np.array(measurement_order)
    n_meas = jacobian_matrix.shape[0]

    # Directory to save individual Jacobians
    output_dir = "jacobian_files"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = "jacobian_files/"

    # Create a copy to avoid modifying the original
    result = np.zeros_like(vol)
    # Only keep the specified label
    result[vol == grey_matter_index] = grey_matter_index
    gm_mask_flat = (vol.flatten() == grey_matter_index)

    refInds = np.array(base_cfg['prop'])[:,2]
    refIndVec = np.zeros(vol.shape)
    for xind in range(vol.shape[0]):
        for yind in range(vol.shape[1]):
            for zind in range(vol.shape[2]):
                refIndVec[xind,yind,zind] = refInds[int(vol[xind,yind,zind])]
    
    c0 = 0.3 # speed of light in vacuum (m/ns ???) # directly from DOTHUB_makeToastJacboian
    c_medium = c0 / refIndVec.flatten()
    file_path = "735nm_9src_12_det_example.json"
    n_meas = jacobian_matrix.shape[0]
    #whole basis implementation is like DOTHUB_makeToastJacboian
    if use_basis:
        # Create the VoxelBasis instance using the full volume dimensions and desired basis grid.
        # For example, basis_size = (50, 50, 50)
        vBasis = VoxelBasis(vol.shape, basis_size=basis_size)
        nBasisNodes = np.prod(vBasis.basis_size)

        # Compute the mapping vector that maps c_medium (full-volume, flattened) to basis space.
        # This returns a flattened vector of length nBasisNodes.
        mapping_vec = vBasis.Map('M->S', c_medium, flatten=True)
        
        # Prepare lists to store the processed Jacobians.
        J_basis_list = []  # Jacobian in the reduced (basis) space
        J_vol_list = []    # Jacobian mapped back to full volume space
        J_gm_list = []     # Gray matter (GM) Jacobian after applying vol2gm

        for idx in range(n_meas):
            # Get the flattened Jacobian row for the current measurement.
            # Here, since there is only one wavelength/channel, Jtmp has shape (vol.size,)
            Jtmp = jacobian_matrix[idx]

            # Map from full volume (M) to basis (S) space.
            Jtmp_basis = vBasis.Map('M->S', Jtmp, flatten=True)  # shape: (nBasisNodes,)

            # Multiply elementwise by the mapping vector (this mimics the MATLAB repmat operation)
            Jtmp_basis_scaled = Jtmp_basis * mapping_vec  # still shape: (nBasisNodes,)
            J_basis_list.append(Jtmp_basis_scaled)

            # Map each measurement back from basis space (S) to full volume (M) space.
            Jtmp_vol = vBasis.Map('S->M', Jtmp_basis_scaled, flatten=True)  # shape: (vol.size,)
            J_vol_list.append(Jtmp_vol)

            # Compute the GM Jacobian by multiplying vol2gm (shape: [nGM, vol.size])
            # by the full-volume Jacobian vector.
            # The result J_gm has shape (nGM,)
            J_gm = Jtmp_vol[gm_mask_flat] 
            J_gm_list.append(J_gm)

        # Convert lists to NumPy arrays.
        jacobian_basis = np.array(J_basis_list)  # Shape: (n_meas, nBasisNodes)
        jacobian_vol = None
        jacobian_gm    = np.array(J_gm_list)       # Shape: (n_meas, nGM)

        print("BASIS MODE:")
        print("Jacobian_basis shape:", jacobian_basis.shape)
        #print("Jacobian_vol shape:", jacobian_vol.shape)
        print("Jacobian_gm shape:", jacobian_gm.shape)
        print("saved to: ",file_path[:-5]+'_jacobian')
        file_name = file_path[:-5]+'_jacobian'
        np.savez(output_dir+file_name+'_basis.npz',  jacobian=jacobian_basis)
        np.savez(output_dir+file_name+'_basis_gm.npz', jacobian=jacobian_gm)
    else:
        # VOLUME MODE: directly scale each flattened Jacobian row by c_medium.
        jacobian_vol = jacobian_matrix * c_medium  # elementwise multiplication (broadcasting)
        
        # Compute GM Jacobian for each measurement.
        J_gm_list = []
        for idx in range(n_meas):
            Jtmp = jacobian_vol[idx]  # shape: (vol.size,)
            J_gm = Jtmp[gm_mask_flat]
            J_gm_list.append(J_gm)
        jacobian_gm = np.array(J_gm_list)
        jacobian_basis = None
        print("VOLUME MODE:")
        print("Jacobian_vol shape:", jacobian_vol.shape)
        print("Jacobian_gm shape:", jacobian_gm.shape)
        print("saved to: ",file_path[:-5]+'_jacobian')
        file_name = file_path[:-5]+'_jacobian'
        np.savez(output_dir+file_name+'_data.npz',  jacobian=jacobian_matrix)
        np.savez(output_dir+file_name+'_gm.npz', jacobian=jacobian_gm)


    # Print final Jacobian shape
    print(f"Final Jacobian matrix shape: {jacobian_matrix.shape}")
    
        
    measurement_order_list = [{"source": int(src), "detector": int(det)} for src, det in measurement_order]
    with open('measurement_order.json', 'w') as f:
        json.dump(measurement_order_list, f, indent=4)

    # Plot the sensitivity of a specific SD pair (first one)
    if len(jacobian_matrix) > 0:
        # Reshape the first row back to volume for visualization
        sd_pair_idx = 0
        jac_volume = jacobian_matrix[sd_pair_idx].reshape(vol.shape)
        
        # Get the source and detector indices for this pair
        src_idx, det_idx = measurement_order[sd_pair_idx]
        
        # Visualize a slice
        mid_slice = jac_volume.shape[1] // 2
        plt.figure(figsize=(10, 8))
        
        # Add small epsilon to avoid log10(0)
        jac_for_log = np.abs(jac_volume) + 1e-10
        
        plt.imshow(np.log10(jac_for_log[:, mid_slice, :]), cmap='jet')
        plt.colorbar(label='log10(Sensitivity)')
        plt.title(f'Jacobian for Source {src_idx+1}, Detector {det_idx+1}')
        plt.xlabel('Z dimension')
        plt.ylabel('X dimension')
        plt.savefig('jac_src_det_sens.png')
        plt.show()
        