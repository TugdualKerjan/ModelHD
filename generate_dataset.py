#!/usr/bin/env python3
"""
Generate HuggingFace dataset from processed shots summary.

This script reads the processed_shots_summary.json file and creates a HuggingFace dataset
with 2D q-profile features, MHD amplitude, and width data.
"""

import json
import h5py
import numpy as np
from datasets import Dataset, Features, Value
from scipy import interpolate
import os

def load_and_process_shot(filepath: str, time_start: float, time_end: float, n_points: int = 1000):
    """Load and process a single shot file."""
    print(f"Processing {os.path.basename(filepath)}...")
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Extract time arrays
            astra_time = f['astra/time'][0]
            width_time = f['width/time'][0]
            mhd_time = f['mhd/time'][0]
            
            # Extract data
            astra_q = f['astra/q'][:]  # Full 2D q-profile
            astra_rhopol = f['astra/rhopol'][0, :] if 'astra/rhopol' in f else np.linspace(0, 1, astra_q.shape[1])
            width_width = f['width/width'][0]
            mhd_amplitude = f['mhd/amplitude'][0]
            
            # Create common time grid
            common_time = np.linspace(time_start, time_end, n_points)
            
            # Initialize output arrays
            n_radial = astra_q.shape[1]
            q_profile_interp = np.zeros((n_points, n_radial))
            
            # Interpolate q-profile for each radial position
            for rad_idx in range(n_radial):
                astra_q_rad = astra_q[:, rad_idx]
                astra_mask = ~np.isnan(astra_time) & ~np.isnan(astra_q_rad)
                
                if np.sum(astra_mask) > 1:
                    interp_func = interpolate.interp1d(
                        astra_time[astra_mask], astra_q_rad[astra_mask],
                        kind='linear', bounds_error=False, fill_value=np.nan
                    )
                    q_profile_interp[:, rad_idx] = interp_func(common_time)
                else:
                    q_profile_interp[:, rad_idx] = np.nan
            
            # Interpolate width data
            width_mask = ~np.isnan(width_time) & ~np.isnan(width_width)
            if np.sum(width_mask) > 1:
                width_interp_func = interpolate.interp1d(
                    width_time[width_mask], width_width[width_mask],
                    kind='linear', bounds_error=False, fill_value=np.nan
                )
                width_interp = width_interp_func(common_time)
            else:
                width_interp = np.full(n_points, np.nan)
            
            # Interpolate MHD amplitude data
            mhd_mask = ~np.isnan(mhd_time) & ~np.isnan(mhd_amplitude)
            if np.sum(mhd_mask) > 1:
                mhd_interp_func = interpolate.interp1d(
                    mhd_time[mhd_mask], mhd_amplitude[mhd_mask],
                    kind='linear', bounds_error=False, fill_value=np.nan
                )
                mhd_amplitude_interp = mhd_interp_func(common_time)
            else:
                mhd_amplitude_interp = np.full(n_points, np.nan)
            
            return {
                'time': common_time,
                'q_profile': q_profile_interp,
                'rhopol': astra_rhopol,
                'width': width_interp,
                'mhd_amplitude': mhd_amplitude_interp,
                'n_radial': n_radial
            }
            
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None

def create_dataset_from_summary(summary_file: str = "processed_shots_summary.json", 
                               dataset_name: str = "plasma_diagnostics_dataset"):
    """Create HuggingFace dataset from processed shots summary."""
    
    # Load the summary
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    print(f"Creating dataset from {summary['n_shots']} approved shots...")
    
    # Process all shots
    dataset_data = {
        'shot_number': [],
        'filepath': [],
        'time_start': [],
        'time_end': [],
        'duration': [],
        'time': [],
        'q_profile': [],
        'rhopol': [],
        'width': [],
        'mhd_amplitude': []
    }
    
    n_radial_positions = None
    n_time_points = None
    
    for shot_info in summary['approved_shots']:
        shot_data = load_and_process_shot(
            shot_info['filepath'],
            shot_info['time_start'],
            shot_info['time_end'],
            shot_info['n_points']
        )
        
        if shot_data is None:
            print(f"Skipping shot {shot_info['shot_number']} due to processing error")
            continue
        
        # Set dimensions from first successful shot
        if n_radial_positions is None:
            n_radial_positions = shot_data['n_radial']
            n_time_points = len(shot_data['time'])
            print(f"Dataset dimensions: {n_time_points} time points, {n_radial_positions} radial positions")
        
        # Check consistency
        if shot_data['n_radial'] != n_radial_positions:
            print(f"Warning: Shot {shot_info['shot_number']} has {shot_data['n_radial']} radial positions, expected {n_radial_positions}")
            continue
        
        # Add to dataset
        dataset_data['shot_number'].append(shot_info['shot_number'])
        dataset_data['filepath'].append(shot_info['filepath'])
        dataset_data['time_start'].append(shot_info['time_start'])
        dataset_data['time_end'].append(shot_info['time_end'])
        dataset_data['duration'].append(shot_info['duration'])
        dataset_data['time'].append(shot_data['time'].tolist())
        dataset_data['q_profile'].append(shot_data['q_profile'].tolist())
        dataset_data['rhopol'].append(shot_data['rhopol'].tolist())
        dataset_data['width'].append(shot_data['width'].tolist())
        dataset_data['mhd_amplitude'].append(shot_data['mhd_amplitude'].tolist())
        
        print(f"✓ Added shot {shot_info['shot_number']}")
    
    if len(dataset_data['shot_number']) == 0:
        raise ValueError("No shots were successfully processed")
    
    print(f"\nSuccessfully processed {len(dataset_data['shot_number'])} shots")
    
    # Define HuggingFace dataset features
    features = Features({
        'shot_number': Value('string'),
        'filepath': Value('string'),
        'time_start': Value('float64'),
        'time_end': Value('float64'),
        'duration': Value('float64'),
        'time': [Value('float64')],  # 1D array
        'q_profile': [[Value('float64')]],  # 2D array  
        'rhopol': [Value('float64')],  # 1D array
        'width': [Value('float64')],  # 1D array
        'mhd_amplitude': [Value('float64')]  # 1D array
    })
    
    # Create dataset
    print("\nCreating HuggingFace dataset...")
    dataset = Dataset.from_dict(dataset_data, features=features)
    
    # Save dataset
    print(f"Saving dataset to {dataset_name}/")
    dataset.save_to_disk(dataset_name)
    
    # Create and save metadata
    metadata = {
        'description': 'Plasma diagnostics dataset with 2D q-profiles, width, and MHD amplitude',
        'n_samples': len(dataset_data['shot_number']),
        'n_time_points': n_time_points,
        'n_radial_positions': n_radial_positions,
        'shot_numbers': dataset_data['shot_number'],
        'time_ranges': summary['time_ranges'],
        'features': {
            'shot_number': 'Unique identifier for each plasma shot',
            'filepath': 'Path to original H5 file',
            'time_start': 'Start time of the analysis window (seconds)',
            'time_end': 'End time of the analysis window (seconds)',
            'duration': 'Duration of the analysis window (seconds)',
            'time': f'Time grid with {n_time_points} points (seconds)',
            'q_profile': f'2D q-safety factor profile evolution ({n_time_points} x {n_radial_positions})',
            'rhopol': f'Normalized radial coordinate grid ({n_radial_positions} points)',
            'width': f'Mode width evolution ({n_time_points} points)',
            'mhd_amplitude': f'MHD amplitude evolution ({n_time_points} points)'
        },
        'physics_info': {
            'q_profile': 'Safety factor profile - critical for MHD stability',
            'rhopol': 'Normalized poloidal flux coordinate (0=magnetic axis, 1=edge)',
            'width': 'Characteristic width of MHD modes',
            'mhd_amplitude': 'Amplitude of MHD fluctuations'
        }
    }
    
    with open(f"{dataset_name}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Dataset saved to {dataset_name}/")
    print(f"✓ Metadata saved to {dataset_name}/metadata.json")
    print("\nDataset summary:")
    print(f"  - {len(dataset_data['shot_number'])} plasma shots")
    print(f"  - Shape per shot: q_profile({n_time_points}, {n_radial_positions}), width({n_time_points}), mhd_amplitude({n_time_points})")
    print(f"  - Total size: {dataset.shape}")
    print(f"  - Features: {list(dataset.features.keys())}")
    
    return dataset

def main():
    """Main function."""
    print("Plasma Diagnostics Dataset Generator")
    print("=" * 50)
    
    # Check if summary file exists
    summary_file = "processed_shots_summary.json"
    if not os.path.exists(summary_file):
        print(f"❌ Summary file '{summary_file}' not found!")
        print("Please run the simple_plasma_processor.py script first to generate the summary.")
        return
    
    try:
        # Create the dataset
        create_dataset_from_summary(summary_file)
        
        print("\n🎉 Dataset creation completed successfully!")
        print("\nYou can now load the dataset with:")
        print("```python")
        print("from datasets import load_from_disk")
        print("dataset = load_from_disk('plasma_diagnostics_dataset')")
        print("```")
        
    except Exception as e:
        print(f"\n❌ Error creating dataset: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
