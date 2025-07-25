#!/usr/bin/env python3
"""
Simple plasma data processor for visual inspection.

This script processes H5 files and displays interactive plots for inspection.
Can be extended to create datasets later.
"""

import os
import glob
import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import interpolate
import warnings
import json
import time

warnings.filterwarnings('ignore')

class SimplePlasmaProcessor:
    """Simple processor for plasma diagnostic data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.processed_shots = []
        
    def find_h5_files(self):
        """Find all H5 files in the data directory."""
        pattern = os.path.join(self.data_dir, "*.h5")
        files = glob.glob(pattern)
        print(f"Found {len(files)} H5 files:")
        for file in files:
            print(f"  - {os.path.basename(file)}")
        return files
    
    def extract_shot_number(self, filepath: str) -> str:
        """Extract shot number from filename."""
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        if len(parts) >= 2 and parts[0] == 'shot':
            return parts[1]
        return filename.replace('.h5', '')
    
    def process_single_file(self, filepath: str):
        """Process a single H5 file and create interactive plot."""
        shot_number = self.extract_shot_number(filepath)
        print(f"\nProcessing Shot {shot_number}")
        print("-" * 40)
        
        try:
            with h5py.File(filepath, 'r') as f:
                # Check structure
                print("Available groups:")
                for key in f.keys():
                    print(f"  - {key}")
                
                # Extract time arrays
                astra_time = f['astra/time'][0] if 'astra/time' in f else None
                width_time = f['width/time'][0] if 'width/time' in f else None
                mhd_time = f['mhd/time'][0] if 'mhd/time' in f else None
                
                if astra_time is None or width_time is None or mhd_time is None:
                    print("❌ Missing required time arrays")
                    return False
                
                # Extract data
                astra_q = f['astra/q'][:] if 'astra/q' in f else None  # All radial positions
                width_width = f['width/width'][0] if 'width/width' in f else None
                mhd_amplitude = f['mhd/amplitude'][0] if 'mhd/amplitude' in f else None
                
                if astra_q is None or width_width is None or mhd_amplitude is None:
                    print("❌ Missing required data arrays")
                    return False
                
                print(f"ASTRA q-profile shape: {astra_q.shape}")
                print(f"Number of radial positions: {astra_q.shape[1] if astra_q.ndim > 1 else 'N/A'}")
                
                # Find common time window
                astra_valid = astra_time[~np.isnan(astra_time)]
                width_valid = width_time[~np.isnan(width_time)]
                mhd_valid = mhd_time[~np.isnan(mhd_time)]
                
                time_start = max(astra_valid.min(), width_valid.min(), mhd_valid.min())
                time_end = min(astra_valid.max(), width_valid.max(), mhd_valid.max())
                
                print(f"Time window: {time_start:.4f} - {time_end:.4f} seconds")
                print(f"Duration: {time_end - time_start:.4f} seconds")
                
                if time_start >= time_end:
                    print("❌ No overlapping time range")
                    return False
                
                # Create common time grid
                common_time = np.linspace(time_start, time_end, 1000)
                
                # Interpolate data
                datasets = {}
                
                # ASTRA q-profile (use middle radial position for 1D plot)
                if astra_q.ndim > 1:
                    astra_q_mid = astra_q[:, 30]  # Middle radial position for plotting
                    astra_q_core = astra_q[:, 5]   # Core position
                    astra_q_edge = astra_q[:, -5]  # Edge position
                else:
                    astra_q_mid = astra_q
                    astra_q_core = astra_q
                    astra_q_edge = astra_q
                
                astra_mask = ~np.isnan(astra_time) & ~np.isnan(astra_q_mid)
                if np.sum(astra_mask) > 1:
                    astra_interp = interpolate.interp1d(
                        astra_time[astra_mask], astra_q_mid[astra_mask],
                        kind='linear', bounds_error=False, fill_value=np.nan
                    )
                    datasets['ASTRA q-profile (mid)'] = astra_interp(common_time)
                    
                    # Also interpolate core and edge if available
                    if astra_q.ndim > 1:
                        astra_mask_core = ~np.isnan(astra_time) & ~np.isnan(astra_q_core)
                        astra_mask_edge = ~np.isnan(astra_time) & ~np.isnan(astra_q_edge)
                        
                        if np.sum(astra_mask_core) > 1:
                            astra_interp_core = interpolate.interp1d(
                                astra_time[astra_mask_core], astra_q_core[astra_mask_core],
                                kind='linear', bounds_error=False, fill_value=np.nan
                            )
                            datasets['q-core'] = astra_interp_core(common_time)
                        
                        if np.sum(astra_mask_edge) > 1:
                            astra_interp_edge = interpolate.interp1d(
                                astra_time[astra_mask_edge], astra_q_edge[astra_mask_edge],
                                kind='linear', bounds_error=False, fill_value=np.nan
                            )
                            datasets['q-edge'] = astra_interp_edge(common_time)
                
                # Width
                width_mask = ~np.isnan(width_time) & ~np.isnan(width_width)
                if np.sum(width_mask) > 1:
                    width_interp = interpolate.interp1d(
                        width_time[width_mask], width_width[width_mask],
                        kind='linear', bounds_error=False, fill_value=np.nan
                    )
                    datasets['Width'] = width_interp(common_time)
                
                # MHD Amplitude
                mhd_mask = ~np.isnan(mhd_time) & ~np.isnan(mhd_amplitude)
                if np.sum(mhd_mask) > 1:
                    mhd_interp = interpolate.interp1d(
                        mhd_time[mhd_mask], mhd_amplitude[mhd_mask],
                        kind='linear', bounds_error=False, fill_value=np.nan
                    )
                    datasets['MHD Amplitude'] = mhd_interp(common_time)
                
                # Create interactive plot
                n_plots = len(datasets)
                fig = make_subplots(
                    rows=n_plots, cols=1,
                    shared_xaxes=True,
                    subplot_titles=[f"{name} (Shot {shot_number})" for name in datasets.keys()],
                    vertical_spacing=0.08
                )
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                
                for i, (name, data) in enumerate(datasets.items(), 1):
                    fig.add_trace(
                        go.Scatter(
                            x=common_time,
                            y=data,
                            mode='lines',
                            name=name,
                            line=dict(color=colors[(i-1) % len(colors)], width=2),
                            hovertemplate=f'<b>{name}</b><br>Time: %{{x:.4f}} s<br>Value: %{{y:.6f}}<extra></extra>'
                        ),
                        row=i, col=1
                    )
                
                # Update layout
                fig.update_layout(
                    height=200 * n_plots + 200,  # Dynamic height based on number of plots
                    title=dict(
                        text=f"Plasma Diagnostics - Shot {shot_number}",
                        x=0.5,
                        font=dict(size=16)
                    ),
                    showlegend=True,
                    hovermode='x unified'
                )
                
                # Update axes
                fig.update_xaxes(title_text="Time (seconds)", row=n_plots, col=1)
                
                # Set y-axis labels
                for i, (name, _) in enumerate(datasets.items(), 1):
                    if 'q-' in name or 'q-profile' in name:
                        fig.update_yaxes(title_text="q-value", row=i, col=1)
                    elif 'Width' in name:
                        fig.update_yaxes(title_text="Width", row=i, col=1)
                    elif 'Amplitude' in name:
                        fig.update_yaxes(title_text="Amplitude", row=i, col=1)
                
                # Show plot
                print("\n📊 Displaying interactive plot...")
                print("   Please review the plot in your browser/plot window")
                
                # Configure plotly to open in browser
                import plotly.io as pio
                pio.renderers.default = "browser"
                
                fig.show()
                
                # Wait a moment for plot to render and give user time to review
                print("   Waiting 3 seconds for plot to load...")
                time.sleep(3)
                print("   Plot should now be visible in your browser. Please review the data quality.")
                
                # Store processed data info
                shot_data = {
                    'shot_number': shot_number,
                    'filepath': filepath,
                    'time_start': time_start,
                    'time_end': time_end,
                    'duration': time_end - time_start,
                    'n_points': len(common_time),
                    'valid_interpolation': True
                }
                
                print("✅ Processing successful")
                return shot_data
                
        except Exception as e:
            print(f"❌ Error processing {filepath}: {str(e)}")
            return False
    
    def process_all_files(self, interactive=True):
        """Process all files with optional interactive inspection."""
        files = self.find_h5_files()
        
        if not files:
            print("No H5 files found!")
            return
        
        approved_shots = []
        
        for i, filepath in enumerate(files, 1):
            print(f"\n{'='*50}")
            print(f"Processing file {i}/{len(files)}")
            print(f"{'='*50}")
            
            shot_data = self.process_single_file(filepath)
            
            if shot_data:
                if interactive:
                    print(f"\n{'='*60}")
                    print(f"REVIEW SHOT {shot_data['shot_number']}")
                    print(f"{'='*60}")
                    print(f"Time range: {shot_data['time_start']:.4f} - {shot_data['time_end']:.4f} seconds")
                    print(f"Duration: {shot_data['duration']:.4f} seconds")
                    print(f"Data points: {shot_data['n_points']}")
                    print("\nPlease check the interactive plot for data quality issues:")
                    print("- Look for discontinuities or artifacts")
                    print("- Check if interpolation looks reasonable") 
                    print("- Verify all three signals are present and meaningful")
                    
                    while True:
                        response = input(f"\n🔍 Approve Shot {shot_data['shot_number']} for dataset? (y/n/q): ").lower().strip()
                        if response in ['y', 'yes']:
                            approved_shots.append(shot_data)
                            print(f"✅ Shot {shot_data['shot_number']} approved")
                            break
                        elif response in ['n', 'no']:
                            print(f"❌ Shot {shot_data['shot_number']} rejected")
                            break
                        elif response in ['q', 'quit']:
                            print("Processing stopped by user")
                            self.save_summary(approved_shots)
                            return approved_shots
                        else:
                            print("Please enter 'y' for yes, 'n' for no, or 'q' to quit")
                else:
                    approved_shots.append(shot_data)
        
        self.save_summary(approved_shots)
        return approved_shots
    
    def save_summary(self, approved_shots):
        """Save summary of processed shots."""
        if not approved_shots:
            print("No shots approved for dataset")
            return
        
        summary = {
            'n_shots': len(approved_shots),
            'shot_numbers': [shot['shot_number'] for shot in approved_shots],
            'time_ranges': {
                'min_start': min(shot['time_start'] for shot in approved_shots),
                'max_end': max(shot['time_end'] for shot in approved_shots),
                'min_duration': min(shot['duration'] for shot in approved_shots),
                'max_duration': max(shot['duration'] for shot in approved_shots)
            },
            'approved_shots': approved_shots
        }
        
        # Save summary
        with open('processed_shots_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n📊 Summary saved to processed_shots_summary.json")
        print(f"   - Total approved shots: {len(approved_shots)}")
        print(f"   - Shot numbers: {', '.join(summary['shot_numbers'])}")

def main():
    """Main function."""
    print("Simple Plasma Data Processor")
    print("="*40)
    print("This script will process H5 files and display interactive plots")
    print("for visual inspection of plasma diagnostic data.\n")
    
    processor = SimplePlasmaProcessor()
    
    # Check if data directory exists
    if not os.path.exists(processor.data_dir):
        print(f"❌ Data directory '{processor.data_dir}' not found!")
        print("Please ensure your H5 files are in the 'data' directory")
        return
    
    # Process all files
    approved_shots = processor.process_all_files(interactive=True)
    
    if approved_shots:
        print("\n🎉 Processing complete!")
        print(f"   Approved {len(approved_shots)} shots for dataset creation")
        print("\nNext steps:")
        print("1. Review processed_shots_summary.json")
        print("2. Use the full process_plasma_data.py script to create HuggingFace dataset")
    else:
        print("\n❌ No shots were approved")

if __name__ == "__main__":
    main()
