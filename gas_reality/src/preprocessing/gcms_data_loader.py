# gcms_data_loader.py
# Advanced GCMS data loader for 6-species gas spectroscopy analysis

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class GCMSDataLoader:
    """
    Advanced GCMS data loader for real experimental spectroscopy data
    Handles 6 chemical species: SOF2, SO2F2, SO2, NO, NO2, NF3
    Supports multiple voltage levels (22kV, 24kV, 36kV) and time series
    """
    
    def __init__(self, gcms_path: str = "E:/generate_mixture/gcms"):
        self.gcms_path = Path(gcms_path)
        self.species_names = ['SOF2', 'SO2F2', 'SO2', 'NO', 'NO2', 'NF3']
        self.voltage_levels = ['22kv', '24kv', '36kv']
        
        # Data storage
        self.spectral_data = {}
        self.concentration_data = {}
        self.metadata = {}
        
        print(f"GCMS Data Loader initialized")
        print(f"Target path: {self.gcms_path}")
        print(f"Chemical species: {', '.join(self.species_names)}")
        print(f"Voltage levels: {', '.join(self.voltage_levels)}")
        
    def extract_file_metadata(self, filename: str) -> Dict:
        """Extract voltage and time information from filename"""
        # Parse pattern: 22kv0h.CSV, 24kv108h.CSV, etc.
        pattern = r'(\d+)kv(\d+)h\.CSV'
        match = re.match(pattern, filename)
        
        if match:
            voltage = int(match.group(1))
            time_hours = int(match.group(2))
            return {
                'voltage': voltage,
                'time_hours': time_hours,
                'voltage_level': f"{voltage}kv",
                'filename': filename
            }
        else:
            raise ValueError(f"Cannot parse filename: {filename}")
    
    def load_spectral_files(self) -> Dict:
        """Load all spectral CSV files from GCMS directory"""
        print("\n" + "="*60)
        print("LOADING SPECTRAL DATA")
        print("="*60)
        
        spectral_files = list(self.gcms_path.glob("*.CSV"))
        print(f"Found {len(spectral_files)} spectral files")
        
        loaded_data = {}
        
        for file_path in spectral_files:
            try:
                # Extract metadata
                metadata = self.extract_file_metadata(file_path.name)
                
                # Load spectral data
                df = pd.read_csv(file_path)
                
                # Check data format and standardize column names
                if len(df.columns) >= 2:
                    # Assume first column is wavenumber/frequency, second is intensity
                    df_clean = pd.DataFrame({
                        'wavenumber': df.iloc[:, 0].values,
                        'intensity': df.iloc[:, 1].values
                    })
                    
                    # Store data with metadata
                    file_key = f"{metadata['voltage']}kv_{metadata['time_hours']}h"
                    loaded_data[file_key] = {
                        'data': df_clean,
                        'metadata': metadata,
                        'filename': file_path.name
                    }
                    
                    print(f"OK {file_path.name}: {len(df_clean)} points, "
                          f"Range: {df_clean['wavenumber'].min():.1f}-{df_clean['wavenumber'].max():.1f}")
                
                else:
                    print(f"ERROR {file_path.name}: Invalid format (need at least 2 columns)")
                    
            except Exception as e:
                print(f"ERROR {file_path.name}: Error loading - {e}")
        
        self.spectral_data = loaded_data
        print(f"\nSuccessfully loaded {len(loaded_data)} spectral files")
        
        return loaded_data
    
    def load_concentration_files(self) -> Dict:
        """Load concentration data from Excel files"""
        print("\n" + "="*60)
        print("LOADING CONCENTRATION DATA")
        print("="*60)
        
        hanliang_path = self.gcms_path / "hanliang"
        excel_files = list(hanliang_path.glob("*.xlsx"))
        print(f"Found {len(excel_files)} concentration files")
        
        concentration_data = {}
        
        for file_path in excel_files:
            try:
                # Extract voltage from filename
                voltage_match = re.search(r'(\d+)kv', file_path.name)
                if not voltage_match:
                    print(f"ERROR {file_path.name}: Cannot extract voltage")
                    continue
                
                voltage = int(voltage_match.group(1))
                voltage_key = f"{voltage}kv"
                
                # Read Excel file
                df = pd.read_excel(file_path)
                print(f"OK {file_path.name}: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"  Columns: {list(df.columns)}")
                
                # Process concentration data
                processed_data = self.process_concentration_data(df, voltage)
                concentration_data[voltage_key] = processed_data
                
            except Exception as e:
                print(f"ERROR {file_path.name}: Error loading - {e}")
        
        self.concentration_data = concentration_data
        print(f"\nSuccessfully loaded concentration data for {len(concentration_data)} voltage levels")
        
        return concentration_data
    
    def process_concentration_data(self, df: pd.DataFrame, voltage: int) -> Dict:
        """Process and standardize concentration data"""
        processed = {
            'voltage': voltage,
            'raw_data': df.copy(),
            'time_series': {},
            'species_data': {}
        }
        
        # Try to identify time and species columns
        # Look for time-related columns (including unnamed first column which is often time)
        time_cols = [col for col in df.columns if any(keyword in col.lower() 
                    for keyword in ['time', 'hour', 'h', '时间', 'unnamed'])]
        
        # Look for species columns
        species_cols = {}
        for species in self.species_names:
            # Exact match first, then partial match
            exact_match = [col for col in df.columns if col.upper() == species.upper()]
            if exact_match:
                species_cols[species] = exact_match[0]
            else:
                # For partial matches, be more careful with SO2 vs SO2F2
                if species == 'SO2':
                    # Only match exact SO2, not SO2F2
                    matching_cols = [col for col in df.columns if col.upper() == 'SO2']
                else:
                    matching_cols = [col for col in df.columns if species.upper() in col.upper()]
                if matching_cols:
                    species_cols[species] = matching_cols[0]
        
        print(f"    Time columns: {time_cols}")
        print(f"    Species columns: {species_cols}")
        
        # If we have time data, create time series
        if time_cols and species_cols:
            time_col = time_cols[0]
            
            for species, col_name in species_cols.items():
                if col_name in df.columns:
                    time_series = df[[time_col, col_name]].dropna()
                    processed['time_series'][species] = time_series
                    processed['species_data'][species] = {
                        'values': time_series[col_name].values,
                        'times': time_series[time_col].values,
                        'mean': time_series[col_name].mean(),
                        'std': time_series[col_name].std(),
                        'range': (time_series[col_name].min(), time_series[col_name].max())
                    }
        
        return processed
    
    def create_unified_dataset(self) -> Dict:
        """Create unified dataset combining spectral and concentration data"""
        print("\n" + "="*60)
        print("CREATING UNIFIED DATASET")
        print("="*60)
        
        if not self.spectral_data or not self.concentration_data:
            raise ValueError("Must load both spectral and concentration data first")
        
        unified_data = {
            'samples': [],
            'metadata': {
                'n_samples': 0,
                'n_features': 0,
                'voltage_levels': self.voltage_levels,
                'species': self.species_names,
                'time_range': (0, 120)  # hours
            }
        }
        
        sample_count = 0
        
        for voltage_level in self.voltage_levels:
            print(f"\nProcessing {voltage_level} data...")
            
            # Get spectral data for this voltage
            spectral_keys = [k for k in self.spectral_data.keys() if k.startswith(voltage_level)]
            
            # Get concentration data for this voltage
            if voltage_level not in self.concentration_data:
                print(f"  Warning: No concentration data for {voltage_level}")
                continue
            
            conc_data = self.concentration_data[voltage_level]
            
            for spectral_key in spectral_keys:
                spectral_info = self.spectral_data[spectral_key]
                time_hours = spectral_info['metadata']['time_hours']
                
                # Find matching concentration data
                concentrations = self.find_concentration_at_time(conc_data, time_hours)
                
                if concentrations is not None:
                    sample = {
                        'spectral_data': spectral_info['data'],
                        'concentrations': concentrations,
                        'voltage': int(voltage_level.replace('kv', '')),
                        'time_hours': time_hours,
                        'sample_id': f"{voltage_level}_{time_hours}h",
                        'filename': spectral_info['filename']
                    }
                    
                    unified_data['samples'].append(sample)
                    sample_count += 1
                    
                    print(f"  OK {spectral_key}: {len(spectral_info['data'])} spectral points")
                else:
                    print(f"  ERROR {spectral_key}: No matching concentration data")
        
        unified_data['metadata']['n_samples'] = sample_count
        if sample_count > 0:
            first_sample = unified_data['samples'][0]
            unified_data['metadata']['n_features'] = len(first_sample['spectral_data'])
        
        print(f"\nUnified dataset created:")
        print(f"  Total samples: {sample_count}")
        print(f"  Voltage levels: {len(set(s['voltage'] for s in unified_data['samples']))}")
        print(f"  Time points: {len(set(s['time_hours'] for s in unified_data['samples']))}")
        
        return unified_data
    
    def find_concentration_at_time(self, conc_data: Dict, target_time: int) -> Optional[Dict]:
        """Find concentration values at specific time point"""
        # Simple approach: find closest time point for each species
        concentrations = {}
        
        for species in self.species_names:
            if species in conc_data['species_data']:
                species_data = conc_data['species_data'][species]
                times = species_data['times']
                values = species_data['values']
                
                # Find closest time point
                if len(times) > 0:
                    closest_idx = np.argmin(np.abs(times - target_time))
                    concentrations[species] = values[closest_idx]
                else:
                    concentrations[species] = 0.0
            else:
                concentrations[species] = 0.0
        
        return concentrations if any(v > 0 for v in concentrations.values()) else None
    
    def save_processed_data(self, output_dir: str = "data/processed"):
        """Save processed data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving processed data to {output_path}")
        
        # Save spectral data summary
        spectral_summary = []
        for key, data in self.spectral_data.items():
            meta = data['metadata']
            spectral_summary.append({
                'sample_id': key,
                'voltage': meta['voltage'],
                'time_hours': meta['time_hours'],
                'filename': meta['filename'],
                'n_points': len(data['data']),
                'wavenumber_min': data['data']['wavenumber'].min(),
                'wavenumber_max': data['data']['wavenumber'].max()
            })
        
        pd.DataFrame(spectral_summary).to_csv(output_path / "spectral_summary.csv", index=False)
        
        # Save concentration data summary
        conc_summary = []
        for voltage_key, data in self.concentration_data.items():
            for species, species_data in data['species_data'].items():
                conc_summary.append({
                    'voltage': data['voltage'],
                    'species': species,
                    'mean_concentration': species_data['mean'],
                    'std_concentration': species_data['std'],
                    'min_concentration': species_data['range'][0],
                    'max_concentration': species_data['range'][1],
                    'n_timepoints': len(species_data['values'])
                })
        
        pd.DataFrame(conc_summary).to_csv(output_path / "concentration_summary.csv", index=False)
        
        print("OK Data summaries saved")

def main():
    """Main function to demonstrate data loading"""
    print("GCMS EXPERIMENTAL DATA LOADER")
    print("="*50)
    
    # Initialize loader
    loader = GCMSDataLoader()
    
    # Load data
    spectral_data = loader.load_spectral_files()
    concentration_data = loader.load_concentration_files()
    
    # Create unified dataset
    unified_data = loader.create_unified_dataset()
    
    # Save processed data
    loader.save_processed_data()
    
    print("\n" + "="*50)
    print("DATA LOADING COMPLETED")
    print("="*50)
    
    return loader, unified_data

if __name__ == "__main__":
    loader, unified_data = main()