# spectral_preprocessor.py
# Advanced spectral preprocessing for 6-species GCMS data

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, medfilt
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class SpectralPreprocessor:
    """
    Advanced spectral preprocessing for real GCMS experimental data
    Handles noise reduction, baseline correction, normalization, and feature engineering
    """
    
    def __init__(self, target_wavenumber_range: Tuple[float, float] = (500, 4000), 
                 target_resolution: int = 2048):
        self.target_range = target_wavenumber_range
        self.target_resolution = target_resolution
        self.reference_wavenumbers = None
        self.scalers = {}
        
        print(f"Spectral Preprocessor initialized")
        print(f"Target range: {target_wavenumber_range[0]}-{target_wavenumber_range[1]} cm-1")
        print(f"Target resolution: {target_resolution} points")
    
    def create_reference_grid(self, all_wavenumbers: List[np.ndarray]) -> np.ndarray:
        """Create a unified reference wavenumber grid"""
        print("\nCreating reference wavenumber grid...")
        
        # Find common range across all spectra
        min_wavenumber = max(arr.min() for arr in all_wavenumbers)
        max_wavenumber = min(arr.max() for arr in all_wavenumbers)
        
        # Adjust to target range if specified
        min_wavenumber = max(min_wavenumber, self.target_range[0])
        max_wavenumber = min(max_wavenumber, self.target_range[1])
        
        # Create uniform grid
        self.reference_wavenumbers = np.linspace(min_wavenumber, max_wavenumber, 
                                               self.target_resolution)
        
        print(f"Reference grid: {min_wavenumber:.1f} - {max_wavenumber:.1f} cm-1")
        print(f"Resolution: {self.target_resolution} points")
        print(f"Step size: {(max_wavenumber - min_wavenumber) / self.target_resolution:.2f} cm-1")
        
        return self.reference_wavenumbers
    
    def interpolate_spectrum(self, wavenumbers: np.ndarray, intensities: np.ndarray,
                           method: str = 'linear') -> np.ndarray:
        """Interpolate spectrum to reference grid"""
        if self.reference_wavenumbers is None:
            raise ValueError("Must create reference grid first")
        
        # Remove duplicates and sort
        unique_indices = np.unique(wavenumbers, return_index=True)[1]
        wavenumbers_clean = wavenumbers[unique_indices]
        intensities_clean = intensities[unique_indices]
        
        # Sort by wavenumber
        sort_indices = np.argsort(wavenumbers_clean)
        wavenumbers_sorted = wavenumbers_clean[sort_indices]
        intensities_sorted = intensities_clean[sort_indices]
        
        # Interpolate
        interpolator = interp1d(wavenumbers_sorted, intensities_sorted,
                              kind=method, bounds_error=False, fill_value=0)
        interpolated = interpolator(self.reference_wavenumbers)
        
        return interpolated
    
    def remove_baseline(self, spectrum: np.ndarray, method: str = 'polynomial', 
                       order: int = 3) -> np.ndarray:
        """Remove baseline from spectrum"""
        if method == 'polynomial':
            # Fit polynomial to lower envelope
            x = np.arange(len(spectrum))
            
            # Use lower percentile points for baseline estimation
            percentile = 10
            window_size = len(spectrum) // 20
            baseline_points = []
            baseline_indices = []
            
            for i in range(0, len(spectrum), window_size):
                window = spectrum[i:i+window_size]
                if len(window) > 0:
                    threshold = np.percentile(window, percentile)
                    local_indices = np.where(window <= threshold)[0]
                    if len(local_indices) > 0:
                        baseline_points.extend(window[local_indices])
                        baseline_indices.extend(i + local_indices)
            
            if len(baseline_points) > order + 1:
                # Fit polynomial to baseline points
                baseline_poly = np.polyfit(baseline_indices, baseline_points, order)
                baseline = np.polyval(baseline_poly, x)
                return spectrum - baseline
            
        # Fallback: simple linear baseline
        baseline = np.linspace(spectrum[0], spectrum[-1], len(spectrum))
        return spectrum - baseline
    
    def smooth_spectrum(self, spectrum: np.ndarray, method: str = 'savgol',
                       window_length: int = 11, polyorder: int = 3) -> np.ndarray:
        """Apply smoothing to spectrum"""
        if method == 'savgol':
            # Ensure window_length is odd and valid
            if window_length % 2 == 0:
                window_length += 1
            window_length = min(window_length, len(spectrum))
            if window_length < polyorder + 1:
                window_length = polyorder + 1
                if window_length % 2 == 0:
                    window_length += 1
            
            return savgol_filter(spectrum, window_length, polyorder)
        
        elif method == 'median':
            kernel_size = min(5, len(spectrum))
            if kernel_size % 2 == 0:
                kernel_size -= 1
            return medfilt(spectrum, kernel_size=kernel_size)
        
        elif method == 'moving_average':
            window_size = min(window_length, len(spectrum))
            return np.convolve(spectrum, np.ones(window_size)/window_size, mode='same')
        
        return spectrum
    
    def normalize_spectrum(self, spectrum: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize spectrum"""
        if method == 'minmax':
            min_val = spectrum.min()
            max_val = spectrum.max()
            if max_val > min_val:
                return (spectrum - min_val) / (max_val - min_val)
            
        elif method == 'zscore':
            mean_val = spectrum.mean()
            std_val = spectrum.std()
            if std_val > 0:
                return (spectrum - mean_val) / std_val
        
        elif method == 'l2':
            norm = np.linalg.norm(spectrum)
            if norm > 0:
                return spectrum / norm
        
        elif method == 'area':
            area = np.trapz(spectrum)
            if area > 0:
                return spectrum / area
        
        return spectrum
    
    def extract_features(self, spectrum: np.ndarray) -> Dict:
        """Extract statistical and spectral features"""
        features = {
            # Basic statistics
            'mean': np.mean(spectrum),
            'std': np.std(spectrum),
            'max': np.max(spectrum),
            'min': np.min(spectrum),
            'range': np.max(spectrum) - np.min(spectrum),
            'median': np.median(spectrum),
            'skewness': self._calculate_skewness(spectrum),
            'kurtosis': self._calculate_kurtosis(spectrum),
            
            # Spectral features
            'total_intensity': np.sum(spectrum),
            'peak_count': self._count_peaks(spectrum),
            'peak_intensity_ratio': self._peak_intensity_ratio(spectrum),
            'spectral_centroid': self._spectral_centroid(spectrum),
            'spectral_bandwidth': self._spectral_bandwidth(spectrum),
            'spectral_rolloff': self._spectral_rolloff(spectrum),
        }
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _count_peaks(self, spectrum: np.ndarray, prominence: float = 0.1) -> int:
        """Count significant peaks"""
        from scipy.signal import find_peaks
        threshold = prominence * (spectrum.max() - spectrum.min()) + spectrum.min()
        peaks, _ = find_peaks(spectrum, height=threshold)
        return len(peaks)
    
    def _peak_intensity_ratio(self, spectrum: np.ndarray) -> float:
        """Ratio of peak intensity to mean intensity"""
        return spectrum.max() / (spectrum.mean() + 1e-10)
    
    def _spectral_centroid(self, spectrum: np.ndarray) -> float:
        """Calculate spectral centroid"""
        if self.reference_wavenumbers is not None:
            weighted_sum = np.sum(self.reference_wavenumbers * spectrum)
            total_intensity = np.sum(spectrum)
            return weighted_sum / (total_intensity + 1e-10)
        return 0
    
    def _spectral_bandwidth(self, spectrum: np.ndarray) -> float:
        """Calculate spectral bandwidth"""
        if self.reference_wavenumbers is not None:
            centroid = self._spectral_centroid(spectrum)
            weighted_deviation = np.sum(spectrum * (self.reference_wavenumbers - centroid) ** 2)
            total_intensity = np.sum(spectrum)
            return np.sqrt(weighted_deviation / (total_intensity + 1e-10))
        return 0
    
    def _spectral_rolloff(self, spectrum: np.ndarray, rolloff_point: float = 0.95) -> float:
        """Calculate spectral rolloff"""
        if self.reference_wavenumbers is not None:
            total_energy = np.sum(spectrum)
            threshold = rolloff_point * total_energy
            
            cumulative_energy = np.cumsum(spectrum)
            rolloff_index = np.where(cumulative_energy >= threshold)[0]
            
            if len(rolloff_index) > 0:
                return self.reference_wavenumbers[rolloff_index[0]]
        return 0
    
    def process_spectrum(self, wavenumbers: np.ndarray, intensities: np.ndarray,
                        baseline_removal: bool = True,
                        smoothing: bool = True,
                        normalization: str = 'minmax',
                        extract_features: bool = False) -> Dict:
        """Complete preprocessing pipeline for a single spectrum"""
        
        # 1. Interpolate to reference grid
        interpolated = self.interpolate_spectrum(wavenumbers, intensities)
        
        # 2. Remove baseline
        if baseline_removal:
            interpolated = self.remove_baseline(interpolated)
        
        # 3. Smooth spectrum
        if smoothing:
            interpolated = self.smooth_spectrum(interpolated)
        
        # 4. Normalize
        if normalization:
            interpolated = self.normalize_spectrum(interpolated, method=normalization)
        
        result = {
            'processed_spectrum': interpolated,
            'wavenumbers': self.reference_wavenumbers
        }
        
        # 5. Extract features if requested
        if extract_features:
            result['features'] = self.extract_features(interpolated)
        
        return result
    
    def process_dataset(self, dataset: Dict, **processing_options) -> Dict:
        """Process entire dataset of spectra"""
        print("\n" + "="*60)
        print("SPECTRAL PREPROCESSING PIPELINE")
        print("="*60)
        
        samples = dataset['samples']
        
        # Create reference grid from all spectra
        all_wavenumbers = []
        for sample in samples:
            wavenumbers = sample['spectral_data']['wavenumber'].values
            all_wavenumbers.append(wavenumbers)
        
        self.create_reference_grid(all_wavenumbers)
        
        # Process each spectrum
        processed_samples = []
        features_matrix = []
        
        print(f"\nProcessing {len(samples)} spectra...")
        
        for i, sample in enumerate(samples):
            wavenumbers = sample['spectral_data']['wavenumber'].values
            intensities = sample['spectral_data']['intensity'].values
            
            # Process spectrum
            processed = self.process_spectrum(wavenumbers, intensities, 
                                           extract_features=True, **processing_options)
            
            # Create enhanced sample
            enhanced_sample = sample.copy()
            enhanced_sample['processed_spectrum'] = processed['processed_spectrum']
            enhanced_sample['spectral_features'] = processed['features']
            
            processed_samples.append(enhanced_sample)
            
            # Collect features for feature matrix
            feature_vector = list(processed['features'].values())
            features_matrix.append(feature_vector)
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(samples)} spectra")
        
        # Create processed dataset
        processed_dataset = {
            'samples': processed_samples,
            'reference_wavenumbers': self.reference_wavenumbers,
            'features_matrix': np.array(features_matrix),
            'feature_names': list(processed['features'].keys()),
            'processing_params': processing_options,
            'metadata': dataset.get('metadata', {})
        }
        
        processed_dataset['metadata']['n_features_spectral'] = len(self.reference_wavenumbers)
        processed_dataset['metadata']['n_features_extracted'] = len(features_matrix[0]) if features_matrix else 0
        
        print(f"\nPreprocessing completed:")
        print(f"  Processed spectra: {len(processed_samples)}")
        print(f"  Spectral features: {len(self.reference_wavenumbers)}")
        print(f"  Extracted features: {len(features_matrix[0]) if features_matrix else 0}")
        
        return processed_dataset
    
    def save_processed_data(self, processed_dataset: Dict, output_dir: str = "data/processed"):
        """Save processed dataset"""
        from pathlib import Path
        import joblib
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving processed data to {output_path}")
        
        # Save spectral matrix
        spectral_matrix = np.array([s['processed_spectrum'] for s in processed_dataset['samples']])
        np.save(output_path / "spectral_matrix.npy", spectral_matrix)
        
        # Save features matrix
        np.save(output_path / "features_matrix.npy", processed_dataset['features_matrix'])
        
        # Save reference wavenumbers
        np.save(output_path / "reference_wavenumbers.npy", processed_dataset['reference_wavenumbers'])
        
        # Save sample metadata
        metadata_list = []
        for sample in processed_dataset['samples']:
            metadata = {
                'sample_id': sample['sample_id'],
                'voltage': sample['voltage'],
                'time_hours': sample['time_hours'],
                'filename': sample['filename']
            }
            # Add concentrations
            for species, conc in sample['concentrations'].items():
                metadata[f'{species}_concentration'] = conc
            
            metadata_list.append(metadata)
        
        pd.DataFrame(metadata_list).to_csv(output_path / "sample_metadata.csv", index=False)
        
        # Save feature names
        pd.DataFrame({'feature_name': processed_dataset['feature_names']}).to_csv(
            output_path / "feature_names.csv", index=False)
        
        # Save preprocessor
        joblib.dump(self, output_path / "spectral_preprocessor.pkl")
        
        print("OK Processed data saved successfully")

def main():
    """Demonstration of spectral preprocessing"""
    print("SPECTRAL PREPROCESSING DEMONSTRATION")
    print("="*50)
    
    # This would typically be called after loading data
    preprocessor = SpectralPreprocessor()
    
    print("Preprocessor ready for dataset processing")
    print("Use process_dataset() method with loaded GCMS data")

if __name__ == "__main__":
    main()