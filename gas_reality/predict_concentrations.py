# predict_concentrations.py
# Real-time prediction tool for gas concentrations using trained PLS models

import sys
from pathlib import Path

# Add source directory to path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
import joblib
import os
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class GasConcentrationPredictor:
    """
    Real-time gas concentration predictor using trained PLS models
    Supports both individual and multi-target model predictions
    """
    
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = Path(models_dir)
        self.species_names = ['SOF2', 'SO2F2', 'SO2', 'NO', 'NO2', 'NF3']
        
        # Model storage
        self.individual_models = {}
        self.multi_target_model = None
        self.preprocessor = None
        self.reference_wavenumbers = None
        
        print(f"Gas Concentration Predictor initialized")
        print(f"Models directory: {models_dir}")
        print(f"Target species: {', '.join(self.species_names)}")
    
    def load_models(self):
        """Load all available trained models"""
        print("\nLoading trained models...")
        
        # Load preprocessor
        preprocessor_path = self.models_dir.parent / "processed" / "spectral_preprocessor.pkl"
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
            self.reference_wavenumbers = self.preprocessor.reference_wavenumbers
            print(f"OK Preprocessor loaded")
        else:
            print(f"ERROR Preprocessor not found at {preprocessor_path}")
        
        # Load reference wavenumbers if preprocessor not available
        if self.reference_wavenumbers is None:
            ref_path = self.models_dir.parent / "processed" / "reference_wavenumbers.npy"
            if ref_path.exists():
                self.reference_wavenumbers = np.load(ref_path)
                print(f"OK Reference wavenumbers loaded")
        
        # Load individual models
        individual_dir = self.models_dir / "individual"
        if individual_dir.exists():
            for species in self.species_names:
                model_path = individual_dir / f"{species}_pls_model.pkl"
                if model_path.exists():
                    self.individual_models[species] = joblib.load(model_path)
                    print(f"OK Individual model loaded: {species}")
                else:
                    print(f"ERROR Individual model not found: {species}")
        
        # Load multi-target model
        multi_target_path = self.models_dir / "multi_target_pls_model.pkl"
        if multi_target_path.exists():
            self.multi_target_model = joblib.load(multi_target_path)
            print(f"OK Multi-target model loaded")
        else:
            print(f"ERROR Multi-target model not found")
        
        # Check what's available
        available_individual = len(self.individual_models)
        available_multi = self.multi_target_model is not None
        
        print(f"\nModels loaded:")
        print(f"  Individual models: {available_individual}/{len(self.species_names)}")
        print(f"  Multi-target model: {'OK' if available_multi else 'ERROR'}")
        print(f"  Preprocessor: {'OK' if self.preprocessor else 'ERROR'}")
        
        if available_individual == 0 and not available_multi:
            raise ValueError("No trained models found. Please train models first.")
    
    def preprocess_spectrum(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Preprocess raw spectrum for prediction"""
        if self.preprocessor is not None:
            # Use trained preprocessor
            result = self.preprocessor.process_spectrum(wavenumbers, intensities)
            return result['processed_spectrum']
        
        elif self.reference_wavenumbers is not None:
            # Basic preprocessing without trained preprocessor
            print("Warning: Using basic preprocessing (no trained preprocessor)")
            
            # Interpolate to reference grid
            interpolator = interp1d(wavenumbers, intensities, 
                                  kind='linear', bounds_error=False, fill_value=0)
            interpolated = interpolator(self.reference_wavenumbers)
            
            # Basic normalization
            interpolated = (interpolated - interpolated.min()) / (interpolated.max() - interpolated.min() + 1e-10)
            
            return interpolated
        
        else:
            raise ValueError("No preprocessing method available")
    
    def predict_with_individual_models(self, processed_spectrum: np.ndarray) -> Dict:
        """Predict concentrations using individual models"""
        if not self.individual_models:
            return {}
        
        predictions = {}
        
        for species, model_info in self.individual_models.items():
            try:
                # Prepare input
                X = processed_spectrum.reshape(1, -1)
                
                # Adjust dimensions if needed
                expected_features = model_info['model'].n_features_in_
                if X.shape[1] != expected_features:
                    if X.shape[1] > expected_features:
                        X = X[:, :expected_features]
                    else:
                        padding = np.zeros((1, expected_features - X.shape[1]))
                        X = np.hstack([X, padding])
                
                # Scale and predict
                X_scaled = model_info['scaler_X'].transform(X)
                y_pred_scaled = model_info['model'].predict(X_scaled)
                y_pred = model_info['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
                
                # Ensure positive
                y_pred = max(0, y_pred)
                predictions[species] = y_pred
                
            except Exception as e:
                print(f"Warning: Failed to predict {species} with individual model: {e}")
                predictions[species] = 0.0
        
        return predictions
    
    def predict_with_multi_target_model(self, processed_spectrum: np.ndarray) -> Dict:
        """Predict concentrations using multi-target model"""
        if self.multi_target_model is None:
            return {}
        
        try:
            # Prepare input
            X = processed_spectrum.reshape(1, -1)
            
            # Adjust dimensions if needed
            expected_features = self.multi_target_model['model'].n_features_in_
            if X.shape[1] != expected_features:
                if X.shape[1] > expected_features:
                    X = X[:, :expected_features]
                else:
                    padding = np.zeros((1, expected_features - X.shape[1]))
                    X = np.hstack([X, padding])
            
            # Scale and predict
            X_scaled = self.multi_target_model['scaler_X'].transform(X)
            Y_pred_scaled = self.multi_target_model['model'].predict(X_scaled)
            Y_pred = self.multi_target_model['scaler_Y'].inverse_transform(Y_pred_scaled)[0]
            
            # Ensure positive
            Y_pred = np.maximum(Y_pred, 0)
            
            # Create predictions dictionary
            predictions = {}
            for i, species in enumerate(self.species_names):
                predictions[species] = Y_pred[i]
            
            return predictions
            
        except Exception as e:
            print(f"Warning: Failed to predict with multi-target model: {e}")
            return {}
    
    def predict_from_file(self, file_path: str) -> Dict:
        """Predict concentrations from spectrum file"""
        print(f"\nPredicting concentrations from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load spectrum data
        df = pd.read_csv(file_path)
        
        # Extract wavenumbers and intensities
        if 'wavenumber' in df.columns and 'intensity' in df.columns:
            wavenumbers = df['wavenumber'].values
            intensities = df['intensity'].values
        elif len(df.columns) >= 2:
            wavenumbers = df.iloc[:, 0].values
            intensities = df.iloc[:, 1].values
        else:
            raise ValueError("Cannot identify wavenumber and intensity columns")
        
        print(f"  Loaded spectrum: {len(wavenumbers)} points")
        print(f"  Wavenumber range: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm-1")
        
        # Preprocess spectrum
        processed_spectrum = self.preprocess_spectrum(wavenumbers, intensities)
        
        # Make predictions
        predictions = self.predict_concentrations(processed_spectrum)
        
        return predictions
    
    def predict_concentrations(self, processed_spectrum: np.ndarray) -> Dict:
        """Predict concentrations using all available models"""
        results = {}
        
        # Individual model predictions
        individual_preds = self.predict_with_individual_models(processed_spectrum)
        if individual_preds:
            results['individual'] = individual_preds
        
        # Multi-target model predictions
        multi_target_preds = self.predict_with_multi_target_model(processed_spectrum)
        if multi_target_preds:
            results['multi_target'] = multi_target_preds
        
        # Ensemble prediction (average of available models)
        if len(results) > 1:
            ensemble_preds = {}
            for species in self.species_names:
                values = []
                if 'individual' in results and species in results['individual']:
                    values.append(results['individual'][species])
                if 'multi_target' in results and species in results['multi_target']:
                    values.append(results['multi_target'][species])
                
                if values:
                    ensemble_preds[species] = np.mean(values)
                else:
                    ensemble_preds[species] = 0.0
            
            results['ensemble'] = ensemble_preds
        
        return results
    
    def display_predictions(self, predictions: Dict, normalize: bool = True):
        """Display prediction results in a formatted way"""
        print("\n" + "="*60)
        print("GAS CONCENTRATION PREDICTIONS")
        print("="*60)
        
        for model_type, pred_dict in predictions.items():
            print(f"\n{model_type.upper()} Model Predictions:")
            print("-" * 40)
            
            if normalize:
                # Normalize to percentages
                total = sum(pred_dict.values())
                if total > 0:
                    for species in self.species_names:
                        conc = pred_dict.get(species, 0)
                        percentage = (conc / total) * 100
                        print(f"  {species:6s}: {conc:.4f} ({percentage:5.1f}%)")
                else:
                    for species in self.species_names:
                        print(f"  {species:6s}: 0.0000 (  0.0%)")
            else:
                # Show raw concentrations
                for species in self.species_names:
                    conc = pred_dict.get(species, 0)
                    print(f"  {species:6s}: {conc:.4f}")
    
    def save_predictions(self, predictions: Dict, output_file: str, input_file: str = None):
        """Save predictions to CSV file"""
        # Prepare data for saving
        result_data = []
        
        for model_type, pred_dict in predictions.items():
            row = {'Model': model_type}
            
            # Add raw concentrations
            for species in self.species_names:
                row[f'{species}_concentration'] = pred_dict.get(species, 0)
            
            # Add percentages
            total = sum(pred_dict.values())
            if total > 0:
                for species in self.species_names:
                    percentage = (pred_dict.get(species, 0) / total) * 100
                    row[f'{species}_percentage'] = percentage
            else:
                for species in self.species_names:
                    row[f'{species}_percentage'] = 0.0
            
            if input_file:
                row['Input_file'] = input_file
            
            result_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(result_data)
        df.to_csv(output_file, index=False)
        print(f"\nOK Predictions saved to: {output_file}")

def quick_predict(file_path: str, models_dir: str = "data/models") -> Dict:
    """Quick prediction function for single file"""
    predictor = GasConcentrationPredictor(models_dir)
    predictor.load_models()
    predictions = predictor.predict_from_file(file_path)
    predictor.display_predictions(predictions)
    return predictions

def main():
    """Main function for command-line usage"""
    print("GAS CONCENTRATION PREDICTION TOOL")
    print("="*50)
    
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        # Initialize predictor
        predictor = GasConcentrationPredictor()
        predictor.load_models()
        
        # Make predictions
        predictions = predictor.predict_from_file(file_path)
        predictor.display_predictions(predictions)
        
        # Save results
        output_file = f"predictions_{Path(file_path).stem}.csv"
        predictor.save_predictions(predictions, output_file, file_path)
        
    else:
        print("Usage: python predict_concentrations.py <spectrum_file.csv>")
        print("\nExample files to try:")
        test_dir = Path("E:/generate_mixture/gcms")
        if test_dir.exists():
            csv_files = list(test_dir.glob("*.CSV"))[:3]
            for f in csv_files:
                print(f"  python predict_concentrations.py {f}")

if __name__ == "__main__":
    main()