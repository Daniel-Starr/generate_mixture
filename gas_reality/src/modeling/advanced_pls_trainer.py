# advanced_pls_trainer.py
# Advanced PLS regression trainer for 6-species gas concentration prediction

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, 
                                   GridSearchCV, train_test_split)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class AdvancedPLSTrainer:
    """
    Advanced PLS regression trainer for 6-species gas concentration prediction
    Features: Multi-target PLS, advanced cross-validation, ensemble methods
    """
    
    def __init__(self, species_names: List[str] = None):
        if species_names is None:
            self.species_names = ['SOF2', 'SO2F2', 'SO2', 'NO', 'NO2', 'NF3']
        else:
            self.species_names = species_names
        
        self.n_species = len(self.species_names)
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        
        print(f"Advanced PLS Trainer initialized")
        print(f"Target species: {', '.join(self.species_names)}")
        print(f"Number of species: {self.n_species}")
    
    def prepare_training_data(self, processed_dataset: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Prepare training data from processed dataset"""
        print("\n" + "="*60)
        print("PREPARING TRAINING DATA")
        print("="*60)
        
        samples = processed_dataset['samples']
        
        # Extract spectral features
        X_spectral = np.array([s['processed_spectrum'] for s in samples])
        
        # Extract additional features if available
        if 'features_matrix' in processed_dataset:
            X_features = processed_dataset['features_matrix']
            X_combined = np.hstack([X_spectral, X_features])
            feature_types = ['spectral'] * X_spectral.shape[1] + ['extracted'] * X_features.shape[1]
        else:
            X_combined = X_spectral
            feature_types = ['spectral'] * X_spectral.shape[1]
        
        # Extract concentrations
        Y = np.zeros((len(samples), self.n_species))
        voltage_data = []
        time_data = []
        
        for i, sample in enumerate(samples):
            concentrations = sample['concentrations']
            for j, species in enumerate(self.species_names):
                Y[i, j] = concentrations.get(species, 0.0)
            
            voltage_data.append(sample['voltage'])
            time_data.append(sample['time_hours'])
        
        # Create metadata
        metadata = {
            'sample_ids': [s['sample_id'] for s in samples],
            'voltages': np.array(voltage_data),
            'times': np.array(time_data),
            'feature_types': feature_types,
            'n_spectral_features': X_spectral.shape[1],
            'n_extracted_features': X_features.shape[1] if 'features_matrix' in processed_dataset else 0
        }
        
        print(f"Training data prepared:")
        print(f"  Samples: {X_combined.shape[0]}")
        print(f"  Spectral features: {X_spectral.shape[1]}")
        print(f"  Extracted features: {metadata['n_extracted_features']}")
        print(f"  Total features: {X_combined.shape[1]}")
        print(f"  Target species: {self.n_species}")
        
        # Data quality checks
        print(f"\nData quality checks:")
        print(f"  X shape: {X_combined.shape}")
        print(f"  Y shape: {Y.shape}")
        print(f"  X range: [{X_combined.min():.3f}, {X_combined.max():.3f}]")
        print(f"  Y range: [{Y.min():.3f}, {Y.max():.3f}]")
        print(f"  Missing values in X: {np.isnan(X_combined).sum()}")
        print(f"  Missing values in Y: {np.isnan(Y).sum()}")
        
        return X_combined, Y, metadata
    
    def create_data_splits(self, X: np.ndarray, Y: np.ndarray, metadata: Dict,
                          test_size: float = 0.2, val_size: float = 0.1,
                          split_strategy: str = 'voltage_stratified') -> Dict:
        """Create train/validation/test splits with different strategies"""
        print(f"\nCreating data splits (strategy: {split_strategy})...")
        
        if split_strategy == 'voltage_stratified':
            # Stratify by voltage level to ensure representation
            voltage_labels = metadata['voltages']
            
            # First split: train+val vs test
            X_trainval, X_test, Y_trainval, Y_test, idx_trainval, idx_test = train_test_split(
                X, Y, np.arange(len(X)), test_size=test_size, 
                stratify=voltage_labels, random_state=42)
            
            # Second split: train vs val
            voltage_trainval = voltage_labels[idx_trainval]
            val_size_adjusted = val_size / (1 - test_size)
            
            X_train, X_val, Y_train, Y_val, idx_train_rel, idx_val_rel = train_test_split(
                X_trainval, Y_trainval, np.arange(len(X_trainval)),
                test_size=val_size_adjusted, stratify=voltage_trainval, random_state=42)
            
        elif split_strategy == 'time_based':
            # Split based on time - early times for training, later for testing
            time_threshold_test = np.percentile(metadata['times'], 100 * (1 - test_size))
            time_threshold_val = np.percentile(metadata['times'], 100 * (1 - test_size - val_size))
            
            test_mask = metadata['times'] >= time_threshold_test
            val_mask = (metadata['times'] >= time_threshold_val) & (metadata['times'] < time_threshold_test)
            train_mask = metadata['times'] < time_threshold_val
            
            X_train, Y_train = X[train_mask], Y[train_mask]
            X_val, Y_val = X[val_mask], Y[val_mask]
            X_test, Y_test = X[test_mask], Y[test_mask]
            
        else:  # random split
            X_trainval, X_test, Y_trainval, Y_test = train_test_split(
                X, Y, test_size=test_size, random_state=42)
            
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_trainval, Y_trainval, test_size=val_size_adjusted, random_state=42)
        
        splits = {
            'X_train': X_train, 'Y_train': Y_train,
            'X_val': X_val, 'Y_val': Y_val,
            'X_test': X_test, 'Y_test': Y_test
        }
        
        print(f"Data splits created:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        return splits
    
    def optimize_pls_components(self, X_train: np.ndarray, Y_train: np.ndarray,
                               X_val: np.ndarray, Y_val: np.ndarray,
                               max_components: int = 20) -> int:
        """Optimize number of PLS components using validation set"""
        print("\nOptimizing PLS components...")
        
        max_components = min(max_components, min(X_train.shape[1], X_train.shape[0] - 1))
        component_range = range(1, max_components + 1)
        
        val_scores = []
        train_scores = []
        
        for n_comp in component_range:
            # Create and train model
            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train)
            Y_train_scaled = scaler_Y.fit_transform(Y_train)
            
            model = PLSRegression(n_components=n_comp)
            model.fit(X_train_scaled, Y_train_scaled)
            
            # Evaluate on training set
            Y_train_pred_scaled = model.predict(X_train_scaled)
            Y_train_pred = scaler_Y.inverse_transform(Y_train_pred_scaled)
            train_r2 = r2_score(Y_train, Y_train_pred, multioutput='uniform_average')
            train_scores.append(train_r2)
            
            # Evaluate on validation set
            X_val_scaled = scaler_X.transform(X_val)
            Y_val_pred_scaled = model.predict(X_val_scaled)
            Y_val_pred = scaler_Y.inverse_transform(Y_val_pred_scaled)
            val_r2 = r2_score(Y_val, Y_val_pred, multioutput='uniform_average')
            val_scores.append(val_r2)
            
            if n_comp % 5 == 0 or n_comp <= 5:
                print(f"  Components {n_comp:2d}: Train R2 = {train_r2:.4f}, Val R2 = {val_r2:.4f}")
        
        # Find optimal components (highest validation score)
        best_idx = np.argmax(val_scores)
        best_components = component_range[best_idx]
        best_val_score = val_scores[best_idx]
        
        print(f"\nOptimal components: {best_components} (Val R2 = {best_val_score:.4f})")
        
        return best_components
    
    def train_individual_models(self, X_train: np.ndarray, Y_train: np.ndarray,
                               n_components: int) -> Dict:
        """Train individual PLS models for each species"""
        print(f"\nTraining individual PLS models (components: {n_components})...")
        
        individual_models = {}
        
        for i, species in enumerate(self.species_names):
            print(f"  Training model for {species}...")
            
            # Extract target for this species
            y_species = Y_train[:, i]
            
            # Create scalers
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            # Scale data
            X_scaled = scaler_X.fit_transform(X_train)
            y_scaled = scaler_y.fit_transform(y_species.reshape(-1, 1)).ravel()
            
            # Train model
            model = PLSRegression(n_components=n_components)
            model.fit(X_scaled, y_scaled)
            
            # Store model and scalers
            individual_models[species] = {
                'model': model,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y
            }
        
        return individual_models
    
    def train_multi_target_model(self, X_train: np.ndarray, Y_train: np.ndarray,
                                n_components: int) -> Dict:
        """Train multi-target PLS model for all species simultaneously"""
        print(f"\nTraining multi-target PLS model (components: {n_components})...")
        
        # Create scalers
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        
        # Scale data
        X_scaled = scaler_X.fit_transform(X_train)
        Y_scaled = scaler_Y.fit_transform(Y_train)
        
        # Train model
        model = PLSRegression(n_components=n_components)
        model.fit(X_scaled, Y_scaled)
        
        multi_target_model = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_Y': scaler_Y
        }
        
        return multi_target_model
    
    def evaluate_models(self, models: Dict, X_test: np.ndarray, Y_test: np.ndarray,
                       model_type: str) -> Dict:
        """Evaluate model performance"""
        print(f"\nEvaluating {model_type} models...")
        
        metrics = {
            'model_type': model_type,
            'species_metrics': {},
            'overall_metrics': {}
        }
        
        if model_type == 'individual':
            # Evaluate individual models
            Y_pred = np.zeros_like(Y_test)
            
            for i, species in enumerate(self.species_names):
                model_info = models[species]
                
                # Scale input
                X_scaled = model_info['scaler_X'].transform(X_test)
                
                # Predict
                y_pred_scaled = model_info['model'].predict(X_scaled)
                y_pred = model_info['scaler_y'].inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                
                Y_pred[:, i] = y_pred
                
                # Calculate metrics for this species
                y_true = Y_test[:, i]
                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                
                metrics['species_metrics'][species] = {
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                }
                
                print(f"  {species}: R2 = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        elif model_type == 'multi_target':
            # Evaluate multi-target model
            X_scaled = models['scaler_X'].transform(X_test)
            Y_pred_scaled = models['model'].predict(X_scaled)
            Y_pred = models['scaler_Y'].inverse_transform(Y_pred_scaled)
            
            # Calculate metrics for each species
            for i, species in enumerate(self.species_names):
                y_true = Y_test[:, i]
                y_pred = Y_pred[:, i]
                
                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                
                metrics['species_metrics'][species] = {
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                }
                
                print(f"  {species}: R2 = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        # Calculate overall metrics
        if Y_pred.shape == Y_test.shape:
            overall_r2 = r2_score(Y_test, Y_pred, multioutput='uniform_average')
            overall_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred, multioutput='uniform_average'))
            overall_mae = mean_absolute_error(Y_test, Y_pred, multioutput='uniform_average')
            
            metrics['overall_metrics'] = {
                'r2': overall_r2,
                'rmse': overall_rmse,
                'mae': overall_mae
            }
            
            print(f"\nOverall performance:")
            print(f"  R2 = {overall_r2:.4f}")
            print(f"  RMSE = {overall_rmse:.4f}")
            print(f"  MAE = {overall_mae:.4f}")
        
        return metrics
    
    def train_complete_pipeline(self, processed_dataset: Dict, **kwargs) -> Dict:
        """Complete training pipeline"""
        print("\n" + "="*60)
        print("ADVANCED PLS TRAINING PIPELINE")
        print("="*60)
        
        # Prepare data
        X, Y, metadata = self.prepare_training_data(processed_dataset)
        
        # Create splits
        splits = self.create_data_splits(X, Y, metadata, **kwargs)
        
        # Optimize components
        best_components = self.optimize_pls_components(
            splits['X_train'], splits['Y_train'],
            splits['X_val'], splits['Y_val']
        )
        
        # Train models
        individual_models = self.train_individual_models(
            splits['X_train'], splits['Y_train'], best_components)
        
        multi_target_model = self.train_multi_target_model(
            splits['X_train'], splits['Y_train'], best_components)
        
        # Evaluate models
        individual_metrics = self.evaluate_models(
            individual_models, splits['X_test'], splits['Y_test'], 'individual')
        
        multi_target_metrics = self.evaluate_models(
            multi_target_model, splits['X_test'], splits['Y_test'], 'multi_target')
        
        # Store results
        self.models = {
            'individual': individual_models,
            'multi_target': multi_target_model,
            'best_components': best_components
        }
        
        self.performance_metrics = {
            'individual': individual_metrics,
            'multi_target': multi_target_metrics
        }
        
        results = {
            'models': self.models,
            'metrics': self.performance_metrics,
            'splits': splits,
            'metadata': metadata,
            'best_components': best_components
        }
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED")
        print("="*60)
        
        return results
    
    def save_models(self, output_dir: str = "data/models"):
        """Save trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving models to {output_path}")
        
        # Save individual models
        individual_dir = output_path / "individual"
        individual_dir.mkdir(exist_ok=True)
        
        for species, model_info in self.models['individual'].items():
            joblib.dump(model_info, individual_dir / f"{species}_pls_model.pkl")
        
        # Save multi-target model
        joblib.dump(self.models['multi_target'], output_path / "multi_target_pls_model.pkl")
        
        # Save performance metrics
        joblib.dump(self.performance_metrics, output_path / "performance_metrics.pkl")
        
        # Save trainer object
        joblib.dump(self, output_path / "pls_trainer.pkl")
        
        print("OK Models saved successfully")

def main():
    """Demonstration of advanced PLS training"""
    print("ADVANCED PLS TRAINER DEMONSTRATION")
    print("="*50)
    
    trainer = AdvancedPLSTrainer()
    
    print("Trainer ready for processed dataset")
    print("Use train_complete_pipeline() method with processed GCMS data")

if __name__ == "__main__":
    main()