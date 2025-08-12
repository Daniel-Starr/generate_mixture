# main_pipeline.py
# Complete pipeline for GCMS real experimental data analysis

import sys
from pathlib import Path

# Add source directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing.gcms_data_loader import GCMSDataLoader
from preprocessing.spectral_preprocessor import SpectralPreprocessor
from modeling.advanced_pls_trainer import AdvancedPLSTrainer

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GCMSAnalysisPipeline:
    """
    Complete pipeline for real GCMS experimental data analysis
    Integrates data loading, preprocessing, and advanced PLS modeling
    """
    
    def __init__(self, gcms_path: str = "E:/generate_mixture/gcms", 
                 output_dir: str = "data"):
        self.gcms_path = gcms_path
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.data_loader = GCMSDataLoader(gcms_path)
        self.preprocessor = SpectralPreprocessor()
        self.trainer = AdvancedPLSTrainer()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.trained_models = None
        
        print(f"GCMS Analysis Pipeline initialized")
        print(f"GCMS data path: {gcms_path}")
        print(f"Output directory: {output_dir}")
        print(f"Timestamp: {self.timestamp}")
    
    def run_complete_pipeline(self, save_intermediate: bool = True, **kwargs):
        """Run the complete analysis pipeline"""
        print("\n" + "="*80)
        print("GCMS REAL EXPERIMENTAL DATA ANALYSIS PIPELINE")
        print("="*80)
        print(f"Pipeline started at: {datetime.now()}")
        
        try:
            # Step 1: Load raw data
            print(f"\n{'='*20} STEP 1: DATA LOADING {'='*20}")
            self.load_data()
            
            if save_intermediate:
                self.save_raw_data()
            
            # Step 2: Preprocess data
            print(f"\n{'='*20} STEP 2: PREPROCESSING {'='*20}")
            self.preprocess_data(**kwargs)
            
            if save_intermediate:
                self.save_processed_data()
            
            # Step 3: Train models
            print(f"\n{'='*20} STEP 3: MODEL TRAINING {'='*20}")
            self.train_models(**kwargs)
            
            # Step 4: Save final results
            print(f"\n{'='*20} STEP 4: SAVING RESULTS {'='*20}")
            self.save_final_results()
            
            # Step 5: Generate report
            print(f"\n{'='*20} STEP 5: GENERATING REPORT {'='*20}")
            self.generate_analysis_report()
            
            print(f"\n{'='*80}")
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print(f"Results saved to: {self.output_dir}")
            print(f"Pipeline finished at: {datetime.now()}")
            print("="*80)
            
        except Exception as e:
            print(f"\nPIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load_data(self):
        """Load GCMS spectral and concentration data"""
        # Load spectral files
        spectral_data = self.data_loader.load_spectral_files()
        
        # Load concentration files
        concentration_data = self.data_loader.load_concentration_files()
        
        # Create unified dataset
        unified_data = self.data_loader.create_unified_dataset()
        
        self.raw_data = {
            'unified_dataset': unified_data,
            'spectral_data': spectral_data,
            'concentration_data': concentration_data,
            'metadata': {
                'n_samples': len(unified_data['samples']),
                'species': self.trainer.species_names,
                'voltage_levels': list(set(s['voltage'] for s in unified_data['samples'])),
                'time_range': (
                    min(s['time_hours'] for s in unified_data['samples']),
                    max(s['time_hours'] for s in unified_data['samples'])
                )
            }
        }
        
        print(f"Raw data loaded:")
        print(f"  Total samples: {self.raw_data['metadata']['n_samples']}")
        print(f"  Voltage levels: {self.raw_data['metadata']['voltage_levels']}")
        print(f"  Time range: {self.raw_data['metadata']['time_range']} hours")
    
    def preprocess_data(self, **preprocessing_options):
        """Preprocess spectral data"""
        # Set default preprocessing options
        default_options = {
            'baseline_removal': True,
            'smoothing': True,
            'normalization': 'minmax'
        }
        
        # Filter out non-preprocessing options
        preprocessing_keys = ['baseline_removal', 'smoothing', 'normalization', 
                            'extract_features', 'target_wavenumber_range', 'target_resolution']
        filtered_options = {k: v for k, v in preprocessing_options.items() 
                          if k in preprocessing_keys}
        default_options.update(filtered_options)
        
        # Process the unified dataset
        self.processed_data = self.preprocessor.process_dataset(
            self.raw_data['unified_dataset'], **default_options)
        
        print(f"Data preprocessing completed:")
        print(f"  Processed samples: {len(self.processed_data['samples'])}")
        print(f"  Spectral features: {self.processed_data['metadata']['n_features_spectral']}")
        print(f"  Extracted features: {self.processed_data['metadata']['n_features_extracted']}")
    
    def train_models(self, **training_options):
        """Train PLS models"""
        # Set default training options
        default_options = {
            'test_size': 0.2,
            'val_size': 0.1,
            'split_strategy': 'voltage_stratified'
        }
        
        # Filter out non-training options
        training_keys = ['test_size', 'val_size', 'split_strategy', 'max_components']
        filtered_options = {k: v for k, v in training_options.items() 
                          if k in training_keys}
        default_options.update(filtered_options)
        
        # Train complete pipeline
        self.trained_models = self.trainer.train_complete_pipeline(
            self.processed_data, **default_options)
        
        print(f"Model training completed:")
        print(f"  Best components: {self.trained_models['best_components']}")
        print(f"  Individual models: {len(self.trained_models['models']['individual'])}")
        print(f"  Multi-target model: {'OK' if 'multi_target' in self.trained_models['models'] else '✗'}")
    
    def save_raw_data(self):
        """Save raw data"""
        raw_dir = self.output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Save unified dataset metadata
        metadata_df = []
        for sample in self.raw_data['unified_dataset']['samples']:
            row = {
                'sample_id': sample['sample_id'],
                'voltage': sample['voltage'],
                'time_hours': sample['time_hours'],
                'filename': sample['filename']
            }
            # Add concentrations
            for species, conc in sample['concentrations'].items():
                row[f'{species}_concentration'] = conc
            metadata_df.append(row)
        
        pd.DataFrame(metadata_df).to_csv(raw_dir / f"raw_metadata_{self.timestamp}.csv", index=False)
        
        # Save summary
        with open(raw_dir / f"raw_summary_{self.timestamp}.json", 'w') as f:
            json.dump(self.raw_data['metadata'], f, indent=2)
        
        print(f"OK Raw data saved to {raw_dir}")
    
    def save_processed_data(self):
        """Save processed data"""
        processed_dir = self.output_dir / "processed"
        self.preprocessor.save_processed_data(self.processed_data, processed_dir)
        
        # Save additional processing metadata
        processing_metadata = {
            'timestamp': self.timestamp,
            'preprocessing_params': self.processed_data.get('processing_params', {}),
            'n_samples': len(self.processed_data['samples']),
            'n_spectral_features': self.processed_data['metadata']['n_features_spectral'],
            'n_extracted_features': self.processed_data['metadata']['n_features_extracted']
        }
        
        with open(processed_dir / f"processing_metadata_{self.timestamp}.json", 'w') as f:
            json.dump(processing_metadata, f, indent=2)
        
        print(f"OK Processed data saved to {processed_dir}")
    
    def save_final_results(self):
        """Save final training results and models"""
        models_dir = self.output_dir / "models"
        results_dir = self.output_dir / "results"
        
        # Save models
        self.trainer.save_models(models_dir)
        
        # Save detailed results
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        metrics_data = []
        for model_type, metrics in self.trained_models['metrics'].items():
            for species, species_metrics in metrics['species_metrics'].items():
                metrics_data.append({
                    'model_type': model_type,
                    'species': species,
                    'r2': species_metrics['r2'],
                    'rmse': species_metrics['rmse'],
                    'mae': species_metrics['mae']
                })
        
        pd.DataFrame(metrics_data).to_csv(
            results_dir / f"performance_metrics_{self.timestamp}.csv", index=False)
        
        # Training summary
        training_summary = {
            'timestamp': self.timestamp,
            'best_components': self.trained_models['best_components'],
            'n_training_samples': self.trained_models['splits']['X_train'].shape[0],
            'n_validation_samples': self.trained_models['splits']['X_val'].shape[0],
            'n_test_samples': self.trained_models['splits']['X_test'].shape[0],
            'individual_model_performance': self.trained_models['metrics']['individual']['overall_metrics'],
            'multi_target_model_performance': self.trained_models['metrics']['multi_target']['overall_metrics']
        }
        
        with open(results_dir / f"training_summary_{self.timestamp}.json", 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"OK Final results saved to {results_dir}")
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = reports_dir / f"analysis_report_{self.timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# GCMS Real Experimental Data Analysis Report\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Pipeline Timestamp:** {self.timestamp}\n\n")
            
            f.write("## 1. Dataset Overview\n\n")
            f.write(f"- **Total Samples:** {self.raw_data['metadata']['n_samples']}\n")
            f.write(f"- **Species Analyzed:** {', '.join(self.trainer.species_names)}\n")
            f.write(f"- **Voltage Levels:** {self.raw_data['metadata']['voltage_levels']}\n")
            f.write(f"- **Time Range:** {self.raw_data['metadata']['time_range']} hours\n\n")
            
            f.write("## 2. Preprocessing Details\n\n")
            f.write(f"- **Spectral Features:** {self.processed_data['metadata']['n_features_spectral']}\n")
            f.write(f"- **Extracted Features:** {self.processed_data['metadata']['n_features_extracted']}\n")
            f.write(f"- **Total Features:** {self.processed_data['metadata']['n_features_spectral'] + self.processed_data['metadata']['n_features_extracted']}\n\n")
            
            f.write("## 3. Model Training Results\n\n")
            f.write(f"- **Optimal PLS Components:** {self.trained_models['best_components']}\n")
            f.write(f"- **Training Samples:** {self.trained_models['splits']['X_train'].shape[0]}\n")
            f.write(f"- **Validation Samples:** {self.trained_models['splits']['X_val'].shape[0]}\n")
            f.write(f"- **Test Samples:** {self.trained_models['splits']['X_test'].shape[0]}\n\n")
            
            f.write("### Individual Model Performance\n\n")
            f.write("| Species | R² | RMSE | MAE |\n")
            f.write("|---------|----|----- |-----|\n")
            
            for species, metrics in self.trained_models['metrics']['individual']['species_metrics'].items():
                f.write(f"| {species} | {metrics['r2']:.4f} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} |\n")
            
            overall_ind = self.trained_models['metrics']['individual']['overall_metrics']
            f.write(f"| **Overall** | **{overall_ind['r2']:.4f}** | **{overall_ind['rmse']:.4f}** | **{overall_ind['mae']:.4f}** |\n\n")
            
            f.write("### Multi-Target Model Performance\n\n")
            f.write("| Species | R² | RMSE | MAE |\n")
            f.write("|---------|----|----- |-----|\n")
            
            for species, metrics in self.trained_models['metrics']['multi_target']['species_metrics'].items():
                f.write(f"| {species} | {metrics['r2']:.4f} | {metrics['rmse']:.4f} | {metrics['mae']:.4f} |\n")
            
            overall_mt = self.trained_models['metrics']['multi_target']['overall_metrics']
            f.write(f"| **Overall** | **{overall_mt['r2']:.4f}** | **{overall_mt['rmse']:.4f}** | **{overall_mt['mae']:.4f}** |\n\n")
            
            f.write("## 4. Files Generated\n\n")
            f.write("- **Raw Data:** `data/raw/`\n")
            f.write("- **Processed Data:** `data/processed/`\n")
            f.write("- **Trained Models:** `data/models/`\n")
            f.write("- **Results:** `data/results/`\n")
            f.write("- **Reports:** `data/reports/`\n\n")
            
            f.write("## 5. Model Usage\n\n")
            f.write("```python\n")
            f.write("import joblib\n")
            f.write("# Load multi-target model\n")
            f.write("model = joblib.load('data/models/multi_target_pls_model.pkl')\n")
            f.write("# Load individual models\n")
            f.write("so2_model = joblib.load('data/models/individual/SO2_pls_model.pkl')\n")
            f.write("```\n\n")
        
        print(f"OK Analysis report generated: {report_file}")

def main():
    """Main execution function"""
    print("GCMS REAL EXPERIMENTAL DATA ANALYSIS")
    print("="*50)
    
    # Create pipeline
    pipeline = GCMSAnalysisPipeline()
    
    # Run complete analysis
    pipeline.run_complete_pipeline(
        # Preprocessing options
        baseline_removal=True,
        smoothing=True,
        normalization='minmax',
        
        # Training options
        test_size=0.2,
        val_size=0.1,
        split_strategy='voltage_stratified'
    )
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()