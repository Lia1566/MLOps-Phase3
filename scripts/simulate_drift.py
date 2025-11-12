"""
Simulate Data Drift and Detect Performance Impact
Generates drifted data and evaluates model performance
"""
import sys
from pathlib import Path

# Add project root to Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Set random seed
np.random.seed(42)

def load_reference_data():
    """Load reference data."""
    ref_path = Path('data/reference/reference_data.csv')
    return pd.read_csv(ref_path)

def generate_drifted_data(reference_df, drift_type='mean_shift', n_samples=500):
    """
    Generate drifted data with different drift scenarios.
    
    Args:
        reference_df: Reference data
        drift_type: Type of drift to simulate
        n_samples: Number of samples to generate
    """
    print(f"\nGenerating {drift_type} drift with {n_samples} samples...")
    
    drifted_data = reference_df.sample(n=n_samples, replace=True).copy()
    
    if drift_type == 'mean_shift':
        # Shift performance indicators down (simulating worse students)
        drifted_data['Class_X_Percentage'] -= 10
        drifted_data['Class_XII_Percentage'] -= 10
        drifted_data['Study_Hours'] -= 2
        
    elif drift_type == 'variance_change':
        # Increase variance in percentages
        noise = np.random.normal(0, 15, n_samples)
        drifted_data['Class_X_Percentage'] += noise
        drifted_data['Class_XII_Percentage'] += noise
        
    elif drift_type == 'distribution_shift':
        # Change categorical distributions
        drifted_data['Coaching_Yes'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Less coaching
        drifted_data['Medium_English'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # Less English
    
    # Clip to valid ranges
    drifted_data['Class_X_Percentage'] = drifted_data['Class_X_Percentage'].clip(0, 100)
    drifted_data['Class_XII_Percentage'] = drifted_data['Class_XII_Percentage'].clip(0, 100)
    drifted_data['Study_Hours'] = drifted_data['Study_Hours'].clip(0, 12)
    
    return drifted_data

def detect_drift_with_evidently(reference_df, current_df):
    """Detect drift using Evidently."""
    from app.drift_detection import DriftDetector
    
    detector = DriftDetector()
    detector.reference_data = reference_df
    
    results = detector.detect_drift(current_df)
    return results

def evaluate_model_performance(data, model_path='models/pipeline_baseline.pkl'):
    """Evaluate model on drifted data."""
    import joblib
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Load model
    model = joblib.load(model_path)
    
    # Make predictions
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]
    
    # Generate pseudo-labels (simulate ground truth degradation)
    # In production, you'd have actual labels
    true_labels = (data['Class_X_Percentage'] > 70).astype(int).values
    
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'f1': f1_score(true_labels, predictions, zero_division=0),
        'mean_probability': float(probabilities.mean())
    }
    
    return metrics, predictions

def visualize_drift(reference_df, drifted_df, drift_type, output_dir='reports/drift'):
    """Create drift visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Data Drift Analysis: {drift_type}', fontsize=16)
    
    # Plot 1: Class X Percentage distribution
    axes[0, 0].hist(reference_df['Class_X_Percentage'], bins=30, alpha=0.5, label='Reference', color='blue')
    axes[0, 0].hist(drifted_df['Class_X_Percentage'], bins=30, alpha=0.5, label='Current', color='red')
    axes[0, 0].set_title('Class X Percentage Distribution')
    axes[0, 0].legend()
    
    # Plot 2: Class XII Percentage distribution
    axes[0, 1].hist(reference_df['Class_XII_Percentage'], bins=30, alpha=0.5, label='Reference', color='blue')
    axes[0, 1].hist(drifted_df['Class_XII_Percentage'], bins=30, alpha=0.5, label='Current', color='red')
    axes[0, 1].set_title('Class XII Percentage Distribution')
    axes[0, 1].legend()
    
    # Plot 3: Study Hours distribution
    axes[1, 0].hist(reference_df['Study_Hours'], bins=20, alpha=0.5, label='Reference', color='blue')
    axes[1, 0].hist(drifted_df['Study_Hours'], bins=20, alpha=0.5, label='Current', color='red')
    axes[1, 0].set_title('Study Hours Distribution')
    axes[1, 0].legend()
    
    # Plot 4: Summary statistics comparison
    features = ['Class_X_Percentage', 'Class_XII_Percentage', 'Study_Hours']
    ref_means = [reference_df[f].mean() for f in features]
    drift_means = [drifted_df[f].mean() for f in features]
    
    x = np.arange(len(features))
    width = 0.35
    axes[1, 1].bar(x - width/2, ref_means, width, label='Reference', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, drift_means, width, label='Current', color='red', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Class X %', 'Class XII %', 'Study Hours'], rotation=45)
    axes[1, 1].set_title('Mean Values Comparison')
    axes[1, 1].legend()
    
    plt.tight_layout()
    output_path = output_dir / f'drift_analysis_{drift_type}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    plt.close()

def generate_drift_report(drift_types=['mean_shift', 'variance_change', 'distribution_shift']):
    """Generate comprehensive drift report."""
    print("\n" + "="*60)
    print("DATA DRIFT SIMULATION AND DETECTION")
    print("="*60)
    
    # Load reference data
    reference_df = load_reference_data()
    print(f"\nReference data loaded: {reference_df.shape}")
    
    # Baseline performance (on reference data)
    print("\nEvaluating baseline performance...")
    baseline_metrics, _ = evaluate_model_performance(reference_df)
    print(f"Baseline Metrics: {json.dumps(baseline_metrics, indent=2)}")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'baseline_metrics': baseline_metrics,
        'drift_scenarios': {}
    }
    
    # Test each drift type
    for drift_type in drift_types:
        print(f"\n{'='*60}")
        print(f"Testing Drift Type: {drift_type.upper()}")
        print(f"{'='*60}")
        
        # Generate drifted data
        drifted_df = generate_drifted_data(reference_df, drift_type)
        
        # Detect drift
        print("\nDetecting drift...")
        drift_results = detect_drift_with_evidently(reference_df, drifted_df)
        print(f"Drift Detected: {drift_results.get('drift_detected', False)}")
        print(f"Drifted Columns: {drift_results.get('number_of_drifted_columns', 0)}/{drift_results.get('number_of_columns', 0)}")
        
        # Evaluate performance
        print("\nEvaluating model performance on drifted data...")
        drifted_metrics, _ = evaluate_model_performance(drifted_df)
        print(f"Drifted Metrics: {json.dumps(drifted_metrics, indent=2)}")
        
        # Calculate performance degradation
        degradation = {
            metric: baseline_metrics[metric] - drifted_metrics[metric]
            for metric in baseline_metrics.keys()
        }
        print(f"\nPerformance Degradation: {json.dumps(degradation, indent=2)}")
        
        # Visualize drift
        visualize_drift(reference_df, drifted_df, drift_type)
        
        # Check thresholds and determine action
        action = determine_action(drift_results, degradation)
        print(f"\nRecommended Action: {action}")
        
        results['drift_scenarios'][drift_type] = {
            'drift_results': drift_results,
            'drifted_metrics': drifted_metrics,
            'degradation': degradation,
            'action': action
        }
        
        # Save drifted data
        output_path = Path(f'data/monitoring/drifted_data_{drift_type}.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        drifted_df.to_csv(output_path, index=False)
        print(f"Saved drifted data: {output_path}")
    
    # Save comprehensive report
    report_path = Path('reports/drift/drift_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comprehensive report: {report_path}")
    
    print("\n" + "="*60)
    print("DRIFT SIMULATION COMPLETE!")
    print("="*60)

def determine_action(drift_results, degradation):
    """Determine recommended action based on drift and degradation."""
    drift_detected = drift_results.get('drift_detected', False)
    accuracy_drop = degradation.get('accuracy', 0)
    f1_drop = degradation.get('f1', 0)
    
    if drift_detected and (accuracy_drop > 0.1 or f1_drop > 0.1):
        return "🔴 CRITICAL: Immediate retraining required. Performance degraded >10%"
    elif drift_detected and (accuracy_drop > 0.05 or f1_drop > 0.05):
        return "🟡 WARNING: Schedule retraining. Performance degraded >5%"
    elif drift_detected:
        return "🟢 MONITOR: Drift detected but performance stable. Continue monitoring"
    else:
        return "✅ OK: No significant drift detected"

if __name__ == "__main__":
    generate_drift_report()
