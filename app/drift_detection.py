""" 
Data Drift Detection using Evidently
Monitors data distribution changes over time
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

try:
    from evidently import Report
    from evidently.metrics import DriftedColumnsCount
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning('Evidenyly not available. Drift detection disabled.')
    
from app.config import config

logger = logging.getLogger(__name__)

class DriftDetector:
    """Class to handle data drift detection."""
    
    def __init__(self, reference_data_path: Optional[Path] = None):
        """
        Initialize drift detector. 
        Args:
            reference_data path: Path to reference data CSV 
        """
        self.reference_data_path = reference_data_path
        self.reference_data = None
        self.drift_threshold = 0.1 # 10% drift threshold
        
        if not EVIDENTLY_AVAILABLE:
            logger.warning('Evidently not installed. Drift detection unavailable.')
            return
        
        # Load reference data if available
        if reference_data_path and reference_data_path.exists():
            self.load_reference_data(reference_data_path)
        else:
            logger.warning(f'Reference data not found at {reference_data_path}')
            
    def load_reference_data(self, path: Path):
        """Load reference data from CSV."""
        try:
            self.reference_data = pd.read_csv(path)
            logger.info(f'Loaded reference data from {path}, shape: {self.reference_data.shape}')
        except Exception as e:
            logger.error(f'Failed to load reference data: {e}')
            self.reference_data = None
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current data to check for drift
            
        Returns:
            Dictionary with drift detection results
        """
        if not EVIDENTLY_AVAILABLE:
            return {
                'error': 'Evidently not available',
                'drift_detected': False
            }
        
        if self.reference_data is None:
            return {
                'error': 'Reference data not loaded',
                'drift_detected': False
            }
        
        try:
            # Create Evidently report
            report = Report(metrics=[
                DriftedColumnsCount()
            ])
            
            # Run the report
            report.run(
                reference_data=self.reference_data,
                current_data=current_data
            )
            
            # Extract results
            result_dict = report.as_dict()
            
            # Parse drift results
            drift_metric = result_dict['metrics'][0]['result']
            
            number_of_columns = drift_metric.get('number_of_columns', 0)
            number_of_drifted = drift_metric.get('number_of_drifted_columns', 0)
            share_drifted = number_of_drifted / number_of_columns if number_of_columns > 0 else 0
            
            drift_results = {
                'drift_detected': number_of_drifted > 0,
                'number_of_columns': number_of_columns,
                'number_of_drifted_columns': number_of_drifted,
                'drifted_columns_count': number_of_drifted,
                'share_of_drifted_columns': share_drifted,
                'drift_share': share_drifted,
                'timestamp': datetime.now().isoformat()
            }
            
            return drift_results
            
        except Exception as e:
            logger.error(f'Drift detection failed: {e}')
            return {
                'error': str(e),
                'drift_detected': False
            }


def get_drift_detector() -> DriftDetector:
    """Get singleton drift detector instance."""
    reference_path = Path('data/reference/reference_data.csv')
    return DriftDetector(reference_data_path=reference_path)