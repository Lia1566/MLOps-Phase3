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
import json

try:
    from evidently import Report
    from evidently.metrics import DriftedColumnsCount
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning('Evidently not available. Drift detection disabled.')
    
logger = logging.getLogger(__name__)

class DriftDetector:
    """Class to handle data drift detection."""
    
    def __init__(self, reference_data_path: Optional[Path] = None):
        """
        Initialize drift detector. 
        Args:
            reference_data_path: Path to reference data CSV 
        """
        self.reference_data_path = reference_data_path
        self.reference_data = None
        self.drift_threshold = 0.1  # 10% drift threshold
        
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
            # Create Evidently report with DriftedColumnsCount
            report = Report(metrics=[
                DriftedColumnsCount()
            ])
            
            # Run the report
            report.run(
                reference_data=self.reference_data,
                current_data=current_data
            )
            
            # Extract results from metric using Evidently 0.7.14 API
            metric = report.metrics[0]
            drift_share = metric.drift_share
            
            # Get number of columns
            number_of_columns = len(current_data.columns)
            number_of_drifted = int(drift_share * number_of_columns)
            
            drift_results = {
                'drift_detected': drift_share > 0,
                'number_of_columns': number_of_columns,
                'number_of_drifted_columns': number_of_drifted,
                'share_of_drifted_columns': drift_share,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Drift detection: {drift_results}")
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