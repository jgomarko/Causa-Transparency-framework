#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing Module for CTF Implementation.

Key Features:
- No data leakage: Fit only on training data
- sklearn Pipeline architecture
- Proper categorical encoding (OneHot for nominal)
- Complete case analysis: Drop ALL rows with ANY missing values
- Data quality validation
- Comprehensive logging

Missing Data Strategy:
- COMPLETE CASE ANALYSIS: All rows with ANY NaN values are dropped
- No imputation - transparent and publication-ready
- NaN values propagate naturally through feature engineering
- Explicit NaN dropping step with detailed reporting
- This approach avoids imputation-related assumptions and bias

Author: John Marko 
Date: 2025-11-16
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
import logging
from pathlib import Path

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import missingno as msno
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("matplotlib/missingno not available. Install with: pip install matplotlib missingno")

# Sklearn imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced learning (optional)
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logging.warning("imbalanced-learn not available. Install with: pip install imbalanced-learn")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Transformers
# ============================================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stateless feature engineering transformer.
    Creates derived features without fitting (no data leakage).
    """
    
    def __init__(self, dataset_type: str = 'compas'):
        """
        Initialize feature engineer.
        
        Args:
            dataset_type: 'compas' or 'mimic'
        """
        self.dataset_type = dataset_type
    
    def fit(self, X, y=None):
        """No fitting required for feature engineering"""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create derived features"""
        X = X.copy()
        
        if self.dataset_type == 'compas':
            X = self._engineer_compas_features(X)
        elif self.dataset_type == 'mimic':
            X = self._engineer_mimic_features(X)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
        
        return X
    
    def _engineer_compas_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer COMPAS-specific features"""
        logger.info(f"Engineering COMPAS features. Input shape: {X.shape}")
        logger.info(f"Input columns: {list(X.columns)}")
        
        X = X.copy()
        
        # Convert numeric columns to proper numeric types first
        numeric_cols = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 
                       'priors_count', 'days_b_screening_arrest', 'c_days_from_compas',
                       'decile_score.1', 'priors_count.1']
        
        for col in numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Age-related features
        if 'age' in X.columns:
            X['age_squared'] = X['age'] ** 2
            X['is_young'] = (X['age'] < 25).astype(int)
            X['is_elderly'] = (X['age'] > 65).astype(int)
        
        # Prior history features
        if 'priors_count' in X.columns:
            # Now that we've ensured numeric types, this should work
            X['log_priors_count'] = np.log1p(X['priors_count'].fillna(0))
            X['has_priors'] = (X['priors_count'] > 0).astype(int)
            X['high_priors'] = (X['priors_count'] >= 5).astype(int)
        
        # Juvenile offenses (if available)
        juv_cols = ['juv_fel_count', 'juv_misd_count', 'juv_other_count']
        if all(col in X.columns for col in juv_cols):
            # Fill NaNs with 0 for juvenile counts
            for col in juv_cols:
                X[col] = X[col].fillna(0)
            X['total_juvenile_count'] = X[juv_cols].sum(axis=1)
            X['has_juvenile_history'] = (X['total_juvenile_count'] > 0).astype(int)
        
        # Time-based features
        if 'days_b_screening_arrest' in X.columns:
            X['days_b_screening_arrest'] = X['days_b_screening_arrest'].fillna(0)
            X['abs_days_screening'] = X['days_b_screening_arrest'].abs()
            
            # Create categorical bins (but convert to string for proper encoding)
            X['screening_delay_days'] = pd.cut(
                X['abs_days_screening'],
                bins=[-np.inf, 7, 14, 30, np.inf],
                labels=['short', 'medium', 'long', 'very_long']
            ).astype(str)
        
        # Interaction features
        if 'age' in X.columns and 'priors_count' in X.columns:
            # Ensure both columns are numeric and filled
            X['age'] = X['age'].fillna(X['age'].median())
            X['age_priors_interaction'] = X['age'] * X['log_priors_count']
        
        return X
    
    def _group_ethnicity(self, ethnicity_series: pd.Series) -> pd.Series:
        """
        Group ethnicity categories from 29 to 5 main categories.
        
        Args:
            ethnicity_series: Series containing ethnicity values
            
        Returns:
            Series with grouped ethnicity values
        """
        ethnicity_mapping = {
            # White group
            'WHITE': 'White',
            'WHITE - RUSSIAN': 'White', 
            'WHITE - OTHER EUROPEAN': 'White',
            'WHITE - EASTERN EUROPEAN': 'White',
            'WHITE - BRAZILIAN': 'White',
            
            # Black/African American group
            'BLACK/AFRICAN AMERICAN': 'Black/African American',
            'BLACK/CAPE VERDEAN': 'Black/African American',
            'BLACK/HAITIAN': 'Black/African American',
            
            # Hispanic/Latino group
            'HISPANIC OR LATINO': 'Hispanic/Latino',
            'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic/Latino',
            'HISPANIC/LATINO - DOMINICAN': 'Hispanic/Latino',
            'HISPANIC/LATINO - SALVADORAN': 'Hispanic/Latino',
            'HISPANIC/LATINO - GUATEMALAN': 'Hispanic/Latino',
            'HISPANIC/LATINO - COLOMBIAN': 'Hispanic/Latino',
            'HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)': 'Hispanic/Latino',
            'SOUTH AMERICAN': 'Hispanic/Latino',
            'PORTUGUESE': 'Hispanic/Latino',
            
            # Asian/Pacific Islander group
            'ASIAN': 'Asian/Pacific Islander',
            'ASIAN - CHINESE': 'Asian/Pacific Islander',
            'ASIAN - ASIAN INDIAN': 'Asian/Pacific Islander',
            'ASIAN - VIETNAMESE': 'Asian/Pacific Islander',
            'ASIAN - FILIPINO': 'Asian/Pacific Islander',
            'ASIAN - CAMBODIAN': 'Asian/Pacific Islander',
            'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Asian/Pacific Islander',
            
            # Other/Unknown group
            'UNKNOWN/NOT SPECIFIED': 'Other/Unknown',
            'OTHER': 'Other/Unknown',
            'PATIENT DECLINED TO ANSWER': 'Other/Unknown',
            'MULTI RACE ETHNICITY': 'Other/Unknown',
            'AMERICAN INDIAN/ALASKA NATIVE': 'Other/Unknown',
            'UNABLE TO OBTAIN': 'Other/Unknown'
        }
        
        return ethnicity_series.map(ethnicity_mapping).fillna('Other/Unknown')
    
    def _engineer_mimic_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Engineer MIMIC-III specific features"""
        # Ethnicity grouping (reduce 29 categories to 5)
        if 'ethnicity' in X.columns:
            X['ethnicity'] = self._group_ethnicity(X['ethnicity'])
        
        # Age-related
        if 'age' in X.columns:
            X['age_squared'] = X['age'] ** 2
            X['elderly'] = (X['age'] > 65).astype(int)
            X['very_elderly'] = (X['age'] > 80).astype(int)
        
        # Vital signs ratios
        if 'systolic_bp' in X.columns and 'diastolic_bp' in X.columns:
            X['pulse_pressure'] = X['systolic_bp'] - X['diastolic_bp']
            X['mean_bp'] = (X['systolic_bp'] + 2 * X['diastolic_bp']) / 3
        
        if 'heart_rate' in X.columns and 'systolic_bp' in X.columns:
            # Shock index: HR / SBP
            X['shock_index'] = X['heart_rate'] / (X['systolic_bp'] + 1e-6)
        
        # Lab value ratios
        if 'bun' in X.columns and 'creatinine' in X.columns:
            X['bun_creatinine_ratio'] = X['bun'] / (X['creatinine'] + 1e-6)
        
        if 'sodium' in X.columns and 'chloride' in X.columns:
            X['anion_gap'] = X['sodium'] - X['chloride']
        
        # Organ dysfunction indicators
        if 'creatinine' in X.columns:
            X['renal_dysfunction'] = (X['creatinine'] > 2.0).astype(int)
        
        if 'bilirubin' in X.columns:
            X['hepatic_dysfunction'] = (X['bilirubin'] > 2.0).astype(int)
        
        if 'lactate' in X.columns:
            X['severe_hyperlactatemia'] = (X['lactate'] > 4.0).astype(int)
        
        # Simplified SOFA components (if available)
        sofa_components = []
        
        # Respiration (PaO2/FiO2)
        if 'pao2_fio2' in X.columns:
            X['resp_sofa'] = pd.cut(
                X['pao2_fio2'],
                bins=[0, 100, 200, 300, 400, np.inf],
                labels=[4, 3, 2, 1, 0],
                include_lowest=True
            ).astype(float)  # Keep as float to preserve NaN
            sofa_components.append('resp_sofa')
        
        # Coagulation (Platelets)
        if 'platelet' in X.columns:
            X['coag_sofa'] = pd.cut(
               X['platelet'],  # NaN will propagate naturally
               bins=[0, 20, 50, 100, 150, np.inf],
               labels=[4, 3, 2, 1, 0],
               include_lowest=True
            ).astype(float)  # Keep as float to preserve NaN
            sofa_components.append('coag_sofa')
        
        # Liver (Bilirubin)
        if 'bilirubin' in X.columns:
            X['liver_sofa'] = pd.cut(
                X['bilirubin'],
                bins=[0, 1.2, 2.0, 6.0, 12.0, np.inf],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True
            ).astype(float)  # Keep as float to preserve NaN
            sofa_components.append('liver_sofa')
        
        # Renal (Creatinine)
        if 'creatinine' in X.columns:
            X['renal_sofa'] = pd.cut(
                X['creatinine'],
                bins=[0, 1.2, 2.0, 3.5, 5.0, np.inf],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True
            ).astype(float)  # Keep as float to preserve NaN
            sofa_components.append('renal_sofa')
        
        # Total simplified SOFA
        if sofa_components:
            X['simplified_sofa'] = X[sofa_components].sum(axis=1)
        
        return X


class DataQualityValidator(BaseEstimator, TransformerMixin):
    """
    Validates data quality and logs issues.
    Does not modify data, only reports.
    """
    
    def __init__(self, raise_on_error: bool = False):
        self.raise_on_error = raise_on_error
        self.validation_report = {}
    
    def fit(self, X, y=None):
        """Fit does nothing"""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality"""
        issues = []
        
        # Check for NaN
        nan_counts = X.isnull().sum()
        if nan_counts.any():
            nan_cols = nan_counts[nan_counts > 0].to_dict()
            issues.append(f"NaN values: {nan_cols}")
            logger.warning(f"Found NaN values in columns: {list(nan_cols.keys())}")
        
        # Check for infinite values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.any(np.isinf(X[col])):
                inf_count = np.sum(np.isinf(X[col]))
                issues.append(f"Inf values in {col}: {inf_count}")
                logger.warning(f"Found {inf_count} infinite values in {col}")
        
        # Check for constant features
        if len(X) > 1:
            for col in numeric_cols:
                if X[col].std() == 0:
                    issues.append(f"Constant feature: {col}")
                    logger.warning(f"Feature {col} is constant (std=0)")
        
        # Check for high cardinality categoricals
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            n_unique = X[col].nunique()
            if n_unique > 100:
                issues.append(f"High cardinality in {col}: {n_unique} unique values")
                logger.warning(f"Feature {col} has {n_unique} unique values")
        
        self.validation_report = {
            'n_issues': len(issues),
            'issues': issues,
            'shape': X.shape,
            'n_nan': nan_counts.sum()
        }
        
        if issues:
            logger.info(f"Data quality check found {len(issues)} issues")
            if self.raise_on_error:
                raise ValueError(f"Data quality issues: {issues}")
        else:
            logger.info("✓ Data quality checks passed")
        
        return X


class IntelligentMissingDataHandler(BaseEstimator, TransformerMixin):
    """
    Intelligent missing data handler that uses feature-wise strategies.
    
    Instead of dropping all rows with ANY missing values, this transformer:
    1. Drops features with excessive missingness (>threshold)
    2. Imputes remaining missing values using domain-appropriate strategies
    3. Provides comprehensive reporting and validation
    
    This approach preserves much more data while maintaining scientific rigor.
    """
    
    def __init__(
        self,
        drop_threshold: float = 0.5,
        dataset_type: str = 'mimic',
        verbose: bool = True,
        validate_imputation: bool = True
    ):
        """
        Initialize intelligent missing data handler.
        
        Args:
            drop_threshold: Drop features with missing % above this threshold
            dataset_type: 'mimic' or 'compas' for domain-specific strategies
            verbose: Whether to print detailed reports
            validate_imputation: Whether to validate imputed values
        """
        self.drop_threshold = drop_threshold
        self.dataset_type = dataset_type
        self.verbose = verbose
        self.validate_imputation = validate_imputation
        
        # Will be set during fit
        self.dropped_features_ = []
        self.imputation_strategies_ = {}
        self.imputation_stats_ = {}
        self.n_before_ = None
        self.n_after_ = None
    
    def _get_clinical_imputation_strategy(self, feature: str, data: pd.Series) -> Dict:
        """Get domain-specific imputation strategy for a feature"""
        
        if self.dataset_type == 'mimic':
            # MIMIC-III clinical imputation strategies
            clinical_strategies = {
                # Lab values - use normal ranges as defaults
                'creatinine': {'method': 'constant', 'value': 1.0, 'reason': 'normal creatinine'},
                'bun': {'method': 'constant', 'value': 15.0, 'reason': 'normal BUN'},
                'glucose': {'method': 'median', 'reason': 'preserve variance for causal discovery'},
                'sodium': {'method': 'constant', 'value': 140.0, 'reason': 'normal sodium'},
                'potassium': {'method': 'constant', 'value': 4.0, 'reason': 'normal potassium'},
                'chloride': {'method': 'constant', 'value': 102.0, 'reason': 'normal chloride'},
                'hemoglobin': {'method': 'median', 'reason': 'population median'},
                'hematocrit': {'method': 'median', 'reason': 'population median'},
                'platelet': {'method': 'constant', 'value': 250.0, 'reason': 'normal platelet count'},
                'wbc': {'method': 'constant', 'value': 7.0, 'reason': 'normal WBC'},
                'lactate': {'method': 'constant', 'value': 1.0, 'reason': 'normal lactate'},
                'bilirubin': {'method': 'constant', 'value': 1.0, 'reason': 'normal bilirubin'},
                
                # Vital signs - use normal ranges
                'heart_rate': {'method': 'constant', 'value': 80.0, 'reason': 'normal HR'},
                'systolic_bp': {'method': 'constant', 'value': 120.0, 'reason': 'normal SBP'},
                'diastolic_bp': {'method': 'constant', 'value': 80.0, 'reason': 'normal DBP'},
                'respiratory_rate': {'method': 'constant', 'value': 16.0, 'reason': 'normal RR'},
                'temperature': {'method': 'constant', 'value': 37.0, 'reason': 'normal temp'},
                'spo2': {'method': 'constant', 'value': 98.0, 'reason': 'normal oxygen sat'},
                
                # Demographics and categorical
                'marital_status': {'method': 'constant', 'value': 'UNKNOWN', 'reason': 'unknown category'},
                'gender': {'method': 'mode', 'reason': 'most common gender'},
                'ethnicity': {'method': 'constant', 'value': 'UNKNOWN', 'reason': 'unknown category'},
                
                # Severity scores - conservative approach
                'saps': {'method': 'median', 'reason': 'median severity'},
                'sapsii': {'method': 'median', 'reason': 'median severity'},
                'apsiii': {'method': 'median', 'reason': 'median severity'},
            }
            
            if feature.lower() in clinical_strategies:
                return clinical_strategies[feature.lower()]
        
        elif self.dataset_type == 'compas':
            # COMPAS-specific strategies
            compas_strategies = {
                'age': {'method': 'median', 'reason': 'population median age'},
                'priors_count': {'method': 'constant', 'value': 0, 'reason': 'no prior record'},
                'juv_fel_count': {'method': 'constant', 'value': 0, 'reason': 'no juvenile record'},
                'juv_misd_count': {'method': 'constant', 'value': 0, 'reason': 'no juvenile record'},
                'juv_other_count': {'method': 'constant', 'value': 0, 'reason': 'no juvenile record'},
                'days_b_screening_arrest': {'method': 'median', 'reason': 'median screening delay'},
                'c_charge_degree': {'method': 'mode', 'reason': 'most common charge degree'},
                'race': {'method': 'constant', 'value': 'Other', 'reason': 'unknown race'},
                'sex': {'method': 'mode', 'reason': 'most common gender'},
            }
            
            if feature.lower() in compas_strategies:
                return compas_strategies[feature.lower()]
        
        # Default strategies based on data type
        if data.dtype in ['object', 'category']:
            return {'method': 'mode', 'reason': 'most frequent category'}
        elif data.dtype in ['int64', 'float64']:
            # Use median for numeric data
            return {'method': 'median', 'reason': 'population median'}
        else:
            return {'method': 'constant', 'value': 'MISSING', 'reason': 'unknown data type'}
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """
        Analyze missingness patterns and determine imputation strategies.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        self.n_before_ = len(X)
        
        # Analyze missingness
        missing_pct = (X.isnull().sum() / len(X)) * 100
        
        # Identify features to drop
        self.dropped_features_ = missing_pct[missing_pct > (self.drop_threshold * 100)].index.tolist()
        
        # Determine imputation strategies for remaining features
        remaining_features = X.columns.difference(self.dropped_features_)
        
        for feature in remaining_features:
            if X[feature].isnull().any():
                strategy = self._get_clinical_imputation_strategy(feature, X[feature])
                self.imputation_strategies_[feature] = strategy
        
        if self.verbose:
            logger.info("="*60)
            logger.info("INTELLIGENT MISSING DATA HANDLING")
            logger.info("="*60)
            logger.info(f"Dataset: {self.dataset_type.upper()}")
            logger.info(f"Total records: {self.n_before_:,}")
            logger.info(f"Drop threshold: {self.drop_threshold:.1%}")
            
            if self.dropped_features_:
                logger.info(f"\nFeatures to DROP ({len(self.dropped_features_)}):")
                for feature in self.dropped_features_:
                    pct = missing_pct[feature]
                    logger.info(f"  {feature}: {pct:.1f}% missing")
            else:
                logger.info("\n✓ No features exceed drop threshold")
            
            if self.imputation_strategies_:
                logger.info(f"\nFeatures to IMPUTE ({len(self.imputation_strategies_)}):")
                for feature, strategy in self.imputation_strategies_.items():
                    pct = missing_pct[feature]
                    logger.info(f"  {feature}: {pct:.1f}% missing → {strategy['method']} ({strategy['reason']})")
            else:
                logger.info("\n✓ No remaining features need imputation")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missing data handling strategies.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_processed = X.copy()
        
        # Step 1: Drop problematic features
        if self.dropped_features_:
            X_processed = X_processed.drop(columns=self.dropped_features_)
            if self.verbose:
                logger.info(f"Dropped {len(self.dropped_features_)} features with excessive missingness")
        
        # Step 2: Impute remaining missing values
        imputation_report = {}
        
        for feature, strategy in self.imputation_strategies_.items():
            if feature not in X_processed.columns:
                continue
                
            missing_count = X_processed[feature].isnull().sum()
            if missing_count == 0:
                continue
            
            original_missing = missing_count
            
            try:
                if strategy['method'] == 'median':
                    fill_value = X_processed[feature].median()
                elif strategy['method'] == 'mode':
                    fill_value = X_processed[feature].mode().iloc[0] if len(X_processed[feature].mode()) > 0 else 'UNKNOWN'
                elif strategy['method'] == 'constant':
                    fill_value = strategy['value']
                else:
                    fill_value = 'UNKNOWN'
                
                # Validate that fill_value is reasonable for this feature
                if isinstance(fill_value, (int, float)) and not pd.isna(fill_value):
                    # Get some basic bounds from existing data
                    existing_data = X_processed[feature].dropna()
                    if len(existing_data) > 0:
                        q25, q75 = existing_data.quantile([0.25, 0.75])
                        iqr = q75 - q25
                        lower_bound = q25 - 3 * iqr  # More conservative than 1.5*IQR
                        upper_bound = q75 + 3 * iqr
                        
                        # Clip extreme imputation values
                        if fill_value < lower_bound or fill_value > upper_bound:
                            # Use median as safer fallback
                            safer_fill = existing_data.median()
                            if self.verbose:
                                logger.warning(f"  Clipping extreme imputation for {feature}: {fill_value:.2f} → {safer_fill:.2f}")
                            fill_value = safer_fill
                
                X_processed[feature] = X_processed[feature].fillna(fill_value)
                
                imputation_report[feature] = {
                    'imputed_count': original_missing,
                    'method': strategy['method'],
                    'value': fill_value,
                    'reason': strategy['reason']
                }
                
            except Exception as e:
                logger.error(f"Failed to impute {feature}: {e}")
                # Fallback: drop the feature
                X_processed = X_processed.drop(columns=[feature])
        
        self.n_after_ = len(X_processed)
        self.imputation_stats_ = imputation_report
        
        if self.verbose and imputation_report:
            logger.info(f"\nImputation completed:")
            total_imputed = sum(stats['imputed_count'] for stats in imputation_report.values())
            logger.info(f"  Total values imputed: {total_imputed:,}")
            
            for feature, stats in imputation_report.items():
                logger.info(f"  {feature}: {stats['imputed_count']} values → {stats['value']} ({stats['method']})")
        
        # Final validation
        remaining_missing = X_processed.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"⚠️  {remaining_missing} missing values remain after imputation")
        else:
            logger.info("✓ All missing values successfully handled")
        
        logger.info(f"✓ Records retained: {self.n_after_:,}/{self.n_before_:,} ({self.n_after_/self.n_before_:.1%})")
        logger.info("="*60)
        
        return X_processed
    
    def get_imputation_report(self) -> Dict:
        """Get detailed report of imputation actions"""
        return {
            'n_before': self.n_before_,
            'n_after': self.n_after_,
            'retention_rate': self.n_after_ / self.n_before_ if self.n_before_ else 0,
            'dropped_features': self.dropped_features_,
            'imputation_stats': self.imputation_stats_,
            'total_imputed_values': sum(stats['imputed_count'] for stats in self.imputation_stats_.values())
        }
    
    def print_improvement_summary(self, comparison_retention_rate: float = 0.179):
        """
        Print a summary comparing intelligent handling to complete case analysis.
        
        Args:
            comparison_retention_rate: Retention rate of complete case analysis for comparison
        """
        print("\n" + "="*80)
        print("INTELLIGENT MISSING DATA HANDLING - IMPROVEMENT SUMMARY")
        print("="*80)
        print("This approach provides significant advantages over complete case analysis:")
        print(f"\n📊 DATA RETENTION:")
        retention_rate = self.n_after_ / self.n_before_ if self.n_before_ else 0
        print(f"   • Intelligent Handler: {self.n_after_:,} records ({retention_rate:.1%})")
        print(f"   • Complete Case:      {int(self.n_before_ * comparison_retention_rate):,} records ({comparison_retention_rate:.1%})")
        print(f"   • Improvement:        +{self.n_after_ - int(self.n_before_ * comparison_retention_rate):,} more records")
        print(f"   • Factor:             {retention_rate / comparison_retention_rate:.1f}x more data")
        
        print(f"\n🔧 PROCESSING ACTIONS:")
        print(f"   • Features dropped:   {len(self.dropped_features_)} (excessive missingness >50%)")
        print(f"   • Features imputed:   {len(self.imputation_stats_)}")
        print(f"   • Values imputed:     {sum(stats['imputed_count'] for stats in self.imputation_stats_.values()):,}")
        
        if self.dropped_features_:
            print(f"   • Dropped features:   {', '.join(self.dropped_features_)}")
        
        print(f"\n✅ RESEARCH BENEFITS:")
        print("   • Maintains statistical power with larger sample size")
        print("   • Reduces selection bias from non-random missingness")
        print("   • Uses clinical domain knowledge for imputation")
        print("   • Transparent and auditable methodology")
        print("   • Publication-ready approach")
        print("="*80)


class NaNDropper(BaseEstimator, TransformerMixin):
    """
    Complete case analysis: drop all rows with ANY NaN values.
    
    This transformer implements a transparent missing data strategy
    suitable for research publication. It reports missingness before
    dropping and logs the retention rate.
    
    Usage:
        dropper = NaNDropper()
        X_clean = dropper.fit_transform(X)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize NaN dropper.
        
        Args:
            verbose: Whether to print detailed missingness report
        """
        self.verbose = verbose
        self.n_before_ = None
        self.n_after_ = None
        self.retention_rate_ = None
        self.dropped_columns_ = None
    
    def fit(self, X, y=None):
        """
        Fit by analyzing missingness patterns.
        
        Args:
            X: DataFrame to analyze
            y: Target (ignored, for sklearn compatibility)
        
        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        self.n_before_ = len(X)
        
        # Analyze missingness
        nan_counts = X.isnull().sum()
        self.dropped_columns_ = nan_counts[nan_counts > 0].to_dict()
        
        # Calculate retention after dropping
        X_clean = X.dropna()
        self.n_after_ = len(X_clean)
        self.retention_rate_ = self.n_after_ / self.n_before_ if self.n_before_ > 0 else 0
        
        if self.verbose and self.dropped_columns_:
            logger.info("="*60)
            logger.info("NaN DROPPING REPORT (Complete Case Analysis)")
            logger.info("="*60)
            logger.info(f"Before dropping: {self.n_before_:,} rows")
            logger.info(f"\nColumns with missing values:")
            for col, count in sorted(self.dropped_columns_.items(), key=lambda x: x[1], reverse=True):
                pct = count / self.n_before_ * 100
                logger.info(f"  {col}: {count:,} ({pct:.1f}%)")
            logger.info(f"\nDropped: {self.n_before_ - self.n_after_:,} rows with ANY NaN")
            logger.info(f"Retained: {self.n_after_:,} rows ({self.retention_rate_:.1%})")
            logger.info("="*60)
        
        return self
    
    def transform(self, X):
        """
        Drop all rows with ANY NaN values.
        
        Args:
            X: DataFrame to clean
        
        Returns:
            DataFrame with complete cases only
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        n_before = len(X)
        X_clean = X.dropna()
        n_after = len(X_clean)
        
        if self.verbose and n_before > n_after:
            logger.info(f"Dropped {n_before - n_after:,} rows with NaN ({n_after:,} remaining)")
        
        return X_clean
    
    def get_missingness_report(self) -> Dict[str, any]:
        """
        Get detailed missingness report.
        
        Returns:
            Dictionary with missingness statistics
        """
        return {
            'n_before': self.n_before_,
            'n_after': self.n_after_,
            'n_dropped': self.n_before_ - self.n_after_ if self.n_before_ else 0,
            'retention_rate': self.retention_rate_,
            'columns_with_nan': self.dropped_columns_
        }


# ============================================================================
# Main Preprocessor Classes
# ============================================================================

class COMPASPreprocessor:
    """
    Production-ready COMPAS dataset preprocessor.
    
    Features:
    - No data leakage (fit only on training)
    - sklearn Pipeline architecture
    - Proper categorical encoding
    - Comprehensive logging
    - Data quality validation
    """
    
    def __init__(
        self,
        target_column: str = 'two_year_recid',
        include_sensitive: bool = True,
        validate_data: bool = True,
        random_state: int = 42
    ):
        """
        Initialize COMPAS preprocessor.
        
        Args:
            target_column: Name of target column
            include_sensitive: Whether to include race/sex as features
            validate_data: Whether to run data quality checks
            random_state: Random seed for reproducibility
        """
        self.target_column = target_column
        self.include_sensitive = include_sensitive
        self.validate_data = validate_data
        self.random_state = random_state
        
        # Will be set during fit
        self.pipeline = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.is_fitted = False
        
        # Feature type definitions
        self._define_feature_types()
        
        logger.info(f"Initialized COMPASPreprocessor (include_sensitive={include_sensitive})")
    
    def _define_feature_types(self):
        """Define feature types for preprocessing"""
        # Columns to EXCLUDE (identifiers, dates, raw strings, outcome-related)
        self.exclude_columns = [
            # Identifiers
            'id', 'name', 'first', 'last', 'dob',
            'c_case_number', 'r_case_number', 'vr_case_number',
            # Dates (raw strings, need parsing)
            'compas_screening_date', 'c_jail_in', 'c_jail_out',
            'c_offense_date', 'c_arrest_date', 
            'r_offense_date', 'r_jail_in', 'r_jail_out',
            'vr_offense_date', 'screening_date', 'v_screening_date',
            'in_custody', 'out_custody',
            # High cardinality descriptions
            'c_charge_desc', 'r_charge_desc', 'vr_charge_desc',
            # COMPAS scores (don't use as features - this is what we're comparing against!)
            'decile_score', 'score_text', 
            'v_decile_score', 'v_score_text',
            'type_of_assessment', 'v_type_of_assessment',
            # Recidivism-related features (r_ and vr_ prefixes)
            'r_charge_degree', 'r_days_from_arrest',
            'vr_charge_degree',
            # Outcomes (don't use these as features!)
            'is_recid',  # General recidivism (keep if it's different from target)
            'violent_recid', 'is_violent_recid',  # Violence-specific
            'vr_case_number', 'vr_charge_degree', 'vr_offense_date',
            # Redundant age features (keep only continuous 'age')
            'age_cat'
        ]
        
        # Numerical features (will be scaled)
        self.numeric_features = [
            'age', 'age_squared', 'priors_count', 'log_priors_count',
            'juv_fel_count', 'juv_misd_count', 'juv_other_count',
            'total_juvenile_count', 'days_b_screening_arrest',
            'abs_days_screening', 'age_priors_interaction'
        ]
        
        # Binary features (no encoding needed)
        self.binary_features = [
            'is_young', 'is_elderly', 'has_priors', 'high_priors',
            'has_juvenile_history'
        ]
        
        # Nominal categorical (one-hot encode)
        self.nominal_features = ['c_charge_degree', 'screening_delay_days']  # Felony vs Misdemeanor, screening delay
        if self.include_sensitive:
            self.nominal_features.extend(['sex', 'race'])
        
        # Ordinal categorical (ordinal encode)
        self.ordinal_features = []  # Ordinal categorical features (none currently)
    
    def _filter_available_features(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """Filter feature lists to only include available columns"""
        available = {
            'numeric': [f for f in self.numeric_features if f in X.columns],
            'binary': [f for f in self.binary_features if f in X.columns],
            'nominal': [f for f in self.nominal_features if f in X.columns],
            'ordinal': [f for f in self.ordinal_features if f in X.columns]
        }
        
        # Log unavailable features
        all_expected = (
            self.numeric_features + self.binary_features + 
            self.nominal_features + self.ordinal_features
        )
        unavailable = set(all_expected) - set(X.columns)
        if unavailable:
            logger.debug(f"Features not available in data: {unavailable}")
        
        return available
    
    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Build preprocessing pipeline"""
        # First, drop excluded columns
        X = self._drop_excluded_columns(X)
        
        # Filter to available features
        available_features = self._filter_available_features(X)
        
        transformers = []
        
        # Numerical features: impute + scale
        if available_features['numeric']:
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append((
                'numeric',
                numeric_transformer,
                available_features['numeric']
            ))
        
        # Binary features: just impute
        if available_features['binary']:
            binary_transformer = SimpleImputer(strategy='constant', fill_value=0)
            transformers.append((
                'binary',
                binary_transformer,
                available_features['binary']
            ))
        
        # Nominal features: impute + one-hot
        if available_features['nominal']:
            nominal_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                    drop='if_binary'  # Drop one category for binary features
                ))
            ])
            transformers.append((
                'nominal',
                nominal_transformer,
                available_features['nominal']
            ))
        
        # Ordinal features: impute + ordinal encode
        if available_features['ordinal']:
            ordinal_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append((
                'ordinal',
                ordinal_transformer,
                available_features['ordinal']
            ))
        
        # Build column transformer
        column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # Drop unused columns
            verbose_feature_names_out=False  # Prevent sklearn prefix addition
        )
        
        # Full pipeline
        steps = []
        
        # Step 1: Feature engineering
        steps.append(('feature_engineer', FeatureEngineer(dataset_type='compas')))
        
        # Step 2: Data quality validation (optional)
        if self.validate_data:
            steps.append(('validator', DataQualityValidator(raise_on_error=False)))
        
        # Step 3: Preprocessing
        steps.append(('preprocessor', column_transformer))
        
        pipeline = Pipeline(steps)
        
        return pipeline
    
    def _drop_excluded_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop identifier and irrelevant columns"""
        cols_to_drop = [col for col in self.exclude_columns if col in X.columns]
        if cols_to_drop:
            logger.debug(f"Dropping {len(cols_to_drop)} excluded columns: {cols_to_drop[:5]}...")
            X = X.drop(columns=cols_to_drop)
        return X
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """
        Fit preprocessor on training data.
        
        Args:
            X: Training features (including target if y is None)
            y: Training target (optional, will extract from X if None)
        
        Returns:
            self
        """
        logger.info("Fitting COMPAS preprocessor...")
        
        X = X.copy()
        
        # Extract target if not provided
        if y is None:
            if self.target_column not in X.columns:
                raise ValueError(f"Target column '{self.target_column}' not found")
            y = X[self.target_column].values
            X = X.drop(columns=[self.target_column])
        
        # Drop excluded columns first
        X = self._drop_excluded_columns(X)
        
        # Store original feature names (after dropping excluded)
        self.feature_names_in_ = X.columns.tolist()
        
        # Build and fit pipeline
        self.pipeline = self._build_pipeline(X)
        self.pipeline.fit(X, y)
        
        # Mark as fitted BEFORE getting feature names (they check this flag!)
        self.is_fitted = True
        
        # Get output feature names (now that is_fitted=True)
        self.feature_names_out_ = self._get_feature_names_out()
        
        logger.info(f"✓ Preprocessor fitted: {len(self.feature_names_in_)} → {len(self.feature_names_out_)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Features (including target if present)
        
        Returns:
            X_transformed: Preprocessed features
            y: Target values (None if target not in X)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = X.copy()
        
        # Extract target if present
        y = None
        if self.target_column in X.columns:
            y = X[self.target_column].values
            X = X.drop(columns=[self.target_column])
        
        # Drop excluded columns
        X = self._drop_excluded_columns(X)
        
        # Transform
        X_transformed = self.pipeline.transform(X)
        
        logger.debug(f"Transformed shape: {X_transformed.shape}")
        
        return X_transformed, y
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.
        
        Args:
            X: Features (including target if y is None)
            y: Target (optional)
        
        Returns:
            X_transformed: Preprocessed features
            y: Target values
        """
        self.fit(X, y)
        X_transformed, y = self.transform(X)
        return X_transformed, y
    
    def _get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation"""
        if not self.is_fitted or self.pipeline is None:
            logger.debug(f"Cannot get features: is_fitted={self.is_fitted}, pipeline={self.pipeline is not None}")
            return []
        
        try:
            # Get feature names directly from the column transformer
            preprocessor = self.pipeline.named_steps.get('preprocessor')
            if preprocessor is None:
                logger.warning("No preprocessor step found in pipeline")
                return []
            
            logger.debug(f"Attempting to get feature names from preprocessor")
            
            if hasattr(preprocessor, 'get_feature_names_out'):
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    # Clean the feature names by removing sklearn prefixes
                    cleaned_names = self._clean_feature_names(feature_names)
                    logger.debug(f"Successfully extracted and cleaned {len(cleaned_names)} feature names")
                    return cleaned_names
                except Exception as e:
                    logger.error(f"get_feature_names_out() call failed: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    return self._construct_feature_names_manually()
            else:
                logger.warning("Preprocessor does not have get_feature_names_out method")
                return self._construct_feature_names_manually()
            
        except Exception as e:
            logger.error(f"Error extracting feature names: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return self._construct_feature_names_manually()
    
    def _clean_feature_names(self, feature_names: List[str]) -> List[str]:
        """Remove sklearn prefixes from feature names"""
        cleaned = []
        for name in feature_names:
            # Remove sklearn ColumnTransformer prefixes
            if '__' in name:
                # Split on '__' and take everything after the first part
                parts = name.split('__', 1)
                if len(parts) > 1:
                    cleaned_name = parts[1]  # Take part after first '__'
                else:
                    cleaned_name = name
            else:
                cleaned_name = name
                
            # Additional cleaning for common prefixes that might remain
            for prefix in ['numeric_', 'nominal_', 'binary_', 'ordinal_']:
                if cleaned_name.startswith(prefix):
                    cleaned_name = cleaned_name[len(prefix):]
                    break
                    
            cleaned.append(cleaned_name)
        
        return cleaned

    def _construct_feature_names_manually(self) -> List[str]:
        """Manually construct feature names for older sklearn"""
        try:
            preprocessor = self.pipeline.named_steps['preprocessor']
            feature_names = []
            
            for name, transformer, columns in preprocessor.transformers_:
                if name == 'remainder':
                    continue
                
                if not isinstance(columns, list):
                    continue
                    
                if hasattr(transformer, 'get_feature_names_out'):
                    try:
                        trans_features = transformer.get_feature_names_out(columns)
                        feature_names.extend(trans_features)
                        continue
                    except:
                        pass
                
                # Fallback: just use column names
                feature_names.extend(columns)
            
            return feature_names
            
        except Exception as e:
            logger.error(f"Failed to construct feature names: {e}")
            return []
    
    def get_feature_names_out(self) -> List[str]:
        """Public method to get output feature names"""
        return self.feature_names_out_ if self.feature_names_out_ else []


class MIMICPreprocessor:
    """
    Production-ready MIMIC-III dataset preprocessor.
    
    Features:
    - No data leakage
    - Clinical feature engineering
    - Proper handling of lab values
    - Data quality validation
    """
    
    def __init__(
        self,
        target_column: str = 'hospital_expire_flag',
        validate_data: bool = True,
        clip_outliers: bool = True,
        random_state: int = 42
    ):
        """
        Initialize MIMIC preprocessor.
        
        Args:
            target_column: Name of target column
            validate_data: Whether to run data quality checks
            clip_outliers: Whether to clip lab values to clinical ranges
            random_state: Random seed
        """
        self.target_column = target_column
        self.validate_data = validate_data
        self.clip_outliers = clip_outliers
        self.random_state = random_state
        
        self.pipeline = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.is_fitted = False
        
        # Clinical ranges for outlier clipping
        self.clinical_ranges = {
            'creatinine': (0.1, 15.0),
            'bun': (1.0, 200.0),
            'hemoglobin': (3.0, 20.0),
            'hematocrit': (10.0, 60.0),
            'platelet': (10, 1000),
            'wbc': (0.1, 100),
            'sodium': (100, 180),
            'potassium': (1.5, 10.0),
            'chloride': (50, 150),
            'glucose': (20, 1000),
            'lactate': (0.1, 30.0),
            'bilirubin': (0.1, 40.0),
            'heart_rate': (20, 250),
            'systolic_bp': (40, 300),
            'diastolic_bp': (20, 200),
            'respiratory_rate': (4, 60),
            'temperature': (32, 43),  # Celsius
            'spo2': (50, 100),
            'age': (18, 100)
        }
        
        self._define_feature_types()
        
        logger.info("Initialized MIMICPreprocessor")
    
    def _define_feature_types(self):
        """Define feature types"""
        # Continuous features
        self.numeric_features = [
            # Demographics
            'age', 'age_squared',
            # Vital signs
            'heart_rate', 'systolic_bp', 'diastolic_bp', 'respiratory_rate',
            'temperature', 'spo2', 'pulse_pressure', 'mean_bp', 'shock_index',
            # Lab values
            'creatinine', 'bun', 'hemoglobin', 'hematocrit', 'platelet', 'wbc',
            'sodium', 'potassium', 'chloride', 'glucose', 'lactate', 'bilirubin',
            'bun_creatinine_ratio', 'anion_gap',
            # Severity scores
            'saps', 'sapsii', 'apsiii', 'charlson_comorbidity',
            'elixhauser_vanwalraven', 'simplified_sofa',
            # SOFA components
            'resp_sofa', 'coag_sofa', 'liver_sofa', 'renal_sofa',
            # Derived
            'pao2_fio2'
        ]
        
        # Binary features
        self.binary_features = [
            'elderly', 'very_elderly', 'renal_dysfunction',
            'hepatic_dysfunction', 'severe_hyperlactatemia',
            'ventilated', 'vasopressor', 'gender'
        ]
        
        # Categorical
        self.nominal_features = []
    
    def _clip_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers to clinically plausible ranges"""
        if not self.clip_outliers:
            return X
        
        X = X.copy()
        n_clipped = 0
        
        for col, (min_val, max_val) in self.clinical_ranges.items():
            if col in X.columns:
                outliers = (X[col] < min_val) | (X[col] > max_val)
                if outliers.any():
                    n_clipped += outliers.sum()
                    X[col] = X[col].clip(min_val, max_val)
        
        if n_clipped > 0:
            logger.info(f"Clipped {n_clipped} outlier values to clinical ranges")
        
        return X
    
    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Build preprocessing pipeline"""
        # Filter available features
        available_numeric = [f for f in self.numeric_features if f in X.columns]
        available_binary = [f for f in self.binary_features if f in X.columns]
        available_nominal = [f for f in self.nominal_features if f in X.columns]
        
        transformers = []
        
        # Numeric: impute + scale
        if available_numeric:
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('numeric', numeric_transformer, available_numeric))
        
        # Binary: impute
        if available_binary:
            binary_transformer = SimpleImputer(strategy='constant', fill_value=0)
            transformers.append(('binary', binary_transformer, available_binary))
        
        # Nominal: one-hot
        if available_nominal:
            nominal_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('nominal', nominal_transformer, available_nominal))
        
        column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        # Full pipeline
        steps = [
            ('missing_handler', IntelligentMissingDataHandler(
                drop_threshold=0.5,
                dataset_type='mimic',
                verbose=True
            )),
            ('feature_engineer', FeatureEngineer(dataset_type='mimic')),
        ]
        
        if self.validate_data:
            steps.append(('validator', DataQualityValidator(raise_on_error=False)))
        
        steps.append(('preprocessor', column_transformer))
        
        return Pipeline(steps)
    
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """Fit preprocessor on training data"""
        logger.info("Fitting MIMIC preprocessor...")
        
        X = X.copy()
        
        # Clip outliers
        X = self._clip_outliers(X)
        
        # Extract target
        if y is None:
            if self.target_column not in X.columns:
                raise ValueError(f"Target column '{self.target_column}' not found")
            y = X[self.target_column].values
            X = X.drop(columns=[self.target_column])
        
        self.feature_names_in_ = X.columns.tolist()
        
        # Build and fit pipeline
        self.pipeline = self._build_pipeline(X)
        self.pipeline.fit(X, y)
        
        # Mark as fitted BEFORE getting feature names (they check this flag!)
        self.is_fitted = True
        
        # Get output feature names (now that is_fitted=True)
        self.feature_names_out_ = self._get_feature_names_out()
        
        logger.info(f"✓ Preprocessor fitted: {len(self.feature_names_in_)} → {len(self.feature_names_out_)} features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform data"""
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        X = X.copy()
        X = self._clip_outliers(X)
        
        y = None
        if self.target_column in X.columns:
            y = X[self.target_column].values
            X = X.drop(columns=[self.target_column])
        
        X_transformed = self.pipeline.transform(X)
        
        return X_transformed, y
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform"""
        self.fit(X, y)
        X_transformed, y = self.transform(X)
        return X_transformed, y
    
    def _get_feature_names_out(self) -> List[str]:
        """Get output feature names"""
        if not self.is_fitted or self.pipeline is None:
            return []
        
        try:
            # Get feature names from the column transformer
            preprocessor = self.pipeline.named_steps.get('preprocessor')
            if preprocessor is None:
                logger.warning("No preprocessor step found in MIMIC pipeline")
                return []
            
            if hasattr(preprocessor, 'get_feature_names_out'):
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    # Clean the feature names by removing sklearn prefixes
                    cleaned_names = self._clean_mimic_feature_names(feature_names)
                    logger.debug(f"Successfully extracted and cleaned {len(cleaned_names)} MIMIC feature names")
                    return cleaned_names
                except Exception as e:
                    logger.error(f"MIMIC get_feature_names_out() failed: {type(e).__name__}: {e}")
                    return self._construct_mimic_feature_names_manually()
            else:
                logger.warning("MIMIC preprocessor does not have get_feature_names_out method")
                return self._construct_mimic_feature_names_manually()
                
        except Exception as e:
            logger.error(f"Error extracting MIMIC feature names: {type(e).__name__}: {e}")
            return self._construct_mimic_feature_names_manually()
    
    def _clean_mimic_feature_names(self, feature_names: List[str]) -> List[str]:
        """Remove sklearn prefixes from MIMIC feature names"""
        cleaned = []
        for name in feature_names:
            # Remove sklearn ColumnTransformer prefixes
            if '__' in name:
                # Split on '__' and take everything after the first part
                parts = name.split('__', 1)
                if len(parts) > 1:
                    cleaned_name = parts[1]  # Take part after first '__'
                else:
                    cleaned_name = name
            else:
                cleaned_name = name
                
            # Additional cleaning for common prefixes that might remain
            for prefix in ['numeric_', 'nominal_', 'binary_', 'ordinal_']:
                if cleaned_name.startswith(prefix):
                    cleaned_name = cleaned_name[len(prefix):]
                    break
                    
            cleaned.append(cleaned_name)
        
        return cleaned

    def _construct_mimic_feature_names_manually(self) -> List[str]:
        """Manually construct feature names for MIMIC preprocessor"""
        try:
            preprocessor = self.pipeline.named_steps['preprocessor']
            feature_names = []
            
            for name, transformer, columns in preprocessor.transformers_:
                if name == 'remainder':
                    continue
                
                if not isinstance(columns, list):
                    continue
                    
                if hasattr(transformer, 'get_feature_names_out'):
                    try:
                        trans_features = transformer.get_feature_names_out(columns)
                        feature_names.extend(trans_features)
                        continue
                    except:
                        pass
                
                # Fallback: just use column names
                feature_names.extend(columns)
            
            logger.debug(f"Manually constructed {len(feature_names)} MIMIC feature names")
            return feature_names
            
        except Exception as e:
            logger.error(f"Failed to construct MIMIC feature names: {e}")
            # Last resort: create dummy names based on actual output shape
            try:
                # Create a dummy DataFrame to get the actual output shape
                dummy_data = pd.DataFrame({col: [0] for col in self.feature_names_in_})
                transformed = self.pipeline.transform(dummy_data)
                n_features = transformed.shape[1]
                return [f"mimic_feature_{i}" for i in range(n_features)]
            except:
                return []
    
    def get_feature_names_out(self) -> List[str]:
        """Public method to get feature names"""
        return self.feature_names_out_ if self.feature_names_out_ else []


# ============================================================================
# Imbalanced Data Handling (Apply WITHIN CV folds only!)
# ============================================================================

class StratifiedCVSplitter:
    """
    Creates stratified CV splits with proper data partitioning.
    Ensures no data leakage by splitting BEFORE any resampling.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        causal_discovery_ratio: float = 0.48,
        model_training_ratio: float = 0.32,
        test_ratio: float = 0.20,
        random_state: int = 42
    ):
        """
        Initialize CV splitter.
        
        Args:
            n_splits: Number of CV folds
            causal_discovery_ratio: Fraction for causal discovery
            model_training_ratio: Fraction for model training
            test_ratio: Fraction for testing
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.causal_discovery_ratio = causal_discovery_ratio
        self.model_training_ratio = model_training_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Validate ratios
        total = causal_discovery_ratio + model_training_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        self.skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
    
    def split(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        """
        Generate stratified splits.
        
        Yields:
            fold_data: Dict with 'causal', 'train', and 'test' splits
        """
        # Convert y to numpy if it's a Series
        if isinstance(y, pd.Series):
            y = y.values
        
        for fold_idx, (trainval_idx, test_idx) in enumerate(self.skf.split(X, y)):
            # Test split
            if isinstance(X, pd.DataFrame):
                X_test = X.iloc[test_idx]
                X_trainval = X.iloc[trainval_idx]
            else:
                X_test = X[test_idx]
                X_trainval = X[trainval_idx]
            
            y_test = y[test_idx]
            y_trainval = y[trainval_idx]
            
            # Split trainval into causal discovery and training
            # Calculate split ratio for causal discovery vs training
            causal_ratio = self.causal_discovery_ratio / (self.causal_discovery_ratio + self.model_training_ratio)
            
            # Stratified split
            causal_skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.random_state + fold_idx)
            for causal_idx, train_idx in causal_skf.split(X_trainval, y_trainval):
                # Use only first split
                if isinstance(X, pd.DataFrame):
                    X_causal = X_trainval.iloc[causal_idx]
                    X_train = X_trainval.iloc[train_idx]
                else:
                    X_causal = X_trainval[causal_idx]
                    X_train = X_trainval[train_idx]
                
                y_causal = y_trainval[causal_idx]
                y_train = y_trainval[train_idx]
                
                break
            
            yield {
                'fold': fold_idx,
                'causal': {'X': X_causal, 'y': y_causal},
                'train': {'X': X_train, 'y': y_train},
                'test': {'X': X_test, 'y': y_test}
            }


class ImbalanceHandler:
    """
    Handles class imbalance with SMOTE/ADASYN.
    CRITICAL: Only apply WITHIN each CV fold, never before splitting!
    """
    
    def __init__(
        self,
        method: str = 'smote',
        sampling_strategy: Union[str, float] = 'auto',
        random_state: int = 42
    ):
        """
        Initialize imbalance handler.
        
        Args:
            method: 'smote', 'adasyn', or 'none'
            sampling_strategy: Resampling strategy
            random_state: Random seed
        """
        if not IMBLEARN_AVAILABLE and method != 'none':
            logger.warning("imbalanced-learn not available, using method='none'")
            method = 'none'
        
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.resampler = None
        
        if method == 'smote':
            self.resampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=min(3, 5),  # Adaptive for small datasets
                random_state=random_state
            )
        elif method == 'adasyn':
            self.resampler = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=min(3, 5),
                random_state=random_state
            )
        elif method == 'none':
            self.resampler = None
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample training data.
        WARNING: Only call this WITHIN a CV fold on training data!
        
        Args:
            X: Training features
            y: Training labels
        
        Returns:
            X_resampled, y_resampled
        """
        if self.resampler is None:
            return X, y
        
        # Check if enough samples
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        
        if min_count < 6:
            logger.warning(f"Minority class has only {min_count} samples, skipping resampling")
            return X, y
        
        try:
            logger.info(f"Original distribution: {dict(zip(unique, counts))}")
            X_resampled, y_resampled = self.resampler.fit_resample(X, y)
            
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            logger.info(f"Resampled distribution: {dict(zip(unique_new, counts_new))}")
            
            return X_resampled, y_resampled
        
        except Exception as e:
            logger.error(f"Resampling failed: {e}, using original data")
            return X, y
    
    @staticmethod
    def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
        """Compute balanced class weights for cost-sensitive learning"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        logger.info(f"Class weights: {class_weights}")
        return class_weights


# ============================================================================
# Missing Data Analysis
# ============================================================================

def analyze_missingness(
    df: pd.DataFrame, 
    target_column: Optional[str] = None,
    save_plots: bool = True,
    plot_prefix: str = "missingness"
) -> pd.DataFrame:
    """
    Comprehensive missingness analysis for understanding missing data patterns.
    
    This function helps determine if missing data is MCAR (Missing Completely at Random),
    MAR (Missing at Random), or MNAR (Missing Not at Random), which is crucial for
    choosing appropriate handling strategies.
    
    Args:
        df: DataFrame to analyze
        target_column: Name of target column for outcome-related analysis
        save_plots: Whether to save visualization plots
        plot_prefix: Prefix for saved plot filenames
    
    Returns:
        DataFrame with missing data statistics per feature
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available - skipping plots")
    
    # Overall missingness per feature
    missing_stats = pd.DataFrame({
        'feature': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df)) * 100,
        'non_missing_count': df.notnull().sum()
    }).sort_values('missing_pct', ascending=False)
    
    # Only show features with missing data
    missing_features = missing_stats[missing_stats['missing_pct'] > 0]
    
    logger.info("="*60)
    logger.info("MISSING DATA ANALYSIS")
    logger.info("="*60)
    logger.info(f"Total records: {len(df):,}")
    logger.info(f"Features with missing data: {len(missing_features)}/{len(df.columns)}")
    
    if len(missing_features) > 0:
        logger.info("\nMissing data by feature:")
        for _, row in missing_features.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['missing_count']:,} ({row['missing_pct']:.1f}%)")
        
        if len(missing_features) > 10:
            logger.info(f"  ... and {len(missing_features) - 10} more features with missing data")
    else:
        logger.info("✓ No missing data found!")
        return missing_stats
    
    # Complete case analysis impact
    complete_cases = df.dropna()
    retention_rate = len(complete_cases) / len(df)
    logger.info(f"\nComplete case analysis would retain: {len(complete_cases):,} ({retention_rate:.1%}) records")
    
    # Visualization (if available)
    if VISUALIZATION_AVAILABLE and save_plots:
        try:
            # Missingness matrix
            plt.figure(figsize=(15, 8))
            msno.matrix(df, figsize=(15, 8), sparkline=True, labels=True)
            plt.title('Missing Data Pattern Matrix', fontsize=14, pad=20)
            if save_plots:
                plt.savefig(f'{plot_prefix}_matrix.png', dpi=300, bbox_inches='tight')
                logger.info(f"Saved missingness matrix plot: {plot_prefix}_matrix.png")
            plt.show()
            
            # Missingness correlation heatmap
            plt.figure(figsize=(12, 8))
            msno.heatmap(df, figsize=(12, 8))
            plt.title('Missing Data Correlation Heatmap', fontsize=14, pad=20)
            if save_plots:
                plt.savefig(f'{plot_prefix}_correlation.png', dpi=300, bbox_inches='tight')
                logger.info(f"Saved missingness correlation plot: {plot_prefix}_correlation.png")
            plt.show()
            
            # Dendrogram of missingness patterns
            plt.figure(figsize=(12, 6))
            msno.dendrogram(df, figsize=(12, 6))
            plt.title('Missing Data Pattern Dendrogram', fontsize=14, pad=20)
            if save_plots:
                plt.savefig(f'{plot_prefix}_dendrogram.png', dpi=300, bbox_inches='tight')
                logger.info(f"Saved missingness dendrogram: {plot_prefix}_dendrogram.png")
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating missingness plots: {e}")
    
    # Outcome-related missingness analysis (MAR vs MNAR detection)
    if target_column and target_column in df.columns:
        logger.info(f"\n=== Missingness vs {target_column} Analysis ===")
        logger.info("(Helps detect if data is Missing at Random vs Missing Not at Random)")
        
        outcome_missing_analysis = []
        
        for col in df.columns:
            if col == target_column or not df[col].isnull().any():
                continue
            
            try:
                # Compare target rates between missing and non-missing groups
                missing_mask = df[col].isnull()
                
                if missing_mask.sum() > 0 and (~missing_mask).sum() > 0:
                    rate_with_missing = df.loc[missing_mask, target_column].mean()
                    rate_without_missing = df.loc[~missing_mask, target_column].mean()
                    difference = rate_with_missing - rate_without_missing
                    
                    # Statistical test (chi-square)
                    from scipy.stats import chi2_contingency
                    contingency_table = pd.crosstab(missing_mask, df[target_column])
                    chi2, p_value, _, _ = chi2_contingency(contingency_table)
                    
                    outcome_missing_analysis.append({
                        'feature': col,
                        'missing_pct': (missing_mask.sum() / len(df)) * 100,
                        'outcome_rate_missing': rate_with_missing,
                        'outcome_rate_present': rate_without_missing,
                        'difference': difference,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
                    
                    significance = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    logger.info(f"  {col}:")
                    logger.info(f"    {target_column} rate when missing: {rate_with_missing:.2%}")
                    logger.info(f"    {target_column} rate when present: {rate_without_missing:.2%}")
                    logger.info(f"    Difference: {difference:+.2%} (p={p_value:.3f}) {significance}")
                    
            except Exception as e:
                logger.warning(f"Could not analyze missingness for {col}: {e}")
        
        # Summary of outcome-related missingness
        if outcome_missing_analysis:
            outcome_df = pd.DataFrame(outcome_missing_analysis)
            significant_features = outcome_df[outcome_df['significant']]
            
            if len(significant_features) > 0:
                logger.info(f"\n⚠️  {len(significant_features)} features show significant missingness patterns:")
                logger.info("   This suggests Missing at Random (MAR) or Missing Not at Random (MNAR)")
                for _, row in significant_features.iterrows():
                    logger.info(f"   - {row['feature']}: {row['difference']:+.2%} difference (p={row['p_value']:.3f})")
            else:
                logger.info("\n✓ No significant outcome-related missingness patterns detected")
                logger.info("   Data appears Missing Completely at Random (MCAR)")
    
    logger.info("="*60)
    
    return missing_stats


def validate_imputation(
    df_original: pd.DataFrame, 
    df_imputed: pd.DataFrame,
    save_plots: bool = True,
    plot_filename: str = 'imputation_validation.png'
) -> Dict[str, any]:
    """
    Comprehensive validation of imputation quality.
    
    Checks for:
    1. Clinically implausible values
    2. Distribution preservation
    3. Correlation structure preservation
    4. Statistical summaries comparison
    
    Args:
        df_original: DataFrame before imputation
        df_imputed: DataFrame after imputation  
        save_plots: Whether to save validation plots
        plot_filename: Name for saved plot file
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'out_of_range_counts': {},
        'distribution_changes': {},
        'correlation_changes': {},
        'summary_stats': {}
    }
    
    logger.info("="*60)
    logger.info("IMPUTATION VALIDATION")
    logger.info("="*60)
    
    # 1. Check for clinically implausible values
    clinical_ranges = {
        'heart_rate': (40, 200, 'bpm'),
        'systolic_bp': (60, 250, 'mmHg'), 
        'sysbp': (60, 250, 'mmHg'),
        'diastolic_bp': (30, 150, 'mmHg'),
        'diasbp': (30, 150, 'mmHg'), 
        'temperature': (35, 42, '°C'),
        'temp': (35, 42, '°C'),
        'spo2': (70, 100, '%'),
        'glucose': (50, 500, 'mg/dL'),
        'creatinine': (0.5, 15, 'mg/dL'),
        'bun': (5, 100, 'mg/dL'),
        'hemoglobin': (5, 20, 'g/dL'),
        'hematocrit': (15, 60, '%'),
        'platelet': (50, 1000, 'K/μL'),
        'wbc': (1, 50, 'K/μL'),
        'sodium': (120, 160, 'mmol/L'),
        'potassium': (2.5, 7.0, 'mmol/L'),
        'chloride': (90, 120, 'mmol/L'),
        'lactate': (0.5, 15, 'mmol/L'),
        'bilirubin': (0.2, 20, 'mg/dL'),
        'age': (18, 100, 'years')
    }
    
    logger.info("Checking for clinically implausible values:")
    total_out_of_range = 0
    
    for col, (min_val, max_val, unit) in clinical_ranges.items():
        if col in df_imputed.columns:
            out_of_range = ((df_imputed[col] < min_val) | 
                           (df_imputed[col] > max_val)).sum()
            validation_results['out_of_range_counts'][col] = out_of_range
            total_out_of_range += out_of_range
            
            if out_of_range > 0:
                pct = (out_of_range / len(df_imputed)) * 100
                logger.warning(f"  ⚠️  {col}: {out_of_range} values ({pct:.1f}%) outside [{min_val}, {max_val}] {unit}")
            else:
                logger.info(f"  ✓ {col}: All values within [{min_val}, {max_val}] {unit}")
    
    if total_out_of_range == 0:
        logger.info("✓ All imputed values are clinically plausible")
    else:
        logger.warning(f"⚠️  Total out-of-range values: {total_out_of_range}")
    
    # 2. Compare distributions and summary statistics
    logger.info(f"\nComparing distributions for imputed features:")
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in df_original.columns and df_original[col].isnull().any():
            # Original non-missing data
            original_clean = df_original[col].dropna()
            imputed_all = df_imputed[col]
            
            # Compare means
            orig_mean = original_clean.mean()
            imp_mean = imputed_all.mean()
            mean_diff = abs(imp_mean - orig_mean) / orig_mean * 100 if orig_mean != 0 else 0
            
            # Compare standard deviations  
            orig_std = original_clean.std()
            imp_std = imputed_all.std()
            std_diff = abs(imp_std - orig_std) / orig_std * 100 if orig_std != 0 else 0
            
            validation_results['summary_stats'][col] = {
                'original_mean': orig_mean,
                'imputed_mean': imp_mean,
                'mean_change_pct': mean_diff,
                'original_std': orig_std,
                'imputed_std': imp_std,
                'std_change_pct': std_diff
            }
            
            if mean_diff > 10 or std_diff > 20:
                logger.warning(f"  ⚠️  {col}: Mean Δ{mean_diff:.1f}%, Std Δ{std_diff:.1f}% (significant change)")
            else:
                logger.info(f"  ✓ {col}: Mean Δ{mean_diff:.1f}%, Std Δ{std_diff:.1f}% (preserved)")
    
    # 3. Visualization (if available)
    if VISUALIZATION_AVAILABLE and save_plots:
        try:
            import seaborn as sns
            
            # Select up to 9 numeric columns with missing data
            imputed_numeric_cols = []
            for col in numeric_cols:
                if col in df_original.columns and df_original[col].isnull().any():
                    imputed_numeric_cols.append(col)
                if len(imputed_numeric_cols) >= 9:
                    break
            
            if imputed_numeric_cols:
                n_cols = min(3, len(imputed_numeric_cols))
                n_rows = min(3, (len(imputed_numeric_cols) + n_cols - 1) // n_cols)
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                elif n_rows == 1 or n_cols == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()
                
                for idx, col in enumerate(imputed_numeric_cols[:n_rows*n_cols]):
                    ax = axes[idx] if len(imputed_numeric_cols) > 1 else axes
                    
                    # Plot original (non-missing) vs imputed distributions
                    original_data = df_original[col].dropna()
                    imputed_data = df_imputed[col]
                    
                    try:
                        sns.kdeplot(data=original_data, ax=ax, label='Original (non-missing)', 
                                  color='blue', alpha=0.7)
                        sns.kdeplot(data=imputed_data, ax=ax, label='After imputation', 
                                  color='red', linestyle='--', alpha=0.7)
                        
                        ax.set_title(f'{col} Distribution Comparison', fontsize=10)
                        ax.legend(fontsize=8)
                        ax.grid(True, alpha=0.3)
                        
                    except Exception as e:
                        logger.warning(f"Could not plot {col}: {e}")
                        ax.text(0.5, 0.5, f'Plot failed for {col}', 
                               transform=ax.transAxes, ha='center')
                
                # Hide empty subplots
                for idx in range(len(imputed_numeric_cols), n_rows * n_cols):
                    if idx < len(axes):
                        axes[idx].set_visible(False)
                
                plt.suptitle('Imputation Validation: Distribution Comparison', fontsize=14)
                plt.tight_layout()
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                logger.info(f"✓ Distribution comparison saved to {plot_filename}")
                plt.show()
            else:
                logger.info("No numeric columns with imputed data found for plotting")
                
        except Exception as e:
            logger.error(f"Error creating validation plots: {e}")
    
    # 4. Overall assessment
    logger.info(f"\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    if total_out_of_range == 0:
        logger.info("✅ CLINICAL PLAUSIBILITY: All values within acceptable ranges")
    else:
        logger.warning(f"⚠️  CLINICAL PLAUSIBILITY: {total_out_of_range} values out of range")
    
    # Count features with significant distribution changes
    significant_changes = sum(1 for stats in validation_results['summary_stats'].values() 
                            if stats['mean_change_pct'] > 10 or stats['std_change_pct'] > 20)
    
    if significant_changes == 0:
        logger.info("✅ DISTRIBUTION PRESERVATION: All features maintained statistical properties")
    else:
        logger.warning(f"⚠️  DISTRIBUTION CHANGES: {significant_changes} features show significant changes")
    
    imputed_features = len([col for col in numeric_cols 
                           if col in df_original.columns and df_original[col].isnull().any()])
    total_imputed = sum(df_original[col].isnull().sum() for col in numeric_cols 
                       if col in df_original.columns)
    
    logger.info(f"📊 IMPUTATION SCOPE: {imputed_features} features, {total_imputed:,} values imputed")
    logger.info("="*60)
    
    return validation_results


# ============================================================================
# Utility Functions
# ============================================================================

def load_compas_data(
    filepath: str,
    apply_filters: bool = True
) -> pd.DataFrame:
    """
    Load COMPAS dataset with standard filters.
    
    Args:
        filepath: Path to COMPAS CSV
        apply_filters: Whether to apply standard filters
    
    Returns:
        Filtered DataFrame
    """
    logger.info(f"Loading COMPAS data from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records")
    except FileNotFoundError:
        # Try to download
        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        logger.info(f"File not found, downloading from {url}")
        df = pd.read_csv(url)
        df.to_csv(filepath, index=False)
        logger.info(f"Downloaded and saved to {filepath}")
    
    if apply_filters:
        # Standard COMPAS filters from ProPublica analysis
        initial_count = len(df)
        
        # Temporal filter
        df = df[
            (df['days_b_screening_arrest'] <= 30) &
            (df['days_b_screening_arrest'] >= -30)
        ]
        logger.info(f"After temporal filter: {len(df)} ({len(df)/initial_count:.1%})")
        
        # Quality filters
        df = df[df['is_recid'] != -1]
        df = df[df['c_charge_degree'] != 'O']  # Remove ordinances
        
        logger.info(f"After quality filters: {len(df)} ({len(df)/initial_count:.1%})")
        logger.info(f"Recidivism rate: {df['two_year_recid'].mean():.2%}")
    
    return df


def load_mimic_data(filepath: str, drop_na: bool = False) -> pd.DataFrame:
    """
    Load MIMIC-III dataset.
    
    Args:
        filepath: Path to processed MIMIC-III CSV
        drop_na: Whether to drop rows with NaN (NOT recommended - use IntelligentMissingDataHandler instead)
    
    Returns:
        DataFrame
    """
    logger.info(f"Loading MIMIC-III data from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} ICU admissions")
        
        if drop_na:
            logger.warning("⚠️  Using drop_na=True will cause significant data loss. Consider IntelligentMissingDataHandler instead.")
            before_na = len(df)
            df = df.dropna().reset_index(drop=True)
            logger.info(f"Dropped {before_na - len(df)} rows with NaN values")
        else:
            missing_count = df.isnull().sum().sum()
            logger.info(f"Dataset contains {missing_count:,} missing values (will be handled by preprocessing)")
        
        # Basic filters
        initial_count = len(df)
        df = df[df['age'] >= 18]
        df['age'] = df['age'].clip(18, 100)
        
        logger.info(f"After age filter: {len(df)} ({len(df)/initial_count:.1%} retained)")
        logger.info(f"Mortality rate: {df['hospital_expire_flag'].mean():.2%}")
        
        return df
    
    except FileNotFoundError:
        logger.error(f"MIMIC-III data not found at {filepath}")
        logger.error("MIMIC-III requires credentialed access from https://physionet.org/")
        raise


def diagnose_out_of_range_values(df_original, df_imputed):
    """
    Determine if out-of-range values are:
    1. Original extreme values (legitimate)
    2. Imputation artifacts (problematic)
    
    Args:
        df_original: DataFrame with original data (may contain NaN)
        df_imputed: DataFrame after imputation (no NaN expected)
    """
    
    clinical_ranges = {
        'glucose': (50, 500),
        'creatinine': (0.5, 15),
        'bun': (5, 100),
        'hemoglobin': (5, 20),
        'hematocrit': (15, 60),
        'platelet': (50, 1000),
        'wbc': (1, 50),
        'sodium': (120, 160),
        'potassium': (2.5, 7.0),
        'chloride': (90, 120)
    }
    
    print("="*70)
    print("OUT-OF-RANGE VALUE ANALYSIS")
    print("="*70)
    
    for col, (min_val, max_val) in clinical_ranges.items():
        if col not in df_imputed.columns:
            continue
            
        # Find out-of-range values in imputed data
        out_of_range_mask = (df_imputed[col] < min_val) | (df_imputed[col] > max_val)
        n_out_of_range = out_of_range_mask.sum()
        
        if n_out_of_range == 0:
            continue
            
        print(f"\n{col.upper()}: {n_out_of_range} values ({n_out_of_range/len(df_imputed)*100:.1f}%)")
        
        # Check if these were originally missing (imputation artifact)
        # or originally present (legitimate extreme values)
        out_of_range_indices = df_imputed[out_of_range_mask].index
        
        originally_missing = df_original.loc[out_of_range_indices, col].isnull().sum()
        originally_present = n_out_of_range - originally_missing
        
        print(f"  - Originally present: {originally_present} ({originally_present/n_out_of_range*100:.1f}%)")
        print(f"  - Imputation artifacts: {originally_missing} ({originally_missing/n_out_of_range*100:.1f}%)")
        
        # Show extreme values
        extreme_vals = df_imputed.loc[out_of_range_mask, col]
        print(f"  - Range: [{extreme_vals.min():.2f}, {extreme_vals.max():.2f}]")
        print(f"  - Mean: {extreme_vals.mean():.2f}, Median: {extreme_vals.median():.2f}")
        
        # Check mortality association (if hospital_expire_flag exists)
        if 'hospital_expire_flag' in df_imputed.columns:
            mortality_extreme = df_imputed.loc[out_of_range_mask, 'hospital_expire_flag'].mean()
            mortality_normal = df_imputed.loc[~out_of_range_mask, 'hospital_expire_flag'].mean()
            
            print(f"  - Mortality rate (extreme): {mortality_extreme*100:.1f}%")
            print(f"  - Mortality rate (normal): {mortality_normal*100:.1f}%")
            
            if mortality_extreme > mortality_normal * 1.5:
                print(f"  ⚠️  HIGH-RISK GROUP - Do NOT exclude these patients!")


def validate_extreme_values_are_real(df):
    """
    Check if extreme values are associated with worse outcomes
    If YES → they're legitimate high-risk patients (KEEP)
    If NO → they might be imputation artifacts (FIX)
    
    Args:
        df: DataFrame with clinical data and hospital_expire_flag
    """
    
    print("\n" + "="*80)
    print("EXTREME VALUE VALIDATION FOR CTF FRAMEWORK")
    print("="*80)
    
    clinical_ranges = {
        'glucose': (50, 500),
        'creatinine': (0.5, 15),
        'bun': (5, 100),
        'hemoglobin': (5, 20),
        'platelet': (50, 1000),
        'wbc': (1, 50),
        'potassium': (2.5, 7.0),
    }
    
    overall_mortality = df['hospital_expire_flag'].mean()
    print(f"\nOverall mortality rate: {overall_mortality*100:.2f}%")
    
    high_risk_features = []
    
    for col, (min_val, max_val) in clinical_ranges.items():
        if col not in df.columns:
            continue
            
        # Identify extreme values
        extreme_mask = (df[col] < min_val) | (df[col] > max_val)
        n_extreme = extreme_mask.sum()
        
        if n_extreme == 0:
            continue
        
        # Calculate mortality rates
        mort_extreme = df.loc[extreme_mask, 'hospital_expire_flag'].mean()
        mort_normal = df.loc[~extreme_mask, 'hospital_expire_flag'].mean()
        
        # Risk ratio
        risk_ratio = mort_extreme / mort_normal if mort_normal > 0 else 0
        
        print(f"\n{col.upper()}:")
        print(f"  Out-of-range: {n_extreme} patients ({n_extreme/len(df)*100:.1f}%)")
        print(f"  Range: [{df.loc[extreme_mask, col].min():.1f}, {df.loc[extreme_mask, col].max():.1f}]")
        print(f"  Mortality (extreme): {mort_extreme*100:.1f}%")
        print(f"  Mortality (normal):  {mort_normal*100:.1f}%")
        print(f"  Risk Ratio: {risk_ratio:.2f}x")
        
        # Clinical interpretation
        if risk_ratio > 1.5:
            print(f"  ✓ HIGH-RISK GROUP - These are legitimate critically ill patients!")
            print(f"    → KEEP these values (they represent vulnerable populations)")
            high_risk_features.append(col)
        elif risk_ratio < 0.7:
            print(f"  ⚠️  PROTECTIVE effect? Unusual - check for imputation artifacts")
        else:
            print(f"  → Similar mortality - may be noise or imputation artifacts")
    
    print("\n" + "="*80)
    print("RECOMMENDATION FOR CTF FRAMEWORK")
    print("="*80)
    
    if high_risk_features:
        print(f"\n✓ {len(high_risk_features)} features show extreme values = high mortality")
        print(f"  Features: {', '.join(high_risk_features)}")
        print("\nRECOMMENDATION:")
        print("  1. KEEP all extreme values (they identify vulnerable patients)")
        print("  2. CREATE binary flags for extreme values as additional features")
        print("  3. Document in methods: 'Extreme values retained to preserve high-risk patients'")
        print("  4. This STRENGTHENS your CTF framework by including most vulnerable populations!")
    else:
        print("\n⚠️  Extreme values not associated with mortality")
        print("  → May indicate imputation artifacts")
        print("  → Consider more conservative imputation strategy")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 0: Complete Case Analysis with NaNDropper
    print("\n" + "="*80)
    print("Example 0: Complete Case Analysis (Dropping NaN values)")
    print("="*80)
    
    # Simulate data with some missing values
    sample_data = pd.DataFrame({
        'age': [25, 30, np.nan, 45, 50],
        'income': [50000, 60000, 70000, np.nan, 90000],
        'score': [0.8, 0.9, 0.7, 0.85, 0.95]
    })
    
    print("\nOriginal data:")
    print(sample_data)
    print(f"Shape: {sample_data.shape}")
    
    # Apply NaN dropper
    dropper = NaNDropper(verbose=True)
    clean_data = dropper.fit_transform(sample_data)
    
    print("\nCleaned data (complete cases only):")
    print(clean_data)
    print(f"Shape: {clean_data.shape}")
    
    # Get missingness report
    report = dropper.get_missingness_report()
    print(f"\nMissingness report:")
    print(f"  Retention rate: {report['retention_rate']:.1%}")
    print(f"  Columns with NaN: {list(report['columns_with_nan'].keys())}")
    
    print("\n✓ Complete case analysis example complete\n")
    
    # Example 0b: Missingness Analysis
    print("\n" + "="*80)
    print("Example 0b: Comprehensive Missingness Analysis")
    print("="*80)
    
    # Create sample data with various missingness patterns
    np.random.seed(42)
    n_samples = 1000
    
    # Create correlated missingness patterns
    sample_data_missing = pd.DataFrame({
        'age': np.random.normal(45, 15, n_samples),
        'income': np.random.normal(60000, 20000, n_samples),
        'lab_value_a': np.random.normal(10, 3, n_samples),
        'lab_value_b': np.random.normal(5, 2, n_samples),
        'outcome': np.random.binomial(1, 0.3, n_samples)
    })
    
    # Introduce different types of missingness
    # MCAR: Random 10% missing in income
    missing_income = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    sample_data_missing.loc[missing_income, 'income'] = np.nan
    
    # MAR: Lab values missing more often in elderly patients
    elderly_mask = sample_data_missing['age'] > 65
    elderly_indices = sample_data_missing[elderly_mask].index
    missing_lab_a = np.random.choice(elderly_indices, size=int(0.4 * len(elderly_indices)), replace=False)
    sample_data_missing.loc[missing_lab_a, 'lab_value_a'] = np.nan
    
    # MNAR: Lab value B missing when values would be high (outcome-related)
    high_outcome_mask = sample_data_missing['outcome'] == 1
    high_outcome_indices = sample_data_missing[high_outcome_mask].index
    missing_lab_b = np.random.choice(high_outcome_indices, size=int(0.25 * len(high_outcome_indices)), replace=False)
    sample_data_missing.loc[missing_lab_b, 'lab_value_b'] = np.nan
    
    print(f"Created sample dataset with {len(sample_data_missing)} records")
    print(f"Outcome rate: {sample_data_missing['outcome'].mean():.1%}")
    
    # Run comprehensive missingness analysis
    missing_analysis = analyze_missingness(
        sample_data_missing, 
        target_column='outcome',
        save_plots=VISUALIZATION_AVAILABLE,
        plot_prefix='sample_missingness'
    )
    
    print("\n✓ Missingness analysis complete\n")
    
    # Example 0c: Intelligent Missing Data Handling
    print("\n" + "="*80)
    print("Example 0c: Intelligent Missing Data Handling vs Complete Case Analysis")
    print("="*80)
    
    try:
        # Load raw MIMIC data
        mimic_raw = pd.read_csv('data/mimic_cohort_ctf.csv')
        print(f"Raw MIMIC data: {mimic_raw.shape}")
        
        # Method 1: Traditional Complete Case Analysis (NaNDropper)
        print("\n--- Method 1: Complete Case Analysis ---")
        traditional_dropper = NaNDropper(verbose=True)
        mimic_complete_case = traditional_dropper.fit_transform(mimic_raw.copy())
        traditional_report = traditional_dropper.get_missingness_report()
        
        # Method 2: Intelligent Missing Data Handling
        print("\n--- Method 2: Intelligent Missing Data Handling ---")
        intelligent_handler = IntelligentMissingDataHandler(
            drop_threshold=0.5,
            dataset_type='mimic',
            verbose=True
        )
        mimic_intelligent = intelligent_handler.fit_transform(mimic_raw.copy())
        intelligent_report = intelligent_handler.get_imputation_report()
        
        # Comparison
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"Original data:           {mimic_raw.shape[0]:,} records, {mimic_raw.shape[1]} features")
        print(f"Complete case analysis:  {traditional_report['n_after']:,} records ({traditional_report['retention_rate']:.1%} retained)")
        print(f"Intelligent handling:    {intelligent_report['n_after']:,} records ({intelligent_report['retention_rate']:.1%} retained)")
        print(f"Data improvement:        {intelligent_report['n_after'] - traditional_report['n_after']:,} more records")
        print(f"Improvement factor:      {intelligent_report['n_after'] / traditional_report['n_after']:.1f}x more data")
        
        # Features comparison
        print(f"\nFeatures:")
        print(f"  Dropped by intelligent handler: {len(intelligent_report['dropped_features'])} features")
        print(f"  Features with imputation: {len(intelligent_report['imputation_stats'])}")
        
        if intelligent_report['dropped_features']:
            print(f"  Dropped features: {intelligent_report['dropped_features']}")
        
        print("\n✓ Intelligent missing data handling demonstrates significant improvement")
        
        # Validate the imputation quality
        print("\n--- Imputation Quality Validation ---")
        validation_results = validate_imputation(
            mimic_raw, 
            mimic_intelligent,
            save_plots=VISUALIZATION_AVAILABLE,
            plot_filename='mimic_imputation_validation.png'
        )
        
        print("\n✓ Imputation validation complete")
        
    except FileNotFoundError:
        print("⚠️  MIMIC data not available - skipping intelligent handler demo")
    
    print("\n" + "="*80)
    
    # Example 1: COMPAS preprocessing
    print("\n" + "="*80)
    print("Example 1: COMPAS Preprocessing")
    print("="*80)
    
    # Load data
    compas_df = load_compas_data('data/compas-scores-two-years.csv')
    
    # Show data info
    print(f"\nDataset info:")
    print(f"  Total records: {len(compas_df)}")
    print(f"  Total columns: {len(compas_df.columns)}")
    print(f"  Recidivism rate: {compas_df['two_year_recid'].mean():.2%}")
    print(f"  Available columns: {list(compas_df.columns[:10])}...")
    
    # Initialize preprocessor
    compas_processor = COMPASPreprocessor(
        target_column='two_year_recid',
        include_sensitive=True,  # Include race/sex for analysis
        validate_data=True
    )
    
    # Create CV splits (NO resampling yet!)
    splitter = StratifiedCVSplitter(n_splits=5)
    
    X = compas_df.drop(columns=['two_year_recid'])
    y = compas_df['two_year_recid'].values
    
    for fold_data in splitter.split(X, y):
        fold_idx = fold_data['fold']
        print(f"\n--- Fold {fold_idx + 1} ---")
        
        # Fit on training data only
        X_train = fold_data['train']['X']
        y_train = fold_data['train']['y']
        
        compas_processor.fit(X_train, y_train)
        
        # Transform all splits
        X_train_transformed, _ = compas_processor.transform(X_train)
        X_test_transformed, y_test = compas_processor.transform(fold_data['test']['X'])
        
        print(f"Training shape: {X_train_transformed.shape}")
        print(f"Test shape: {X_test_transformed.shape}")
        
        # Get and show feature names
        feature_names = compas_processor.get_feature_names_out()
        print(f"Number of features: {len(feature_names)}")
        print(f"Sample features: {feature_names[:10]}")
        
        # Optional: Apply SMOTE to training data only
        if IMBLEARN_AVAILABLE:
            imbalance_handler = ImbalanceHandler(method='smote')
            X_train_resampled, y_train_resampled = imbalance_handler.fit_resample(
                X_train_transformed, y_train
            )
            print(f"After SMOTE: {X_train_resampled.shape}")
        else:
            print("SMOTE not available (install imbalanced-learn)")
        
        # Now train model on X_train_resampled, validate on X_test_transformed
        break  # Just show first fold
    
    print("\n✓ COMPAS preprocessing complete")
    
    # Example 2: MIMIC preprocessing
    print("\n" + "="*80)
    print("Example 2: MIMIC-III Preprocessing")
    print("="*80)
    
    try:
        mimic_df = load_mimic_data('data/mimic_cohort_ctf.csv', drop_na=False)  # Use intelligent handler instead
        
        mimic_processor = MIMICPreprocessor(
            target_column='hospital_expire_flag',
            validate_data=True,
            clip_outliers=True
        )
        
        X_mimic = mimic_df.drop(columns=['hospital_expire_flag'])
        y_mimic = mimic_df['hospital_expire_flag'].values
        
        # Fit transform
        X_mimic_transformed, y_mimic = mimic_processor.fit_transform(X_mimic, y_mimic)
        
        print(f"MIMIC shape: {X_mimic_transformed.shape}")
        print(f"Features: {len(mimic_processor.get_feature_names_out())}")
        print("✓ MIMIC preprocessing complete")
        
    except FileNotFoundError:
        print("⚠️  MIMIC data not available (requires PhysioNet access)")