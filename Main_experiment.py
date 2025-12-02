#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main CTF Experiment Pipeline.

Implements complete nested cross-validation methodology:
1. Three-way data split (test, causal discovery, training)
2. Ensemble causal discovery (PC, GES, NOTEARS)
3. Model training (LR, RF, XGB, DNN)
4. CTF metric evaluation (CII, CCM, TE, CS)
5. Comparative analysis (CII vs SHAP/LIME)

Follows methodology from: "The Causal Transparency Framework: 
A Multi-Metric Approach to Algorithmic Accountability"

Author: John Marko
Date: 2025-11-18
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing
import sys
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import (
    load_compas_data,
    load_mimic_data,
    COMPASPreprocessor,
    MIMICPreprocessor,
    StratifiedCVSplitter
)

# Import our modules
from models import ModelFactory, evaluate_model
from causal_discovery import EnsembleCausalDiscovery, save_graph, visualize_graph
from Ctf_metrics import CTFFramework
from comparative_analysis import (
    SHAPExplainer,
    LIMEExplainer,
    ComparativeAnalysis,
    SHAP_AVAILABLE,
    LIME_AVAILABLE
)
from quick_diagnostic import quick_fold_diagnostic, compare_folds
from visualization import PublicationGraphVisualizer, visualize_causal_graph
from roc_visualization import ROCVisualizer
from domain_constraints import apply_domain_constraints, get_tier_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ctf_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def robust_data_cleaning(X_data, feature_names=None, use_median=True):
    """
    Robust data preparation that handles edge cases for any data.
    
    Args:
        X_data: Input data (numpy array or pandas DataFrame)
        feature_names: Feature names (optional)
        use_median: Use median imputation instead of mean
        
    Returns:
        Clean pandas DataFrame or numpy array
    """
    import pandas as pd
    import numpy as np
    
    # Convert to DataFrame if needed
    if isinstance(X_data, np.ndarray):
        if feature_names is not None:
            df = pd.DataFrame(X_data, columns=feature_names)
            return_df = True
        else:
            df = pd.DataFrame(X_data)
            return_df = False
    else:
        df = X_data.copy()
        return_df = True
    
    # Step 1: Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Step 2: Robust imputation
    for col in df.columns:
        if df[col].isna().any():
            if use_median:
                fill_val = df[col].median()
            else:
                fill_val = df[col].mean()
                
            if pd.isna(fill_val):
                fill_val = 0  # Fallback for entirely NaN columns
                
            df[col] = df[col].fillna(fill_val)
    
    # Step 3: Ensure numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Step 4: Final NaN check and fill
    if df.isna().any().any():
        logger.warning("Found NaNs after numeric conversion - filling with 0")
        df = df.fillna(0)
    
    # Step 5: Verify clean data
    assert not df.isna().any().any(), "NaN values still present after robust cleaning"
    
    return df if return_df else df.values


def prepare_data_for_causal_discovery(X_causal, feature_names):
    """
    Robust data preparation that handles edge cases.
    """
    return robust_data_cleaning(X_causal, feature_names, use_median=True)


def evaluate_model_with_consistent_threshold(model, X_test, y_test, model_name):
    """
    Balanced evaluation using F1-optimal threshold.
    
    F1 gives equal weight to precision and recall, providing a better
    balance for clinical deployment where alert fatigue is a concern.
    
    Also enforces minimum precision floor to prevent extreme recall-only solutions.
    """
    from sklearn.metrics import precision_recall_curve, fbeta_score, accuracy_score
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Get precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Remove last element (precision=1, recall=0 at threshold > max(proba))
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    
    # Compute F1 scores for each threshold (BALANCED approach - not F2!)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Find F1-optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Enforce minimum precision floor (15%) to avoid alert fatigue
    min_precision = 0.15
    if precisions[optimal_idx] < min_precision:
        # Find threshold that gives at least min_precision
        valid_idx = np.where(precisions >= min_precision)[0]
        if len(valid_idx) > 0:
            # Among those meeting precision floor, find highest F1
            best_f1_idx = valid_idx[np.argmax(f1_scores[valid_idx])]
            optimal_idx = best_f1_idx
            optimal_threshold = thresholds[optimal_idx]
            logger.info(f"Precision floor ({min_precision}) applied")
    
    # Apply threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'f2': fbeta_score(y_test, y_pred, beta=2.0, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'log_loss': log_loss(y_test, y_pred_proba),
        'threshold': float(optimal_threshold)
    }
    
    # Log performance with clinical interpretation
    logger.info(f"\n{model_name} Performance (F1-optimal):")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"  Threshold: {optimal_threshold:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.4f} (positive predictive value)")
    logger.info(f"  Recall:    {metrics['recall']:.4f} (sensitivity)")
    logger.info(f"  F1-Score:  {metrics['f1']:.4f} ← BALANCED METRIC")
    logger.info(f"  F2-Score:  {metrics['f2']:.4f} (recall-weighted)")
    
    # Clinical interpretation
    alert_rate = y_pred.mean()
    logger.info(f"  Alert Rate: {alert_rate:.1%} of patients flagged as high-risk")
    
    return {
        'optimal_threshold': optimal_threshold,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'metrics': metrics
    }


# ============================================================================
# Experiment Configuration
# ============================================================================

class ExperimentConfig:
    """Configuration for CTF experiments."""
    
    def __init__(
        self,
        dataset: str = 'compas',
        models: List[str] = ['lr', 'rf', 'xgb', 'dnn'],
        n_folds: int = 5,
        causal_discovery_ratio: float = 0.6,
        test_ratio: float = 0.2,
        hyperopt_iterations: int = 50,
        n_perturbations: int = 50,
        random_state: int = 42,
        output_dir: str = 'results',
        apply_domain_constraints: bool = False
    ):
        """
        Initialize experiment configuration.
        
        Args:
            dataset: 'compas' or 'mimic'
            models: List of model types to train
            n_folds: Number of CV folds
            causal_discovery_ratio: Ratio of data for causal discovery
            test_ratio: Ratio for test set
            hyperopt_iterations: Iterations for hyperparameter optimization
            n_perturbations: Perturbations for counterfactual stability
            random_state: Random seed
            output_dir: Output directory
            apply_domain_constraints: Whether to apply domain knowledge constraints to causal graphs
        """
        self.dataset = dataset
        self.models = models
        self.n_folds = n_folds
        self.causal_discovery_ratio = causal_discovery_ratio
        self.test_ratio = test_ratio
        self.hyperopt_iterations = hyperopt_iterations
        self.n_perturbations = n_perturbations
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.apply_domain_constraints = apply_domain_constraints
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / 'graphs').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'dataset': self.dataset,
            'models': self.models,
            'n_folds': self.n_folds,
            'causal_discovery_ratio': self.causal_discovery_ratio,
            'test_ratio': self.test_ratio,
            'hyperopt_iterations': self.hyperopt_iterations,
            'n_perturbations': self.n_perturbations,
            'random_state': self.random_state,
            'apply_domain_constraints': self.apply_domain_constraints
        }


# ============================================================================
# Main Experiment Pipeline
# ============================================================================

class CTFExperiment:
    """Complete CTF experiment pipeline."""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results = {
            'config': config.to_dict(),
            'folds': [],
            'summary': {}
        }
        
        # Storage for diagnostic analysis
        self.fold_diagnostics = {}
        
        # Initialize ROC visualizer
        self.roc_viz = ROCVisualizer()
        
        logger.info("="*80)
        logger.info("CTF EXPERIMENT INITIALIZED")
        logger.info("="*80)
        logger.info(f"Dataset: {config.dataset}")
        logger.info(f"Models: {config.models}")
        logger.info(f"Folds: {config.n_folds}")
        logger.info(f"Output: {config.output_dir}")
        logger.info("="*80)
    
    def load_and_preprocess_data(
        self
    ) -> Tuple[pd.DataFrame, np.ndarray, Any]:
        """
        Load raw data without preprocessing.
        
        Returns:
            (X_raw, y_raw, preprocessor_class)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Data Loading (Raw)")
        logger.info("="*80)
        
        if self.config.dataset == 'compas':
            # Load COMPAS
            data = load_compas_data('data/compas-scores-two-years.csv')
            
            preprocessor_class = COMPASPreprocessor(
                target_column='two_year_recid',
                include_sensitive=True,
                validate_data=True
            )
            
            X_raw = data.drop(columns=['two_year_recid'])
            y_raw = data['two_year_recid'].values
            
            logger.info(f"COMPAS raw data loaded: {X_raw.shape}")
            logger.info(f"Recidivism rate: {y_raw.mean():.2%}")
            
        elif self.config.dataset == 'mimic':
            # Load MIMIC
            data = load_mimic_data('data/mimic_cohort_ctf.csv')
            
            preprocessor_class = MIMICPreprocessor(
                target_column='hospital_expire_flag',
                validate_data=True,
                clip_outliers=True
            )
            
            X_raw = data.drop(columns=['hospital_expire_flag'])
            y_raw = data['hospital_expire_flag'].values
            
            logger.info(f"MIMIC raw data loaded: {X_raw.shape}")
            logger.info(f"Mortality rate: {y_raw.mean():.2%}")
        
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
        
        return X_raw, y_raw, preprocessor_class
    
    def get_preprocessor(self):
        """
        Get a fresh preprocessor instance for the current dataset.
        
        Returns:
            New preprocessor instance
        """
        if self.config.dataset == 'compas':
            return COMPASPreprocessor(
                target_column='two_year_recid',
                include_sensitive=True,
                validate_data=True
            )
        elif self.config.dataset == 'mimic':
            return MIMICPreprocessor(
                target_column='hospital_expire_flag',
                validate_data=True,
                clip_outliers=True
            )
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
    
    def three_way_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fold_idx: int
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Create three-way split: test, causal discovery, training.
        
        Args:
            X: Features
            y: Labels
            fold_idx: Fold index for random seed
            
        Returns:
            Dictionary with test, causal, and train splits
        """
        from sklearn.model_selection import train_test_split
        
        # First split: test set
        X_remaining, X_test, y_remaining, y_test = train_test_split(
            X, y,
            test_size=self.config.test_ratio,
            stratify=y,
            random_state=self.config.random_state + fold_idx
        )
        
        # Second split: causal discovery vs training
        causal_size = self.config.causal_discovery_ratio
        train_size = 1 - causal_size
        
        X_causal, X_train, y_causal, y_train = train_test_split(
            X_remaining, y_remaining,
            test_size=train_size,
            stratify=y_remaining,
            random_state=self.config.random_state + fold_idx + 1000
        )
        
        logger.info(f"\nThree-way split:")
        logger.info(f"  Test:            {X_test.shape} ({len(X_test)/len(X):.1%})")
        logger.info(f"  Causal Discovery: {X_causal.shape} ({len(X_causal)/len(X):.1%})")
        logger.info(f"  Training:        {X_train.shape} ({len(X_train)/len(X):.1%})")
        
        return {
            'test': {'X': X_test, 'y': y_test},
            'causal': {'X': X_causal, 'y': y_causal},
            'train': {'X': X_train, 'y': y_train}
        }
    
    def run_causal_discovery(
        self,
        X_causal: pd.DataFrame,
        y_causal: np.ndarray,
        feature_names: List[str],
        fold_idx: int
    ):
        """
        Run ensemble causal discovery.
        
        Args:
            X_causal: Features for causal discovery
            y_causal: Target labels for causal discovery
            feature_names: Feature names
            fold_idx: Fold index
            
        Returns:
            Consensus causal graph
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Ensemble Causal Discovery")
        logger.info("="*80)
        
        # CRITICAL: Include target variable in causal discovery
        # Use consistent 'target' name (X_causal already has 'target' column from fold processing)
        target_name = 'target'
        
        # Combine features and target for causal discovery
        X_causal_with_target = X_causal.copy()
        
        # Check if target column already exists (from fold processing)
        if target_name not in X_causal_with_target.columns:
            # Add target variable if not present
            if hasattr(y_causal, 'astype'):
                y_causal_numeric = y_causal.astype(np.float64)
            else:
                y_causal_numeric = np.array(y_causal, dtype=np.float64)
            X_causal_with_target[target_name] = y_causal_numeric
            logger.info(f"Added target column '{target_name}' to causal discovery data")
        else:
            logger.info(f"Target column '{target_name}' already exists in causal discovery data")
        
        # Ensure all columns are numeric
        for col in X_causal_with_target.columns:
            X_causal_with_target[col] = pd.to_numeric(X_causal_with_target[col], errors='coerce')
        
        # Check for any remaining non-numeric data
        if X_causal_with_target.isnull().any().any():
            logger.warning("Found NaN values after numeric conversion - filling with column means")
            X_causal_with_target = X_causal_with_target.fillna(X_causal_with_target.mean())
        
        logger.info(f"Data types after conversion: {dict(X_causal_with_target.dtypes)}")
        
        # Update feature names to include target (avoid duplicates)
        logger.info(f"Original feature names count: {len(feature_names)}")
        logger.info(f"Target name: {target_name}")
        logger.info(f"Target in features? {target_name in feature_names}")
        
        if target_name in feature_names:
            feature_names_with_target = list(feature_names)
            logger.info(f"Target '{target_name}' already in feature names - using existing list")
        else:
            feature_names_with_target = list(feature_names) + [target_name]
            logger.info(f"Added target '{target_name}' to feature names")
        
        # Additional check for any duplicates in the final list
        if len(feature_names_with_target) != len(set(feature_names_with_target)):
            logger.warning("Duplicate feature names detected - removing duplicates")
            feature_names_with_target = list(dict.fromkeys(feature_names_with_target))  # Preserve order
            logger.info(f"Final feature names after deduplication: {len(feature_names_with_target)} features")
        
        logger.info(f"Causal discovery data:")
        logger.info(f"  Features: {len(feature_names)}")
        logger.info(f"  Target: {target_name}")
        logger.info(f"  Total variables: {len(feature_names_with_target)}")
        logger.info(f"  Samples: {len(X_causal_with_target)}")
        
        # Initialize ensemble
        ensemble = EnsembleCausalDiscovery(
            algorithms=['pc', 'notears'],  # Use available algorithms
            stability_threshold=0.3  # Lower threshold to preserve target connections
        )
        
        # Run causal discovery WITH target included
        consensus_graph, edge_stability = ensemble.fit(
            X_causal_with_target, 
            feature_names_with_target
        )
        
        logger.info(f"\nConsensus graph (before domain constraints):")
        logger.info(f"  Nodes: {consensus_graph.number_of_nodes()}")
        logger.info(f"  Edges: {consensus_graph.number_of_edges()}")
        logger.info(f"  Target node present: {target_name in consensus_graph.nodes}")
        
        # Apply domain constraints to clean up impossible edges (if enabled)
        if self.config.apply_domain_constraints:
            logger.info("\n" + "-"*60)
            logger.info("APPLYING DOMAIN CONSTRAINTS")
            logger.info("-"*60)
            
            import networkx as nx
            
            # Validate consistency before NetworkX conversion
            logger.info(f"Pre-conversion validation:")
            logger.info(f"  Graph nodes: {consensus_graph.number_of_nodes()}")
            logger.info(f"  Feature names count: {len(feature_names_with_target)}")
            logger.info(f"  Graph node list: {sorted(list(consensus_graph.nodes()))}")
            logger.info(f"  Feature name list: {feature_names_with_target}")
            
            # Check for duplicates one more time
            if len(feature_names_with_target) != len(set(feature_names_with_target)):
                duplicates = [name for name in feature_names_with_target if feature_names_with_target.count(name) > 1]
                logger.error(f"Still have duplicates in feature names: {set(duplicates)}")
                raise ValueError("Cannot proceed with duplicate feature names")
            
            # Convert graph to adjacency matrix for domain constraints
            adj_matrix_before = nx.to_numpy_array(consensus_graph, nodelist=feature_names_with_target)
            
            # Apply domain knowledge constraints
            adj_matrix_after = apply_domain_constraints(
                adj_matrix_before, 
                feature_names_with_target, 
                dataset=self.config.dataset,
                stability_scores=edge_stability
            )
            
            # DEBUG: Check glucose edge after domain constraints
            if 'glucose' in feature_names_with_target and 'target' in feature_names_with_target:
                glucose_idx = feature_names_with_target.index('glucose')
                target_idx = feature_names_with_target.index('target')
                logger.info(f"DEBUG: After domain constraints: glucose->target = {adj_matrix_after[glucose_idx, target_idx]}")
            
            # Reconstruct graph from cleaned adjacency matrix
            consensus_graph = nx.DiGraph()
            consensus_graph.add_nodes_from(feature_names_with_target)
            
            # Add edges back with preserved weights
            for i, source in enumerate(feature_names_with_target):
                for j, target in enumerate(feature_names_with_target):
                    if adj_matrix_after[i, j] > 0:
                        # Try to preserve original edge weight if it existed
                        if adj_matrix_before[i, j] == 1:
                            # Find original weight from the edge data
                            try:
                                original_weight = edge_stability[i, j] if hasattr(edge_stability, 'shape') else 0.5
                            except:
                                original_weight = 0.5
                        else:
                            original_weight = 0.5
                        
                        consensus_graph.add_edge(source, target, weight=max(0.1, min(1.0, original_weight)))
        else:
            logger.info("\n" + "-"*60)
            logger.info("DOMAIN CONSTRAINTS DISABLED - Using raw causal discovery results")
            logger.info("-"*60)
        
        logger.info(f"\nConsensus graph (after domain constraints):")
        logger.info(f"  Nodes: {consensus_graph.number_of_nodes()}")
        logger.info(f"  Edges: {consensus_graph.number_of_edges()}")
        logger.info(f"  Target node present: {target_name in consensus_graph.nodes}")
        
        # Count edges TO target
        if target_name in consensus_graph.nodes:
            edges_to_target = list(consensus_graph.predecessors(target_name))
            logger.info(f"  Edges to target: {len(edges_to_target)}")
            if edges_to_target:
                logger.info(f"    Causal influences: {edges_to_target}")  # Show all
                
                # Debug glucose specifically
                glucose_in_edges = [edge for edge in edges_to_target if 'glucose' in edge.lower()]
                logger.info(f"    DEBUG: Glucose edges to target: {glucose_in_edges}")
                logger.info(f"    DEBUG: Has glucose->target edge: {consensus_graph.has_edge('glucose', target_name) if 'glucose' in feature_names_with_target else 'glucose not in features'}")
        
        # Save graph
        graph_path = self.config.output_dir / 'graphs' / f'fold_{fold_idx}_graph.graphml'
        save_graph(consensus_graph, str(graph_path))
        
        # Create publication-quality visualization
        fig_path = self.config.output_dir / 'figures' / f'fold_{fold_idx}_graph.png'
        dataset_title = 'COMPAS Recidivism' if self.config.dataset == 'compas' else 'MIMIC Mortality'
        graph_title = f"Causal Graph - Fold {fold_idx + 1}: {dataset_title}"
        
        visualize_causal_graph(
            consensus_graph,
            filepath=str(fig_path),
            title=graph_title,
            target_node=target_name,
            edge_stability=None,
            dataset='auto'
        )
        
        return consensus_graph, edge_stability
    
    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        fold_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for DNN)
            y_val: Validation labels
            fold_idx: Fold index
            
        Returns:
            Dictionary of trained models
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Model Training")
        logger.info("="*80)
        
        # --- ROBUST: Data Cleaning ---
        X_train = robust_data_cleaning(X_train, use_median=False)  # Use mean for training consistency
        
        # Ensure y is numeric (handle 'F'/'M' in target if present)
        if y_train.dtype == object or (len(y_train) > 0 and isinstance(y_train[0], str)):
            # Simple mapping if it's gender or similar binary
            y_train = pd.Series(y_train).map({'F': 0, 'M': 1, '0': 0, '1': 1}).fillna(0).values
        
        y_train = y_train.astype(int)
        # Convert back to numpy for sklearn (only if DataFrame)
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        # -----------------------------------
        
        trained_models = {}
        
        for model_name in self.config.models:
            logger.info(f"\n--- Training {model_name.upper()} ---")
            
            try:
                if model_name == 'lr':
                    model, params = ModelFactory.create_logistic_regression(
                        X_train, y_train,
                        n_iter=self.config.hyperopt_iterations
                    )
                
                elif model_name == 'rf':
                    model, params = ModelFactory.create_random_forest(
                        X_train, y_train,
                        n_iter=self.config.hyperopt_iterations,
                        cv=5,
                        random_state=42,
                        n_jobs=1  # Reduced for stability
                    )
                
                elif model_name == 'xgb':
                    model, params = ModelFactory.create_xgboost(
                        X_train, y_train,
                        n_iter=self.config.hyperopt_iterations
                    )
                
                elif model_name == 'dnn':
                    # Need validation split for DNN
                    if X_val is None:
                        from sklearn.model_selection import train_test_split
                        X_train_dnn, X_val, y_train_dnn, y_val = train_test_split(
                            X_train, y_train,
                            test_size=0.2,
                            stratify=y_train,
                            random_state=self.config.random_state
                        )
                    else:
                        X_train_dnn = X_train
                        y_train_dnn = y_train
                    
                    dnn_result = ModelFactory.create_dnn(
                        X_train_dnn, y_train_dnn,
                        X_val, y_val,
                        max_epochs=100,
                        patience=10
                    )
                    model = dnn_result['wrapped_model']  # Use wrapped model for sklearn compatibility
                    params = dnn_result['params']
                
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                trained_models[model_name] = {
                    'model': model,
                    'params': params
                }
                
                # Save model
                model_path = self.config.output_dir / 'models' / f'fold_{fold_idx}_{model_name}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                
                logger.info(f"{model_name.upper()} training complete")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return trained_models
    
    def evaluate_ctf_metrics(
        self,
        model: Any,
        model_name: str,
        causal_graph: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        feature_types: Optional[List[str]] = None,
        optimal_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate CTF metrics for a model.
        
        Args:
            model: Trained model
            model_name: Model name
            causal_graph: Causal graph
            X_train: Training data (for CII)
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
            feature_names: Feature names
            feature_types: Feature types for CS
            optimal_threshold: Optimal threshold (stored but not currently used by CTF)
            
        Returns:
            CTF metrics results
        """
        logger.info(f"\n--- Evaluating CTF Metrics for {model_name.upper()} ---")
        
        # --- BUG WAS HERE: Using original names instead of graph name ---
        # The graph was built using the column name 'target', so we must use that.
        target_name_in_graph = 'target' 
        # ----------------------------------------------------------------
        
        # Initialize CTF framework
        ctf = CTFFramework(
            causal_graph=causal_graph,
            model=model,
            feature_names=feature_names,
            target_name=target_name_in_graph  # Pass 'target', not original dataset name
        )
        
        # Evaluate all metrics
        ctf_results = ctf.evaluate(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_types=feature_types
        )
        
        # Get summary
        summary_df = ctf.summary()
        logger.info(f"\nCTF Summary for {model_name}:")
        logger.info(summary_df.to_string())
        
        return ctf_results
    
    def compare_with_shap_lime(
        self,
        model: Any,
        model_name: str,
        X_train: np.ndarray,
        X_test: np.ndarray,
        cii_scores: Dict[str, float],
        feature_names: List[str],
        fold_idx: int
    ) -> Dict[str, Any]:
        """
        Compare CII with SHAP and LIME.
        
        Args:
            model: Trained model
            model_name: Model name
            X_train: Training data
            X_test: Test data
            cii_scores: CII scores
            feature_names: Feature names
            fold_idx: Fold index
            
        Returns:
            Comparison results
        """
        logger.info(f"\n--- Comparing CII with SHAP/LIME for {model_name.upper()} ---")
        
        # SHAP
        shap_scores = None
        if SHAP_AVAILABLE:
            try:
                logger.info("Computing SHAP values...")
                # Determine SHAP explainer type based on model
                if model_name in ['rf', 'xgb']:
                    shap_model_type = 'tree'
                elif model_name == 'lr':
                    shap_model_type = 'linear'
                else:  # DNN and other models
                    shap_model_type = 'kernel'
                
                shap_explainer = SHAPExplainer(
                    model, X_train, feature_names,
                    model_type=shap_model_type
                )
                shap_scores = shap_explainer.compute_feature_importance(X_test)
                logger.info(f"SHAP complete: {len(shap_scores)} features")
            except Exception as e:
                logger.warning(f"SHAP failed: {e}")
        
        # LIME
        lime_scores = None
        if LIME_AVAILABLE:
            try:
                logger.info("Computing LIME values...")
                lime_explainer = LIMEExplainer(
                    model, X_train, feature_names
                )
                lime_scores = lime_explainer.compute_feature_importance(
                    X_test, n_samples=50
                )
                logger.info(f"LIME complete: {len(lime_scores)} features")
            except Exception as e:
                logger.warning(f"LIME failed: {e}")
        
        # Comparative analysis
        comparison = ComparativeAnalysis(
            cii_scores=cii_scores,
            shap_scores=shap_scores,
            lime_scores=lime_scores,
            feature_names=feature_names
        )
        
        # Compare
        comparison_results = comparison.compare_all()
        logger.info(f"\nComparison Results:")
        logger.info(comparison_results.to_string())
        
        # Visualize
        fig_path = self.config.output_dir / 'figures' / f'fold_{fold_idx}_{model_name}_comparison.png'
        comparison.visualize_comparison(str(fig_path))
        
        return {
            'shap_scores': shap_scores,
            'lime_scores': lime_scores,
            'comparison_table': comparison_results.to_dict()
        }
    
    def run_fold(
        self,
        fold_idx: int,
        X_raw: pd.DataFrame,
        y_raw: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Run one fold of the experiment.
        
        Args:
            fold_idx: Fold index
            X_raw: Raw features
            y_raw: Raw labels
            feature_names: Raw feature names
            
        Returns:
            Fold results
        """
        logger.info("\n" + "="*80)
        logger.info(f"FOLD {fold_idx + 1}/{self.config.n_folds}")
        logger.info("="*80)
        
        fold_results = {
            'fold': fold_idx,
            'models': {}
        }
        
        # 1. Split RAW data
        splits = self.three_way_split(X_raw.values, y_raw, fold_idx)
        
        # 2. Fit Preprocessor on TRAIN + CAUSAL (Training environment)
        # Combine train and causal for fitting scaler/imputer to maximize data usage
        # WITHOUT touching Test data
        X_fit = np.concatenate([splits['train']['X'], splits['causal']['X']])
        y_fit = np.concatenate([splits['train']['y'], splits['causal']['y']])
        
        # Re-initialize new preprocessor for this fold
        preprocessor = self.get_preprocessor()
        
        # Create DataFrame for fitting with proper column alignment
        logger.info(f"X_fit shape: {X_fit.shape}, feature_names length: {len(feature_names)}")
        
        # Ensure column count matches
        if X_fit.shape[1] != len(feature_names):
            logger.error(f"Column mismatch: X_fit has {X_fit.shape[1]} columns, feature_names has {len(feature_names)}")
            # Take only the columns that exist
            n_cols = min(X_fit.shape[1], len(feature_names))
            X_fit = X_fit[:, :n_cols]
            feature_names_fit = feature_names[:n_cols]
        else:
            feature_names_fit = feature_names
        
        X_fit_df = pd.DataFrame(X_fit, columns=feature_names_fit)
        logger.info(f"Created DataFrame with columns: {list(X_fit_df.columns)}")
        logger.info(f"DataFrame dtypes: {dict(X_fit_df.dtypes)}")
        
        preprocessor.fit(X_fit_df, y_fit)
        
        # Get processed feature names
        processed_feature_names = preprocessor.get_feature_names_out()
        
        # 3. Transform all splits
        X_train_df = pd.DataFrame(splits['train']['X'], columns=feature_names_fit)
        X_causal_df = pd.DataFrame(splits['causal']['X'], columns=feature_names_fit)  
        X_test_df = pd.DataFrame(splits['test']['X'], columns=feature_names_fit)
        
        X_train_proc, _ = preprocessor.transform(X_train_df)
        X_causal_proc, _ = preprocessor.transform(X_causal_df)
        X_test_proc, _ = preprocessor.transform(X_test_df)
        
        # Update splits with processed data
        splits['train']['X'] = X_train_proc
        splits['causal']['X'] = X_causal_proc
        splits['test']['X'] = X_test_proc
        
        # --- ROBUST: Data Preparation for Causal Discovery ---
        
        # Use processed feature names (no need to check for duplicates after preprocessing)
        # Prepare causal discovery data with robust NaN handling
        X_causal_df = prepare_data_for_causal_discovery(
            splits['causal']['X'],
            processed_feature_names
        )

        # Add the clean 'target' column
        X_causal_df['target'] = splits['causal']['y']
        logger.info(f"Final columns for causal discovery: {list(X_causal_df.columns)}")
            
        # Update discovery names to match the actual dataframe
        discovery_names = list(X_causal_df.columns)
        
        # Run discovery (Make sure fold_idx is passed!)
        causal_graph, edge_stability = self.run_causal_discovery(
            X_causal_df, splits['causal']['y'], discovery_names, fold_idx
        )
        
        fold_results['causal_graph'] = {
            'n_edges': causal_graph.number_of_edges(),
            'edge_stability': edge_stability
        }
        
        # 3. Train models
        trained_models = self.train_models(
            splits['train']['X'],
            splits['train']['y'],
            fold_idx=fold_idx
        )
        
        # --- ROBUST: Test Data Cleaning ---
        X_test_eval = robust_data_cleaning(splits['test']['X'], use_median=False)  # Consistent with training
        y_test_eval = splits['test']['y']

        # Clean y_test
        if y_test_eval.dtype == object or (len(y_test_eval) > 0 and isinstance(y_test_eval[0], str)):
            y_test_eval = pd.Series(y_test_eval).map({'F': 0, 'M': 1, '0': 0, '1': 1}).fillna(0).values
        y_test_eval = y_test_eval.astype(int)
        
        # Convert back to numpy (only if DataFrame)
        if isinstance(X_test_eval, pd.DataFrame):
            X_test_eval = X_test_eval.values
        # ------------------------------------------------------------

        # 4. Evaluate each model
        for model_name, model_info in trained_models.items():
            model = model_info['model']
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating {model_name.upper()}")
            logger.info(f"{'='*80}")
            
            # Predictive performance with consistent threshold
            eval_results = evaluate_model_with_consistent_threshold(
                model,
                X_test_eval,  # Use the sanitized version
                y_test_eval,  # Use the sanitized version
                model_name=model_name.upper()
            )
            perf_metrics = eval_results['metrics']
            optimal_threshold = eval_results['optimal_threshold']
            
            # Capture probabilities for ROC visualization
            y_prob = eval_results['y_pred_proba']
            self.roc_viz.add_fold_data(
                model_name=model_name,
                fold_idx=fold_idx,
                y_true=y_test_eval,
                y_score=y_prob
            )
            
            # CTF metrics
            X_train_df = pd.DataFrame(splits['train']['X'], columns=processed_feature_names)
            
            ctf_results = self.evaluate_ctf_metrics(
                model=model,
                model_name=model_name,
                causal_graph=causal_graph,
                X_train=X_train_df,
                y_train=splits['train']['y'],
                X_test=X_test_eval,  # Use sanitized version
                y_test=y_test_eval,  # Use sanitized version
                feature_names=processed_feature_names,
                optimal_threshold=optimal_threshold  # Pass the optimal threshold
            )
            
            # ADD DIAGNOSTIC FOR FIRST MODEL ONLY (to avoid duplicate diagnostics)
            if model_name == 'lr' and ctf_results.get('cii'):
                cii_scores = ctf_results['cii']['viz_scaled']
                
                # Run diagnostic to check for identical Gini bug
                diag = quick_fold_diagnostic(
                    fold_idx=fold_idx,
                    consensus_graph=causal_graph,
                    cii_scores=cii_scores,
                    feature_names=processed_feature_names
                )
                self.fold_diagnostics[fold_idx] = diag
            
            # Comparative analysis
            if ctf_results.get('cii'):
                cii_scores = ctf_results['cii']['viz_scaled']
                
                comparison_results = self.compare_with_shap_lime(
                    model=model,
                    model_name=model_name,
                    X_train=splits['train']['X'],
                    X_test=X_test_eval,  # Use sanitized version
                    cii_scores=cii_scores,
                    feature_names=processed_feature_names,
                    fold_idx=fold_idx
                )
            else:
                comparison_results = None
            
            # Store results
            fold_results['models'][model_name] = {
                'params': model_info['params'],
                'performance': perf_metrics,
                'optimal_threshold': optimal_threshold,
                'ctf_metrics': ctf_results,
                'comparison': comparison_results
            }
        
        return fold_results
    
    def run_experiment(self):
        """Run complete experiment."""
        start_time = datetime.now()
        logger.info(f"\nExperiment started: {start_time}")
        
        # Load raw data
        X_raw, y_raw, preprocessor_class = self.load_and_preprocess_data()
        
        # Get raw feature names
        feature_names = list(X_raw.columns)
        
        # Run folds
        robustness_results = []
        for fold_idx in range(self.config.n_folds):
            fold_results = self.run_fold(fold_idx, X_raw, y_raw, feature_names)
            self.results['folds'].append(fold_results)
            
            # Run robustness stress test for the first fold (representative analysis)
            if fold_idx == 0:
                logger.info("\n" + "="*80)
                logger.info("RUNNING ROBUSTNESS STRESS TEST")
                logger.info("="*80)
                
                # Get test data for robustness analysis
                splits = self.three_way_split(X_raw.values, y_raw, fold_idx)
                preprocessor = self.get_preprocessor()
                
                # Fit preprocessor on train+causal
                X_fit = np.concatenate([splits['train']['X'], splits['causal']['X']])
                y_fit = np.concatenate([splits['train']['y'], splits['causal']['y']])
                X_fit_df = pd.DataFrame(X_fit, columns=feature_names[:X_fit.shape[1]])
                preprocessor.fit(X_fit_df, y_fit)
                
                # Transform test data
                X_test_df = pd.DataFrame(splits['test']['X'], columns=feature_names[:splits['test']['X'].shape[1]])
                X_test_processed = preprocessor.transform(X_test_df)
                if isinstance(X_test_processed, pd.DataFrame):
                    X_test_processed = X_test_processed.values
                
                # Run stress test for each model
                for model_name in self.config.models:
                    if model_name in fold_results['models'] and 'trained_model' in fold_results['models'][model_name]:
                        model = fold_results['models'][model_name]['trained_model']
                        robustness_data = self.run_robustness_stress_test(
                            model, X_test_processed, feature_names, model_name
                        )
                        robustness_results.append(robustness_data)
                
                # Generate robustness comparison plot
                if robustness_results:
                    self.plot_robustness_comparison(robustness_results)
        
        # DIAGNOSTIC: Compare all folds to detect identical Gini bug
        if self.fold_diagnostics:
            logger.info("\n\n" + "="*80)
            logger.info("RUNNING DIAGNOSTIC ANALYSIS...")
            logger.info("="*80)
            diagnosis = compare_folds(self.fold_diagnostics)
            logger.info(f"\nFinal diagnosis: {diagnosis}")
            
            # Store diagnosis in results
            self.results['diagnosis'] = {
                'result': diagnosis,
                'diagnostics': self.fold_diagnostics
            }
        
        # Generate ROC visualizations
        logger.info("\n" + "="*80)
        logger.info("GENERATING ROC VISUALIZATIONS...")
        logger.info("="*80)
        
        # 1. Generate Stability Plots (One per model, consolidated across folds)
        for model_name in self.config.models:
            save_path = self.config.output_dir / 'figures' / f'{model_name}_stability_roc.png'
            self.roc_viz.plot_model_stability(model_name, str(save_path))
            
        # 2. Generate Fold Comparison Plots (One per fold, comparing models)
        for fold_idx in range(self.config.n_folds):
            save_path = self.config.output_dir / 'figures' / f'fold_{fold_idx}_comparison_roc.png'
            self.roc_viz.plot_fold_comparison(fold_idx, str(save_path))
            
        # 3. Generate Comprehensive Summary
        save_path = self.config.output_dir / 'figures' / f'{self.config.dataset}_roc_summary.png'
        self.roc_viz.plot_comprehensive_summary(str(save_path))
        
        # 4. Log ROC summary statistics
        roc_stats = self.roc_viz.get_summary_statistics()
        logger.info("\nROC Summary Statistics:")
        for model_name, stats in roc_stats.items():
            logger.info(f"  {model_name.upper()}: AUC = {stats['mean_auc']:.3f} ± {stats['std_auc']:.3f}")
        
        # Aggregate results
        self.aggregate_results()
        
        # Save results
        self.save_results()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("="*80)
        logger.info(f"Duration: {duration}")
        logger.info(f"Results saved to: {self.config.output_dir}")
        logger.info("="*80)
    
    def aggregate_results(self):
        """Aggregate results across folds."""
        logger.info("\n" + "="*80)
        logger.info("AGGREGATING RESULTS")
        logger.info("="*80)
        
        # Get model names
        model_names = self.config.models
        
        # Aggregate for each model
        aggregated = {}
        
        for model_name in model_names:
            model_results = []
            
            for fold_results in self.results['folds']:
                if model_name in fold_results['models']:
                    model_results.append(fold_results['models'][model_name])
            
            if not model_results:
                continue
            
            # Aggregate performance metrics
            perf_metrics = ['auc_roc', 'accuracy', 'f1', 'precision', 'recall', 'f2']
            perf_agg = {}
            
            for metric in perf_metrics:
                values = [r['performance'].get(metric, np.nan) for r in model_results]
                perf_agg[metric] = {
                    'mean': np.nanmean(values),
                    'std': np.nanstd(values),
                    'values': values
                }
            
            # Aggregate optimal thresholds
            thresholds = [r.get('optimal_threshold', np.nan) for r in model_results]
            perf_agg['threshold'] = {
                'mean': np.nanmean(thresholds),
                'std': np.nanstd(thresholds),
                'values': thresholds
            }
            
            # Aggregate CTF metrics
            ctf_metrics = {}
            
            # CII
            cii_gini = [
                r['ctf_metrics']['cii']['distribution']['gini']
                for r in model_results if r['ctf_metrics'].get('cii')
            ]
            if cii_gini:
                ctf_metrics['cii_gini'] = {
                    'mean': np.mean(cii_gini),
                    'std': np.std(cii_gini)
                }
            
            # CCM
            ccm_total = [
                r['ctf_metrics']['ccm']['ccm_total']
                for r in model_results if r['ctf_metrics'].get('ccm')
            ]
            if ccm_total:
                ctf_metrics['ccm_total'] = {
                    'mean': np.mean(ccm_total),
                    'std': np.std(ccm_total)
                }
            
            # TE
            te_score = [
                r['ctf_metrics']['te']['te_score']
                for r in model_results if r['ctf_metrics'].get('te')
            ]
            if te_score:
                ctf_metrics['te_score'] = {
                    'mean': np.mean(te_score),
                    'std': np.std(te_score)
                }
            
            # Enhanced CS
            enhanced_cs_scores = [
                r['ctf_metrics']['cs']['enhanced_cs_score']
                for r in model_results if r['ctf_metrics'].get('cs') and 'enhanced_cs_score' in r['ctf_metrics']['cs']
            ]
            
            discriminative_power = [
                r['ctf_metrics']['cs']['discriminative_power']
                for r in model_results if r['ctf_metrics'].get('cs') and 'discriminative_power' in r['ctf_metrics']['cs']
            ]
            
            cs_robustness = [
                r['ctf_metrics']['cs']['cs_robustness']
                for r in model_results if r['ctf_metrics'].get('cs') and 'cs_robustness' in r['ctf_metrics']['cs']
            ]
            
            if enhanced_cs_scores:
                ctf_metrics['enhanced_cs_score'] = {
                    'mean': np.mean(enhanced_cs_scores),
                    'std': np.std(enhanced_cs_scores)
                }
                
            if discriminative_power:
                ctf_metrics['discriminative_power'] = {
                    'mean': np.mean(discriminative_power),
                    'std': np.std(discriminative_power)
                }
                
            if cs_robustness:
                ctf_metrics['cs_robustness'] = {
                    'mean': np.mean(cs_robustness),
                    'std': np.std(cs_robustness)
                }
            
            # Fallback for legacy cs_score if enhanced version not available
            legacy_cs_scores = [
                r['ctf_metrics']['cs']['cs_score']
                for r in model_results if r['ctf_metrics'].get('cs') and 'cs_score' in r['ctf_metrics']['cs']
            ]
            
            if legacy_cs_scores:
                ctf_metrics['cs_score'] = {
                    'mean': np.mean(legacy_cs_scores),
                    'std': np.std(legacy_cs_scores)
                }
            
            aggregated[model_name] = {
                'performance': perf_agg,
                'ctf_metrics': ctf_metrics
            }
        
        self.results['summary'] = aggregated
        
        # Print summary
        logger.info("\nSummary Statistics:")
        for model_name, model_agg in aggregated.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  AUC-ROC: {model_agg['performance']['auc_roc']['mean']:.4f} ± {model_agg['performance']['auc_roc']['std']:.4f}")
            logger.info(f"  Recall:  {model_agg['performance']['recall']['mean']:.4f} ± {model_agg['performance']['recall']['std']:.4f} ← CRITICAL")
            logger.info(f"  F2-Score: {model_agg['performance']['f2']['mean']:.4f} ± {model_agg['performance']['f2']['std']:.4f} (recall-weighted)")
            logger.info(f"  Threshold: {model_agg['performance']['threshold']['mean']:.3f} ± {model_agg['performance']['threshold']['std']:.3f}")
            
            if 'cii_gini' in model_agg['ctf_metrics']:
                logger.info(f"  CII Gini: {model_agg['ctf_metrics']['cii_gini']['mean']:.4f} ± {model_agg['ctf_metrics']['cii_gini']['std']:.4f}")
            if 'ccm_total' in model_agg['ctf_metrics']:
                logger.info(f"  CCM: {model_agg['ctf_metrics']['ccm_total']['mean']:.4f} ± {model_agg['ctf_metrics']['ccm_total']['std']:.4f}")
            if 'te_score' in model_agg['ctf_metrics']:
                logger.info(f"  TE: {model_agg['ctf_metrics']['te_score']['mean']:.4f} ± {model_agg['ctf_metrics']['te_score']['std']:.4f}")
            # Enhanced CS metrics
            if 'enhanced_cs_score' in model_agg['ctf_metrics']:
                logger.info(f"  Enhanced CS: {model_agg['ctf_metrics']['enhanced_cs_score']['mean']:.4f} ± {model_agg['ctf_metrics']['enhanced_cs_score']['std']:.4f}")
            if 'discriminative_power' in model_agg['ctf_metrics']:
                logger.info(f"  CS Discriminative Power: {model_agg['ctf_metrics']['discriminative_power']['mean']:.4f} ± {model_agg['ctf_metrics']['discriminative_power']['std']:.4f}")
            if 'cs_robustness' in model_agg['ctf_metrics']:
                logger.info(f"  CS Robustness: {model_agg['ctf_metrics']['cs_robustness']['mean']:.4f} ± {model_agg['ctf_metrics']['cs_robustness']['std']:.4f}")
            # Legacy CS fallback
            elif 'cs_score' in model_agg['ctf_metrics']:
                logger.info(f"  CS (Legacy): {model_agg['ctf_metrics']['cs_score']['mean']:.4f} ± {model_agg['ctf_metrics']['cs_score']['std']:.4f}")
    
    def run_robustness_stress_test(self, model, X_test, feature_names, model_name):
        """
        Runs stability analysis across a range of noise levels (0.1 to 1.0 standard deviations).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from Ctf_metrics import CounterfactualStability
        
        # Define the stress levels (10% noise to 100% noise)
        noise_scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        logger.info(f"Running Robustness Stress Test for {model_name}...")
        
        # Initialize the multi-scale calculator
        cs_calculator = CounterfactualStability(
            num_perturbations=20,  # Higher samples for smoother curves
            perturbation_scales=noise_scales
        )
        
        # Compute
        results = cs_calculator.compute(model, X_test)
        
        # Extract the curve
        stability_curve = results.get('curve', [])
        
        # Return data for plotting later
        return {
            'scales': noise_scales,
            'stability': stability_curve,
            'model': model_name
        }

    def plot_robustness_comparison(self, all_model_results):
        """
        Generates the final "Robustness Analysis" figure for your paper.
        Compares how LR, RF, XGB, and DNN degrade as noise increases.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        # Plot style
        styles = {
            'lr': {'color': '#2ecc71', 'marker': 'o', 'label': 'Logistic Regression (Linear)'},  # Green
            'rf': {'color': '#f1c40f', 'marker': 's', 'label': 'Random Forest'},                # Yellow
            'xgb': {'color': '#e67e22', 'marker': '^', 'label': 'XGBoost'},                     # Orange
            'dnn': {'color': '#9b59b6', 'marker': 'D', 'label': 'Deep Neural Network'}          # Purple
        }
        
        for res in all_model_results:
            name = res['model']
            style = styles.get(name, {'color': 'gray', 'marker': 'x', 'label': name})
            
            plt.plot(
                res['scales'], 
                res['stability'], 
                marker=style['marker'], 
                color=style['color'], 
                linewidth=2, 
                markersize=8,
                label=style['label']
            )
        
        # Formatting
        plt.title('Causal Stability: Robustness under Feature Perturbation', fontsize=14, fontweight='bold')
        plt.xlabel('Perturbation Scale (Standard Deviations of Noise)', fontsize=12)
        plt.ylabel('Stability Score (1 - Prediction Change)', fontsize=12)
        plt.ylim(0.5, 1.05)  # Zoom in on the relevant range
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        
        # Save
        save_path = self.config.output_dir / 'figures' / 'robustness_stress_test.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Robustness curve saved to {save_path}")

    def save_results(self):
        """Save results to files."""
        # Save as JSON (without model objects)
        results_json = {
            'config': self.results['config'],
            'summary': self.results['summary'],
            'n_folds': len(self.results['folds'])
        }
        
        json_path = self.config.output_dir / 'results_summary.json'
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        logger.info(f"Results saved to {json_path}")
        
        # Save full results as pickle
        pickle_path = self.config.output_dir / 'results_full.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"Full results saved to {pickle_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    # Run experiments on both datasets
    for dataset_name in ['compas', 'mimic']:
        logger.info(f"Starting experiment on {dataset_name.upper()} dataset")
        logger.info("=" * 80)
        
        # Configuration for current dataset
        config = ExperimentConfig(
            dataset=dataset_name,
            models=['lr', 'rf', 'xgb', 'dnn'],  # DNN re-enabled with fixes
            n_folds=5,
            causal_discovery_ratio=0.6,
            test_ratio=0.2,
            hyperopt_iterations=10,  # Reduced for faster execution
            n_perturbations=10,  # Reduced for faster execution
            random_state=42,
            output_dir=f'results/ctf_experiment_{dataset_name}',
            apply_domain_constraints=True  # Enable domain constraints
        )
        
        # Create experiment
        experiment = CTFExperiment(config)
        
        # Run
        experiment.run_experiment()
        
        logger.info(f"\n✅ {dataset_name.upper()} experiment completed!")
        logger.info("=" * 80)
    
    logger.info("\n✅ All experiments completed successfully!")


if __name__ == "__main__":
    main()