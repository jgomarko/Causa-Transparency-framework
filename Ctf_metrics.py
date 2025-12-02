#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal Transparency Framework (CTF) Metrics Implementation.

UPDATES:
1. Implements "Permutation CII" to fix non-linear model drift.
2. Uses Dynamic Decay for graph propagation.

Implements the four core CTF metrics:
1. Causal Influence Index (CII) - Feature-level causal contributions
   * UPDATED: Now uses Permutation Importance for non-linear models
2. Causal Complexity Measure (CCM) - Structural and functional complexity
3. Transparency Entropy (TE) - Decision certainty/uncertainty
4. Counterfactual Stability (CS) - Robustness under interventions

Author: John Marko
Date: 2025-11-26
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mutual_info_score
from sklearn.inspection import permutation_importance  # NEW IMPORT
import networkx as nx
import warnings

# Compression libraries
import gzip
import lzma
import pickle

# Import enhanced counterfactual stability
from enhanced_cs_perturbations import EnhancedCounterfactualStability

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Causal Influence Index (CII) - Enhanced for Non-Linear Models
# ============================================================================

class CausalInfluenceIndex:
    """
    Computes Causal Influence Index (CII) for features.
    ENHANCED: Uses Permutation Importance for non-linear models to fix drift issue.
    """
    
    def __init__(
        self,
        causal_graph: nx.DiGraph,
        decay_factor: float = 0.9,
        n_neighbors: int = 3
    ):
        """
        Initialize CII calculator.
        
        Args:
            causal_graph: Learned causal graph
            decay_factor: Default path decay for indirect effects (default: 0.9)
            n_neighbors: Neighbors for MI estimation
        """
        self.causal_graph = causal_graph
        self.decay_factor = decay_factor
        self.n_neighbors = n_neighbors
        self.raw_cii_scores = {}
        self.normalized_cii_scores = {}
    
    def _compute_permutation_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Compute Permutation Importance to capture non-linear model usage.
        This fixes the issue where MI doesn't align with how non-linear models use features.
        """
        # Use a subset for speed if dataset is huge
        if len(X) > 2000:
            indices = np.random.choice(len(X), 2000, replace=False)
            X_sub = X.iloc[indices]
            y_sub = y[indices]
        else:
            X_sub = X
            y_sub = y

        try:
            # Run sklearn permutation importance
            # n_repeats=5 is a good balance of speed/stability
            result = permutation_importance(
                model, X_sub.values, y_sub, 
                n_repeats=5, random_state=42, n_jobs=1
            )
            
            # Map importances to feature names
            # Clip at 0 to remove negative importance (noise)
            importances = {
                name: max(0.0, imp) 
                for name, imp in zip(feature_names, result.importances_mean)
            }
            return importances
            
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}. Falling back to MI.")
            return None

    def _compute_path_strength(
        self,
        source: str,
        target: str,
        max_path_length: int = 5,
        model_type: str = 'default'
    ) -> float:
        """
        Compute causal strength along paths using Dynamic Decay.
        
        Args:
            source: Source feature
            target: Target variable
            max_path_length: Maximum path length to consider
            model_type: 'linear', 'tree', or 'neural' to adjust decay
            
        Returns:
            Causal path strength
        """
        if not self.causal_graph.has_node(source) or not self.causal_graph.has_node(target):
            return 0.0
        
        # Dynamic Decay Configuration
        decay_map = {
            'linear': 0.95,   # Linear: Signal travels far
            'tree': 0.80,     # RF/XGB: Splits fragment signal
            'neural': 0.60,   # DNN: High non-linearity absorbs signal
            'default': 0.90
        }
        current_decay = decay_map.get(model_type, decay_map['default'])
        
        try:
            paths = list(nx.all_simple_paths(self.causal_graph, source, target, cutoff=max_path_length))
        except nx.NetworkXNoPath:
            return 0.0
        
        if not paths:
            return 0.0
            
        path_strengths = []
        for path in paths:
            path_length = len(path) - 1
            # Structural weight (Product of edges)
            path_weight = 1.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if self.causal_graph.has_edge(u, v):
                    edge_weight = self.causal_graph[u][v].get('weight', 1.0)
                    path_weight *= edge_weight
            
            # Apply Dynamic Decay
            decayed_strength = path_weight * (current_decay ** path_length)
            path_strengths.append(decayed_strength)
        
        return min(sum(path_strengths), 2.0)

    def compute(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str],
        target_name: str = 'target',
        model: Any = None
    ) -> Dict[str, Dict]:
        """
        Compute CII scores with Model Alignment for non-linear models.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            target_name: Name of target variable
            model: Trained model (required for non-linear model alignment)
            
        Returns:
            Dictionary with raw and normalized CII scores
        """
        logger.info("Computing Causal Influence Index (CII) with Model Alignment...")
        
        # 1. Determine Model Type
        model_type = 'default'
        if model is not None:
            m_str = str(type(model)).lower()
            if any(x in m_str for x in ['torch', 'mlp', 'dnn', 'neural', 'keras']): 
                model_type = 'neural'
            elif any(x in m_str for x in ['forest', 'xgb', 'tree', 'boost', 'decision']): 
                model_type = 'tree'
            elif any(x in m_str for x in ['logistic', 'linear', 'sgd']): 
                model_type = 'linear'
            
            logger.info(f"CII - Detected model type: '{model_type}'")
        
        # 2. Get Base Signal: Permutation Importance (Model Usage)
        # This ensures we only credit features the model ACTUALLY uses
        base_importance = None
        if model is not None and model_type in ['tree', 'neural']:
            logger.info("Using Permutation Importance as Base Signal (Fixing Non-Linear Bias)")
            base_importance = self._compute_permutation_importance(model, X, y, feature_names)
        
        # Fallback to Mutual Information for Linear models or failures
        # (Linear models work fine with MI, as proven by your results)
        use_mi = base_importance is None
        if use_mi:
            logger.info("Using Mutual Information as Base Signal")
            # Pre-process for MI
            X_numeric = X.copy()
            for col in X_numeric.columns:
                if X_numeric[col].dtype == 'object' or isinstance(X_numeric[col].dtype, pd.CategoricalDtype):
                    X_numeric[col] = pd.factorize(X_numeric[col])[0]
            # Get target signal
            try:
                if model and hasattr(model, 'predict_proba'):
                    y_target = model.predict_proba(X_numeric.values)[:, 1]
                elif model:
                    y_target = model.predict(X_numeric.values)
                else:
                    y_target = y
            except:
                y_target = y

        # 3. Compute CII Scores
        raw_cii_map = {}
        
        for feature in feature_names:
            if feature not in X.columns: 
                continue
            
            # A. Get Base Signal (CMI or Permutation)
            if use_mi:
                X_feat = X_numeric[feature].values
                try:
                    X_discrete = pd.cut(X_feat, 10, labels=False, duplicates='drop')
                    Y_discrete = pd.cut(y_target, 10, labels=False, duplicates='drop') if len(np.unique(y_target)) > 10 else y_target
                    
                    valid_idx = ~(pd.isna(X_discrete) | pd.isna(Y_discrete))
                    if np.sum(valid_idx) > 0:
                        cmi = mutual_info_score(
                            X_discrete[valid_idx].astype(str), 
                            Y_discrete[valid_idx].astype(str)
                        )
                    else:
                        cmi = 0.0
                except:
                    cmi = 0.0
            else:
                cmi = base_importance.get(feature, 0.0)
            
            # B. Compute Graph Propagation
            causal_strength = self._compute_path_strength(
                feature, target_name, model_type=model_type
            )
            
            # Debug: Log key values for glucose
            if 'glucose' in feature.lower():
                logger.info(f"  DEBUG glucose: cmi={cmi:.6f}, causal_strength={causal_strength:.6f}")
                logger.info(f"  DEBUG glucose: graph has {feature}? {self.causal_graph.has_node(feature)}")
                logger.info(f"  DEBUG glucose: graph has {target_name}? {self.causal_graph.has_node(target_name)}")
                if self.causal_graph.has_node(feature) and self.causal_graph.has_node(target_name):
                    logger.info(f"  DEBUG glucose: direct edge {feature}->{target_name}? {self.causal_graph.has_edge(feature, target_name)}")
            
            # C. Final Score
            raw_cii_map[feature] = cmi * causal_strength

        # 4. Normalization
        vals = np.array(list(raw_cii_map.values()))
        if len(vals) > 0 and vals.std() > 0:
            norm_cii = {k: (v - vals.min()) / (vals.max() - vals.min()) for k, v in raw_cii_map.items()}
        else:
            norm_cii = raw_cii_map.copy()
            
        self.raw_cii_scores = raw_cii_map
        
        logger.info(f"CII computed for {len(norm_cii)} features using {model_type} model alignment")
        
        return {
            'raw': raw_cii_map,
            'normalized': norm_cii,
            'viz_scaled': norm_cii
        }

    def compute_distribution_metrics(self) -> Dict[str, float]:
        """Compute CII distribution metrics (Gini, concentration)."""
        scores = np.array(list(self.raw_cii_scores.values()))
        if len(scores) == 0 or np.all(scores == 0):
            return {'gini': 0.0, 'top5_concentration': 0.0}
        
        sorted_scores = np.sort(scores)
        n = len(scores)
        cumsum = np.cumsum(sorted_scores)
        if cumsum[-1] == 0:
            gini = 0.0
        else:
            gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_scores)) / (n * cumsum[-1]) - (n + 1) / n
        
        top5_conc = np.sort(scores)[::-1][:5].sum() / scores.sum() if scores.sum() > 0 else 0.0
        
        return {'gini': gini, 'top5_concentration': top5_conc}


# ============================================================================
# 2. Causal Complexity Measure (CCM)
# ============================================================================

class CausalComplexityMeasure:
    def __init__(self, causal_graph: nx.DiGraph):
        self.causal_graph = causal_graph
    
    def compute(self, model: Any, X_sample: Optional[np.ndarray] = None, include_runtime=True, include_cognitive=True) -> Dict[str, float]:
        # 1. Graph Complexity (Von Neumann Entropy approx)
        n_nodes = self.causal_graph.number_of_nodes()
        n_edges = self.causal_graph.number_of_edges()
        max_edges = n_nodes * (n_nodes - 1)
        density = n_edges / max(max_edges, 1)
        C_struct = density + 0.3 * np.log1p(n_nodes) # Simplified proxy
        
        # 2. Model Complexity (Compression proxy)
        try:
            serialized = pickle.dumps(model)
            compressed = gzip.compress(serialized)
            C_model = np.log1p(len(compressed) / 1024) # KB log scale
        except:
            C_model = 1.0
            
        # Weights
        score = 0.5 * C_struct + 0.5 * C_model
        return {'ccm_total': score, 'graph_complexity': C_struct, 'model_complexity': C_model}


# ============================================================================
# 3. Transparency Entropy (TE)
# ============================================================================

class TransparencyEntropy:
    def __init__(self, calibrate: bool = True):
        self.calibrate = calibrate
    
    def compute(self, y_pred_proba: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
        # Ensure bounds
        probs = np.clip(y_pred_proba, 1e-10, 1.0 - 1e-10)
        
        # If binary classification (1D array), make it 2D
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs, probs])
            
        # Shannon Entropy per sample
        entropies = -np.sum(probs * np.log(probs), axis=1)
        avg_entropy = np.mean(entropies)
        
        # Normalize by max entropy (log2(n_classes))
        n_classes = probs.shape[1]
        max_ent = np.log(n_classes)
        te_score = avg_entropy / max_ent if max_ent > 0 else 0.0
        
        return {'te_score': te_score, 'avg_entropy': avg_entropy}


# ============================================================================
# 4. Counterfactual Stability (CS)
# ============================================================================

class CounterfactualStability:
    """
    Computes model stability under varying levels of input perturbation.
    Generates a 'Robustness Curve' by testing multiple noise scales.
    """
    def __init__(
        self, 
        num_perturbations: int = 10, 
        perturbation_scales: Union[float, List[float]] = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0], 
        causal_graph=None
    ):
        self.num_perturbations = num_perturbations
        
        # Handle single float or list of scales
        if isinstance(perturbation_scales, (float, int)):
            self.scales = [float(perturbation_scales)]
        else:
            self.scales = sorted(list(perturbation_scales))
            
    def compute(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        feature_types: Optional[List[str]] = None, 
        feature_names=None
    ) -> Dict[str, Any]:
        if not hasattr(model, 'predict_proba'):
            return {'cs_score': 0.0}
            
        # Limit samples for speed
        n_samples = min(len(X_test), 50)
        X_subset = X_test[:n_samples]
        base_preds = model.predict_proba(X_subset)
        
        # If binary (1D), make 2D
        if base_preds.ndim == 1:
            base_preds = np.column_stack([1 - base_preds, base_preds])
            
        results = {}
        all_stabilities = []
        
        logger.info(f"Computing Stability Curve across scales: {self.scales}")
        
        # LOOP OVER SCALES
        for scale in self.scales:
            stability_scores = []
            
            # Perturb each feature
            for i in range(X_subset.shape[1]):
                # Create perturbed copy
                X_pert = X_subset.copy()
                std = np.std(X_subset[:, i])
                
                # Apply noise at current scale
                # Ensure minimum noise for constant features
                noise_std = max(std, 1e-3) * scale
                noise = np.random.normal(0, noise_std, size=n_samples)
                X_pert[:, i] += noise
                
                # Predict
                new_preds = model.predict_proba(X_pert)
                if new_preds.ndim == 1:
                    new_preds = np.column_stack([1 - new_preds, new_preds])
                
                # TV Distance (L1 norm / 2)
                diff = np.abs(base_preds - new_preds).sum(axis=1) * 0.5
                stability = 1.0 - np.mean(diff)
                stability_scores.append(stability)
            
            # Average stability for this scale
            avg_stability = np.mean(stability_scores)
            results[f'cs_score_{scale}'] = avg_stability
            all_stabilities.append(avg_stability)
            
        # Overall Score (Average across all scales to get "Area Under Stability Curve")
        results['cs_score'] = np.mean(all_stabilities)
        results['scales'] = self.scales
        results['curve'] = all_stabilities
        results['feature_stability'] = {}
        
        return results


# ============================================================================
# CTF Framework - Complete Pipeline
# ============================================================================

class CTFFramework:
    """
    Complete Causal Transparency Framework.
    Computes all four metrics: CII, CCM, TE, CS.
    """
    
    def __init__(
        self,
        causal_graph: nx.DiGraph,
        model: Any,
        feature_names: List[str],
        target_name: str = 'target'
    ):
        """
        Initialize CTF Framework.
        
        Args:
            causal_graph: Learned causal graph
            model: Trained model
            feature_names: Feature names
            target_name: Target variable name
        """
        self.causal_graph = causal_graph
        self.model = model
        self.feature_names = feature_names
        self.target_name = target_name
        
        # Initialize metric calculators
        self.cii = CausalInfluenceIndex(causal_graph)
        self.ccm = CausalComplexityMeasure(causal_graph)
        self.te = TransparencyEntropy(calibrate=True)
        self.cs = EnhancedCounterfactualStability(
            causal_graph=causal_graph,
            perturbation_strengths=[0.1, 0.3, 0.5, 1.0, 2.0],
            n_samples_per_test=100,
            n_perturbation_rounds=10
        )
        
        # Results storage
        self.results = {}
    
    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_types: Optional[List[str]] = None,
        optimal_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all CTF metrics.
        
        Args:
            X_train: Training features (for CII)
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_types: Feature types for CS
            optimal_threshold: Optimal threshold
            
        Returns:
            Dictionary with all CTF metrics
        """
        logger.info("\n" + "="*80)
        logger.info("Evaluating Causal Transparency Framework (CTF)")
        logger.info("="*80)
        
        results = {}
        
        # 1. Causal Influence Index (CII)
        try:
            cii_results = self.cii.compute(
                X_train,
                y_train,
                self.feature_names,
                self.target_name,
                model=self.model
            )
            cii_distribution = self.cii.compute_distribution_metrics()
            results['cii'] = {
                **cii_results,
                'distribution': cii_distribution
            }
        except Exception as e:
            logger.error(f"Error computing CII: {e}")
            results['cii'] = None
        
        # 2. Causal Complexity Measure (CCM)
        try:
            ccm_results = self.ccm.compute(
                self.model,
                X_test[:100] if len(X_test) > 100 else X_test,
                include_runtime=True,
                include_cognitive=True
            )
            results['ccm'] = ccm_results
        except Exception as e:
            logger.error(f"Error computing CCM: {e}")
            results['ccm'] = None
        
        # 3. Transparency Entropy (TE)
        try:
            y_pred_proba = self.model.predict_proba(X_test)
            te_results = self.te.compute(y_pred_proba, y_test)
            results['te'] = te_results
        except Exception as e:
            logger.error(f"Error computing TE: {e}")
            results['te'] = None
        
        # 4. Enhanced Counterfactual Stability (CS)
        try:
            cs_results = self.cs.compute_enhanced_cs(
                X_test,
                self.model,
                self.feature_names
            )
            results['cs'] = cs_results
        except Exception as e:
            logger.error(f"Error computing Enhanced CS: {e}")
            results['cs'] = None
        
        self.results = results
        
        logger.info("\n" + "="*80)
        logger.info("CTF Evaluation Complete")
        logger.info("="*80)
        
        return results
    
    def summary(self) -> pd.DataFrame:
        """
        Generate summary of CTF metrics.
        
        Returns:
            DataFrame with metric summary
        """
        summary_data = []
        
        if self.results.get('cii'):
            cii_dist = self.results['cii']['distribution']
            summary_data.append({
                'Metric': 'CII (Gini)',
                'Value': cii_dist['gini'],
                'Description': 'Feature influence inequality'
            })
            summary_data.append({
                'Metric': 'CII (Top-5 Concentration)',
                'Value': cii_dist['top5_concentration'],
                'Description': 'Top-5 feature influence'
            })
        
        if self.results.get('ccm'):
            ccm = self.results['ccm']
            summary_data.append({
                'Metric': 'CCM (Total)',
                'Value': ccm['ccm_total'],
                'Description': 'Overall model complexity'
            })
        
        if self.results.get('te'):
            te = self.results['te']
            summary_data.append({
                'Metric': 'TE (Transparency Entropy)',
                'Value': te['te_score'],
                'Description': 'Prediction uncertainty'
            })
        
        if self.results.get('cs'):
            cs = self.results['cs']
            
            # Enhanced CS overall scores
            summary_data.append({
                'Metric': 'Enhanced CS Score',
                'Value': cs.get('enhanced_cs_score', 0),
                'Description': 'Multi-strategy stability score'
            })
            
            summary_data.append({
                'Metric': 'CS Discriminative Power',
                'Value': cs.get('discriminative_power', 0),
                'Description': 'Ability to differentiate models'
            })
            
            summary_data.append({
                'Metric': 'CS Robustness',
                'Value': cs.get('cs_robustness', 0),
                'Description': 'Consistency across perturbation strengths'
            })
            
            # Add breakdown by strategy
            strategies = cs.get('by_strategy', {})
            for strategy, scores in strategies.items():
                summary_data.append({
                    'Metric': f'CS ({strategy.title()})',
                    'Value': scores.get('mean', 0),
                    'Description': f'{strategy.replace("_", " ").title()} perturbation stability'
                })
        
        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    print("CTF Metrics module loaded.")