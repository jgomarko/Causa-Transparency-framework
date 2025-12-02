#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Counterfactual Stability with Discriminative Perturbations
==================================================================

Implements strengthened perturbation strategies to make Counterfactual Stability (CS) 
more discriminative across different model architectures and complexities.

Current CS shows ceiling effects (0.99+ for all models). This module introduces:
1. Multi-scale adversarial perturbations
2. Feature interaction perturbations  
3. Causal pathway-aware perturbations
4. Model-specific perturbation strategies
5. Ensemble perturbation testing

Author: John Marko
Date: 2025-11-30
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, uniform, lognorm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedCounterfactualStability:
    """
    Enhanced Counterfactual Stability with discriminative perturbations.
    
    Implements multiple perturbation strategies to reveal model robustness differences:
    - Adversarial perturbations (gradient-based for differentiable models)
    - Multi-scale noise (uniform, gaussian, heavy-tailed) 
    - Feature interaction perturbations
    - Causal pathway-aware perturbations
    - Realistic distributional shifts
    """
    
    def __init__(
        self, 
        causal_graph: Optional[nx.DiGraph] = None,
        perturbation_strengths: List[float] = [0.1, 0.3, 0.5, 1.0, 2.0],
        n_samples_per_test: int = 100,
        n_perturbation_rounds: int = 10
    ):
        """
        Initialize Enhanced CS calculator.
        
        Args:
            causal_graph: Causal graph for pathway-aware perturbations
            perturbation_strengths: Scaling factors for perturbation magnitude
            n_samples_per_test: Number of samples to perturb per test
            n_perturbation_rounds: Number of perturbation rounds for averaging
        """
        self.causal_graph = causal_graph
        self.perturbation_strengths = perturbation_strengths
        self.n_samples_per_test = n_samples_per_test
        self.n_perturbation_rounds = n_perturbation_rounds
        
        # Perturbation strategy weights
        self.strategy_weights = {
            'gaussian': 0.2,
            'adversarial': 0.25, 
            'interaction': 0.2,
            'pathway': 0.2,
            'distributional': 0.15
        }
    
    def _get_model_gradients(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract gradients for adversarial perturbations (if model supports it).
        
        Args:
            model: Trained model
            X: Input features
            
        Returns:
            Gradients w.r.t. inputs or None if not available
        """
        try:
            # Check if model has gradient information
            model_str = str(type(model)).lower()
            
            if 'neural' in model_str or 'mlp' in model_str:
                # For sklearn MLPClassifier, approximate gradients via finite differences
                return self._approximate_gradients_finite_diff(model, X)
            
            elif 'logistic' in model_str:
                # For logistic regression, use analytical gradients
                return self._logistic_gradients(model, X)
            
            elif 'xgb' in model_str or 'forest' in model_str:
                # Tree models: use feature importance as proxy for gradient direction
                return self._tree_importance_gradients(model, X)
            
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Gradient extraction failed: {e}")
            return None
    
    def _approximate_gradients_finite_diff(self, model: Any, X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Approximate gradients using finite differences."""
        n_samples, n_features = X.shape
        gradients = np.zeros_like(X)
        
        # Get base predictions
        if hasattr(model, 'predict_proba'):
            base_pred = model.predict_proba(X)[:, 1] if model.predict_proba(X).shape[1] > 1 else model.predict_proba(X).flatten()
        else:
            base_pred = model.predict(X)
        
        # Compute partial derivatives
        for j in range(min(n_features, 20)):  # Limit for efficiency
            X_plus = X.copy()
            X_plus[:, j] += eps
            
            if hasattr(model, 'predict_proba'):
                pred_plus = model.predict_proba(X_plus)[:, 1] if model.predict_proba(X_plus).shape[1] > 1 else model.predict_proba(X_plus).flatten()
            else:
                pred_plus = model.predict(X_plus)
            
            gradients[:, j] = (pred_plus - base_pred) / eps
        
        return gradients
    
    def _logistic_gradients(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Compute analytical gradients for logistic regression."""
        try:
            # Get predictions and coefficients
            probs = model.predict_proba(X)[:, 1]
            coef = model.coef_[0] if hasattr(model, 'coef_') else np.ones(X.shape[1])
            
            # Gradient: coef * prob * (1 - prob)
            grad_magnitude = probs * (1 - probs)
            gradients = np.outer(grad_magnitude, coef)
            
            return gradients
        except:
            return self._approximate_gradients_finite_diff(model, X)
    
    def _tree_importance_gradients(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Use feature importance as proxy gradients for tree models."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                importance = np.ones(X.shape[1]) / X.shape[1]
            
            # Use sign of features weighted by importance
            gradients = np.sign(X) * importance.reshape(1, -1)
            return gradients
        except:
            return np.random.randn(*X.shape) * 0.1
    
    def gaussian_perturbations(
        self, 
        X: np.ndarray, 
        model: Any, 
        strength: float = 1.0
    ) -> float:
        """
        Standard Gaussian noise perturbations with enhanced strength.
        
        Args:
            X: Input features
            model: Trained model
            strength: Perturbation strength multiplier
            
        Returns:
            Stability score (0-1, higher = more stable)
        """
        if hasattr(model, 'predict_proba'):
            base_preds = model.predict_proba(X)
            if base_preds.shape[1] > 1:
                base_preds = base_preds[:, 1]
            else:
                base_preds = base_preds.flatten()
        else:
            base_preds = model.predict(X)
        
        stabilities = []
        
        for _ in range(self.n_perturbation_rounds):
            # Enhanced noise with strength scaling
            noise_std = np.std(X, axis=0) * strength
            noise = np.random.normal(0, noise_std, X.shape)
            X_pert = X + noise
            
            # Get perturbed predictions
            if hasattr(model, 'predict_proba'):
                pert_preds = model.predict_proba(X_pert)
                if pert_preds.shape[1] > 1:
                    pert_preds = pert_preds[:, 1]
                else:
                    pert_preds = pert_preds.flatten()
            else:
                pert_preds = model.predict(X_pert)
            
            # Total Variation Distance
            tv_distance = np.mean(np.abs(base_preds - pert_preds))
            stability = max(0, 1 - tv_distance)
            stabilities.append(stability)
        
        return np.mean(stabilities)
    
    def adversarial_perturbations(
        self, 
        X: np.ndarray, 
        model: Any, 
        strength: float = 1.0
    ) -> float:
        """
        Adversarial perturbations using model gradients.
        
        Args:
            X: Input features
            model: Trained model  
            strength: Attack strength
            
        Returns:
            Stability score under adversarial attack
        """
        gradients = self._get_model_gradients(model, X)
        
        if gradients is None:
            # Fallback to random perturbations
            return self.gaussian_perturbations(X, model, strength * 2)
        
        # Get base predictions
        if hasattr(model, 'predict_proba'):
            base_preds = model.predict_proba(X)
            if base_preds.shape[1] > 1:
                base_preds = base_preds[:, 1]
            else:
                base_preds = base_preds.flatten()
        else:
            base_preds = model.predict(X)
        
        stabilities = []
        
        for _ in range(self.n_perturbation_rounds):
            # Gradient-based adversarial perturbation
            grad_norm = np.linalg.norm(gradients, axis=1, keepdims=True) + 1e-8
            normalized_grads = gradients / grad_norm
            
            # Scale by feature std and strength
            feature_scales = np.std(X, axis=0)
            adversarial_noise = normalized_grads * feature_scales * strength * 0.5
            
            X_adv = X + adversarial_noise
            
            # Get adversarial predictions
            if hasattr(model, 'predict_proba'):
                adv_preds = model.predict_proba(X_adv)
                if adv_preds.shape[1] > 1:
                    adv_preds = adv_preds[:, 1]
                else:
                    adv_preds = adv_preds.flatten()
            else:
                adv_preds = model.predict(X_adv)
            
            # Stability under adversarial attack
            tv_distance = np.mean(np.abs(base_preds - adv_preds))
            stability = max(0, 1 - tv_distance)
            stabilities.append(stability)
        
        return np.mean(stabilities)
    
    def interaction_perturbations(
        self, 
        X: np.ndarray, 
        model: Any, 
        strength: float = 1.0
    ) -> float:
        """
        Feature interaction perturbations - perturb correlated features together.
        
        Args:
            X: Input features
            model: Trained model
            strength: Perturbation strength
            
        Returns:
            Stability score under interaction perturbations
        """
        # Compute feature correlations
        corr_matrix = np.corrcoef(X.T)
        
        if hasattr(model, 'predict_proba'):
            base_preds = model.predict_proba(X)
            if base_preds.shape[1] > 1:
                base_preds = base_preds[:, 1]
            else:
                base_preds = base_preds.flatten()
        else:
            base_preds = model.predict(X)
        
        stabilities = []
        
        for _ in range(self.n_perturbation_rounds):
            X_pert = X.copy()
            
            # Find highly correlated feature pairs
            n_features = X.shape[1]
            for i in range(min(n_features, 10)):  # Limit for efficiency
                for j in range(i + 1, min(n_features, 10)):
                    if abs(corr_matrix[i, j]) > 0.3:  # Moderate correlation threshold
                        # Perturb correlated features together
                        noise_i = np.random.normal(0, np.std(X[:, i]) * strength * 0.8)
                        noise_j = np.random.normal(0, np.std(X[:, j]) * strength * 0.8)
                        
                        # Correlated noise
                        X_pert[:, i] += noise_i
                        X_pert[:, j] += noise_j * np.sign(corr_matrix[i, j])
            
            # Get perturbed predictions
            if hasattr(model, 'predict_proba'):
                pert_preds = model.predict_proba(X_pert)
                if pert_preds.shape[1] > 1:
                    pert_preds = pert_preds[:, 1]
                else:
                    pert_preds = pert_preds.flatten()
            else:
                pert_preds = model.predict(X_pert)
            
            tv_distance = np.mean(np.abs(base_preds - pert_preds))
            stability = max(0, 1 - tv_distance)
            stabilities.append(stability)
        
        return np.mean(stabilities)
    
    def pathway_aware_perturbations(
        self, 
        X: np.ndarray, 
        model: Any, 
        feature_names: List[str],
        strength: float = 1.0
    ) -> float:
        """
        Causal pathway-aware perturbations using the causal graph.
        
        Args:
            X: Input features
            model: Trained model
            feature_names: Feature names
            strength: Perturbation strength
            
        Returns:
            Stability score under pathway-aware perturbations
        """
        if self.causal_graph is None:
            return self.gaussian_perturbations(X, model, strength)
        
        if hasattr(model, 'predict_proba'):
            base_preds = model.predict_proba(X)
            if base_preds.shape[1] > 1:
                base_preds = base_preds[:, 1]
            else:
                base_preds = base_preds.flatten()
        else:
            base_preds = model.predict(X)
        
        stabilities = []
        
        for _ in range(self.n_perturbation_rounds):
            X_pert = X.copy()
            
            # Perturb along causal pathways
            for i, feature in enumerate(feature_names[:X.shape[1]]):
                if feature in self.causal_graph.nodes():
                    # Get causal descendants
                    descendants = list(nx.descendants(self.causal_graph, feature))
                    
                    # Stronger perturbation for causally important features
                    if len(descendants) > 0:
                        causal_importance = len(descendants) / len(self.causal_graph.nodes())
                        enhanced_strength = strength * (1 + causal_importance)
                    else:
                        enhanced_strength = strength * 0.5  # Weaker for leaf nodes
                    
                    # Apply perturbation
                    noise = np.random.normal(0, np.std(X[:, i]) * enhanced_strength * 0.7)
                    X_pert[:, i] += noise
            
            # Get perturbed predictions
            if hasattr(model, 'predict_proba'):
                pert_preds = model.predict_proba(X_pert)
                if pert_preds.shape[1] > 1:
                    pert_preds = pert_preds[:, 1]
                else:
                    pert_preds = pert_preds.flatten()
            else:
                pert_preds = model.predict(X_pert)
            
            tv_distance = np.mean(np.abs(base_preds - pert_preds))
            stability = max(0, 1 - tv_distance)
            stabilities.append(stability)
        
        return np.mean(stabilities)
    
    def distributional_shift_perturbations(
        self, 
        X: np.ndarray, 
        model: Any, 
        strength: float = 1.0
    ) -> float:
        """
        Realistic distributional shift perturbations.
        
        Args:
            X: Input features
            model: Trained model
            strength: Shift magnitude
            
        Returns:
            Stability under distributional shifts
        """
        if hasattr(model, 'predict_proba'):
            base_preds = model.predict_proba(X)
            if base_preds.shape[1] > 1:
                base_preds = base_preds[:, 1]
            else:
                base_preds = base_preds.flatten()
        else:
            base_preds = model.predict(X)
        
        stabilities = []
        
        for _ in range(self.n_perturbation_rounds):
            X_pert = X.copy()
            
            for j in range(X.shape[1]):
                feature_values = X[:, j]
                
                # Apply different distributional shifts
                shift_type = np.random.choice(['location', 'scale', 'skew'])
                
                if shift_type == 'location':
                    # Location shift (bias)
                    shift = np.std(feature_values) * strength * 0.5
                    X_pert[:, j] += shift
                    
                elif shift_type == 'scale':
                    # Scale shift (variance change)
                    scale_factor = 1 + strength * 0.3
                    X_pert[:, j] = feature_values * scale_factor
                    
                elif shift_type == 'skew':
                    # Skewness shift (heavy tails)
                    if np.random.random() < 0.5:
                        # Heavy tail noise
                        heavy_noise = lognorm.rvs(
                            s=strength * 0.2, 
                            scale=np.std(feature_values) * 0.3, 
                            size=len(feature_values)
                        )
                        X_pert[:, j] += heavy_noise
            
            # Get predictions under distributional shift
            if hasattr(model, 'predict_proba'):
                pert_preds = model.predict_proba(X_pert)
                if pert_preds.shape[1] > 1:
                    pert_preds = pert_preds[:, 1]
                else:
                    pert_preds = pert_preds.flatten()
            else:
                pert_preds = model.predict(X_pert)
            
            tv_distance = np.mean(np.abs(base_preds - pert_preds))
            stability = max(0, 1 - tv_distance)
            stabilities.append(stability)
        
        return np.mean(stabilities)
    
    def compute_enhanced_cs(
        self, 
        X_test: np.ndarray, 
        model: Any,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute Enhanced Counterfactual Stability across multiple strategies.
        
        Args:
            X_test: Test features
            model: Trained model
            feature_names: Feature names for pathway analysis
            
        Returns:
            Comprehensive CS results with strategy breakdown
        """
        # Limit sample size for efficiency
        n_samples = min(len(X_test), self.n_samples_per_test)
        X_subset = X_test[:n_samples]
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_subset.shape[1])]
        
        logger.info(f"Computing Enhanced CS on {n_samples} samples with {len(self.perturbation_strengths)} strength levels")
        
        results = {
            'metadata': {
                'n_samples': n_samples,
                'n_features': X_subset.shape[1],
                'perturbation_strengths': self.perturbation_strengths,
                'strategy_weights': self.strategy_weights
            },
            'by_strategy': {},
            'by_strength': {},
            'overall_scores': {}
        }
        
        # Test each perturbation strategy across strength levels
        for strength in self.perturbation_strengths:
            logger.info(f"  Testing perturbation strength: {strength}")
            
            strength_results = {}
            
            # 1. Enhanced Gaussian perturbations
            gaussian_stability = self.gaussian_perturbations(X_subset, model, strength)
            strength_results['gaussian'] = gaussian_stability
            
            # 2. Adversarial perturbations
            adversarial_stability = self.adversarial_perturbations(X_subset, model, strength)
            strength_results['adversarial'] = adversarial_stability
            
            # 3. Feature interaction perturbations
            interaction_stability = self.interaction_perturbations(X_subset, model, strength)
            strength_results['interaction'] = interaction_stability
            
            # 4. Pathway-aware perturbations
            pathway_stability = self.pathway_aware_perturbations(X_subset, model, feature_names, strength)
            strength_results['pathway'] = pathway_stability
            
            # 5. Distributional shift perturbations
            distributional_stability = self.distributional_shift_perturbations(X_subset, model, strength)
            strength_results['distributional'] = distributional_stability
            
            results['by_strength'][strength] = strength_results
            
            # Weighted average for this strength
            weighted_stability = sum(
                stability * self.strategy_weights[strategy]
                for strategy, stability in strength_results.items()
            )
            results['overall_scores'][strength] = weighted_stability
        
        # Aggregate by strategy (average across strengths)
        for strategy in self.strategy_weights.keys():
            strategy_scores = [
                results['by_strength'][strength][strategy]
                for strength in self.perturbation_strengths
            ]
            results['by_strategy'][strategy] = {
                'mean': np.mean(strategy_scores),
                'std': np.std(strategy_scores),
                'min': np.min(strategy_scores),
                'scores_by_strength': strategy_scores
            }
        
        # Final Enhanced CS score (weighted average across all strategies and strengths)
        all_scores = list(results['overall_scores'].values())
        results['enhanced_cs_score'] = np.mean(all_scores)
        results['cs_robustness'] = 1 - np.std(all_scores)  # Higher when stable across strengths
        results['cs_vulnerability'] = 1 - np.min(all_scores)  # Worst-case stability
        
        # Discriminative power (how much variation across strengths)
        results['discriminative_power'] = np.std(all_scores)
        
        logger.info(f"Enhanced CS Score: {results['enhanced_cs_score']:.4f}")
        logger.info(f"CS Robustness: {results['cs_robustness']:.4f}")
        logger.info(f"Discriminative Power: {results['discriminative_power']:.4f}")
        
        return results


def compare_enhanced_vs_original_cs(models_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Compare Enhanced CS vs Original CS across different models.
    
    Args:
        models_data: Dictionary with model results
        
    Returns:
        Comparison DataFrame
    """
    comparison_results = []
    
    for model_name, data in models_data.items():
        original_cs = data.get('original_cs', 0.99)  # Typical high value
        enhanced_cs = data.get('enhanced_cs_score', 0.8)
        discriminative_power = data.get('discriminative_power', 0.1)
        
        comparison_results.append({
            'Model': model_name,
            'Original_CS': original_cs,
            'Enhanced_CS': enhanced_cs,
            'Improvement': original_cs - enhanced_cs,  # Lower is worse (more discriminative)
            'Discriminative_Power': discriminative_power,
            'Most_Vulnerable_To': min(data.get('by_strategy', {}), key=lambda x: data['by_strategy'][x]['mean']) if 'by_strategy' in data else 'Unknown'
        })
    
    return pd.DataFrame(comparison_results)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("=== ENHANCED COUNTERFACTUAL STABILITY TEST ===")
    
    # Create synthetic test data
    np.random.seed(42)
    X_test = np.random.randn(200, 10)
    
    # Create a simple test model
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    y_train = np.random.binomial(1, 0.3, 200)
    
    # Test models
    models = {
        'Logistic_Regression': LogisticRegression().fit(X_test, y_train),
        'Random_Forest': RandomForestClassifier(n_estimators=50, random_state=42).fit(X_test, y_train)
    }
    
    # Initialize enhanced CS
    enhanced_cs = EnhancedCounterfactualStability(
        perturbation_strengths=[0.2, 0.5, 1.0, 1.5, 2.5],
        n_samples_per_test=50,
        n_perturbation_rounds=5
    )
    
    # Test each model
    results_by_model = {}
    
    for model_name, model in models.items():
        print(f"\n=== Testing {model_name} ===")
        
        cs_results = enhanced_cs.compute_enhanced_cs(X_test, model)
        results_by_model[model_name] = cs_results
        
        print(f"Enhanced CS: {cs_results['enhanced_cs_score']:.4f}")
        print(f"Discriminative Power: {cs_results['discriminative_power']:.4f}")
        print("Strategy Vulnerabilities:")
        for strategy, scores in cs_results['by_strategy'].items():
            print(f"  {strategy}: {scores['mean']:.4f} ± {scores['std']:.4f}")
    
    # Generate comparison
    print(f"\n=== MODEL COMPARISON ===")
    comparison_df = compare_enhanced_vs_original_cs(results_by_model)
    print(comparison_df.to_string(index=False, float_format='{:.4f}'.format))
    
    print(f"\n=== ENHANCED CS IMPLEMENTATION COMPLETE ===")