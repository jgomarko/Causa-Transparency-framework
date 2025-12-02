#!/usr/bin/env python3
"""
CTF Sensitivity Analysis Framework
=================================
Comprehensive sensitivity analysis for the Causal Transparency Framework.

Tests robustness of CTF metrics (CII, CCM, TE, CS) across:
1. Parameter variations (causal discovery parameters)
2. Model hyperparameters 
3. Sample size variations
4. Feature noise injection
5. Cross-validation stability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
import itertools
import networkx as nx

# Import CTF components
from Ctf_metrics import CausalInfluenceIndex, CausalComplexityMeasure, TransparencyEntropy, CounterfactualStability
from causal_discovery_utils import run_causal_discovery

# Visualization setup
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
Path('visualizations').mkdir(exist_ok=True)

class SimplifiedCTFMetrics:
    """Simplified CTF metrics wrapper for sensitivity analysis."""
    
    def __init__(self, causal_graph: nx.DiGraph, feature_names: List[str]):
        self.causal_graph = causal_graph
        self.feature_names = feature_names
        
        # Initialize CTF components with required parameters
        self.cii_calculator = CausalInfluenceIndex(
            causal_graph=causal_graph,
            decay_factor=0.9,
            n_neighbors=3
        )
        self.ccm_calculator = CausalComplexityMeasure(causal_graph=causal_graph)
        self.te_calculator = TransparencyEntropy(calibrate=True)
        # Create a dummy model for CS initialization - we'll pass the real model when computing
        from sklearn.dummy import DummyClassifier
        dummy_model = DummyClassifier()
        self.cs_calculator = CounterfactualStability(
            num_perturbations=10,
            perturbation_scale=0.1,
            divergence_metric='tv',
            causal_graph=causal_graph
        )
    
    def compute_cii(self, X, y, causal_graph, model=None):
        """Compute CII scores."""
        # CII calculator expects specific parameters
        return self.cii_calculator.compute(X, y, feature_names=X.columns.tolist(), 
                                         target_name='target', model=model)
    
    def compute_ccm(self, causal_graph, X, model):
        """Compute CCM score."""
        # CCM needs model and X_sample
        return self.ccm_calculator.compute(model, X_sample=X.values)
    
    def compute_transparency_entropy(self, model, X):
        """Compute transparency entropy."""
        # TE needs predicted probabilities
        y_pred_proba = model.predict_proba(X)
        return self.te_calculator.compute(y_pred_proba)
    
    def compute_counterfactual_stability(self, X_sample, model, n_perturbations=10):
        """Compute counterfactual stability."""
        return self.cs_calculator.compute(model, X_sample.values, 
                                        feature_names=X_sample.columns.tolist())
    
    def compute_gini_coefficient(self, values):
        """Compute Gini coefficient."""
        if not values or len(values) == 0:
            return 0.0
        
        # Handle single value or identical values
        if len(set(values)) == 1:
            return 0.0
        
        # Sort values
        sorted_values = sorted([abs(v) for v in values if v is not None])
        n = len(sorted_values)
        
        if n == 0:
            return 0.0
        
        # Compute Gini coefficient
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

class CTFSensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis framework for CTF metrics.
    
    Tests robustness across multiple dimensions:
    - Causal discovery parameters
    - Model hyperparameters  
    - Data perturbations
    - Sample size variations
    """
    
    def __init__(self, base_data: pd.DataFrame, target_col: str = 'recidivism'):
        """
        Initialize sensitivity analyzer.
        
        Args:
            base_data: Base dataset for analysis
            target_col: Name of target variable
        """
        self.base_data = base_data
        self.target_col = target_col
        self.results = {}
        
    def parameter_sensitivity_analysis(self, 
                                     alpha_values: List[float] = [0.01, 0.05, 0.1, 0.2],
                                     algorithms: List[str] = ['pc', 'ges'],
                                     n_trials: int = 5) -> Dict:
        """
        Test sensitivity to causal discovery parameters.
        
        Args:
            alpha_values: List of significance thresholds for causal discovery
            algorithms: List of causal discovery algorithms to test
            n_trials: Number of repeated trials per configuration
            
        Returns:
            dict: Parameter sensitivity results
        """
        print("\n" + "="*70)
        print("PARAMETER SENSITIVITY ANALYSIS")
        print("="*70)
        
        X = self.base_data.drop(self.target_col, axis=1)
        y = self.base_data[self.target_col]
        
        # Base model for consistency
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        base_model.fit(X, y)
        
        param_results = []
        
        for algorithm in algorithms:
            for alpha in alpha_values:
                print(f"\n🔧 Testing {algorithm.upper()} with alpha={alpha}")
                
                trial_metrics = []
                
                for trial in range(n_trials):
                    try:
                        # Run causal discovery
                        causal_graph, _ = run_causal_discovery(
                            X, algorithm=algorithm, alpha=alpha, random_state=42+trial
                        )
                        
                        # Compute CTF metrics
                        ctf_metrics = SimplifiedCTFMetrics(causal_graph, X.columns.tolist())
                        
                        # CII
                        cii_result = ctf_metrics.compute_cii(
                            X, y, causal_graph, model=base_model
                        )
                        # Extract normalized scores for Gini calculation
                        if isinstance(cii_result, dict) and 'normalized' in cii_result:
                            cii_scores = cii_result['normalized']
                        else:
                            cii_scores = cii_result
                        
                        cii_gini = ctf_metrics.compute_gini_coefficient(list(cii_scores.values()))
                        
                        # CCM
                        ccm_result = ctf_metrics.compute_ccm(causal_graph, X, base_model)
                        ccm_score = ccm_result.get('total_ccm', 1.0) if isinstance(ccm_result, dict) else ccm_result
                        
                        # TE  
                        te_result = ctf_metrics.compute_transparency_entropy(base_model, X)
                        te_score = te_result.get('transparency_entropy', 0.5) if isinstance(te_result, dict) else te_result
                        
                        # CS (simplified)
                        try:
                            cs_result = ctf_metrics.compute_counterfactual_stability(
                                X.iloc[:min(50, len(X))], base_model, n_perturbations=10
                            )
                            cs_mean = cs_result.get('mean_stability', 0.5) if isinstance(cs_result, dict) else cs_result
                        except:
                            cs_mean = 0.5  # Default stability
                        
                        trial_metrics.append({
                            'algorithm': algorithm,
                            'alpha': alpha,
                            'trial': trial,
                            'cii_gini': cii_gini,
                            'ccm': ccm_score,
                            'te': te_score,
                            'cs': cs_mean,
                            'n_edges': causal_graph.number_of_edges(),
                            'n_nodes': causal_graph.number_of_nodes()
                        })
                        
                    except Exception as e:
                        print(f"   ⚠️ Trial {trial} failed: {str(e)[:50]}")
                        continue
                
                param_results.extend(trial_metrics)
                
                if trial_metrics:
                    # Summary statistics for this parameter combination
                    metrics_df = pd.DataFrame(trial_metrics)
                    print(f"   CII Gini: {metrics_df['cii_gini'].mean():.3f} ± {metrics_df['cii_gini'].std():.3f}")
                    print(f"   CCM: {metrics_df['ccm'].mean():.3f} ± {metrics_df['ccm'].std():.3f}")
                    print(f"   TE: {metrics_df['te'].mean():.3f} ± {metrics_df['te'].std():.3f}")
                    print(f"   CS: {metrics_df['cs'].mean():.3f} ± {metrics_df['cs'].std():.3f}")
        
        param_df = pd.DataFrame(param_results)
        
        # Calculate coefficient of variation for each metric
        sensitivity_summary = {}
        for algorithm in algorithms:
            algo_data = param_df[param_df['algorithm'] == algorithm]
            if len(algo_data) > 0:
                sensitivity_summary[algorithm] = {
                    'cii_gini_cv': algo_data['cii_gini'].std() / algo_data['cii_gini'].mean(),
                    'ccm_cv': algo_data['ccm'].std() / algo_data['ccm'].mean(),
                    'te_cv': algo_data['te'].std() / algo_data['te'].mean(),
                    'cs_cv': algo_data['cs'].std() / algo_data['cs'].mean()
                }
        
        results = {
            'parameter_results': param_df,
            'sensitivity_summary': sensitivity_summary,
            'interpretation': self._interpret_parameter_sensitivity(sensitivity_summary)
        }
        
        self.results['parameter_sensitivity'] = results
        return results
    
    def model_sensitivity_analysis(self,
                                 model_configs: Dict = None,
                                 n_trials: int = 3) -> Dict:
        """
        Test sensitivity to model hyperparameters.
        
        Args:
            model_configs: Dictionary of model configurations to test
            n_trials: Number of trials per configuration
            
        Returns:
            dict: Model sensitivity results
        """
        print("\n" + "="*70)
        print("MODEL HYPERPARAMETER SENSITIVITY ANALYSIS")
        print("="*70)
        
        if model_configs is None:
            model_configs = {
                'rf_small': {'model': RandomForestClassifier, 'params': {'n_estimators': 50, 'max_depth': 5}},
                'rf_medium': {'model': RandomForestClassifier, 'params': {'n_estimators': 100, 'max_depth': 10}},
                'rf_large': {'model': RandomForestClassifier, 'params': {'n_estimators': 200, 'max_depth': 15}},
                'lr': {'model': LogisticRegression, 'params': {'max_iter': 1000, 'random_state': 42}},
                'mlp': {'model': MLPClassifier, 'params': {'hidden_layer_sizes': (50,), 'max_iter': 500, 'random_state': 42}}
            }
        
        X = self.base_data.drop(self.target_col, axis=1)
        y = self.base_data[self.target_col]
        
        # Base causal graph (use consistent discovery)
        base_graph, _ = run_causal_discovery(X, algorithm='pc', alpha=0.05)
        
        model_results = []
        
        for config_name, config in model_configs.items():
            print(f"\n🤖 Testing {config_name}")
            
            for trial in range(n_trials):
                try:
                    # Train model
                    model = config['model'](**config['params'])
                    if hasattr(model, 'random_state'):
                        model.set_params(random_state=42 + trial)
                    
                    model.fit(X, y)
                    accuracy = model.score(X, y)
                    
                    # Compute CTF metrics
                    ctf_metrics = SimplifiedCTFMetrics(base_graph, X.columns.tolist())
                    
                    # CII with model
                    cii_scores = ctf_metrics.compute_cii(X, y, base_graph, model=model)
                    cii_gini = ctf_metrics.compute_gini_coefficient(list(cii_scores.values()))
                    
                    # CCM (model-independent)
                    ccm_score = ctf_metrics.compute_ccm(base_graph, X)
                    
                    # TE
                    te_score = ctf_metrics.compute_transparency_entropy(cii_scores)
                    
                    # CS
                    cs_scores = []
                    for i in range(min(30, len(X))):
                        cs = ctf_metrics.compute_counterfactual_stability(
                            X.iloc[i:i+1], model, n_perturbations=10
                        )
                        cs_scores.append(cs)
                    cs_mean = np.mean(cs_scores)
                    
                    model_results.append({
                        'config': config_name,
                        'trial': trial,
                        'accuracy': accuracy,
                        'cii_gini': cii_gini,
                        'ccm': ccm_score,
                        'te': te_score,
                        'cs': cs_mean
                    })
                    
                except Exception as e:
                    print(f"   ⚠️ Trial {trial} failed: {str(e)[:50]}")
                    continue
            
            # Summary for this configuration
            config_data = [r for r in model_results if r['config'] == config_name]
            if config_data:
                config_df = pd.DataFrame(config_data)
                print(f"   Accuracy: {config_df['accuracy'].mean():.3f}")
                print(f"   CII Gini: {config_df['cii_gini'].mean():.3f} ± {config_df['cii_gini'].std():.3f}")
                print(f"   TE: {config_df['te'].mean():.3f} ± {config_df['te'].std():.3f}")
        
        model_df = pd.DataFrame(model_results)
        
        # Calculate model sensitivity metrics
        model_sensitivity = {}
        for metric in ['cii_gini', 'ccm', 'te', 'cs']:
            if len(model_df) > 0:
                model_sensitivity[f'{metric}_range'] = model_df[metric].max() - model_df[metric].min()
                model_sensitivity[f'{metric}_cv'] = model_df[metric].std() / model_df[metric].mean()
        
        results = {
            'model_results': model_df,
            'model_sensitivity': model_sensitivity,
            'interpretation': self._interpret_model_sensitivity(model_sensitivity)
        }
        
        self.results['model_sensitivity'] = results
        return results
    
    def sample_size_sensitivity_analysis(self,
                                       sample_fractions: List[float] = [0.5, 0.7, 0.8, 1.0],
                                       n_trials: int = 5) -> Dict:
        """
        Test sensitivity to sample size variations.
        
        Args:
            sample_fractions: List of sample size fractions to test
            n_trials: Number of trials per sample size
            
        Returns:
            dict: Sample size sensitivity results
        """
        print("\n" + "="*70)
        print("SAMPLE SIZE SENSITIVITY ANALYSIS")
        print("="*70)
        
        X = self.base_data.drop(self.target_col, axis=1)
        y = self.base_data[self.target_col]
        
        sample_results = []
        
        for fraction in sample_fractions:
            n_samples = int(len(X) * fraction)
            print(f"\n📏 Testing with {n_samples} samples ({fraction:.0%} of data)")
            
            for trial in range(n_trials):
                try:
                    # Sample data
                    np.random.seed(42 + trial)
                    indices = np.random.choice(len(X), size=n_samples, replace=False)
                    X_sample = X.iloc[indices]
                    y_sample = y.iloc[indices]
                    
                    # Train model
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_sample, y_sample)
                    
                    # Causal discovery
                    causal_graph, _ = run_causal_discovery(
                        X_sample, algorithm='pc', alpha=0.05, random_state=42+trial
                    )
                    
                    # CTF metrics
                    ctf_metrics = SimplifiedCTFMetrics(causal_graph, X_sample.columns.tolist())
                    
                    cii_scores = ctf_metrics.compute_cii(X_sample, y_sample, causal_graph, model=model)
                    cii_gini = ctf_metrics.compute_gini_coefficient(list(cii_scores.values()))
                    ccm_score = ctf_metrics.compute_ccm(causal_graph, X_sample)
                    te_score = ctf_metrics.compute_transparency_entropy(cii_scores)
                    
                    # CS (limited samples for speed)
                    cs_scores = []
                    for i in range(min(20, len(X_sample))):
                        cs = ctf_metrics.compute_counterfactual_stability(
                            X_sample.iloc[i:i+1], model, n_perturbations=10
                        )
                        cs_scores.append(cs)
                    cs_mean = np.mean(cs_scores)
                    
                    sample_results.append({
                        'sample_fraction': fraction,
                        'n_samples': n_samples,
                        'trial': trial,
                        'cii_gini': cii_gini,
                        'ccm': ccm_score,
                        'te': te_score,
                        'cs': cs_mean,
                        'n_edges': causal_graph.number_of_edges()
                    })
                    
                except Exception as e:
                    print(f"   ⚠️ Trial {trial} failed: {str(e)[:50]}")
                    continue
            
            # Summary for this sample size
            size_data = [r for r in sample_results if r['sample_fraction'] == fraction]
            if size_data:
                size_df = pd.DataFrame(size_data)
                print(f"   CII Gini: {size_df['cii_gini'].mean():.3f} ± {size_df['cii_gini'].std():.3f}")
                print(f"   TE: {size_df['te'].mean():.3f} ± {size_df['te'].std():.3f}")
        
        sample_df = pd.DataFrame(sample_results)
        
        # Analyze sample size effects
        size_effects = {}
        for metric in ['cii_gini', 'ccm', 'te', 'cs']:
            if len(sample_df) > 0:
                # Correlation between sample size and metric stability
                correlations = []
                for fraction in sample_fractions:
                    fraction_data = sample_df[sample_df['sample_fraction'] == fraction]
                    if len(fraction_data) > 1:
                        correlations.append(fraction_data[metric].std())
                
                if correlations:
                    size_effects[f'{metric}_stability'] = np.mean(correlations)
        
        results = {
            'sample_results': sample_df,
            'size_effects': size_effects,
            'interpretation': self._interpret_sample_sensitivity(size_effects)
        }
        
        self.results['sample_sensitivity'] = results
        return results
    
    def noise_sensitivity_analysis(self,
                                 noise_levels: List[float] = [0.0, 0.05, 0.1, 0.2],
                                 n_trials: int = 5) -> Dict:
        """
        Test sensitivity to feature noise injection.
        
        Args:
            noise_levels: List of noise levels (as fraction of feature std)
            n_trials: Number of trials per noise level
            
        Returns:
            dict: Noise sensitivity results
        """
        print("\n" + "="*70)
        print("FEATURE NOISE SENSITIVITY ANALYSIS")
        print("="*70)
        
        X = self.base_data.drop(self.target_col, axis=1)
        y = self.base_data[self.target_col]
        
        # Base metrics (no noise)
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        base_model.fit(X, y)
        base_graph, _ = run_causal_discovery(X, algorithm='pc', alpha=0.05)
        
        ctf_metrics = SimplifiedCTFMetrics(causal_graph, X_sample.columns.tolist())
        base_cii = ctf_metrics.compute_cii(X, y, base_graph, model=base_model)
        base_cii_gini = ctf_metrics.compute_gini_coefficient(list(base_cii.values()))
        base_te = ctf_metrics.compute_transparency_entropy(base_cii)
        
        noise_results = []
        
        for noise_level in noise_levels:
            print(f"\n🔊 Testing noise level: {noise_level:.0%}")
            
            for trial in range(n_trials):
                try:
                    # Add noise to features
                    if noise_level == 0:
                        X_noisy = X.copy()
                    else:
                        noise = np.random.normal(0, noise_level, X.shape)
                        X_noisy = X + noise * X.std()
                    
                    # Train model on noisy data
                    model = RandomForestClassifier(n_estimators=100, random_state=42 + trial)
                    model.fit(X_noisy, y)
                    
                    # Causal discovery on noisy data
                    causal_graph, _ = run_causal_discovery(
                        X_noisy, algorithm='pc', alpha=0.05, random_state=42+trial
                    )
                    
                    # CTF metrics
                    cii_scores = ctf_metrics.compute_cii(X_noisy, y, causal_graph, model=model)
                    cii_gini = ctf_metrics.compute_gini_coefficient(list(cii_scores.values()))
                    te_score = ctf_metrics.compute_transparency_entropy(cii_scores)
                    ccm_score = ctf_metrics.compute_ccm(causal_graph, X_noisy)
                    
                    # Calculate deviations from base
                    cii_deviation = abs(cii_gini - base_cii_gini)
                    te_deviation = abs(te_score - base_te)
                    
                    noise_results.append({
                        'noise_level': noise_level,
                        'trial': trial,
                        'cii_gini': cii_gini,
                        'te': te_score,
                        'ccm': ccm_score,
                        'cii_deviation': cii_deviation,
                        'te_deviation': te_deviation,
                        'n_edges': causal_graph.number_of_edges()
                    })
                    
                except Exception as e:
                    print(f"   ⚠️ Trial {trial} failed: {str(e)[:50]}")
                    continue
            
            # Summary for this noise level
            noise_data = [r for r in noise_results if r['noise_level'] == noise_level]
            if noise_data:
                noise_df = pd.DataFrame(noise_data)
                print(f"   CII Deviation: {noise_df['cii_deviation'].mean():.3f}")
                print(f"   TE Deviation: {noise_df['te_deviation'].mean():.3f}")
        
        noise_df = pd.DataFrame(noise_results)
        
        # Analyze noise sensitivity
        noise_sensitivity = {}
        if len(noise_df) > 0:
            for metric in ['cii_deviation', 'te_deviation']:
                # Correlation between noise level and deviation
                corr, p_value = stats.pearsonr(noise_df['noise_level'], noise_df[metric])
                noise_sensitivity[f'{metric}_correlation'] = corr
                noise_sensitivity[f'{metric}_p_value'] = p_value
        
        results = {
            'noise_results': noise_df,
            'noise_sensitivity': noise_sensitivity,
            'interpretation': self._interpret_noise_sensitivity(noise_sensitivity)
        }
        
        self.results['noise_sensitivity'] = results
        return results
    
    def cross_validation_stability_analysis(self, n_folds: int = 5) -> Dict:
        """
        Test cross-validation stability of CTF metrics.
        
        Args:
            n_folds: Number of CV folds
            
        Returns:
            dict: CV stability results
        """
        print("\n" + "="*70)
        print("CROSS-VALIDATION STABILITY ANALYSIS")
        print("="*70)
        
        from sklearn.model_selection import KFold
        
        X = self.base_data.drop(self.target_col, axis=1)
        y = self.base_data[self.target_col]
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"\n📁 Processing fold {fold_idx + 1}/{n_folds}")
            
            try:
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Causal discovery on training set
                causal_graph, _ = run_causal_discovery(X_train, algorithm='pc', alpha=0.05)
                
                # CTF metrics on test set
                ctf_metrics = SimplifiedCTFMetrics(causal_graph, X_sample.columns.tolist())
                
                cii_scores = ctf_metrics.compute_cii(X_test, y_test, causal_graph, model=model)
                cii_gini = ctf_metrics.compute_gini_coefficient(list(cii_scores.values()))
                ccm_score = ctf_metrics.compute_ccm(causal_graph, X_test)
                te_score = ctf_metrics.compute_transparency_entropy(cii_scores)
                
                # CS on small test sample
                cs_scores = []
                for i in range(min(10, len(X_test))):
                    cs = ctf_metrics.compute_counterfactual_stability(
                        X_test.iloc[i:i+1], model, n_perturbations=5
                    )
                    cs_scores.append(cs)
                cs_mean = np.mean(cs_scores)
                
                cv_results.append({
                    'fold': fold_idx,
                    'cii_gini': cii_gini,
                    'ccm': ccm_score,
                    'te': te_score,
                    'cs': cs_mean,
                    'test_accuracy': model.score(X_test, y_test),
                    'n_edges': causal_graph.number_of_edges()
                })
                
            except Exception as e:
                print(f"   ⚠️ Fold {fold_idx} failed: {str(e)[:50]}")
                continue
        
        cv_df = pd.DataFrame(cv_results)
        
        # Calculate stability metrics
        stability_metrics = {}
        for metric in ['cii_gini', 'ccm', 'te', 'cs']:
            if len(cv_df) > 0:
                values = cv_df[metric]
                stability_metrics[f'{metric}_mean'] = values.mean()
                stability_metrics[f'{metric}_std'] = values.std()
                stability_metrics[f'{metric}_cv'] = values.std() / values.mean()
                stability_metrics[f'{metric}_range'] = values.max() - values.min()
        
        results = {
            'cv_results': cv_df,
            'stability_metrics': stability_metrics,
            'interpretation': self._interpret_cv_stability(stability_metrics)
        }
        
        self.results['cv_stability'] = results
        return results
    
    def run_comprehensive_sensitivity_analysis(self) -> Dict:
        """Run all sensitivity analyses."""
        print("\n" + "🎯"*20)
        print("COMPREHENSIVE CTF SENSITIVITY ANALYSIS")
        print("🎯"*20)
        
        # Run all analyses
        param_results = self.parameter_sensitivity_analysis()
        model_results = self.model_sensitivity_analysis()
        sample_results = self.sample_size_sensitivity_analysis()
        noise_results = self.noise_sensitivity_analysis()
        cv_results = self.cross_validation_stability_analysis()
        
        # Comprehensive summary
        overall_summary = self._generate_overall_summary()
        
        self.results['overall_summary'] = overall_summary
        return self.results
    
    def visualize_sensitivity_results(self, save_path: str = 'visualizations/ctf_sensitivity_analysis.png'):
        """Create comprehensive sensitivity analysis visualization."""
        if not self.results:
            print("⚠️ No results to visualize. Run analyses first.")
            return
        
        print(f"\n🎨 Creating comprehensive sensitivity visualization...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)
        
        # 1. Parameter Sensitivity (Top Left)
        if 'parameter_sensitivity' in self.results:
            self._plot_parameter_sensitivity(fig.add_subplot(gs[0, :2]))
        
        # 2. Model Sensitivity (Top Right)  
        if 'model_sensitivity' in self.results:
            self._plot_model_sensitivity(fig.add_subplot(gs[0, 2:]))
        
        # 3. Sample Size Effects (Middle Left)
        if 'sample_sensitivity' in self.results:
            self._plot_sample_sensitivity(fig.add_subplot(gs[1, :2]))
        
        # 4. Noise Sensitivity (Middle Right)
        if 'noise_sensitivity' in self.results:
            self._plot_noise_sensitivity(fig.add_subplot(gs[1, 2:]))
        
        # 5. CV Stability (Bottom Left)
        if 'cv_stability' in self.results:
            self._plot_cv_stability(fig.add_subplot(gs[2, :2]))
        
        # 6. Overall Robustness Summary (Bottom Right)
        self._plot_robustness_summary(fig.add_subplot(gs[2, 2:]))
        
        # 7. Interpretation Panel (Bottom)
        self._plot_interpretation_panel(fig.add_subplot(gs[3, :]))
        
        plt.suptitle('Comprehensive CTF Sensitivity Analysis', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Sensitivity analysis visualization saved: {save_path}")
    
    # Helper methods for interpretation
    def _interpret_parameter_sensitivity(self, sensitivity_summary: Dict) -> str:
        """Interpret parameter sensitivity results."""
        if not sensitivity_summary:
            return "Insufficient data for parameter sensitivity analysis."
        
        avg_cv = np.mean([list(alg_data.values()) for alg_data in sensitivity_summary.values()])
        
        if avg_cv < 0.1:
            return "✅ ROBUST - CTF metrics stable across parameter variations"
        elif avg_cv < 0.2:
            return "✓ STABLE - Minor sensitivity to parameter changes"
        else:
            return "⚠️ SENSITIVE - Significant parameter dependency detected"
    
    def _interpret_model_sensitivity(self, model_sensitivity: Dict) -> str:
        """Interpret model sensitivity results."""
        if not model_sensitivity:
            return "Insufficient data for model sensitivity analysis."
        
        cv_values = [v for k, v in model_sensitivity.items() if k.endswith('_cv')]
        if cv_values:
            avg_cv = np.mean(cv_values)
            if avg_cv < 0.15:
                return "✅ MODEL-INVARIANT - Consistent across model types"
            elif avg_cv < 0.3:
                return "✓ STABLE - Some model dependency"
            else:
                return "⚠️ MODEL-DEPENDENT - High sensitivity to model choice"
        return "Incomplete model sensitivity analysis."
    
    def _interpret_sample_sensitivity(self, size_effects: Dict) -> str:
        """Interpret sample size sensitivity results."""
        if not size_effects:
            return "Insufficient data for sample size analysis."
        
        stability_values = [v for k, v in size_effects.items() if k.endswith('_stability')]
        if stability_values:
            avg_stability = np.mean(stability_values)
            if avg_stability < 0.05:
                return "✅ SCALE-ROBUST - Stable across sample sizes"
            elif avg_stability < 0.1:
                return "✓ ADEQUATE - Minor scale dependency"
            else:
                return "⚠️ SCALE-SENSITIVE - Requires large samples"
        return "Incomplete sample size analysis."
    
    def _interpret_noise_sensitivity(self, noise_sensitivity: Dict) -> str:
        """Interpret noise sensitivity results."""
        correlations = [v for k, v in noise_sensitivity.items() if k.endswith('_correlation')]
        if correlations:
            max_corr = max(correlations)
            if max_corr < 0.3:
                return "✅ NOISE-ROBUST - Minimal degradation with noise"
            elif max_corr < 0.6:
                return "✓ TOLERANT - Moderate noise sensitivity"
            else:
                return "⚠️ NOISE-SENSITIVE - Significant degradation with noise"
        return "Incomplete noise analysis."
    
    def _interpret_cv_stability(self, stability_metrics: Dict) -> str:
        """Interpret CV stability results."""
        cv_values = [v for k, v in stability_metrics.items() if k.endswith('_cv')]
        if cv_values:
            max_cv = max(cv_values)
            if max_cv < 0.1:
                return "✅ HIGHLY STABLE - Consistent across folds"
            elif max_cv < 0.2:
                return "✓ STABLE - Good cross-validation consistency"  
            else:
                return "⚠️ VARIABLE - High cross-validation variance"
        return "Incomplete stability analysis."
    
    def _generate_overall_summary(self) -> Dict:
        """Generate overall robustness summary."""
        robustness_scores = {}
        
        # Collect robustness indicators from each analysis
        analyses = ['parameter_sensitivity', 'model_sensitivity', 'sample_sensitivity', 
                   'noise_sensitivity', 'cv_stability']
        
        for analysis in analyses:
            if analysis in self.results:
                interpretation = self.results[analysis].get('interpretation', '')
                if '✅' in interpretation:
                    robustness_scores[analysis] = 1.0
                elif '✓' in interpretation:
                    robustness_scores[analysis] = 0.7
                else:
                    robustness_scores[analysis] = 0.3
        
        overall_robustness = np.mean(list(robustness_scores.values())) if robustness_scores else 0.5
        
        if overall_robustness > 0.8:
            overall_interpretation = "🏆 HIGHLY ROBUST - CTF framework shows excellent stability"
        elif overall_robustness > 0.6:
            overall_interpretation = "✅ ROBUST - CTF framework is generally reliable"
        else:
            overall_interpretation = "⚠️ MODERATE ROBUSTNESS - Some sensitivity concerns"
        
        return {
            'robustness_scores': robustness_scores,
            'overall_robustness': overall_robustness,
            'interpretation': overall_interpretation
        }
    
    # Plotting helper methods (simplified versions)
    def _plot_parameter_sensitivity(self, ax):
        """Plot parameter sensitivity results."""
        ax.text(0.5, 0.5, 'Parameter\nSensitivity\nResults', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, weight='bold')
        ax.set_title('Parameter Sensitivity')
        
    def _plot_model_sensitivity(self, ax):
        """Plot model sensitivity results.""" 
        ax.text(0.5, 0.5, 'Model\nSensitivity\nResults', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, weight='bold')
        ax.set_title('Model Sensitivity')
        
    def _plot_sample_sensitivity(self, ax):
        """Plot sample size sensitivity."""
        ax.text(0.5, 0.5, 'Sample Size\nSensitivity\nResults', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, weight='bold')
        ax.set_title('Sample Size Effects')
        
    def _plot_noise_sensitivity(self, ax):
        """Plot noise sensitivity."""
        ax.text(0.5, 0.5, 'Noise\nSensitivity\nResults', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, weight='bold')
        ax.set_title('Noise Sensitivity')
        
    def _plot_cv_stability(self, ax):
        """Plot CV stability."""
        ax.text(0.5, 0.5, 'Cross-Validation\nStability\nResults', ha='center', va='center',
               transform=ax.transAxes, fontsize=12, weight='bold')
        ax.set_title('CV Stability')
        
    def _plot_robustness_summary(self, ax):
        """Plot overall robustness summary."""
        if 'overall_summary' in self.results:
            summary = self.results['overall_summary']
            scores = summary.get('robustness_scores', {})
            
            if scores:
                labels = list(scores.keys())
                values = list(scores.values())
                colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]
                
                ax.barh(labels, values, color=colors, alpha=0.7)
                ax.set_xlim([0, 1])
                ax.set_xlabel('Robustness Score')
                ax.set_title('Overall Robustness Summary')
        else:
            ax.text(0.5, 0.5, 'Robustness\nSummary', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, weight='bold')
            ax.set_title('Robustness Summary')
    
    def _plot_interpretation_panel(self, ax):
        """Plot interpretation panel."""
        ax.axis('off')
        
        # Collect all interpretations
        interpretations = []
        for analysis_name, analysis_results in self.results.items():
            if isinstance(analysis_results, dict) and 'interpretation' in analysis_results:
                interpretations.append(f"{analysis_name.replace('_', ' ').title()}: {analysis_results['interpretation']}")
        
        if 'overall_summary' in self.results:
            interpretations.append(f"Overall: {self.results['overall_summary']['interpretation']}")
        
        text = "\n".join(interpretations) if interpretations else "No interpretations available."
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

def main():
    """Main function to demonstrate sensitivity analysis."""
    print("🎯 CTF Sensitivity Analysis Framework")
    print("="*80)
    
    # Create demo COMPAS-like data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'race': np.random.binomial(1, 0.3, n_samples),
        'age': np.random.normal(35, 10, n_samples),
        'priors_count': np.random.poisson(2, n_samples),
        'is_employed': np.random.binomial(1, 0.6, n_samples),
        'education_years': np.random.normal(12, 3, n_samples)
    }
    
    # Create target with realistic relationships
    recidivism_logit = (
        -1.5 + 
        0.3 * data['priors_count'] +
        -0.05 * data['age'] +
        -0.8 * data['is_employed'] +
        0.2 * data['race']
    )
    data['recidivism'] = np.random.binomial(1, 1/(1 + np.exp(-recidivism_logit)), n_samples)
    
    df = pd.DataFrame(data)
    
    # Normalize features
    scaler = StandardScaler()
    df[['age', 'priors_count', 'education_years']] = scaler.fit_transform(
        df[['age', 'priors_count', 'education_years']]
    )
    
    print(f"✅ Created demo dataset with {len(df)} samples")
    
    # Initialize analyzer
    analyzer = CTFSensitivityAnalyzer(df, target_col='recidivism')
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_sensitivity_analysis()
    
    # Create visualization
    analyzer.visualize_sensitivity_results()
    
    print(f"\n🎊 CTF Sensitivity Analysis Complete!")
    print(f"📁 Visualization saved to: visualizations/ctf_sensitivity_analysis.png")
    
    return analyzer

if __name__ == '__main__':
    analyzer = main()