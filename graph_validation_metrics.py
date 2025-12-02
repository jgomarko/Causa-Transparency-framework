#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantitative Validation Metrics for Causal Graph Comparison
===========================================================

Implements comprehensive metrics to compare discovered causal graphs 
against literature-based reference graphs across multiple dimensions:
- Structural similarity (edge overlap, topology)
- Semantic alignment (pathway preservation)
- Statistical significance testing
- Domain-specific validation for COMPAS and MIMIC-III

Author: John Marko
Date: 2025-11-30
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.metrics import jaccard_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class GraphValidationMetrics:
    """Comprehensive causal graph validation against literature references."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results_cache = {}
    
    def structural_similarity(
        self, 
        discovered_graph: nx.DiGraph, 
        reference_graph: nx.DiGraph,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Compute structural similarity metrics between graphs.
        
        Args:
            discovered_graph: Graph learned from data
            reference_graph: Literature-based reference graph
            normalize: Whether to normalize by graph size
            
        Returns:
            Dictionary with structural similarity metrics
        """
        # Ensure both graphs have same nodes
        all_nodes = set(discovered_graph.nodes()) | set(reference_graph.nodes())
        
        # Convert to adjacency matrices
        disc_adj = nx.adjacency_matrix(discovered_graph, nodelist=sorted(all_nodes)).toarray()
        ref_adj = nx.adjacency_matrix(reference_graph, nodelist=sorted(all_nodes)).toarray()
        
        # Edge-level metrics
        disc_edges = disc_adj.flatten()
        ref_edges = ref_adj.flatten()
        
        # Basic overlap metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            ref_edges, disc_edges, average='binary', zero_division=0
        )
        
        # Jaccard similarity
        jaccard = jaccard_score(ref_edges, disc_edges, zero_division=0)
        
        # Structural Hamming Distance
        shd = np.sum(np.abs(disc_adj - ref_adj))
        if normalize:
            max_possible_edges = len(all_nodes) * (len(all_nodes) - 1)
            shd_normalized = shd / max_possible_edges if max_possible_edges > 0 else 0
        else:
            shd_normalized = shd
        
        # True/False Positives/Negatives
        tp = np.sum((disc_edges == 1) & (ref_edges == 1))
        fp = np.sum((disc_edges == 1) & (ref_edges == 0))
        fn = np.sum((disc_edges == 0) & (ref_edges == 1))
        tn = np.sum((disc_edges == 0) & (ref_edges == 0))
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Edge density comparison
        disc_density = nx.density(discovered_graph)
        ref_density = nx.density(reference_graph)
        density_diff = abs(disc_density - ref_density)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'jaccard_similarity': jaccard,
            'specificity': specificity,
            'shd': shd,
            'shd_normalized': shd_normalized,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'discovered_edges': np.sum(disc_edges),
            'reference_edges': np.sum(ref_edges),
            'density_discovered': disc_density,
            'density_reference': ref_density,
            'density_difference': density_diff
        }
    
    def topological_similarity(
        self, 
        discovered_graph: nx.DiGraph, 
        reference_graph: nx.DiGraph
    ) -> Dict[str, float]:
        """
        Compute graph topology similarity metrics.
        
        Args:
            discovered_graph: Discovered causal graph
            reference_graph: Reference literature graph
            
        Returns:
            Dictionary with topological similarity metrics
        """
        # Ensure common nodes
        common_nodes = set(discovered_graph.nodes()) & set(reference_graph.nodes())
        if len(common_nodes) == 0:
            return {'error': 'No common nodes between graphs'}
        
        # Subgraphs with common nodes only
        disc_sub = discovered_graph.subgraph(common_nodes)
        ref_sub = reference_graph.subgraph(common_nodes)
        
        metrics = {}
        
        # Degree distribution similarity
        disc_degrees = [d for n, d in disc_sub.degree()]
        ref_degrees = [d for n, d in ref_sub.degree()]
        
        if len(disc_degrees) > 1 and len(ref_degrees) > 1:
            degree_corr, degree_pval = stats.spearmanr(
                sorted(disc_degrees), sorted(ref_degrees)
            )
            metrics['degree_correlation'] = degree_corr
            metrics['degree_correlation_pvalue'] = degree_pval
        else:
            metrics['degree_correlation'] = 0.0
            metrics['degree_correlation_pvalue'] = 1.0
        
        # In-degree and out-degree correlations
        disc_in_degrees = [disc_sub.in_degree(n) for n in common_nodes]
        ref_in_degrees = [ref_sub.in_degree(n) for n in common_nodes]
        disc_out_degrees = [disc_sub.out_degree(n) for n in common_nodes]
        ref_out_degrees = [ref_sub.out_degree(n) for n in common_nodes]
        
        if len(common_nodes) > 2:
            in_deg_corr, in_deg_pval = stats.spearmanr(disc_in_degrees, ref_in_degrees)
            out_deg_corr, out_deg_pval = stats.spearmanr(disc_out_degrees, ref_out_degrees)
            
            metrics['in_degree_correlation'] = in_deg_corr if not np.isnan(in_deg_corr) else 0.0
            metrics['out_degree_correlation'] = out_deg_corr if not np.isnan(out_deg_corr) else 0.0
            metrics['in_degree_pvalue'] = in_deg_pval if not np.isnan(in_deg_pval) else 1.0
            metrics['out_degree_pvalue'] = out_deg_pval if not np.isnan(out_deg_pval) else 1.0
        else:
            metrics['in_degree_correlation'] = 0.0
            metrics['out_degree_correlation'] = 0.0
            metrics['in_degree_pvalue'] = 1.0
            metrics['out_degree_pvalue'] = 1.0
        
        # Graph diameter and average path length (for weakly connected components)
        try:
            if nx.is_weakly_connected(disc_sub) and nx.is_weakly_connected(ref_sub):
                disc_diameter = nx.diameter(disc_sub.to_undirected())
                ref_diameter = nx.diameter(ref_sub.to_undirected())
                metrics['diameter_similarity'] = 1 - abs(disc_diameter - ref_diameter) / max(disc_diameter, ref_diameter, 1)
                
                disc_avg_path = nx.average_shortest_path_length(disc_sub.to_undirected())
                ref_avg_path = nx.average_shortest_path_length(ref_sub.to_undirected())
                metrics['avg_path_similarity'] = 1 - abs(disc_avg_path - ref_avg_path) / max(disc_avg_path, ref_avg_path, 1)
            else:
                metrics['diameter_similarity'] = 0.0
                metrics['avg_path_similarity'] = 0.0
        except:
            metrics['diameter_similarity'] = 0.0
            metrics['avg_path_similarity'] = 0.0
        
        # Clustering coefficient similarity
        disc_clustering = nx.average_clustering(disc_sub.to_undirected()) if len(common_nodes) > 2 else 0
        ref_clustering = nx.average_clustering(ref_sub.to_undirected()) if len(common_nodes) > 2 else 0
        metrics['clustering_similarity'] = 1 - abs(disc_clustering - ref_clustering)
        
        return metrics
    
    def pathway_similarity(
        self, 
        discovered_graph: nx.DiGraph, 
        reference_graph: nx.DiGraph,
        target_node: str = None
    ) -> Dict[str, float]:
        """
        Compute causal pathway similarity focusing on paths to target.
        
        Args:
            discovered_graph: Discovered causal graph
            reference_graph: Reference literature graph  
            target_node: Target outcome node (e.g., 'target', 'mortality')
            
        Returns:
            Dictionary with pathway similarity metrics
        """
        # Auto-detect target node if not specified
        if target_node is None:
            potential_targets = ['target', 'mortality', 'two_year_recid', 'hospital_expire_flag']
            for target in potential_targets:
                if target in discovered_graph.nodes() and target in reference_graph.nodes():
                    target_node = target
                    break
        
        if target_node is None or target_node not in discovered_graph.nodes() or target_node not in reference_graph.nodes():
            return {'error': f'Target node {target_node} not found in both graphs'}
        
        metrics = {}
        
        # Parents of target node
        disc_parents = set(discovered_graph.predecessors(target_node))
        ref_parents = set(reference_graph.predecessors(target_node))
        
        # Parent overlap metrics
        parent_intersection = len(disc_parents & ref_parents)
        parent_union = len(disc_parents | ref_parents)
        
        metrics['parent_jaccard'] = parent_intersection / parent_union if parent_union > 0 else 0
        metrics['parent_precision'] = parent_intersection / len(disc_parents) if len(disc_parents) > 0 else 0
        metrics['parent_recall'] = parent_intersection / len(ref_parents) if len(ref_parents) > 0 else 0
        metrics['discovered_parents'] = len(disc_parents)
        metrics['reference_parents'] = len(ref_parents)
        
        # Path-based similarity (paths to target of length ≤ 3)
        common_nodes = set(discovered_graph.nodes()) & set(reference_graph.nodes())
        common_nodes.discard(target_node)  # Exclude target from sources
        
        disc_paths = []
        ref_paths = []
        
        for source in common_nodes:
            try:
                # Discovered graph paths
                disc_simple_paths = list(nx.all_simple_paths(
                    discovered_graph, source, target_node, cutoff=3
                ))
                disc_paths.extend(disc_simple_paths)
                
                # Reference graph paths
                ref_simple_paths = list(nx.all_simple_paths(
                    reference_graph, source, target_node, cutoff=3
                ))
                ref_paths.extend(ref_simple_paths)
                
            except nx.NetworkXNoPath:
                continue
        
        # Convert paths to comparable format
        disc_path_tuples = {tuple(path) for path in disc_paths}
        ref_path_tuples = {tuple(path) for path in ref_paths}
        
        # Path similarity
        path_intersection = len(disc_path_tuples & ref_path_tuples)
        path_union = len(disc_path_tuples | ref_path_tuples)
        
        metrics['path_jaccard'] = path_intersection / path_union if path_union > 0 else 0
        metrics['path_precision'] = path_intersection / len(disc_path_tuples) if len(disc_path_tuples) > 0 else 0
        metrics['path_recall'] = path_intersection / len(ref_path_tuples) if len(ref_path_tuples) > 0 else 0
        metrics['discovered_paths'] = len(disc_path_tuples)
        metrics['reference_paths'] = len(ref_path_tuples)
        
        return metrics
    
    def domain_specific_validation(
        self, 
        discovered_graph: nx.DiGraph, 
        reference_graph: nx.DiGraph,
        domain: str = 'general'
    ) -> Dict[str, float]:
        """
        Domain-specific validation metrics for COMPAS and MIMIC-III.
        
        Args:
            discovered_graph: Discovered graph
            reference_graph: Reference graph
            domain: Domain type ('compas', 'mimic', or 'general')
            
        Returns:
            Domain-specific validation metrics
        """
        metrics = {}
        
        if domain.lower() == 'compas':
            # COMPAS-specific critical relationships
            critical_edges = [
                ('priors_count', 'target'),
                ('age', 'target'),
                ('charge_degree', 'target'),
                ('is_violent_recid', 'target')
            ]
            
            # Check for demographic fairness paths that should be blocked
            problematic_paths = [
                ('race', 'target'),
                ('sex', 'target'),
                ('age', 'race'),  # Indirect discrimination
            ]
            
        elif domain.lower() == 'mimic':
            # MIMIC-III critical clinical relationships
            critical_edges = [
                ('lactate', 'mortality'),
                ('blood_pressure', 'mortality'),
                ('temperature', 'mortality'),
                ('heart_rate', 'mortality')
            ]
            
            # Clinical pathway validation
            problematic_paths = [
                ('age', 'mortality'),  # Should be mediated by clinical factors
                ('gender', 'mortality')  # Should be mediated
            ]
            
        else:
            # General domain
            critical_edges = []
            problematic_paths = []
        
        # Critical edge preservation
        if critical_edges:
            critical_preserved = 0
            critical_total = len(critical_edges)
            
            for source, target in critical_edges:
                if (source in discovered_graph.nodes() and target in discovered_graph.nodes() and
                    source in reference_graph.nodes() and target in reference_graph.nodes()):
                    
                    disc_has_edge = discovered_graph.has_edge(source, target)
                    ref_has_edge = reference_graph.has_edge(source, target)
                    
                    if disc_has_edge and ref_has_edge:
                        critical_preserved += 1
            
            metrics['critical_edge_preservation'] = critical_preserved / critical_total if critical_total > 0 else 0
        
        # Problematic path detection
        if problematic_paths:
            problematic_found = 0
            for source, target in problematic_paths:
                if (source in discovered_graph.nodes() and target in discovered_graph.nodes()):
                    try:
                        paths = list(nx.all_simple_paths(discovered_graph, source, target, cutoff=2))
                        if paths:
                            problematic_found += 1
                    except nx.NetworkXNoPath:
                        continue
            
            metrics['problematic_paths_ratio'] = problematic_found / len(problematic_paths) if problematic_paths else 0
        
        return metrics
    
    def comprehensive_validation(
        self, 
        discovered_graph: nx.DiGraph, 
        reference_graph: nx.DiGraph,
        domain: str = 'general',
        target_node: str = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation comparing discovered vs reference graphs.
        
        Args:
            discovered_graph: Graph learned from data
            reference_graph: Literature-based reference graph
            domain: Domain for specialized validation
            target_node: Target outcome node
            
        Returns:
            Comprehensive validation results
        """
        if self.verbose:
            logger.info("Running comprehensive graph validation...")
        
        results = {
            'metadata': {
                'discovered_nodes': len(discovered_graph.nodes()),
                'discovered_edges': len(discovered_graph.edges()),
                'reference_nodes': len(reference_graph.nodes()),
                'reference_edges': len(reference_graph.edges()),
                'common_nodes': len(set(discovered_graph.nodes()) & set(reference_graph.nodes())),
                'domain': domain
            }
        }
        
        # Structural similarity
        try:
            results['structural'] = self.structural_similarity(discovered_graph, reference_graph)
            if self.verbose:
                logger.info(f"Structural F1: {results['structural']['f1_score']:.3f}")
        except Exception as e:
            logger.error(f"Structural similarity failed: {e}")
            results['structural'] = {'error': str(e)}
        
        # Topological similarity
        try:
            results['topological'] = self.topological_similarity(discovered_graph, reference_graph)
            if self.verbose:
                logger.info(f"Degree correlation: {results['topological'].get('degree_correlation', 0):.3f}")
        except Exception as e:
            logger.error(f"Topological similarity failed: {e}")
            results['topological'] = {'error': str(e)}
        
        # Pathway similarity
        try:
            results['pathway'] = self.pathway_similarity(discovered_graph, reference_graph, target_node)
            if self.verbose:
                logger.info(f"Parent overlap: {results['pathway'].get('parent_jaccard', 0):.3f}")
        except Exception as e:
            logger.error(f"Pathway similarity failed: {e}")
            results['pathway'] = {'error': str(e)}
        
        # Domain-specific validation
        try:
            results['domain_specific'] = self.domain_specific_validation(discovered_graph, reference_graph, domain)
        except Exception as e:
            logger.error(f"Domain validation failed: {e}")
            results['domain_specific'] = {'error': str(e)}
        
        # Overall validation score (weighted composite)
        try:
            structural_f1 = results['structural'].get('f1_score', 0)
            parent_jaccard = results['pathway'].get('parent_jaccard', 0)
            degree_corr = results['topological'].get('degree_correlation', 0)
            
            # Ensure correlations are in [0,1] range
            degree_corr = max(0, degree_corr) if not np.isnan(degree_corr) else 0
            
            overall_score = (0.4 * structural_f1 + 0.4 * parent_jaccard + 0.2 * degree_corr)
            results['overall_validation_score'] = overall_score
            
            if self.verbose:
                logger.info(f"Overall validation score: {overall_score:.3f}")
        except Exception as e:
            logger.error(f"Overall score calculation failed: {e}")
            results['overall_validation_score'] = 0.0
        
        return results
    
    def save_validation_results(self, results: Dict, filepath: str):
        """Save validation results to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if self.verbose:
            logger.info(f"Validation results saved to {filepath}")


def create_reference_graphs() -> Dict[str, nx.DiGraph]:
    """Create literature-based reference graphs for COMPAS and MIMIC-III."""
    
    reference_graphs = {}
    
    # COMPAS Reference Graph (Criminal Justice Literature)
    compas_ref = nx.DiGraph()
    
    # Add nodes
    compas_nodes = [
        'priors_count', 'age', 'charge_degree', 'violence_history',
        'criminal_severity_score', 'juvenile_fel_count', 'sex', 'race', 'target'
    ]
    compas_ref.add_nodes_from(compas_nodes)
    
    # Add literature-based edges
    compas_edges = [
        ('priors_count', 'target'),  # Strong predictor
        ('age', 'target'),  # Age-crime curve
        ('charge_degree', 'target'),  # Severity matters
        ('violence_history', 'target'),  # Violence predicts recidivism
        ('criminal_severity_score', 'target'),  # Derived measure
        ('juvenile_fel_count', 'priors_count'),  # Juvenile -> adult pattern
        ('age', 'priors_count'),  # Older -> more history
        ('sex', 'violence_history'),  # Gender patterns in violence
    ]
    compas_ref.add_edges_from(compas_edges)
    
    reference_graphs['compas'] = compas_ref
    
    # MIMIC-III Reference Graph (Clinical Literature)
    mimic_ref = nx.DiGraph()
    
    # Add nodes
    mimic_nodes = [
        'lactate_max', 'blood_pressure', 'temperature_max', 'heart_rate_mean',
        'respiratory_rate_mean', 'glucose_max', 'creatinine_max', 'age', 
        'gender', 'mortality'
    ]
    mimic_ref.add_nodes_from(mimic_nodes)
    
    # Add clinical literature-based edges
    mimic_edges = [
        ('lactate_max', 'mortality'),  # Strong mortality predictor
        ('blood_pressure', 'mortality'),  # Hypotension -> mortality
        ('temperature_max', 'mortality'),  # Fever/hypothermia
        ('heart_rate_mean', 'mortality'),  # Tachycardia
        ('respiratory_rate_mean', 'mortality'),  # Respiratory distress
        ('glucose_max', 'mortality'),  # Hyperglycemia
        ('creatinine_max', 'mortality'),  # Kidney function
        ('age', 'mortality'),  # Age is predictor but should be mediated
        ('age', 'blood_pressure'),  # Age affects BP
        ('age', 'creatinine_max'),  # Age affects kidney function
        ('lactate_max', 'blood_pressure'),  # Lactate affects circulation
        ('heart_rate_mean', 'blood_pressure'),  # Cardiovascular coupling
    ]
    mimic_ref.add_edges_from(mimic_edges)
    
    reference_graphs['mimic'] = mimic_ref
    
    return reference_graphs


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("=== GRAPH VALIDATION METRICS TEST ===")
    
    # Create reference graphs
    ref_graphs = create_reference_graphs()
    
    # Create a simple test discovered graph (COMPAS)
    discovered_compas = nx.DiGraph()
    discovered_compas.add_edges_from([
        ('priors_count', 'target'),
        ('age', 'target'),
        ('sex', 'target'),  # This might be problematic
        ('charge_degree', 'criminal_severity_score'),
        ('criminal_severity_score', 'target')
    ])
    
    # Run validation
    validator = GraphValidationMetrics(verbose=True)
    
    print("\n=== COMPAS VALIDATION ===")
    compas_results = validator.comprehensive_validation(
        discovered_compas, 
        ref_graphs['compas'], 
        domain='compas',
        target_node='target'
    )
    
    print(f"Overall Validation Score: {compas_results['overall_validation_score']:.3f}")
    
    if 'error' not in compas_results['structural']:
        print(f"Structural F1: {compas_results['structural']['f1_score']:.3f}")
        print(f"Structural Precision: {compas_results['structural']['precision']:.3f}")
        print(f"Structural Recall: {compas_results['structural']['recall']:.3f}")
    else:
        print(f"Structural validation error: {compas_results['structural']['error']}")
    
    if 'error' not in compas_results['pathway']:
        print(f"Parent Jaccard: {compas_results['pathway']['parent_jaccard']:.3f}")
        print(f"Path Jaccard: {compas_results['pathway']['path_jaccard']:.3f}")
    else:
        print(f"Pathway validation error: {compas_results['pathway']['error']}")
    
    if 'error' not in compas_results['topological']:
        print(f"Degree Correlation: {compas_results['topological']['degree_correlation']:.3f}")
    else:
        print(f"Topological validation error: {compas_results['topological']['error']}")
    
    # Save results
    validator.save_validation_results(
        compas_results, 
        'results/graph_validation/compas_validation_results.json'
    )
    
    print("\n=== VALIDATION METRICS IMPLEMENTATION COMPLETE ===")