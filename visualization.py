#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL Publication-Quality Causal Graph + ROC Visualizer
Used by ALL experiments (MIMIC, juvenile justice, robustness)

Features:
- 600 DPI output
- Dataset-aware coloring (MIMIC vs COMPAS)
- Curved edges, perfect node placement
- Stability-weighted edges
- Target node highlighting
- Professional legend

Author: John Marko 
Date: 2025-11-26
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
from typing import Dict,List,Tuple,Optional
logger = logging.getLogger(__name__)

# Publication settings
plt.rcParams.update({
    'savefig.dpi': 600,
    'figure.dpi': 150,
    'font.size': 13,
    'axes.titlesize': 18,
    'font.family': 'Arial',
    'axes.labelsize': 14,
})

Path("visualizations").mkdir(exist_ok=True)

# =============================================================================
# Dataset-Specific Node Categories
# =============================================================================
def get_node_categories(G: nx.DiGraph, dataset: str = 'auto') -> Dict[str, List[str]]:
    nodes = list(G.nodes())
    if dataset == 'auto':
        if any('creatinine' in n.lower() for n in nodes):
            dataset = 'mimic'
        elif any('priors_count' in n for n in nodes):
            dataset = 'compas'
        else:
            dataset = 'generic'

    if dataset == 'mimic':
        return {
            'Demographics': ['age', 'gender'],
            'Renal': ['creatinine', 'bun', 'anion_gap'],
            'Electrolytes': ['sodium', 'potassium', 'bicarbonate', 'chloride'],
            'CBC': ['hematocrit', 'hemoglobin', 'wbc', 'platelet'],
            'Metabolic': ['glucose'],
            'Outcome': ['mortality', 'target']
        }
    elif dataset == 'compas':
        return {
            'Demographics': ['age', 'age_cat', 'race', 'sex'],
            'Criminal History': ['priors_count', 'juv_fel_count', 'juv_misd_count'],
            'Current Charge': ['c_charge_degree', 'c_charge_desc'],
            'Outcome': ['recidivism', 'target']
        }
    else:
        return {'Features': nodes}

# =============================================================================
# MAIN: Publication-Quality Graph
# =============================================================================
def visualize_causal_graph(
    G: nx.DiGraph,
    filepath: str = "visualizations/causal_graph.png",
    title: str = "Consensus Causal Graph",
    target_node: str = "target",
    edge_stability: Optional[Dict[Tuple[str, str], float]] = None,
    dataset: str = 'auto'
):
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)

    categories = get_node_categories(G, dataset)
    color_map = {}
    for cat, nodes in categories.items():
        color = {
            'Demographics': '#ff9999',
            'Renal': '#ff6666',
            'Electrolytes': '#66b3ff',
            'CBC': '#99ff99',
            'Metabolic': '#ffff99',
            'Criminal History': '#9467bd',
            'Current Charge': '#8c564b',
            'Outcome': '#1f77b4',
            'Features': '#7f7f7f'
        }.get(cat, '#d3d3d3')
        for node in nodes:
            if node in G.nodes:
                color_map[node] = color

    node_colors = [color_map.get(node, '#d3d3d3') for node in G.nodes]
    
    # Edges: thickness and alpha from stability
    edges = G.edges()
    if edge_stability:
        weights = [edge_stability.get((u, v), 0.5) * 8 for u, v in edges]
        alphas = [edge_stability.get((u, v), 0.5) for u, v in edges]
    else:
        weights = [3] * len(edges)
        alphas = [0.8] * len(edges)

    nx.draw_networkx_edges(G, pos,
                           width=weights,
                           alpha=alphas,
                           arrowsize=20,
                           arrowstyle='->',
                           connectionstyle='arc3,rad=0.1',
                           edge_color='gray')

    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=3000,
                           linewidths=2,
                           edgecolors='black')

    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')

    # Target highlight
    if target_node in G.nodes:
        nx.draw_networkx_nodes(G, pos,
                               nodelist=[target_node],
                               node_color='red',
                               node_size=4000,
                               linewidths=4,
                               edgecolors='darkred')

    # Legend removed per user request

    plt.title(title, pad=20, size=20, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=600)
    plt.close()
    logger.info(f"Graph saved: {filepath}")


# =============================================================================
# Publication Graph Visualizer Class (for compatibility)
# =============================================================================
class PublicationGraphVisualizer:
    """Compatibility class for Main_experiment.py imports."""
    
    def __init__(self):
        pass
    
    def create_publication_graph(self, G, filepath, title="", target_node="target", dataset="auto"):
        """Create publication-quality graph visualization."""
        visualize_causal_graph(G, filepath, title, target_node, dataset=dataset)