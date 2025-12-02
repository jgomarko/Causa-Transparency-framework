#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain Knowledge Constraints for Causal Discovery.

This module applies structural constraints to the learned causal graphs
to ensure:
1. Physical consistency (Age cannot be caused by Crime)
2. Ethical fairness (Race cannot cause Recidivism directly)
3. Methodological robustness (Severing "Proxy" links like Juvenile -> Adult history)

Author: John Marko
Date: 2025-11-26
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_tier_info(feature_names, dataset='compas'):
    """
    Define causal tiers (temporal ordering).
    Variables in earlier tiers can cause variables in later tiers, but not vice-versa.
    
    Args:
        feature_names: List of feature names
        dataset: 'compas' or 'mimic'
        
    Returns:
        Dictionary mapping feature indices to tier numbers
    """
    tiers = {}
    
    for idx, name in enumerate(feature_names):
        name_lower = name.lower()
        
        if dataset == 'compas':
            # Tier 0: Demographics / Immutable (Root Causes)
            if any(x in name_lower for x in ['age', 'sex', 'race', 'gender', 'dob']) and 'age_cat' not in name_lower:
                tiers[idx] = 0
            
            # Tier 1: Juvenile History (Past)
            elif 'juv' in name_lower:
                tiers[idx] = 1
                
            # Tier 2: Adult History (Recent Past)
            elif 'priors' in name_lower:
                tiers[idx] = 2
                
            # Tier 3: Current Charge / Procedural (Present)
            elif any(x in name_lower for x in ['c_charge', 'days_b_screening', 'jail', 'custody']):
                tiers[idx] = 3
                
            # Tier 4: Assessment / Target (Future/Outcome)
            elif any(x in name_lower for x in ['decile', 'score', 'text', 'target', 'recid']):
                tiers[idx] = 4
            
            else:
                tiers[idx] = 2  # Default to middle tier
                
        elif dataset == 'mimic':
            # Tier 0: Demographics (Static)
            if any(x in name_lower for x in ['age', 'gender', 'ethnicity']):
                tiers[idx] = 0
            
            # Tier 1: Admission / History
            elif any(x in name_lower for x in ['admit', 'history', 'prior']):
                tiers[idx] = 1
                
            # Tier 2: Vitals / Labs (Dynamic)
            elif any(x in name_lower for x in ['blood', 'urine', 'wbc', 'bicarbonate', 'sodium', 'potassium']):
                tiers[idx] = 2
                
            # Tier 3: Outcome
            elif any(x in name_lower for x in ['expire', 'death', 'survival', 'target']):
                tiers[idx] = 3
                
            else:
                tiers[idx] = 2
                
    return tiers

def apply_domain_constraints(adj_matrix, feature_names, dataset='compas', stability_scores=None):
    """
    Apply hard constraints to the adjacency matrix based on domain knowledge.
    
    Args:
        adj_matrix: Binary adjacency matrix (numpy array)
        feature_names: List of feature names matching matrix indices
        dataset: 'compas' or 'mimic'
        stability_scores: Stability matrix from ensemble discovery (optional)
        
    Returns:
        Constrained adjacency matrix
    """
    # Create copy to avoid modifying original
    constrained_matrix = adj_matrix.copy()
    n_vars = len(feature_names)
    
    logger.info(f"Applying domain constraints for {dataset.upper()} dataset...")
    
    # Track statistics
    edges_removed = 0
    
    # -------------------------------------------------------------------------
    # 1. UNIVERSAL SINK CONSTRAINT (Critical: Prevent Backwards Causation)
    # -------------------------------------------------------------------------
    # Target outcome cannot cause any other variables (future cannot cause past)
    idx_target = -1
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        if any(target_term in name_lower for target_term in ['target', 'recid', 'expire', 'mortality', 'death']):
            idx_target = i
            break
    
    # If target found, remove ALL outgoing edges
    if idx_target != -1:
        outgoing_count = np.sum(constrained_matrix[idx_target, :])
        if outgoing_count > 0:
            constrained_matrix[idx_target, :] = 0
            edges_removed += int(outgoing_count)
            logger.info(f"  CRITICAL FIX: Removed {int(outgoing_count)} backwards causation edges from Target (impossible future->past causation)")
    
    # -------------------------------------------------------------------------
    # 2. TEMPORAL TIER CONSTRAINTS (Universal)
    # -------------------------------------------------------------------------
    # Variables in later tiers cannot cause variables in earlier tiers
    tiers = get_tier_info(feature_names, dataset)
    
    for i in range(n_vars):
        for j in range(n_vars):
            # If i is in a later tier than j, i cannot cause j
            if tiers[i] > tiers[j]:
                if constrained_matrix[i, j] == 1:
                    constrained_matrix[i, j] = 0
                    edges_removed += 1
    
    logger.info(f"  Removed {edges_removed} edges violating temporal tiers")
    
    # -------------------------------------------------------------------------
    # 2. DATASET SPECIFIC CONSTRAINTS
    # -------------------------------------------------------------------------
    
    if dataset == 'compas':
        constrained_matrix = _apply_compas_constraints(constrained_matrix, feature_names)
    elif dataset == 'mimic':
        constrained_matrix = _apply_mimic_constraints(constrained_matrix, feature_names)
    
    # -------------------------------------------------------------------------
    # 3. RESCUE CRITICAL EDGES
    # -------------------------------------------------------------------------
    # If these edges exist with >0.1 stability, FORCE them to 1.
    # This overrides the strict threshold for known ground truths.
    
    rescue_edges = []
    if dataset == 'compas':
        rescue_edges = [
            ('priors_count', 'target'),
            ('age', 'target'),
            ('juv_fel_count', 'priors_count'), # Optional: restore history link
            ('juv_misd_count', 'priors_count')
        ]
    elif dataset == 'mimic':
        rescue_edges = [
            # 1. Primary Physiological Drivers
            ('creatinine', 'target'),
            ('bun', 'target'),
            ('anion_gap', 'target'),
            
            # 2. Blood Drivers
            ('hemoglobin', 'target'),
            ('platelet', 'target'),
            ('wbc', 'target'),
            
            # 3. The "Frailty" Fix (Age -> Target)
            ('age', 'target'), 
            
            # 4. The "Anion Gap" Formula (NEW UPDATE)
            # Force the ingredients to point to the calculated result
            ('sodium', 'anion_gap'),
            ('chloride', 'anion_gap'),
            
            # 5. Age -> Organs
            ('age', 'creatinine'),
            ('age', 'bun'),
            ('age', 'hematocrit'),
            
            # 6. Blood Parameter Relationships
            # Hemoglobin directly influences hematocrit
            ('hemoglobin', 'hematocrit'),
            # Sodium regulation affects chloride levels
            ('sodium', 'chloride')
        ]
        
    # Apply Rescue
    if stability_scores is not None:
        for u, v in rescue_edges:
            # Find indices
            u_idx = -1
            v_idx = -1
            for idx, name in enumerate(feature_names):
                if u in name: u_idx = idx
                if v in name: v_idx = idx
            
            if u_idx != -1 and v_idx != -1:
                # Check if there is ANY signal (stability > 0.1)
                if stability_scores[u_idx, v_idx] > 0.1:
                    logger.info(f"  RESCUED edge {u} -> {v} (Score: {stability_scores[u_idx, v_idx]:.2f})")
                    constrained_matrix[u_idx, v_idx] = 1
                    # Ensure no reverse edge
                    constrained_matrix[v_idx, u_idx] = 0
        
    return constrained_matrix

def _apply_compas_constraints(adj_matrix, feature_names):
    """
    Specific constraints for COMPAS recidivism data.
    Includes Fairness Blocks, Timeline Blocks, and Force Edges.
    """
    n_vars = len(feature_names)
    edges_removed = 0
    edges_forced = 0
    
    # Identify indices
    idx_target = -1
    idx_race = []
    idx_sex = []
    idx_age = []
    idx_juv = []
    idx_priors = []
    idx_charge_degree = []
    
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        if 'target' in name_lower or 'recid' in name_lower:
            idx_target = i
        if 'race' in name_lower:
            idx_race.append(i)
        if 'sex' in name_lower or 'gender' in name_lower:
            idx_sex.append(i)
        if 'age' in name_lower:
            idx_age.append(i)
        if 'juv' in name_lower:
            idx_juv.append(i)
        if 'priors' in name_lower:
            idx_priors.append(i)
        if 'c_charge_degree' in name_lower:
            idx_charge_degree.append(i)
            
    # -------------------------------------------------------------------------
    # A. FAIRNESS BLOCK: Race/Sex cannot directly cause Target
    # -------------------------------------------------------------------------
    if idx_target != -1:
        # Block Race -> Target
        for r_idx in idx_race:
            if adj_matrix[r_idx, idx_target] == 1:
                adj_matrix[r_idx, idx_target] = 0
                edges_removed += 1
                logger.info(f"  Blocked FAIRNESS violation: {feature_names[r_idx]} -> Target")
        
        # Block Sex -> Target
        for s_idx in idx_sex:
            if adj_matrix[s_idx, idx_target] == 1:
                adj_matrix[s_idx, idx_target] = 0
                edges_removed += 1
                logger.info(f"  Blocked FAIRNESS violation: {feature_names[s_idx]} -> Target")

    # -------------------------------------------------------------------------
    # B. TIMELINE BLOCK: Sever Juvenile -> Adult History
    # This prevents the "Root Cause Fallacy" where Juv history absorbs all importance
    # -------------------------------------------------------------------------
    for j_idx in idx_juv:
        for p_idx in idx_priors:
            # Block Juvenile -> Priors
            if adj_matrix[j_idx, p_idx] == 1:
                adj_matrix[j_idx, p_idx] = 0
                edges_removed += 1
                logger.info(f"  Blocked TIMELINE proxy: {feature_names[j_idx]} -> {feature_names[p_idx]}")
            
            # Block Priors -> Juvenile (Impossible, but ensure it)
            if adj_matrix[p_idx, j_idx] == 1:
                adj_matrix[p_idx, j_idx] = 0
                edges_removed += 1
                
    # -------------------------------------------------------------------------
    # C. FORCE EDGES: Critical predictors -> Target
    # -------------------------------------------------------------------------
    if idx_target != -1:
        # Force Charge Degree -> Target (charge severity is key predictor)
        for cd_idx in idx_charge_degree:
            if adj_matrix[cd_idx, idx_target] == 0:
                adj_matrix[cd_idx, idx_target] = 1
                edges_forced += 1
                logger.info(f"  FORCED edge: {feature_names[cd_idx]} -> Target (charge severity)")

    # -------------------------------------------------------------------------
    # D. ROOT NODE PROTECTION
    # Demographics are root nodes (no incoming edges)
    # -------------------------------------------------------------------------
    root_indices = idx_race + idx_sex + idx_age
    for r_idx in root_indices:
        # Check column r_idx (incoming edges)
        incoming = np.where(adj_matrix[:, r_idx] == 1)[0]
        for src in incoming:
            adj_matrix[src, r_idx] = 0
            edges_removed += 1
            
    logger.info(f"  Total COMPAS constraints applied: {edges_removed} edges removed, {edges_forced} edges forced")
    return adj_matrix

def _apply_mimic_constraints(adj_matrix, feature_names):
    """Specific constraints for MIMIC mortality data."""
    n_vars = len(feature_names)
    edges_removed = 0
    edges_forced = 0
    
    idx_age = []
    idx_target = -1
    idx_creatinine = []
    idx_bun = []
    idx_anion_gap = []
    idx_lactate = []
    idx_glucose = []
    
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        if 'age' in name_lower:
            idx_age.append(i)
        if 'target' in name_lower or 'expire' in name_lower:
            idx_target = i
        if 'creatinine' in name_lower:
            idx_creatinine.append(i)
        if 'bun' in name_lower:
            idx_bun.append(i)
        if 'anion' in name_lower and 'gap' in name_lower:
            idx_anion_gap.append(i)
        if 'lactate' in name_lower:
            idx_lactate.append(i)
        if 'glucose' in name_lower:
            idx_glucose.append(i)
            
    # -------------------------------------------------------------------------
    # FORCE EDGES: Critical biomarkers -> Target
    # These are known clinical indicators that must be preserved
    # -------------------------------------------------------------------------
    if idx_target != -1:
        # Force Creatinine -> Target (kidney function)
        for c_idx in idx_creatinine:
            if adj_matrix[c_idx, idx_target] == 0:
                adj_matrix[c_idx, idx_target] = 1
                edges_forced += 1
                logger.info(f"  FORCED edge: {feature_names[c_idx]} -> Target (kidney function)")
        
        # Force BUN -> Target (kidney function)
        logger.info(f"  DEBUG: idx_bun = {idx_bun}")
        for b_idx in idx_bun:
            logger.info(f"  DEBUG: Checking BUN force edge from {feature_names[b_idx]} (idx {b_idx}) -> target (idx {idx_target})")
            if adj_matrix[b_idx, idx_target] == 0:
                adj_matrix[b_idx, idx_target] = 1
                edges_forced += 1
                logger.info(f"  FORCED edge: {feature_names[b_idx]} -> Target (kidney function)")
            else:
                logger.info(f"  DEBUG: Edge {feature_names[b_idx]} -> Target already exists")
        
        # Force Anion Gap -> Target (acid-base balance)
        for a_idx in idx_anion_gap:
            if adj_matrix[a_idx, idx_target] == 0:
                adj_matrix[a_idx, idx_target] = 1
                edges_forced += 1
                logger.info(f"  FORCED edge: {feature_names[a_idx]} -> Target (acid-base balance)")
        
        # Force Lactate -> Target (tissue hypoxia) - if available
        for l_idx in idx_lactate:
            if adj_matrix[l_idx, idx_target] == 0:
                adj_matrix[l_idx, idx_target] = 1
                edges_forced += 1
                logger.info(f"  FORCED edge: {feature_names[l_idx]} -> Target (tissue hypoxia)")
        
        # Force Glucose -> Target (metabolic dysfunction)
        logger.info(f"  DEBUG: idx_glucose = {idx_glucose}")
        for g_idx in idx_glucose:
            logger.info(f"  DEBUG: Checking glucose force edge from {feature_names[g_idx]} (idx {g_idx}) -> target (idx {idx_target})")
            if adj_matrix[g_idx, idx_target] == 0:
                adj_matrix[g_idx, idx_target] = 1
                edges_forced += 1
                logger.info(f"  FORCED edge: {feature_names[g_idx]} -> Target (metabolic dysfunction)")
            else:
                logger.info(f"  DEBUG: Edge {feature_names[g_idx]} -> Target already exists")
    
    # -------------------------------------------------------------------------
    # STANDARD CONSTRAINTS
    # -------------------------------------------------------------------------
    # Age is root
    for a_idx in idx_age:
        incoming = np.where(adj_matrix[:, a_idx] == 1)[0]
        for src in incoming:
            adj_matrix[src, a_idx] = 0
            edges_removed += 1
            
    # Target is sink (no outgoing edges)
    if idx_target != -1:
        outgoing = np.where(adj_matrix[idx_target, :] == 1)[0]
        for dst in outgoing:
            adj_matrix[idx_target, dst] = 0
            edges_removed += 1
            
    logger.info(f"  Total MIMIC constraints: {edges_removed} edges removed, {edges_forced} edges forced")
    return adj_matrix