#!/usr/bin/env python3
"""
Manual Cross-Reference Fixes
Apply targeted fixes to remaining orphaned nodes based on content analysis.
"""

import json
from pathlib import Path
from typing import Dict, List

# Manual prerequisite mappings based on content analysis
MANUAL_FIXES = {
    "ai_safety": {
        "add_prereqs": ["decision_theory", "uncertainty_quantification"],
        "reason": "AI safety requires decision theory and uncertainty handling"
    },
    "autonomous_systems": {
        "add_prereqs": ["predictive_coding", "control_systems"],
        "reason": "Autonomous systems need predictive coding and control theory"
    },
    "behavioral_economics": {
        "add_prereqs": ["decision_theory", "info_theory_kl_divergence"],
        "reason": "Behavioral economics builds on decision theory and information measures"
    },
    "brain_imaging": {
        "add_prereqs": ["variational_inference", "mcmc_sampling"],
        "reason": "Brain imaging analysis requires inference methods"
    },
    "causal_inference": {
        "add_prereqs": ["graphical_models", "information_bottleneck"],
        "reason": "Causal inference uses graphical models and information theory"
    },
    "climate_decision_making": {
        "add_prereqs": ["decision_theory", "multi_agent_systems"],
        "reason": "Climate decisions involve multi-agent coordination"
    },
    "clinical_applications": {
        "add_prereqs": ["belief_updating", "variational_free_energy"],
        "reason": "Clinical applications depend on belief updating and free energy"
    },
    "cognitive_science": {
        "add_prereqs": ["perception", "belief_updating"],
        "reason": "Cognitive science integrates perception and belief updating"
    },
    "cutting_edge_active_inference": {
        "add_prereqs": ["expected_free_energy", "variational_free_energy"],
        "reason": "Already has all necessary prerequisites listed"
    },
    "empirical_bayes": {
        "add_prereqs": ["hierarchical_models", "bayesian_inference"],
        "reason": "Empirical Bayes builds on hierarchical models and Bayesian inference"
    },
    "game_theory": {
        "add_prereqs": ["decision_theory", "multi_agent_systems"],
        "reason": "Game theory requires decision theory and multi-agent understanding"
    },
    "information_bottleneck": {
        "add_prereqs": ["info_theory_mutual_information", "variational_free_energy"],
        "reason": "Information bottleneck uses mutual information and variational methods"
    },
    "intelligent_tutoring": {
        "add_prereqs": ["belief_updating", "expected_free_energy"],
        "reason": "Intelligent tutoring systems model student beliefs and preferences"
    },
    "market_behavior": {
        "add_prereqs": ["multi_agent_systems", "decision_theory"],
        "reason": "Market behavior involves multi-agent decision making"
    },
    "neuroscience_perception": {
        "add_prereqs": ["perception", "fep_biological_systems"],
        "reason": "Neuroscience perception links perception to biological implementation"
    },
    "optimal_control": {
        "add_prereqs": ["continuous_control", "optimization_methods"],
        "reason": "Optimal control requires continuous control and optimization"
    },
    "personalized_learning": {
        "add_prereqs": ["belief_updating", "expected_free_energy"],
        "reason": "Personalized learning models individual learner beliefs and preferences"
    },
    "policy_gradient": {
        "add_prereqs": ["optimization_methods", "ai_policy_selection"],
        "reason": "Policy gradient is an optimization method for policy selection"
    },
    "posterior_inference": {
        "add_prereqs": ["variational_inference", "bayesian_inference"],
        "reason": "Posterior inference combines Bayesian and variational methods"
    },
    "quantum_cognition": {
        "add_prereqs": ["advanced_probability", "information_geometry"],
        "reason": "Already has appropriate prerequisites"
    },
    "validation": {
        "add_prereqs": ["uncertainty_quantification", "simulation_methods"],
        "reason": "Validation requires uncertainty quantification and simulation"
    },
    "value_alignment": {
        "add_prereqs": ["expected_free_energy", "multi_agent_systems"],
        "reason": "Value alignment builds on expected free energy and multi-agent systems"
    },
}

def apply_manual_fixes(knowledge_dir: str = "knowledge"):
    """Apply manual fixes to orphaned nodes"""
    knowledge_path = Path(knowledge_dir)
    fixes_applied = []
    errors = []
    
    print("🔧 Applying Manual Cross-Reference Fixes")
    print("=" * 70)
    
    for node_id, fix_info in MANUAL_FIXES.items():
        # Find the file
        json_file = None
        for path in knowledge_path.rglob("*.json"):
            if path.name == "learning_paths.json" or path.name.endswith("_completion_report.json"):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get('id') == node_id:
                    json_file = path
                    break
            except:
                continue
        
        if not json_file:
            print(f"❌ {node_id}: File not found")
            errors.append(node_id)
            continue
        
        # Load and update
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            current_prereqs = set(data.get('prerequisites', []))
            add_prereqs = set(fix_info['add_prereqs'])
            new_prereqs = add_prereqs - current_prereqs
            
            if new_prereqs:
                data['prerequisites'] = sorted(list(current_prereqs | new_prereqs))
                
                # Write back
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                print(f"✓ {node_id}: Added {len(new_prereqs)} prerequisites")
                fixes_applied.append({
                    'node_id': node_id,
                    'added': list(new_prereqs),
                    'reason': fix_info['reason']
                })
            else:
                print(f"ℹ️  {node_id}: No new prerequisites needed")
        except Exception as e:
            print(f"❌ {node_id}: Error - {e}")
            errors.append(node_id)
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: Fixed {len(fixes_applied)}, Errors: {len(errors)}")
    print("=" * 70)
    
    return fixes_applied, errors

if __name__ == '__main__':
    fixes, errors = apply_manual_fixes()
    
    # Save log
    log = {
        'timestamp': str(Path(__file__).stat().st_mtime),
        'fixes_applied': fixes,
        'errors': errors
    }
    
    output_dir = Path('output/cross_reference_audit')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / 'manual_fix_log.json'
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Log saved to: {log_file}")
