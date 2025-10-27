#!/usr/bin/env python3
"""
Content Completeness Analysis and Report
Analyzes content sections marked as "very short" and prioritizes for enhancement.
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any

def analyze_content_warnings(audit_report_path: str = "knowledge_audit_report.json"):
    """Analyze warnings from audit report"""
    
    with open(audit_report_path, 'r') as f:
        audit = json.load(f)
    
    # Collect all warnings
    warnings_by_file = {}
    all_section_types = []
    
    for file_path, results in audit.get('detailed_results', {}).items():
        if 'warnings' in results:
            warnings = results['warnings']
            warnings_by_file[file_path] = warnings
            
            # Extract section names
            for warning in warnings:
                if 'is very short' in warning:
                    # Extract section name
                    section = warning.replace(' is very short', '').replace("Content section '", "").replace("'", "")
                    all_section_types.append(section)
    
    # Count section types
    section_counts = Counter(all_section_types)
    
    # Files with most warnings
    files_by_warning_count = sorted(warnings_by_file.items(), 
                                   key=lambda x: len(x[1]), 
                                   reverse=True)
    
    return {
        'total_files_with_warnings': len(warnings_by_file),
        'total_warnings': sum(len(w) for w in warnings_by_file.values()),
        'top_files': files_by_warning_count[:10],
        'common_sections': section_counts.most_common(15)
    }

def generate_prioritization_report():
    """Generate prioritized enhancement report"""
    
    analysis = analyze_content_warnings()
    
    print("=" * 70)
    print("CONTENT COMPLETENESS ANALYSIS")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Files with warnings: {analysis['total_files_with_warnings']}")
    print(f"   Total warnings: {analysis['total_warnings']}")
    
    print(f"\n🔝 Top 10 Files Needing Enhancement:")
    for i, (file_path, warnings) in enumerate(analysis['top_files'], 1):
        print(f"   {i}. {file_path}")
        print(f"      Warnings: {len(warnings)}")
    
    print(f"\n📋 Most Common Short Sections:")
    for section, count in analysis['common_sections']:
        print(f"   - {section}: {count} files")
    
    print(f"\n💡 Recommendations:")
    print(f"   Priority 1: Enhance common sections across many files")
    print(f"   Priority 2: Focus on top 10 files with most warnings")
    print(f"   Target: Reduce warnings to < 300 (< 4 per file average)")
    
    # Save report
    output = {
        'analysis': analysis,
        'priorities': {
            'top_priority_files': [f for f, _ in analysis['top_files'][:5]],
            'common_sections': [s for s, _ in analysis['common_sections'][:5]]
        },
        'recommendations': [
            "Focus on expanding these common sections across all files",
            "Target top 5 files for immediate enhancement",
            "Aim for content completeness score > 0.95",
            "Reduce warnings to < 300 total"
        ]
    }
    
    output_path = Path('output/content_completeness_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n📄 Report saved to: {output_path}")
    print("=" * 70)

if __name__ == '__main__':
    generate_prioritization_report()
