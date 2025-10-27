#!/usr/bin/env python3
"""
Systematic Content Enhancement Script
Automatically finds and enhances files missing critical content sections
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tools.content_enhancer import ContentEnhancer

def find_files_needing_enhancement(knowledge_path: str, gap_report_path: str, section_type: str, limit: int = 10) -> List[str]:
    """Find files that need a specific enhancement type"""

    # Load gap analysis
    with open(gap_report_path, 'r', encoding='utf-8') as f:
        gap_data = json.load(f)

    # Get files with short sections of the target type
    gaps = gap_data.get('gaps_analysis', {}).get('content_structure_gaps', {})
    short_sections = gaps.get('short_content_sections', {})

    count = short_sections.get(section_type, 0)
    if count == 0:
        print(f"No files need {section_type} enhancement")
        return []

    # Get files that need this section
    files_with_issues = gaps.get('files_with_issues', [])

    # Filter for files that have the target section as an issue
    target_files = []
    for file_issue in files_with_issues:
        if any(section_type in warning for warning in file_issue.get('warnings', [])):
            target_files.append(file_issue['file'])
            if len(target_files) >= limit:
                break

    return target_files

def enhance_files_systematically(knowledge_path: str, gap_report_path: str, enhancements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Systematically enhance files with specified enhancements"""

    enhancer = ContentEnhancer(knowledge_path)
    results = {
        'total_enhancements_attempted': 0,
        'successful_enhancements': 0,
        'failed_enhancements': 0,
        'details': []
    }

    for enhancement_spec in enhancements:
        section_type = enhancement_spec['section_type']
        limit = enhancement_spec.get('limit', 5)

        print(f"\n🔍 Finding files needing {section_type} enhancement...")

        # Find files that need this enhancement
        target_files = find_files_needing_enhancement(knowledge_path, gap_report_path, section_type, limit)

        if not target_files:
            print(f"  No files found needing {section_type} enhancement")
            continue

        print(f"  Found {len(target_files)} files to enhance")

        # Enhance each file
        for file_path in target_files:
            print(f"  📝 Enhancing {file_path} with {section_type}...")

            result = enhancer.enhance_file(file_path, section_type, dry_run=False)

            results['total_enhancements_attempted'] += 1
            results['details'].append({
                'file': file_path,
                'section_type': section_type,
                'success': result['success'],
                'message': result.get('error', 'Success')
            })

            if result['success']:
                results['successful_enhancements'] += 1
                print(f"    ✓ Success")
            else:
                results['failed_enhancements'] += 1
                print(f"    ✗ Failed: {result.get('error', 'Unknown error')}")

    return results

def main():
    """Main function for systematic enhancement"""

    knowledge_path = "knowledge"
    gap_report_path = "knowledge_gap_analysis_report.json"

    if not Path(knowledge_path).exists():
        print(f"Knowledge base path does not exist: {knowledge_path}")
        return

    if not Path(gap_report_path).exists():
        print(f"Gap report does not exist: {gap_report_path}")
        return

    # Define enhancement priorities
    enhancements = [
        {
            'section_type': 'further_reading',
            'limit': 15,
            'description': 'Add further reading sections to improve research connections'
        },
        {
            'section_type': 'related_concepts',
            'limit': 15,
            'description': 'Add related concepts sections for better navigation'
        },
        {
            'section_type': 'common_misconceptions',
            'limit': 10,
            'description': 'Add common misconceptions sections to clarify understanding'
        },
        {
            'section_type': 'connections_to_active_inference',
            'limit': 10,
            'description': 'Add Active Inference connections for better context'
        }
    ]

    print("🚀 SYSTEMATIC CONTENT ENHANCEMENT")
    print("=" * 50)

    # Run enhancements
    results = enhance_files_systematically(knowledge_path, gap_report_path, enhancements)

    # Generate report
    print(f"\n📊 ENHANCEMENT RESULTS")
    print("-" * 30)
    print(f"Total Enhancements Attempted: {results['total_enhancements_attempted']}")
    print(f"Successful Enhancements: {results['successful_enhancements']}")
    print(f"Failed Enhancements: {results['failed_enhancements']}")
    print(".1f")

    # Summary by section type
    section_summary = {}
    for detail in results['details']:
        section = detail['section_type']
        section_summary[section] = section_summary.get(section, {'attempted': 0, 'successful': 0})
        section_summary[section]['attempted'] += 1
        if detail['success']:
            section_summary[section]['successful'] += 1

    print("\nBy Section Type:")
    for section, counts in section_summary.items():
        success_rate = counts['successful'] / counts['attempted'] * 100 if counts['attempted'] > 0 else 0
        print(".1f")

    # Save detailed results
    output_file = "systematic_enhancement_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed results saved to: {output_file}")

    # Run final validation
    print(f"\n🔍 Running final validation...")
    os.system("python3 tools/knowledge_validation.py knowledge > validation_after_enhancement.log 2>&1")
    print("✓ Validation complete - check validation_after_enhancement.log")

    print("\n🎯 SYSTEMATIC ENHANCEMENT COMPLETE")
    print("Knowledge base quality has been systematically improved!")

if __name__ == "__main__":
    main()
