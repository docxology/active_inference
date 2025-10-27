#!/usr/bin/env python3
"""
Documentation Audit Script
Performs comprehensive audit of repository documentation coverage.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime

class DocumentationAuditor:
    """Comprehensive documentation audit system"""
    
    def __init__(self, project_root: str = "."):
        """Initialize auditor"""
        self.project_root = Path(project_root)
        self.all_directories = set()
        self.dirs_with_readme = set()
        self.dirs_with_agents = set()
        self.missing_readme = []
        self.missing_agents = []
        self.results = {}
        
    def scan_repository(self):
        """Scan entire repository for documentation"""
        print("Scanning repository structure...")
        
        # Get all directories
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            if any(skip in root for skip in ['__pycache__', '.git', 'venv', '_build', 'egg-info', 'node_modules', '.cursor']):
                continue
                
            path = Path(root)
            self.all_directories.add(path.relative_to(self.project_root))
            
            # Check for README.md
            if 'README.md' in files:
                self.dirs_with_readme.add(path.relative_to(self.project_root))
            
            # Check for AGENTS.md
            if 'AGENTS.md' in files:
                self.dirs_with_agents.add(path.relative_to(self.project_root))
        
        print(f"Found {len(self.all_directories)} directories")
        print(f"  - {len(self.dirs_with_readme)} with README.md")
        print(f"  - {len(self.dirs_with_agents)} with AGENTS.md")
        
    def identify_missing_docs(self):
        """Identify directories missing documentation"""
        print("\nIdentifying missing documentation...")
        
        for directory in sorted(self.all_directories):
            # Skip root
            if directory == Path('.'):
                continue
                
            has_readme = directory in self.dirs_with_readme
            has_agents = directory in self.dirs_with_agents
            
            if not has_readme:
                self.missing_readme.append(str(directory))
            if not has_agents:
                self.missing_agents.append(str(directory))
        
        # Filter out excluded patterns
        self.missing_readme = [d for d in self.missing_readme if not any(ex in d for ex in ['static', 'templates/generated', 'media'])]
        self.missing_agents = [d for d in self.missing_agents if not any(ex in d for ex in ['static', 'templates/generated', 'media'])]
        
        print(f"\nDirectories missing README.md: {len(self.missing_readme)}")
        print(f"Directories missing AGENTS.md: {len(self.missing_agents)}")
        
    def categorize_missing_docs(self) -> Dict[str, List[str]]:
        """Categorize missing documentation by component"""
        categories = {
            'applications': [],
            'knowledge': [],
            'research': [],
            'platform': [],
            'tools': [],
            'visualization': [],
            'tests': [],
            'docs': [],
            'src': [],
            'other': []
        }
        
        for directory in self.missing_readme + self.missing_agents:
            parts = Path(directory).parts
            if len(parts) > 0:
                category = parts[0]
                if category in categories:
                    if directory not in categories[category]:
                        categories[category].append(directory)
                else:
                    categories['other'].append(directory)
        
        return categories
        
    def generate_audit_report(self) -> Dict[str, any]:
        """Generate comprehensive audit report"""
        print("\nGenerating audit report...")
        
        total_dirs = len(self.all_directories) - 1  # Exclude root
        readme_coverage = len(self.dirs_with_readme) / total_dirs * 100 if total_dirs > 0 else 0
        agents_coverage = len(self.dirs_with_agents) / total_dirs * 100 if total_dirs > 0 else 0
        
        categories = self.categorize_missing_docs()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_directories': total_dirs,
                'readme_coverage': {
                    'count': len(self.dirs_with_readme),
                    'percentage': round(readme_coverage, 2)
                },
                'agents_coverage': {
                    'count': len(self.dirs_with_agents),
                    'percentage': round(agents_coverage, 2)
                },
                'missing_readme_count': len(self.missing_readme),
                'missing_agents_count': len(self.missing_agents)
            },
            'missing_documentation': {
                'readme': sorted(self.missing_readme),
                'agents': sorted(self.missing_agents)
            },
            'categorized_missing': {
                category: sorted(dirs) for category, dirs in categories.items() if dirs
            },
            'priority_recommendations': self.generate_priorities(categories)
        }
        
        return report
        
    def generate_priorities(self, categories: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Generate priority recommendations for documentation creation"""
        priorities = []
        
        # Priority 1: Core directories
        core_dirs = ['applications', 'knowledge', 'platform', 'tools', 'research']
        for category in core_dirs:
            if categories.get(category):
                priorities.append({
                    'priority': 'High',
                    'category': category,
                    'count': len(categories[category]),
                    'action': f'Create README.md and AGENTS.md for {category} subdirectories'
                })
        
        # Priority 2: Source code directories
        if categories.get('src'):
            priorities.append({
                'priority': 'Medium',
                'category': 'src',
                'count': len(categories['src']),
                'action': 'Document src directories with implementation details'
            })
        
        # Priority 3: Documentation directories
        if categories.get('docs'):
            priorities.append({
                'priority': 'Medium',
                'category': 'docs',
                'count': len(categories['docs']),
                'action': 'Complete documentation structure for docs subdirectories'
            })
        
        return priorities
        
    def print_report(self, report: Dict[str, any]):
        """Print audit report to console"""
        print("\n" + "="*80)
        print("DOCUMENTATION AUDIT REPORT")
        print("="*80)
        
        print(f"\nTimestamp: {report['timestamp']}")
        
        print("\nSUMMARY:")
        print(f"  Total Directories: {report['summary']['total_directories']}")
        print(f"  README.md Coverage: {report['summary']['readme_coverage']['count']} files ({report['summary']['readme_coverage']['percentage']}%)")
        print(f"  AGENTS.md Coverage: {report['summary']['agents_coverage']['count']} files ({report['summary']['agents_coverage']['percentage']}%)")
        print(f"  Missing README.md: {report['summary']['missing_readme_count']}")
        print(f"  Missing AGENTS.md: {report['summary']['missing_agents_count']}")
        
        print("\nPRIORITY RECOMMENDATIONS:")
        for idx, priority in enumerate(report['priority_recommendations'], 1):
            print(f"\n{idx}. [{priority['priority']}] {priority['action']}")
            print(f"   Category: {priority['category']}")
            print(f"   Count: {priority['count']} directories")
        
        if report['categorized_missing']:
            print("\nCATEGORIZED MISSING DOCUMENTATION:")
            for category, dirs in report['categorized_missing'].items():
                print(f"\n{category.upper()} ({len(dirs)} directories):")
                for directory in dirs[:10]:  # Show first 10
                    print(f"  - {directory}")
                if len(dirs) > 10:
                    print(f"  ... and {len(dirs) - 10} more")

def main():
    """Main execution"""
    auditor = DocumentationAuditor()
    auditor.scan_repository()
    auditor.identify_missing_docs()
    report = auditor.generate_audit_report()
    auditor.print_report(report)
    
    # Save report to file
    report_file = 'documentation_audit_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_file}")

if __name__ == '__main__':
    main()

