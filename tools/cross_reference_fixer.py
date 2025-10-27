#!/usr/bin/env python3
"""
Cross-Reference Fixer
Automatically fixes orphaned nodes by adding appropriate prerequisites based on suggestions.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set
import argparse
from datetime import datetime

class CrossReferenceFixer:
    """Fix cross-references in knowledge base"""
    
    def __init__(self, knowledge_dir: str = "knowledge", dry_run: bool = True):
        self.knowledge_dir = Path(knowledge_dir)
        self.dry_run = dry_run
        self.fixes_applied = []
        self.fixes_skipped = []
        
    def load_audit_report(self, report_path: str) -> Dict:
        """Load cross-reference audit report"""
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_suggestions(self, node_id: str, nodes: Dict) -> List[str]:
        """Get suggested prerequisites for a node"""
        node = nodes[node_id]
        content_type = node.get('content_type', 'unknown')
        tags = set(node.get('tags', []))
        current_prereqs = set(node.get('prerequisites', []))
        
        suggested = []
        
        # Content-based heuristics
        for other_id, other in nodes.items():
            if other_id == node_id or other_id in current_prereqs:
                continue
                
            other_type = other.get('content_type')
            other_tags = set(other.get('tags', []))
            
            # Check relevance
            tag_overlap = len(tags & other_tags)
            
            # Foundation -> Foundation links
            if content_type == 'foundation' and other_type == 'foundation':
                if tag_overlap > 0:
                    suggested.append(other_id)
            
            # Mathematics -> Foundation/Mathematics links
            elif content_type == 'mathematics':
                if other_type in ['foundation', 'mathematics'] and tag_overlap > 0:
                    suggested.append(other_id)
            
            # Implementation -> Foundation/Mathematics links
            elif content_type == 'implementation':
                if other_type in ['foundation', 'mathematics'] and tag_overlap > 0:
                    suggested.append(other_id)
                    
            # Application -> Implementation links
            elif content_type == 'application':
                if other_type == 'implementation' and tag_overlap > 0:
                    suggested.append(other_id)
        
        # Return top 3 most relevant
        return suggested[:3]
    
    def fix_orphaned_nodes(self, report_path: str):
        """Apply fixes to orphaned nodes"""
        print("🔧 Cross-Reference Fixer")
        print("=" * 70)
        
        audit = self.load_audit_report(report_path)
        orphaned = audit['orphaned_nodes']
        
        print(f"\n📋 Found {len(orphaned)} orphaned nodes to fix")
        if self.dry_run:
            print("⚠️  DRY RUN MODE - No files will be modified")
        print()
        
        # Load all nodes
        nodes = {}
        for json_file in self.knowledge_dir.rglob("*.json"):
            if json_file.name in ['learning_paths.json', 'glossary.json', 'faq.json', 
                                 'success_metrics.json', 'repository.json']:
                continue
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'id' in data:
                    nodes[data['id']] = (data, json_file)
            except Exception as e:
                print(f"⚠️  Error loading {json_file}: {e}")
        
        # Apply fixes
        for i, node_id in enumerate(orphaned, 1):
            if node_id not in nodes:
                print(f"{i}. {node_id}: ✗ Not found")
                self.fixes_skipped.append(node_id)
                continue
            
            node_data, node_path = nodes[node_id]
            current_prereqs = set(node_data.get('prerequisites', []))
            
            # Get suggestions
            suggested = self.get_suggestions(node_id, {k: v[0] for k, v in nodes.items()})
            
            if not suggested:
                print(f"{i}. {node_id}: ℹ️  No suggestions available")
                self.fixes_skipped.append(node_id)
                continue
            
            # Filter out already existing prerequisites
            new_prereqs = [p for p in suggested if p not in current_prereqs]
            
            if not new_prereqs:
                print(f"{i}. {node_id}: ✓ Already has prerequisites")
                continue
            
            # Apply fix
            node_data['prerequisites'] = sorted(list(current_prereqs | set(new_prereqs)))
            
            if not self.dry_run:
                # Write updated file
                with open(node_path, 'w', encoding='utf-8') as f:
                    json.dump(node_data, f, indent=2, ensure_ascii=False)
                print(f"{i}. {node_id}: ✓ Added {len(new_prereqs)} prerequisites")
            else:
                print(f"{i}. {node_id}: ✓ Would add {len(new_prereqs)} prerequisites")
                for prereq in new_prereqs:
                    print(f"      + {prereq}")
            
            self.fixes_applied.append({
                'node_id': node_id,
                'added_prereqs': new_prereqs
            })
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Fixed: {len(self.fixes_applied)}")
        print(f"Skipped: {len(self.fixes_skipped)}")
        print(f"Total processed: {len(orphaned)}")
        print()
        
        if self.dry_run:
            print("⚠️  DRY RUN - No changes were made")
            print("Run with --apply to make changes permanent")
        else:
            print("✅ Changes applied successfully")
    
    def save_fix_log(self, output_dir: str = "output/cross_reference_audit"):
        """Save log of applied fixes"""
        log = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': self.dry_run,
            'fixes_applied': self.fixes_applied,
            'fixes_skipped': self.fixes_skipped,
            'total_fixed': len(self.fixes_applied),
            'total_skipped': len(self.fixes_skipped)
        }
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        log_file = output_path / 'fix_log.json'
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Fix log saved to: {log_file}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Fix cross-references in knowledge base')
    parser.add_argument('--apply', action='store_true', help='Apply fixes (default is dry run)')
    parser.add_argument('--report', type=str, default='output/cross_reference_audit/cross_reference_audit.json',
                       help='Path to audit report')
    
    args = parser.parse_args()
    
    fixer = CrossReferenceFixer(dry_run=not args.apply)
    
    if not Path(args.report).exists():
        print(f"❌ Error: Audit report not found at {args.report}")
        print("Please run cross_reference_audit.py first")
        sys.exit(1)
    
    fixer.fix_orphaned_nodes(args.report)
    fixer.save_fix_log()


if __name__ == '__main__':
    main()
