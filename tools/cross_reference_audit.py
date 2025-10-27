#!/usr/bin/env python3
"""
Cross-Reference Audit and Fix Tool
Identifies and fixes orphaned nodes and broken cross-references in the knowledge base.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from datetime import datetime

class CrossReferenceAuditor:
    """Audit and fix cross-references in knowledge base"""
    
    def __init__(self, knowledge_dir: str = "knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.nodes: Dict[str, Dict] = {}
        self.node_locations: Dict[str, Path] = {}
        self.orphaned_nodes: Set[str] = set()
        self.isolated_nodes: Set[str] = set()
        self.reference_graph: Dict[str, Set[str]] = defaultdict(set)
        self.incoming_references: Dict[str, Set[str]] = defaultdict(set)
        
    def load_all_nodes(self):
        """Load all knowledge nodes from JSON files"""
        print("📚 Loading knowledge nodes...")
        
        for json_file in self.knowledge_dir.rglob("*.json"):
            # Skip non-content files
            if json_file.name in ['learning_paths.json', 'glossary.json', 'faq.json', 
                                 'success_metrics.json', 'repository.json']:
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Skip non-node files (metadata only)
                if 'id' not in data:
                    continue
                    
                node_id = data['id']
                self.nodes[node_id] = data
                self.node_locations[node_id] = json_file
                print(f"  ✓ Loaded: {node_id}")
                
            except Exception as e:
                print(f"  ✗ Error loading {json_file}: {e}")
        
        print(f"📊 Loaded {len(self.nodes)} nodes\n")
        
    def analyze_references(self):
        """Build reference graph and identify issues"""
        print("🔍 Analyzing cross-references...")
        
        # Build reference graph
        for node_id, node_data in self.nodes.items():
            prerequisites = node_data.get('prerequisites', [])
            
            for prereq in prerequisites:
                if prereq in self.nodes:
                    # Add outgoing reference
                    self.reference_graph[node_id].add(prereq)
                    # Add incoming reference
                    self.incoming_references[prereq].add(node_id)
        
        # Identify orphaned nodes (no incoming references)
        for node_id in self.nodes:
            if len(self.incoming_references[node_id]) == 0:
                self.orphaned_nodes.add(node_id)
        
        # Identify isolated nodes (no incoming or outgoing references)
        for node_id in self.nodes:
            outgoing = len(self.reference_graph[node_id])
            incoming = len(self.incoming_references[node_id])
            if outgoing == 0 and incoming == 0:
                self.isolated_nodes.add(node_id)
                
        print(f"📈 Analysis complete\n")
        
    def print_statistics(self):
        """Print reference statistics"""
        print("=" * 70)
        print("CROSS-REFERENCE AUDIT REPORT")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   Total nodes: {len(self.nodes)}")
        print(f"   Nodes with incoming references: {len([n for n in self.nodes if len(self.incoming_references[n]) > 0])}")
        print(f"   Orphaned nodes (no incoming refs): {len(self.orphaned_nodes)}")
        print(f"   Isolated nodes (no refs at all): {len(self.isolated_nodes)}")
        
        avg_prereqs = sum(len(self.reference_graph[n]) for n in self.nodes) / len(self.nodes)
        print(f"   Average prerequisites: {avg_prereqs:.2f}")
        
        if self.orphaned_nodes:
            print(f"\n⚠️  Orphaned Nodes ({len(self.orphaned_nodes)}):")
            for node_id in sorted(self.orphaned_nodes):
                node = self.nodes[node_id]
                print(f"   - {node_id:30s} ({node.get('content_type', 'unknown'):15s})")
                
        if self.isolated_nodes:
            print(f"\n⚠️  Isolated Nodes ({len(self.isolated_nodes)}):")
            for node_id in sorted(self.isolated_nodes):
                node = self.nodes[node_id]
                print(f"   - {node_id:30s} ({node.get('content_type', 'unknown'):15s})")
        
        print()
        
    def suggest_fixes(self):
        """Generate suggestions for fixing cross-references"""
        print("💡 Suggested Fixes:\n")
        
        suggestions = []
        
        # Suggest adding prerequisites to orphaned nodes
        for orphan_id in self.orphaned_nodes:
            orphan = self.nodes[orphan_id]
            content_type = orphan.get('content_type', 'unknown')
            suggestions.append(self._suggest_connections(orphan_id, content_type))
        
        return suggestions
    
    def _suggest_connections(self, node_id: str, content_type: str) -> Dict:
        """Suggest connections for a node based on content type and keywords"""
        node = self.nodes[node_id]
        node_tags = node.get('tags', [])
        suggested_prereqs = []
        
        # Content-based suggestions
        if content_type == 'foundation':
            # Foundation nodes should often reference other foundations
            for other_id, other in self.nodes.items():
                if other_id == node_id:
                    continue
                if other.get('content_type') == 'foundation':
                    # Check for keyword overlap
                    other_tags = other.get('tags', [])
                    overlap = set(node_tags) & set(other_tags)
                    if overlap:
                        suggested_prereqs.append(other_id)
                        
        elif content_type == 'mathematics':
            # Math nodes should reference foundations
            for other_id, other in self.nodes.items():
                if other_id == node_id:
                    continue
                other_type = other.get('content_type')
                if other_type in ['foundation', 'mathematics']:
                    other_tags = other.get('tags', [])
                    overlap = set(node_tags) & set(other_tags)
                    if overlap:
                        suggested_prereqs.append(other_id)
                        
        elif content_type == 'implementation':
            # Implementation nodes should reference theory
            for other_id, other in self.nodes.items():
                if other_id == node_id:
                    continue
                other_type = other.get('content_type')
                if other_type in ['foundation', 'mathematics']:
                    other_tags = other.get('tags', [])
                    overlap = set(node_tags) & set(other_tags)
                    if overlap:
                        suggested_prereqs.append(other_id)
                        
        elif content_type == 'application':
            # Application nodes should reference implementations
            for other_id, other in self.nodes.items():
                if other_id == node_id:
                    continue
                other_type = other.get('content_type')
                if other_type in ['implementation', 'mathematics']:
                    other_tags = other.get('tags', [])
                    overlap = set(node_tags) & set(other_tags)
                    if overlap and len(suggested_prereqs) < 3:
                        suggested_prereqs.append(other_id)
        
        # Remove duplicates and limit to top 3
        suggested_prereqs = list(dict.fromkeys(suggested_prereqs))[:3]
        
        return {
            'node_id': node_id,
            'current_prereqs': node.get('prerequisites', []),
            'suggested_prereqs': suggested_prereqs,
            'reason': f"Based on content type and tag overlap"
        }
    
    def print_suggestions(self):
        """Print fix suggestions"""
        suggestions = self.suggest_fixes()
        
        if not suggestions:
            print("✅ No suggestions needed\n")
            return
            
        print("=" * 70)
        print("SUGGESTED FIXES")
        print("=" * 70)
        
        for i, suggestion in enumerate(suggestions, 1):
            node_id = suggestion['node_id']
            node = self.nodes[node_id]
            title = node.get('title', node_id)
            
            print(f"\n{i}. {node_id}")
            print(f"   Title: {title}")
            print(f"   Current prerequisites: {suggestion['current_prereqs']}")
            
            if suggestion['suggested_prereqs']:
                print(f"   Suggested additions:")
                for prereq in suggestion['suggested_prereqs']:
                    prereq_title = self.nodes[prereq].get('title', prereq)
                    print(f"     + {prereq} - {prereq_title}")
            else:
                print(f"   No suggestions available")
        
        print()
    
    def generate_audit_report(self) -> Dict:
        """Generate comprehensive audit report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_nodes': len(self.nodes),
                'orphaned_nodes': len(self.orphaned_nodes),
                'isolated_nodes': len(self.isolated_nodes),
                'avg_prerequisites': sum(len(self.reference_graph[n]) for n in self.nodes) / len(self.nodes) if self.nodes else 0
            },
            'orphaned_nodes': sorted(list(self.orphaned_nodes)),
            'isolated_nodes': sorted(list(self.isolated_nodes)),
            'node_statistics': {
                node_id: {
                    'outgoing_refs': len(self.reference_graph[node_id]),
                    'incoming_refs': len(self.incoming_references[node_id])
                }
                for node_id in self.nodes
            }
        }
    
    def save_report(self, filename: str = 'cross_reference_audit.json'):
        """Save audit report to file"""
        report = self.generate_audit_report()
        
        # Create output directory if needed
        output_dir = Path('output/cross_reference_audit')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Report saved to: {output_file}\n")
        return output_file


def main():
    """Main execution"""
    auditor = CrossReferenceAuditor()
    
    # Load nodes
    auditor.load_all_nodes()
    
    # Analyze references
    auditor.analyze_references()
    
    # Print statistics
    auditor.print_statistics()
    
    # Print suggestions
    auditor.print_suggestions()
    
    # Save report
    auditor.save_report()
    
    print("=" * 70)
    print("✅ CROSS-REFERENCE AUDIT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
