# Interactive Visualization Development Prompt

**"Active Inference for, with, by Generative AI"**

## 🎯 Mission: Develop Interactive Visualization Systems

You are tasked with developing interactive visualization systems for the Active Inference Knowledge Environment that enable users to explore, understand, and analyze complex concepts through dynamic visual interfaces. This involves creating diagrams, dashboards, animations, and comparative analysis tools that make Active Inference accessible and intuitive.

## 📋 Visualization Development Requirements

### Core Visualization Standards (MANDATORY)
1. **Interactive Design**: All visualizations must be interactive and responsive
2. **Accessibility**: Support for keyboard navigation, screen readers, and high contrast
3. **Performance**: Smooth 60fps animations and responsive interactions
4. **Cross-Platform**: Work across desktop, tablet, and mobile devices
5. **Educational Value**: Visualizations must enhance understanding, not just display data
6. **Integration**: Seamless integration with platform knowledge and research systems

### Visualization Architecture Components
```
visualization/
├── diagrams/              # Concept and system diagrams
│   ├── concept_maps.py   # Interactive concept relationship diagrams
│   ├── flow_charts.py    # Process and workflow visualizations
│   ├── network_graphs.py # Knowledge graph visualizations
│   └── system_diagrams.py # Architecture and component diagrams
├── dashboards/           # Interactive exploration interfaces
│   ├── knowledge_dashboard.py    # Knowledge exploration dashboard
│   ├── research_dashboard.py     # Research analysis dashboard
│   ├── learning_dashboard.py     # Learning progress dashboard
│   └── system_dashboard.py       # Platform monitoring dashboard
├── animations/           # Educational animations and demonstrations
│   ├── process_animations.py     # Step-by-step process animations
│   ├── concept_animations.py     # Concept evolution animations
│   ├── simulation_animations.py  # Simulation result animations
│   └── tutorial_animations.py    # Interactive tutorial animations
└── comparative/          # Model and result comparison tools
    ├── model_comparison.py       # Active Inference model comparison
    ├── result_comparison.py      # Experiment result comparison
    ├── performance_comparison.py # Performance metric comparison
    └── sensitivity_analysis.py   # Parameter sensitivity analysis
```

## 🏗️ Visualization Development Framework

### Phase 1: Interactive Diagram Development

#### 1.1 Concept Map Visualization
```python
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, List, Any, Optional
import json

class InteractiveConceptMap:
    """Interactive concept relationship visualization"""

    def __init__(self, knowledge_base: Dict[str, Any]):
        """Initialize concept map with knowledge base data"""
        self.knowledge_base = knowledge_base
        self.graph = nx.DiGraph()
        self.node_positions = {}
        self.visualization_config = self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Get default visualization configuration"""
        return {
            'layout_algorithm': 'force_directed',
            'node_size_range': [20, 50],
            'edge_width_range': [1, 5],
            'color_scheme': 'viridis',
            'animation_duration': 500,
            'interactive_features': {
                'hover': True,
                'click': True,
                'drag': True,
                'zoom': True,
                'pan': True
            }
        }

    def build_concept_graph(self) -> None:
        """Build NetworkX graph from knowledge base"""
        # Add nodes (concepts)
        for concept_id, concept_data in self.knowledge_base.items():
            self.graph.add_node(
                concept_id,
                title=concept_data.get('title', concept_id),
                content_type=concept_data.get('content_type', 'unknown'),
                difficulty=concept_data.get('difficulty', 'intermediate'),
                size=self.calculate_node_size(concept_data)
            )

        # Add edges (relationships)
        for concept_id, concept_data in self.knowledge_base.items():
            prerequisites = concept_data.get('prerequisites', [])
            related_concepts = concept_data.get('related_concepts', [])

            # Add prerequisite relationships
            for prereq in prerequisites:
                if prereq in self.graph:
                    self.graph.add_edge(
                        prereq, concept_id,
                        relationship_type='prerequisite',
                        weight=3
                    )

            # Add related concept relationships
            for related in related_concepts:
                if related in self.graph and related != concept_id:
                    self.graph.add_edge(
                        concept_id, related,
                        relationship_type='related',
                        weight=1
                    )

    def calculate_node_size(self, concept_data: Dict[str, Any]) -> float:
        """Calculate node size based on concept properties"""
        base_size = 30

        # Size based on difficulty
        difficulty_multiplier = {
            'beginner': 0.8,
            'intermediate': 1.0,
            'advanced': 1.2,
            'expert': 1.4
        }

        difficulty = concept_data.get('difficulty', 'intermediate')
        size = base_size * difficulty_multiplier.get(difficulty, 1.0)

        # Size based on content richness
        content = concept_data.get('content', {})
        content_score = len(content) * 2
        size += min(content_score, 20)  # Cap additional size

        return size

    def calculate_layout(self) -> None:
        """Calculate node positions using force-directed layout"""
        if self.visualization_config['layout_algorithm'] == 'force_directed':
            # Use spring layout for concept relationships
            self.node_positions = nx.spring_layout(
                self.graph,
                k=2,  # Optimal distance between nodes
                iterations=50,
                seed=42  # Reproducible layout
            )
        elif self.visualization_config['layout_algorithm'] == 'hierarchical':
            # Use hierarchical layout for learning paths
            self.node_positions = nx.multipartite_layout(
                self.graph,
                subset_key='difficulty'
            )

    def create_interactive_visualization(self) -> go.Figure:
        """Create interactive Plotly visualization"""

        # Build graph if not already built
        if not self.graph.nodes():
            self.build_concept_graph()
            self.calculate_layout()

        # Create node traces
        node_x, node_y, node_text, node_colors, node_sizes = [], [], [], [], []

        for node in self.graph.nodes(data=True):
            node_id, node_data = node
            if node_id in self.node_positions:
                x, y = self.node_positions[node_id]
                node_x.append(x)
                node_y.append(y)
                node_text.append(self.create_node_hover_text(node_id, node_data))
                node_colors.append(self.get_node_color(node_data))
                node_sizes.append(node_data.get('size', 30))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[node[0] for node in self.graph.nodes()],
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale=self.visualization_config['color_scheme'],
                showscale=True,
                colorbar=dict(title="Difficulty Level"),
                line=dict(width=2, color='white')
            ),
            name='Concepts'
        )

        # Create edge traces
        edge_traces = self.create_edge_traces()

        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces)

        # Configure layout
        fig.update_layout(
            title="Interactive Active Inference Concept Map",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )

        # Add interactive features
        self.add_interactive_features(fig)

        return fig

    def create_edge_traces(self) -> List[go.Scatter]:
        """Create edge traces for relationships"""
        edge_traces = []

        for edge in self.graph.edges(data=True):
            source, target, edge_data = edge

            if source in self.node_positions and target in self.node_positions:
                x0, y0 = self.node_positions[source]
                x1, y1 = self.node_positions[target]

                # Create curved edge
                edge_trace = go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(
                        width=self.get_edge_width(edge_data),
                        color=self.get_edge_color(edge_data)
                    ),
                    hoverinfo='text',
                    text=self.create_edge_hover_text(edge_data),
                    showlegend=False
                )
                edge_traces.append(edge_trace)

        return edge_traces

    def get_node_color(self, node_data: Dict[str, Any]) -> str:
        """Get node color based on difficulty and content type"""
        difficulty_colors = {
            'beginner': '#90EE90',      # Light green
            'intermediate': '#FFD700',  # Gold
            'advanced': '#FF6347',      # Tomato red
            'expert': '#8A2BE2'         # Blue violet
        }

        difficulty = node_data.get('difficulty', 'intermediate')
        return difficulty_colors.get(difficulty, '#808080')

    def get_edge_width(self, edge_data: Dict[str, Any]) -> float:
        """Get edge width based on relationship strength"""
        relationship_type = edge_data.get('relationship_type', 'related')
        weight = edge_data.get('weight', 1)

        base_width = 1
        if relationship_type == 'prerequisite':
            base_width = 3

        return base_width * weight

    def get_edge_color(self, edge_data: Dict[str, Any]) -> str:
        """Get edge color based on relationship type"""
        relationship_colors = {
            'prerequisite': '#FF0000',  # Red for prerequisites
            'related': '#808080',       # Gray for related concepts
            'depends_on': '#0000FF'     # Blue for dependencies
        }

        relationship_type = edge_data.get('relationship_type', 'related')
        return relationship_colors.get(relationship_type, '#808080')

    def create_node_hover_text(self, node_id: str, node_data: Dict[str, Any]) -> str:
        """Create hover text for nodes"""
        title = node_data.get('title', node_id)
        content_type = node_data.get('content_type', 'unknown')
        difficulty = node_data.get('difficulty', 'intermediate')

        return f"""
        <b>{title}</b><br>
        ID: {node_id}<br>
        Type: {content_type}<br>
        Difficulty: {difficulty}<br>
        <br>
        Click to explore concept details
        """

    def create_edge_hover_text(self, edge_data: Dict[str, Any]) -> str:
        """Create hover text for edges"""
        relationship_type = edge_data.get('relationship_type', 'related')

        return f"Relationship: {relationship_type}"

    def add_interactive_features(self, fig: go.Figure) -> None:
        """Add interactive features to the visualization"""

        # Add click handling for node exploration
        fig.update_traces(
            hovertemplate="<b>%{text}</b><br>%{hovertext}<extra></extra>"
        )

        # Add zoom and pan capabilities
        fig.update_layout(
            dragmode='pan',
            xaxis=dict(
                autorange=True,
                rangeslider=dict(visible=False)
            ),
            yaxis=dict(
                autorange=True,
                fixedrange=False
            )
        )

        # Add buttons for different views
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(label="Force Layout",
                             method="relayout",
                             args=[{"xaxis.autorange": True, "yaxis.autorange": True}]),
                        dict(label="Hierarchical Layout",
                             method="relayout",
                             args=[{"xaxis.autorange": True, "yaxis.autorange": True}])
                    ]
                )
            ]
        )

    def export_visualization(self, output_path: str, format: str = 'html') -> None:
        """Export visualization to various formats"""
        fig = self.create_interactive_visualization()

        if format == 'html':
            fig.write_html(output_path)
        elif format == 'png':
            fig.write_image(output_path)
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(fig.to_dict(), f, indent=2)

    def integrate_with_knowledge_base(self, knowledge_base) -> None:
        """Integrate with live knowledge base for real-time updates"""
        self.knowledge_base = knowledge_base
        # Rebuild graph with updated data
        self.graph.clear()
        self.build_concept_graph()
        self.calculate_layout()
```

#### 1.2 Flow Chart Visualization
```python
class InteractiveFlowChart:
    """Interactive process and workflow visualization"""

    def __init__(self, process_definition: Dict[str, Any]):
        """Initialize flow chart with process definition"""
        self.process_definition = process_definition
        self.nodes = {}
        self.edges = []
        self.layout_engine = self.initialize_layout_engine()

    def initialize_layout_engine(self):
        """Initialize layout engine for flowchart positioning"""
        # Use dagre or similar layout engine for flowcharts
        return {
            'algorithm': 'hierarchical',
            'direction': 'top-bottom',
            'node_spacing': 100,
            'level_spacing': 150
        }

    def parse_process_definition(self) -> None:
        """Parse process definition into nodes and edges"""
        steps = self.process_definition.get('steps', [])

        for step in steps:
            step_id = step['id']
            self.nodes[step_id] = {
                'label': step['name'],
                'description': step.get('description', ''),
                'type': step.get('type', 'process'),
                'inputs': step.get('inputs', []),
                'outputs': step.get('outputs', []),
                'duration': step.get('duration', 0),
                'status': step.get('status', 'pending')
            }

        # Create edges based on step dependencies
        for step in steps:
            step_id = step['id']
            next_steps = step.get('next_steps', [])

            for next_step in next_steps:
                if next_step in self.nodes:
                    self.edges.append({
                        'source': step_id,
                        'target': next_step,
                        'condition': step.get('condition', ''),
                        'data_flow': step.get('data_flow', [])
                    })

    def calculate_positions(self) -> Dict[str, tuple]:
        """Calculate node positions for flowchart layout"""
        positions = {}

        # Simple hierarchical layout
        levels = self.group_nodes_by_level()

        y_spacing = self.layout_engine['level_spacing']
        x_spacing = self.layout_engine['node_spacing']

        for level_idx, level_nodes in enumerate(levels):
            y = level_idx * y_spacing

            for node_idx, node_id in enumerate(level_nodes):
                x = node_idx * x_spacing - (len(level_nodes) - 1) * x_spacing / 2
                positions[node_id] = (x, y)

        return positions

    def group_nodes_by_level(self) -> List[List[str]]:
        """Group nodes by hierarchical levels"""
        # Simple topological sort for levels
        levels = []
        processed = set()

        # Find root nodes (no incoming edges)
        root_nodes = set(self.nodes.keys())
        for edge in self.edges:
            if edge['target'] in root_nodes:
                root_nodes.discard(edge['target'])

        current_level = list(root_nodes)
        processed.update(current_level)

        while current_level:
            levels.append(current_level)
            next_level = []

            for node in current_level:
                # Find nodes that have this node as prerequisite
                for edge in self.edges:
                    if edge['source'] == node and edge['target'] not in processed:
                        next_level.append(edge['target'])
                        processed.add(edge['target'])

            current_level = list(set(next_level))  # Remove duplicates

        return levels

    def create_flowchart_visualization(self) -> go.Figure:
        """Create interactive flowchart visualization"""
        self.parse_process_definition()
        positions = self.calculate_positions()

        # Create node traces
        node_traces = self.create_node_traces(positions)

        # Create edge traces
        edge_traces = self.create_edge_traces(positions)

        # Combine traces
        all_traces = node_traces + edge_traces

        # Create figure
        fig = go.Figure(data=all_traces)

        # Configure layout
        fig.update_layout(
            title=self.process_definition.get('title', 'Process Flow Chart'),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )

        return fig

    def create_node_traces(self, positions: Dict[str, tuple]) -> List[go.Scatter]:
        """Create node traces for flowchart"""
        traces = []

        # Group nodes by type
        node_types = {}
        for node_id, node_data in self.nodes.items():
            node_type = node_data['type']
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append((node_id, node_data))

        # Create trace for each node type
        for node_type, nodes in node_types.items():
            x_coords, y_coords, texts, colors, sizes = [], [], [], [], []

            for node_id, node_data in nodes:
                if node_id in positions:
                    x, y = positions[node_id]
                    x_coords.append(x)
                    y_coords.append(y)
                    texts.append(self.create_node_text(node_id, node_data))
                    colors.append(self.get_node_type_color(node_type))
                    sizes.append(self.get_node_size(node_data))

            trace = go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers+text',
                text=[node[0] for node in nodes],
                textposition="middle center",
                hovertext=texts,
                marker=dict(
                    size=sizes,
                    color=colors,
                    symbol=self.get_node_symbol(node_type),
                    line=dict(width=2, color='white')
                ),
                name=node_type.title()
            )
            traces.append(trace)

        return traces

    def create_edge_traces(self, positions: Dict[str, tuple]) -> List[go.Scatter]:
        """Create edge traces for flowchart connections"""
        traces = []

        for edge in self.edges:
            source, target = edge['source'], edge['target']

            if source in positions and target in positions:
                x0, y0 = positions[source]
                x1, y1 = positions[target]

                # Create curved edge with arrow
                trace = go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines+markers',
                    line=dict(
                        width=2,
                        color=self.get_edge_color(edge)
                    ),
                    marker=dict(
                        size=8,
                        color=self.get_edge_color(edge),
                        symbol='arrow-bar-up'
                    ),
                    hoverinfo='text',
                    text=self.create_edge_text(edge),
                    showlegend=False
                )
                traces.append(trace)

        return traces

    def get_node_type_color(self, node_type: str) -> str:
        """Get color for node type"""
        type_colors = {
            'start': '#90EE90',      # Light green
            'process': '#FFD700',    # Gold
            'decision': '#FF6347',   # Tomato
            'data': '#87CEEB',       # Sky blue
            'end': '#DC143C'         # Crimson
        }
        return type_colors.get(node_type, '#808080')

    def get_node_symbol(self, node_type: str) -> str:
        """Get symbol for node type"""
        type_symbols = {
            'start': 'circle',
            'process': 'square',
            'decision': 'diamond',
            'data': 'triangle-up',
            'end': 'circle'
        }
        return type_symbols.get(node_type, 'circle')

    def get_node_size(self, node_data: Dict[str, Any]) -> float:
        """Calculate node size based on properties"""
        base_size = 40

        # Size based on duration if available
        duration = node_data.get('duration', 0)
        if duration > 0:
            base_size += min(duration / 10, 20)  # Cap additional size

        return base_size

    def get_edge_color(self, edge: Dict[str, Any]) -> str:
        """Get edge color based on properties"""
        if edge.get('condition'):
            return '#FF6347'  # Red for conditional edges
        else:
            return '#808080'  # Gray for normal edges

    def create_node_text(self, node_id: str, node_data: Dict[str, Any]) -> str:
        """Create hover text for nodes"""
        return f"""
        <b>{node_data['label']}</b><br>
        Type: {node_data['type']}<br>
        Status: {node_data['status']}<br>
        Duration: {node_data.get('duration', 'N/A')} units<br>
        <br>
        {node_data.get('description', '')}
        """

    def create_edge_text(self, edge: Dict[str, Any]) -> str:
        """Create hover text for edges"""
        text = f"Flow: {edge['source']} → {edge['target']}"

        if edge.get('condition'):
            text += f"<br>Condition: {edge['condition']}"

        if edge.get('data_flow'):
            text += f"<br>Data: {', '.join(edge['data_flow'])}"

        return text

    def add_process_animation(self, fig: go.Figure) -> None:
        """Add process flow animation"""
        # Create animation frames for process flow
        frames = []

        # Simulate process execution
        for step in range(len(self.nodes)):
            frame_data = []

            # Update node colors based on execution progress
            for i, (node_id, node_data) in enumerate(self.nodes.items()):
                color = '#90EE90' if i <= step else '#D3D3D3'  # Green for completed, gray for pending

                # Add node to frame
                frame_data.append(go.Scatter(
                    x=[self.positions[node_id][0]],
                    y=[self.positions[node_id][1]],
                    mode='markers+text',
                    text=[node_id],
                    marker=dict(size=40, color=color)
                ))

            frames.append(go.Frame(data=frame_data))

        # Add animation controls
        fig.frames = frames

        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'buttons': [{
                    'label': 'Play Process',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 1000, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 500}
                    }]
                }]
            }]
        )
```

### Phase 2: Dashboard Development Framework

#### 2.1 Knowledge Exploration Dashboard
```python
class KnowledgeExplorationDashboard:
    """Interactive dashboard for exploring knowledge base"""

    def __init__(self, knowledge_base: Dict[str, Any]):
        """Initialize knowledge exploration dashboard"""
        self.knowledge_base = knowledge_base
        self.current_filters = {}
        self.selected_concepts = set()
        self.visualization_components = self.initialize_components()

    def initialize_components(self) -> Dict[str, Any]:
        """Initialize dashboard visualization components"""
        return {
            'concept_map': InteractiveConceptMap(self.knowledge_base),
            'filter_panel': self.create_filter_panel(),
            'detail_panel': self.create_detail_panel(),
            'search_component': self.create_search_component(),
            'navigation_history': []
        }

    def create_filter_panel(self) -> Dict[str, Any]:
        """Create interactive filter panel"""
        # Extract unique values for filters
        content_types = set()
        difficulties = set()
        tags = set()

        for concept in self.knowledge_base.values():
            content_types.add(concept.get('content_type', 'unknown'))
            difficulties.add(concept.get('difficulty', 'unknown'))
            tags.update(concept.get('tags', []))

        return {
            'content_type_filter': {
                'options': sorted(list(content_types)),
                'selected': [],
                'multiselect': True
            },
            'difficulty_filter': {
                'options': sorted(list(difficulties)),
                'selected': [],
                'multiselect': True
            },
            'tag_filter': {
                'options': sorted(list(tags)),
                'selected': [],
                'multiselect': True
            },
            'text_search': {
                'placeholder': 'Search concepts...',
                'value': ''
            }
        }

    def create_detail_panel(self) -> Dict[str, Any]:
        """Create concept detail panel"""
        return {
            'selected_concept': None,
            'content_tabs': ['overview', 'details', 'examples', 'exercises'],
            'related_concepts': [],
            'prerequisites': [],
            'learning_path': []
        }

    def create_search_component(self) -> Dict[str, Any]:
        """Create intelligent search component"""
        return {
            'search_index': self.build_search_index(),
            'autocomplete': True,
            'fuzzy_matching': True,
            'semantic_search': True
        }

    def build_search_index(self) -> Dict[str, Any]:
        """Build search index for fast concept lookup"""
        index = {
            'by_title': {},
            'by_content': {},
            'by_tags': {},
            'by_prerequisites': {},
            'by_difficulty': {}
        }

        for concept_id, concept in self.knowledge_base.items():
            # Index by title
            title = concept.get('title', '').lower()
            if title:
                index['by_title'][title] = concept_id

            # Index by content keywords
            content = concept.get('content', {})
            content_text = json.dumps(content).lower()
            for word in content_text.split():
                if len(word) > 3:  # Skip short words
                    if word not in index['by_content']:
                        index['by_content'][word] = []
                    index['by_content'][word].append(concept_id)

            # Index by tags
            tags = concept.get('tags', [])
            for tag in tags:
                if tag not in index['by_tags']:
                    index['by_tags'][tag] = []
                index['by_tags'][tag].append(concept_id)

            # Index by prerequisites
            prereqs = concept.get('prerequisites', [])
            for prereq in prereqs:
                if prereq not in index['by_prerequisites']:
                    index['by_prerequisites'][prereq] = []
                index['by_prerequisites'][prereq].append(concept_id)

            # Index by difficulty
            difficulty = concept.get('difficulty', 'unknown')
            if difficulty not in index['by_difficulty']:
                index['by_difficulty'][difficulty] = []
            index['by_difficulty'][difficulty].append(concept_id)

        return index

    def apply_filters(self, filters: Dict[str, Any]) -> List[str]:
        """Apply filters to get filtered concept list"""
        matching_concepts = set(self.knowledge_base.keys())

        # Apply content type filter
        if filters.get('content_types'):
            matching_concepts &= set(
                cid for cid, c in self.knowledge_base.items()
                if c.get('content_type') in filters['content_types']
            )

        # Apply difficulty filter
        if filters.get('difficulties'):
            matching_concepts &= set(
                cid for cid, c in self.knowledge_base.items()
                if c.get('difficulty') in filters['difficulties']
            )

        # Apply tag filter
        if filters.get('tags'):
            matching_concepts &= set(
                cid for cid, c in self.knowledge_base.items()
                if any(tag in c.get('tags', []) for tag in filters['tags'])
            )

        # Apply text search
        if filters.get('search_text'):
            search_text = filters['search_text'].lower()
            search_matches = set()

            # Search in titles
            for title, cid in self.visualization_components['search_component']['search_index']['by_title'].items():
                if search_text in title:
                    search_matches.add(cid)

            # Search in content
            for word, cids in self.visualization_components['search_component']['search_index']['by_content'].items():
                if search_text in word:
                    search_matches.update(cids)

            matching_concepts &= search_matches

        return list(matching_concepts)

    def create_dashboard_layout(self) -> Dict[str, Any]:
        """Create complete dashboard layout"""
        # Create main visualization area
        concept_map = self.visualization_components['concept_map']
        main_visualization = concept_map.create_interactive_visualization()

        # Create filter controls
        filter_controls = self.create_filter_controls()

        # Create detail panel
        detail_panel = self.create_detail_view()

        # Arrange in dashboard layout
        dashboard = {
            'layout': {
                'title': 'Active Inference Knowledge Explorer',
                'components': {
                    'main_visualization': main_visualization,
                    'filter_panel': filter_controls,
                    'detail_panel': detail_panel,
                    'search_bar': self.create_search_bar(),
                    'navigation_controls': self.create_navigation_controls()
                },
                'grid_layout': {
                    'rows': 2,
                    'cols': 2,
                    'areas': [
                        ['search_bar', 'filter_panel'],
                        ['main_visualization', 'detail_panel']
                    ]
                }
            },
            'callbacks': {
                'filter_update': self.update_filters,
                'concept_select': self.select_concept,
                'search_update': self.update_search,
                'navigation_update': self.update_navigation
            }
        }

        return dashboard

    def update_filters(self, new_filters: Dict[str, Any]) -> None:
        """Update dashboard filters and refresh visualization"""
        self.current_filters = new_filters
        filtered_concepts = self.apply_filters(new_filters)

        # Update concept map with filtered concepts
        concept_map = self.visualization_components['concept_map']
        concept_map.filter_concepts(filtered_concepts)

        # Update visualization
        self.refresh_visualization()

    def select_concept(self, concept_id: str) -> None:
        """Handle concept selection in dashboard"""
        if concept_id in self.knowledge_base:
            concept_data = self.knowledge_base[concept_id]

            # Update detail panel
            self.visualization_components['detail_panel']['selected_concept'] = concept_data

            # Highlight selected concept in visualization
            concept_map = self.visualization_components['concept_map']
            concept_map.highlight_concept(concept_id)

            # Update related concepts
            related = concept_data.get('related_concepts', [])
            self.visualization_components['detail_panel']['related_concepts'] = related

            # Update prerequisites
            prereqs = concept_data.get('prerequisites', [])
            self.visualization_components['detail_panel']['prerequisites'] = prereqs

            # Add to navigation history
            self.visualization_components['navigation_history'].append(concept_id)

    def update_search(self, search_query: str) -> None:
        """Handle search query updates"""
        if search_query:
            # Use search component for intelligent search
            search_component = self.visualization_components['search_component']
            search_results = search_component.search(search_query)

            # Update current filters with search results
            self.current_filters['search_results'] = search_results
            self.update_filters(self.current_filters)

    def update_navigation(self, navigation_action: str) -> None:
        """Handle navigation actions"""
        if navigation_action == 'back' and self.visualization_components['navigation_history']:
            # Go back in navigation history
            self.visualization_components['navigation_history'].pop()
            if self.visualization_components['navigation_history']:
                last_concept = self.visualization_components['navigation_history'][-1]
                self.select_concept(last_concept)

        elif navigation_action == 'forward':
            # Future: implement forward navigation
            pass

    def refresh_visualization(self) -> None:
        """Refresh dashboard visualization"""
        # Re-render concept map with current filters
        concept_map = self.visualization_components['concept_map']
        concept_map.update_visualization()

        # Update detail panel if needed
        # This would trigger UI updates in a web framework

    def export_dashboard(self, output_path: str, format: str = 'html') -> None:
        """Export dashboard to various formats"""
        dashboard_layout = self.create_dashboard_layout()

        if format == 'html':
            self.export_html_dashboard(dashboard_layout, output_path)
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(dashboard_layout, f, indent=2, default=str)

    def export_html_dashboard(self, dashboard_layout: Dict[str, Any], output_path: str) -> None:
        """Export dashboard as interactive HTML"""
        # This would generate a complete HTML page with embedded visualizations
        # Including JavaScript for interactivity

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard_layout['layout']['title']}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div id="dashboard-container">
                <h1>{dashboard_layout['layout']['title']}</h1>
                <div id="main-visualization"></div>
                <div id="controls">
                    <div id="filter-panel"></div>
                    <div id="detail-panel"></div>
                </div>
            </div>
            <script>
                // Interactive dashboard JavaScript would go here
                // Including Plotly.js integration and event handlers
            </script>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)
```

### Phase 3: Animation Development Framework

#### 3.1 Process Animation System
```python
class ProcessAnimationEngine:
    """Animation engine for process and concept demonstrations"""

    def __init__(self, animation_config: Dict[str, Any]):
        """Initialize animation engine"""
        self.config = animation_config
        self.animation_frames = []
        self.current_frame = 0
        self.is_playing = False
        self.frame_rate = animation_config.get('frame_rate', 30)
        self.duration = animation_config.get('duration', 5.0)  # seconds

    def create_process_animation(self, process_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create animation frames for process demonstration"""
        frames = []
        total_steps = len(process_steps)

        # Calculate timing
        step_duration = self.duration / total_steps

        for step_idx, step in enumerate(process_steps):
            frame_time = step_idx * step_duration

            # Create frame data
            frame = {
                'time': frame_time,
                'step_index': step_idx,
                'step_data': step,
                'visualization_state': self.create_step_visualization(step, step_idx, total_steps),
                'annotations': self.create_step_annotations(step, step_idx),
                'transitions': self.create_step_transitions(step, step_idx, total_steps)
            }

            frames.append(frame)

        self.animation_frames = frames
        return frames

    def create_step_visualization(self, step: Dict[str, Any], step_idx: int, total_steps: int) -> Dict[str, Any]:
        """Create visualization state for animation step"""
        # Calculate progress
        progress = step_idx / (total_steps - 1) if total_steps > 1 else 1.0

        # Create visualization elements
        visualization = {
            'background_color': self.interpolate_color(
                self.config.get('start_color', '#FFFFFF'),
                self.config.get('end_color', '#E0E0E0'),
                progress
            ),
            'active_elements': self.get_active_elements(step),
            'highlight_elements': self.get_highlight_elements(step, progress),
            'data_values': self.get_animated_values(step, progress),
            'layout_changes': self.get_layout_changes(step, progress)
        }

        return visualization

    def create_step_annotations(self, step: Dict[str, Any], step_idx: int) -> List[Dict[str, Any]]:
        """Create annotations for animation step"""
        annotations = []

        # Step title annotation
        annotations.append({
            'text': step.get('title', f'Step {step_idx + 1}'),
            'position': 'top',
            'style': {
                'font_size': 18,
                'font_weight': 'bold',
                'color': '#333333'
            }
        })

        # Step description
        if 'description' in step:
            annotations.append({
                'text': step['description'],
                'position': 'bottom',
                'style': {
                    'font_size': 14,
                    'color': '#666666'
                }
            })

        # Progress indicator
        progress_text = f"Step {step_idx + 1} of {len(self.animation_frames)}"
        annotations.append({
            'text': progress_text,
            'position': 'progress_bar',
            'style': {
                'font_size': 12,
                'color': '#999999'
            }
        })

        return annotations

    def create_step_transitions(self, step: Dict[str, Any], step_idx: int, total_steps: int) -> Dict[str, Any]:
        """Create transition effects between steps"""
        transitions = {
            'duration': 0.5,  # seconds
            'easing': 'ease-in-out',
            'effects': []
        }

        # Add transition effects based on step type
        step_type = step.get('type', 'normal')

        if step_type == 'introduction':
            transitions['effects'].append({
                'type': 'fade_in',
                'target': 'all_elements',
                'duration': 1.0
            })
        elif step_type == 'highlight':
            transitions['effects'].append({
                'type': 'highlight_pulse',
                'target': 'important_elements',
                'duration': 0.8
            })
        elif step_type == 'transition':
            transitions['effects'].append({
                'type': 'slide_transition',
                'direction': step.get('direction', 'left'),
                'duration': 0.6
            })

        return transitions

    def interpolate_color(self, start_color: str, end_color: str, progress: float) -> str:
        """Interpolate between two colors"""
        # Convert hex colors to RGB
        start_rgb = self.hex_to_rgb(start_color)
        end_rgb = self.hex_to_rgb(end_color)

        # Interpolate RGB values
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * progress)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * progress)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * progress)

        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"

    def hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def get_active_elements(self, step: Dict[str, Any]) -> List[str]:
        """Get elements that should be active in this step"""
        return step.get('active_elements', [])

    def get_highlight_elements(self, step: Dict[str, Any], progress: float) -> List[Dict[str, Any]]:
        """Get elements to highlight with intensity based on progress"""
        highlights = []

        for element in step.get('highlight_elements', []):
            highlight = element.copy()
            highlight['intensity'] = progress  # Scale intensity with progress
            highlights.append(highlight)

        return highlights

    def get_animated_values(self, step: Dict[str, Any], progress: float) -> Dict[str, Any]:
        """Get animated data values for the step"""
        animated_values = {}

        # Animate values that change over time
        value_animations = step.get('value_animations', {})

        for value_name, animation_spec in value_animations.items():
            start_value = animation_spec.get('start', 0)
            end_value = animation_spec.get('end', 1)
            animated_values[value_name] = start_value + (end_value - start_value) * progress

        return animated_values

    def get_layout_changes(self, step: Dict[str, Any], progress: float) -> Dict[str, Any]:
        """Get layout changes for the animation step"""
        layout_changes = {}

        # Handle element positioning changes
        position_changes = step.get('position_changes', {})
        for element, change_spec in position_changes.items():
            start_pos = change_spec.get('start', [0, 0])
            end_pos = change_spec.get('end', [0, 0])

            current_pos = [
                start_pos[0] + (end_pos[0] - start_pos[0]) * progress,
                start_pos[1] + (end_pos[1] - start_pos[1]) * progress
            ]

            layout_changes[element] = {'position': current_pos}

        return layout_changes

    def play_animation(self, callback: Optional[callable] = None) -> None:
        """Play the animation sequence"""
        self.is_playing = True
        self.current_frame = 0

        while self.is_playing and self.current_frame < len(self.animation_frames):
            frame = self.animation_frames[self.current_frame]

            # Execute frame
            self.execute_frame(frame)

            # Call callback if provided
            if callback:
                callback(frame, self.current_frame)

            # Wait for next frame
            time.sleep(1.0 / self.frame_rate)
            self.current_frame += 1

        self.is_playing = False

    def execute_frame(self, frame: Dict[str, Any]) -> None:
        """Execute a single animation frame"""
        # This would update the visualization with frame data
        # In a real implementation, this would update the UI components

        visualization_state = frame['visualization_state']
        annotations = frame['annotations']

        # Apply visualization changes
        self.apply_visualization_state(visualization_state)

        # Update annotations
        self.update_annotations(annotations)

    def pause_animation(self) -> None:
        """Pause the animation"""
        self.is_playing = False

    def resume_animation(self) -> None:
        """Resume the animation"""
        if not self.is_playing and self.animation_frames:
            self.is_playing = True
            # Continue from current frame

    def jump_to_frame(self, frame_index: int) -> None:
        """Jump to specific frame in animation"""
        if 0 <= frame_index < len(self.animation_frames):
            self.current_frame = frame_index
            frame = self.animation_frames[frame_index]
            self.execute_frame(frame)

    def export_animation(self, output_path: str, format: str = 'gif') -> None:
        """Export animation to various formats"""
        if format == 'gif':
            self.export_gif_animation(output_path)
        elif format == 'mp4':
            self.export_mp4_animation(output_path)
        elif format == 'json':
            self.export_json_animation(output_path)

    def export_json_animation(self, output_path: str) -> None:
        """Export animation as JSON for web playback"""
        animation_data = {
            'config': self.config,
            'frames': self.animation_frames,
            'metadata': {
                'total_frames': len(self.animation_frames),
                'duration': self.duration,
                'frame_rate': self.frame_rate
            }
        }

        with open(output_path, 'w') as f:
            json.dump(animation_data, f, indent=2, default=str)

    def get_animation_progress(self) -> float:
        """Get current animation progress (0.0 to 1.0)"""
        if not self.animation_frames:
            return 0.0

        return self.current_frame / len(self.animation_frames)

    def set_animation_speed(self, speed_multiplier: float) -> None:
        """Set animation playback speed"""
        self.frame_rate = self.config.get('frame_rate', 30) * speed_multiplier
```

### Phase 4: Comparative Analysis Framework

#### 4.1 Model Comparison System
```python
class ModelComparisonVisualizer:
    """Interactive visualization for comparing Active Inference models"""

    def __init__(self, model_results: Dict[str, Any]):
        """Initialize model comparison visualizer"""
        self.model_results = model_results
        self.comparison_metrics = self.extract_comparison_metrics()
        self.visualization_components = self.initialize_visualization_components()

    def extract_comparison_metrics(self) -> Dict[str, Any]:
        """Extract metrics for model comparison"""
        metrics = {
            'accuracy': {},
            'complexity': {},
            'convergence': {},
            'generalization': {},
            'computational_efficiency': {}
        }

        for model_name, results in self.model_results.items():
            # Extract accuracy metrics
            metrics['accuracy'][model_name] = {
                'train_accuracy': results.get('train_accuracy', []),
                'test_accuracy': results.get('test_accuracy', []),
                'validation_accuracy': results.get('validation_accuracy', [])
            }

            # Extract complexity metrics
            metrics['complexity'][model_name] = {
                'parameter_count': results.get('parameter_count', 0),
                'model_size': results.get('model_size', 0),
                'inference_time': results.get('inference_time', 0)
            }

            # Extract convergence metrics
            metrics['convergence'][model_name] = {
                'epochs_to_converge': results.get('epochs_to_converge', 0),
                'final_loss': results.get('final_loss', 0),
                'training_stability': results.get('training_stability', 0)
            }

        return metrics

    def initialize_visualization_components(self) -> Dict[str, Any]:
        """Initialize visualization components for comparison"""
        return {
            'radar_chart': self.create_radar_comparison(),
            'parallel_coordinates': self.create_parallel_coordinates(),
            'scatter_plots': self.create_scatter_plots(),
            'bar_charts': self.create_bar_charts(),
            'time_series': self.create_time_series_plots(),
            'statistical_summary': self.create_statistical_summary()
        }

    def create_radar_comparison(self) -> go.Figure:
        """Create radar chart for multi-dimensional model comparison"""
        categories = ['Accuracy', 'Efficiency', 'Stability', 'Complexity', 'Generalization']

        fig = go.Figure()

        for model_name, metrics in self.comparison_metrics.items():
            # Normalize metrics to 0-1 scale for radar chart
            normalized_values = self.normalize_metrics_for_radar(metrics, categories)

            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=categories,
                fill='toself',
                name=model_name
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Comparison Radar Chart"
        )

        return fig

    def normalize_metrics_for_radar(self, metrics: Dict[str, Any], categories: List[str]) -> List[float]:
        """Normalize metrics for radar chart display"""
        normalized = []

        for category in categories:
            if category == 'Accuracy':
                # Use test accuracy
                accuracy = metrics.get('accuracy', {}).get('test_accuracy', [0])
                value = accuracy[-1] if accuracy else 0
                normalized.append(value)
            elif category == 'Efficiency':
                # Use inverse of inference time
                inference_time = metrics.get('complexity', {}).get('inference_time', 1)
                efficiency = 1.0 / max(inference_time, 0.001)  # Avoid division by zero
                normalized.append(min(efficiency / 1000, 1.0))  # Scale appropriately
            elif category == 'Stability':
                # Use training stability metric
                stability = metrics.get('convergence', {}).get('training_stability', 0)
                normalized.append(stability)
            elif category == 'Complexity':
                # Use inverse of parameter count
                params = metrics.get('complexity', {}).get('parameter_count', 1000000)
                complexity = 1.0 / max(params, 1)
                normalized.append(min(complexity * 10000000, 1.0))  # Scale appropriately
            elif category == 'Generalization':
                # Use difference between train and test accuracy
                train_acc = metrics.get('accuracy', {}).get('train_accuracy', [0])
                test_acc = metrics.get('accuracy', {}).get('test_accuracy', [0])
                if train_acc and test_acc:
                    generalization = 1.0 - abs(train_acc[-1] - test_acc[-1])
                    normalized.append(max(generalization, 0))
                else:
                    normalized.append(0)

        return normalized

    def create_parallel_coordinates(self) -> go.Figure:
        """Create parallel coordinates plot for detailed comparison"""
        # Prepare data for parallel coordinates
        dimensions = []

        # Accuracy dimension
        accuracy_values = []
        for model_name in self.model_results.keys():
            acc = self.comparison_metrics['accuracy'][model_name]['test_accuracy']
            accuracy_values.append(acc[-1] if acc else 0)

        dimensions.append(dict(
            range=[0, 1],
            label='Test Accuracy',
            values=accuracy_values
        ))

        # Complexity dimension
        complexity_values = []
        for model_name in self.model_results.keys():
            params = self.comparison_metrics['complexity'][model_name]['parameter_count']
            complexity_values.append(params)

        dimensions.append(dict(
            range=[0, max(complexity_values) if complexity_values else 10000],
            label='Parameter Count',
            values=complexity_values
        ))

        # Convergence dimension
        convergence_values = []
        for model_name in self.model_results.keys():
            epochs = self.comparison_metrics['convergence'][model_name]['epochs_to_converge']
            convergence_values.append(epochs)

        dimensions.append(dict(
            range=[0, max(convergence_values) if convergence_values else 100],
            label='Epochs to Converge',
            values=convergence_values
        ))

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=accuracy_values,
                colorscale='Viridis',
                showscale=True
            ),
            dimensions=dimensions
        ))

        fig.update_layout(
            title="Model Comparison - Parallel Coordinates"
        )

        return fig

    def create_scatter_plots(self) -> Dict[str, go.Figure]:
        """Create scatter plots for correlation analysis"""
        plots = {}

        # Accuracy vs Complexity scatter plot
        accuracy_values = []
        complexity_values = []
        model_names = []

        for model_name, metrics in self.comparison_metrics.items():
            acc = metrics['accuracy']['test_accuracy']
            if acc:
                accuracy_values.append(acc[-1])
                complexity_values.append(metrics['complexity']['parameter_count'])
                model_names.append(model_name)

        accuracy_complexity_plot = go.Figure(data=go.Scatter(
            x=complexity_values,
            y=accuracy_values,
            mode='markers+text',
            text=model_names,
            textposition="top center",
            marker=dict(
                size=10,
                color=accuracy_values,
                colorscale='Viridis',
                showscale=True
            )
        ))

        accuracy_complexity_plot.update_layout(
            title="Accuracy vs Model Complexity",
            xaxis_title="Parameter Count",
            yaxis_title="Test Accuracy"
        )

        plots['accuracy_vs_complexity'] = accuracy_complexity_plot

        return plots

    def create_bar_charts(self) -> Dict[str, go.Figure]:
        """Create bar charts for metric comparison"""
        plots = {}

        # Accuracy comparison bar chart
        model_names = list(self.model_results.keys())
        accuracy_values = []

        for model_name in model_names:
            acc = self.comparison_metrics['accuracy'][model_name]['test_accuracy']
            accuracy_values.append(acc[-1] if acc else 0)

        accuracy_bar = go.Figure(data=go.Bar(
            x=model_names,
            y=accuracy_values,
            marker_color='lightblue'
        ))

        accuracy_bar.update_layout(
            title="Model Accuracy Comparison",
            xaxis_title="Model",
            yaxis_title="Test Accuracy"
        )

        plots['accuracy_comparison'] = accuracy_bar

        return plots

    def create_time_series_plots(self) -> Dict[str, go.Figure]:
        """Create time series plots for training progress"""
        plots = {}

        # Training accuracy over time
        training_curves = go.Figure()

        for model_name, metrics in self.comparison_metrics.items():
            train_acc = metrics['accuracy']['train_accuracy']
            if train_acc:
                training_curves.add_trace(go.Scatter(
                    x=list(range(len(train_acc))),
                    y=train_acc,
                    mode='lines',
                    name=f"{model_name} (Train)"
                ))

            test_acc = metrics['accuracy']['test_accuracy']
            if test_acc:
                training_curves.add_trace(go.Scatter(
                    x=list(range(len(test_acc))),
                    y=test_acc,
                    mode='lines+markers',
                    name=f"{model_name} (Test)"
                ))

        training_curves.update_layout(
            title="Training Progress Comparison",
            xaxis_title="Epoch",
            yaxis_title="Accuracy"
        )

        plots['training_progress'] = training_curves

        return plots

    def create_statistical_summary(self) -> Dict[str, Any]:
        """Create statistical summary of model comparisons"""
        summary = {
            'best_performing': {},
            'trade_offs': {},
            'recommendations': {},
            'statistical_tests': {}
        }

        # Find best performing model for each metric
        for metric_category, metrics in self.comparison_metrics.items():
            if metric_category == 'accuracy':
                # Find model with highest test accuracy
                best_model = max(
                    metrics.keys(),
                    key=lambda m: metrics[m]['test_accuracy'][-1] if metrics[m]['test_accuracy'] else 0
                )
                summary['best_performing']['accuracy'] = best_model

            elif metric_category == 'complexity':
                # Find simplest model (least parameters)
                simplest_model = min(
                    metrics.keys(),
                    key=lambda m: metrics[m]['parameter_count']
                )
                summary['best_performing']['simplicity'] = simplest_model

            elif metric_category == 'convergence':
                # Find fastest converging model
                fastest_model = min(
                    metrics.keys(),
                    key=lambda m: metrics[m]['epochs_to_converge']
                )
                summary['best_performing']['convergence'] = fastest_model

        # Identify trade-offs
        summary['trade_offs'] = self.analyze_trade_offs()

        # Generate recommendations
        summary['recommendations'] = self.generate_recommendations()

        return summary

    def analyze_trade_offs(self) -> Dict[str, Any]:
        """Analyze trade-offs between different metrics"""
        trade_offs = {}

        # Accuracy vs Complexity trade-off
        accuracy_values = []
        complexity_values = []

        for model_name, metrics in self.comparison_metrics.items():
            acc = metrics['accuracy']['test_accuracy']
            if acc:
                accuracy_values.append((model_name, acc[-1]))
                complexity_values.append((model_name, metrics['complexity']['parameter_count']))

        # Sort by accuracy and complexity
        accuracy_sorted = sorted(accuracy_values, key=lambda x: x[1], reverse=True)
        complexity_sorted = sorted(complexity_values, key=lambda x: x[1])

        trade_offs['accuracy_complexity'] = {
            'most_accurate': accuracy_sorted[0][0],
            'least_complex': complexity_sorted[0][0],
            'pareto_frontier': self.calculate_pareto_frontier(accuracy_values, complexity_values)
        }

        return trade_offs

    def calculate_pareto_frontier(self, accuracy_data: List[tuple], complexity_data: List[tuple]) -> List[str]:
        """Calculate Pareto frontier for accuracy vs complexity trade-off"""
        # Convert to dictionaries for easier lookup
        accuracy_dict = dict(accuracy_data)
        complexity_dict = dict(complexity_data)

        # Calculate Pareto frontier
        pareto_models = []
        sorted_by_accuracy = sorted(accuracy_dict.items(), key=lambda x: x[1], reverse=True)

        min_complexity = float('inf')
        for model, accuracy in sorted_by_accuracy:
            complexity = complexity_dict[model]
            if complexity < min_complexity:
                pareto_models.append(model)
                min_complexity = complexity

        return pareto_models

    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations based on analysis"""
        recommendations = {
            'best_overall': None,
            'best_for_accuracy': None,
            'best_for_efficiency': None,
            'best_for_simplicity': None,
            'use_case_recommendations': {}
        }

        # Simple recommendation logic
        accuracy_best = self.comparison_metrics['best_performing'].get('accuracy')
        simplicity_best = self.comparison_metrics['best_performing'].get('simplicity')

        if accuracy_best == simplicity_best:
            recommendations['best_overall'] = accuracy_best
        else:
            # More sophisticated logic would be needed here
            recommendations['best_overall'] = accuracy_best

        recommendations['best_for_accuracy'] = accuracy_best
        recommendations['best_for_efficiency'] = self.comparison_metrics['best_performing'].get('convergence')
        recommendations['best_for_simplicity'] = simplicity_best

        # Use case specific recommendations
        recommendations['use_case_recommendations'] = {
            'real_time_application': recommendations['best_for_efficiency'],
            'high_accuracy_needed': recommendations['best_for_accuracy'],
            'resource_constrained': recommendations['best_for_simplicity'],
            'research_exploration': recommendations['best_overall']
        }

        return recommendations

    def create_comparison_dashboard(self) -> Dict[str, Any]:
        """Create comprehensive comparison dashboard"""
        dashboard = {
            'title': 'Active Inference Model Comparison Dashboard',
            'components': {
                'radar_chart': self.visualization_components['radar_chart'],
                'parallel_coords': self.visualization_components['parallel_coordinates'],
                'scatter_plots': self.visualization_components['scatter_plots'],
                'bar_charts': self.visualization_components['bar_charts'],
                'time_series': self.visualization_components['time_series'],
                'statistical_summary': self.visualization_components['statistical_summary']
            },
            'layout': {
                'grid': [
                    ['radar_chart', 'parallel_coords'],
                    ['scatter_plots', 'bar_charts'],
                    ['time_series', 'statistical_summary']
                ]
            },
            'interactivity': {
                'model_selection': True,
                'metric_filtering': True,
                'export_options': ['png', 'pdf', 'html']
            }
        }

        return dashboard

    def export_comparison_report(self, output_path: str, format: str = 'html') -> None:
        """Export comprehensive comparison report"""
        dashboard = self.create_comparison_dashboard()

        if format == 'html':
            self.export_html_comparison_report(dashboard, output_path)
        elif format == 'pdf':
            self.export_pdf_comparison_report(dashboard, output_path)
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(dashboard, f, indent=2, default=str)
```

## 📊 Visualization Quality Standards

### Accessibility Requirements
```python
def ensure_visualization_accessibility(fig: go.Figure, config: Dict[str, Any]) -> go.Figure:
    """Ensure visualization meets accessibility standards"""

    # High contrast color schemes
    accessibility_colors = {
        'primary': '#000000',      # Black
        'secondary': '#FFFFFF',    # White
        'accent': '#005A9C',       # Accessible blue
        'highlight': '#E31B23',    # Accessible red
        'muted': '#6B7280'         # Accessible gray
    }

    # Update color scheme
    fig.update_layout(
        paper_bgcolor=accessibility_colors['secondary'],
        plot_bgcolor=accessibility_colors['secondary'],
        font=dict(
            color=accessibility_colors['primary'],
            size=config.get('font_size', 12)
        )
    )

    # Add alt text for screen readers
    if 'title' in config:
        fig.update_layout(
            title=dict(
                text=config['title'],
                x=0.5,
                xanchor='center'
            )
        )

    # Ensure minimum touch targets for mobile (44px)
    # Ensure keyboard navigation support
    # Add ARIA labels where supported

    return fig

def add_visualization_keyboard_navigation(fig: go.Figure) -> go.Figure:
    """Add keyboard navigation support to visualizations"""

    # Add custom JavaScript for keyboard navigation
    keyboard_script = """
    // Keyboard navigation for Plotly visualizations
    document.addEventListener('keydown', function(event) {
        const plot = document.querySelector('.plotly-graph-div');

        if (!plot) return;

        switch(event.key) {
            case 'ArrowRight':
                // Navigate to next data point
                event.preventDefault();
                break;
            case 'ArrowLeft':
                // Navigate to previous data point
                event.preventDefault();
                break;
            case 'Enter':
                // Select current data point
                event.preventDefault();
                break;
            case 'Escape':
                // Clear selection
                event.preventDefault();
                break;
        }
    });
    """

    # This would be injected into the HTML output
    fig._javascript = keyboard_script

    return fig
```

### Performance Optimization
```python
def optimize_visualization_performance(fig: go.Figure, config: Dict[str, Any]) -> go.Figure:
    """Optimize visualization for performance"""

    # Reduce data points for large datasets
    max_points = config.get('max_data_points', 10000)

    for trace in fig.data:
        if hasattr(trace, 'x') and len(trace.x) > max_points:
            # Downsample data
            step = len(trace.x) // max_points
            trace.x = trace.x[::step]
            if hasattr(trace, 'y'):
                trace.y = trace.y[::step]

    # Use WebGL for large datasets
    if config.get('use_webgl', True):
        for trace in fig.data:
            if hasattr(trace, 'type'):
                if trace.type in ['scatter', 'scattergl']:
                    trace.type = 'scattergl'

    # Optimize layout calculations
    fig.update_layout(
        # Disable expensive features for performance
        hovermode='closest' if not config.get('high_performance', False) else False,
        showlegend=config.get('show_legend', True),
        # Use fixed size for better performance
        autosize=False,
        width=config.get('width', 800),
        height=config.get('height', 600)
    )

    return fig
```

---

**"Active Inference for, with, by Generative AI"** - Creating interactive, accessible visualizations that make complex concepts understandable and enable deep exploration of Active Inference principles.
