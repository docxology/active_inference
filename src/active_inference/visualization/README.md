# Visualization System - Source Code Implementation

This directory contains the source code implementation of the Active Inference visualization system, providing interactive diagrams, educational animations, real-time dashboards, and comparative analysis tools for exploring complex systems.

## Overview

The visualization module provides comprehensive interactive visualization tools for Active Inference concepts, models, and simulations. This includes dynamic diagrams, real-time monitoring dashboards, educational animations, and comparative analysis tools for exploring and understanding complex adaptive systems.

## Module Structure

```
src/active_inference/visualization/
├── __init__.py              # Module initialization and visualization exports
├── diagrams.py              # Interactive diagram generation and management
├── animations.py            # Educational animation system and process visualization
├── dashboards.py            # Real-time monitoring dashboards and interactive exploration
├── comparative.py           # Model comparison and analysis visualization tools
├── diagrams/                # Diagram templates and assets
├── animations/              # Animation assets and resources
├── dashboards/              # Dashboard templates and components
└── comparative/             # Comparison visualization assets and templates
```

## Core Components

### 📈 Interactive Diagrams (`diagrams.py`)
**Dynamic concept visualization and diagram generation**
- Interactive diagram creation and management
- Concept mapping and relationship visualization
- Dynamic diagram updates and real-time rendering
- Integration with knowledge graph and educational content

**Key Methods to Implement:**
```python
def create_diagram(self, diagram_type: DiagramType, title: str, nodes: List[DiagramNode] = None, edges: List[DiagramEdge] = None) -> InteractiveDiagram:
    """Create interactive diagram with comprehensive validation"""

def get_concept_diagram(self, concept: str) -> Optional[InteractiveDiagram]:
    """Get pre-built concept diagram for Active Inference topics"""

def update_node_position(self, node_id: str, position: Tuple[float, float]) -> bool:
    """Update node position with validation and constraint checking"""

def highlight_path(self, node_ids: List[str], color: str = "#ff0000") -> None:
    """Highlight specific path through diagram with visual emphasis"""

def export_diagram(self, diagram_name: str, format: str = "json") -> Optional[Dict[str, Any]]:
    """Export diagram in various formats for sharing and embedding"""

def validate_diagram_structure(self, diagram: InteractiveDiagram) -> Dict[str, Any]:
    """Validate diagram structure and relationships"""

def create_diagram_from_knowledge_graph(self, concept_ids: List[str]) -> InteractiveDiagram:
    """Create diagram from knowledge graph concepts and relationships"""

def implement_diagram_interactivity(self, diagram: InteractiveDiagram) -> Dict[str, Any]:
    """Implement interactive features for diagram exploration"""

def create_diagram_accessibility_features(self, diagram: InteractiveDiagram) -> Dict[str, Any]:
    """Create accessibility features for diagram navigation and understanding"""

def implement_diagram_performance_optimization(self, diagram: InteractiveDiagram) -> Dict[str, Any]:
    """Optimize diagram rendering and interaction performance"""
```

### 🎬 Educational Animations (`animations.py`)
**Step-by-step process demonstrations and educational content**
- Process animation creation and management
- Educational sequence generation and validation
- Animation playback and interaction controls
- Integration with learning management systems

**Key Methods to Implement:**
```python
def create_animation(self, animation_type: AnimationType, name: str, frames: List[AnimationFrame]) -> AnimationSequence:
    """Create animation sequence with comprehensive validation"""

def get_process_animation(self, process: str) -> Optional[AnimationSequence]:
    """Get pre-built process animation for Active Inference concepts"""

def play_animation(self, animation_id: str, speed: float = 1.0) -> Dict[str, Any]:
    """Play animation with playback controls and monitoring"""

def get_frame_at_time(self, animation_id: str, time: float) -> Optional[Dict[str, Any]]:
    """Get specific animation frame at given time"""

def export_animation(self, animation_id: str, format: str = "json") -> Optional[Dict[str, Any]]:
    """Export animation in various formats for sharing"""

def validate_animation_sequence(self, sequence: AnimationSequence) -> Dict[str, Any]:
    """Validate animation sequence for completeness and correctness"""

def create_animation_from_process_description(self, description: Dict[str, Any]) -> AnimationSequence:
    """Create animation from process description and specifications"""

def implement_animation_interactivity(self, sequence: AnimationSequence) -> Dict[str, Any]:
    """Implement interactive features for animation exploration"""

def create_animation_accessibility_features(self, sequence: AnimationSequence) -> Dict[str, Any]:
    """Create accessibility features for animation understanding"""

def optimize_animation_performance(self, sequence: AnimationSequence) -> AnimationSequence:
    """Optimize animation for performance and smooth playback"""
```

### 📋 Real-time Dashboards (`dashboards.py`)
**Interactive exploration and monitoring interfaces**
- Dashboard creation and component management
- Real-time data visualization and monitoring
- Interactive exploration tools and controls
- Integration with simulation and analysis systems

**Key Methods to Implement:**
```python
def add_component(self, component: DashboardComponent) -> None:
    """Add dashboard component with validation and configuration"""

def add_data_source(self, source_id: str, data_function: Callable) -> None:
    """Add data source with validation and monitoring"""

def start_monitoring(self) -> None:
    """Start real-time monitoring with proper resource management"""

def stop_monitoring(self) -> None:
    """Stop monitoring with graceful shutdown and cleanup"""

def get_dashboard_state(self) -> Dict[str, Any]:
    """Get comprehensive dashboard state and component information"""

def export_config(self) -> str:
    """Export dashboard configuration for sharing and reuse"""

def validate_dashboard_configuration(self, config: DashboardConfig) -> Dict[str, Any]:
    """Validate dashboard configuration for completeness"""

def create_dashboard_from_template(self, template: str, data_sources: Dict[str, Any]) -> Dashboard:
    """Create dashboard from template with data source integration"""

def implement_dashboard_interactivity(self, dashboard: Dashboard) -> Dict[str, Any]:
    """Implement interactive features for dashboard exploration"""

def create_dashboard_accessibility_features(self, dashboard: Dashboard) -> Dict[str, Any]:
    """Create accessibility features for dashboard navigation"""
```

### ⚖️ Comparative Analysis (`comparative.py`)
**Side-by-side comparison and model evaluation tools**
- Model comparison and performance analysis
- Interactive comparison interfaces and controls
- Statistical comparison and significance testing
- Integration with benchmarking and evaluation systems

**Key Methods to Implement:**
```python
def compare_models(self, model_a_data: Dict[str, Any], model_b_data: Dict[str, Any], comparison_metrics: List[str] = None) -> ModelComparison:
    """Compare two models with comprehensive statistical analysis"""

def create_comparison_matrix(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create pairwise comparison matrix for multiple models"""

def create_comparison_dashboard(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create interactive comparison dashboard with visualization"""

def validate_comparison_methodology(self, comparison: ModelComparison) -> Dict[str, Any]:
    """Validate comparison methodology and statistical analysis"""

def generate_comparison_insights(self, models: List[Dict[str, Any]]) -> List[str]:
    """Generate insights and recommendations from model comparisons"""

def export_comparison_results(self, comparison: ModelComparison, format: str) -> Any:
    """Export comparison results in various formats for publication"""

def create_comparison_visualization(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create comprehensive visualization for model comparison"""

def implement_comparison_interactivity(self, dashboard: Dict[str, Any]) -> Dict[str, Any]:
    """Implement interactive features for comparison exploration"""

def create_comparison_accessibility_features(self, dashboard: Dict[str, Any]) -> Dict[str, Any]:
    """Create accessibility features for comparison understanding"""
```

## Implementation Architecture

### Visualization Pipeline Architecture
The visualization system implements a comprehensive pipeline with:
- **Content Integration**: Seamless integration with knowledge repository
- **Rendering Engine**: High-performance rendering for interactive visualizations
- **Interaction Management**: Comprehensive interaction handling and controls
- **Export System**: Multiple format export for sharing and publication
- **Accessibility**: Full accessibility support for all visualization types

### Animation System Architecture
The animation system provides:
- **Sequence Management**: Comprehensive animation sequence management
- **Playback Control**: Advanced playback controls and interaction
- **Performance Optimization**: Optimized rendering and memory management
- **Integration**: Integration with educational content and learning systems

## Development Guidelines

### Visualization Standards
- **Educational Value**: Visualizations must support learning objectives
- **Accessibility**: All visualizations must be accessible to diverse users
- **Performance**: Optimize for smooth interaction and real-time updates
- **Accuracy**: Visualizations must accurately represent concepts and data
- **Usability**: Intuitive interfaces with clear interaction patterns

### Quality Standards
- **Code Quality**: Follow established visualization patterns and best practices
- **Performance**: Optimize for rendering performance and memory efficiency
- **Testing**: Comprehensive testing including accessibility and usability
- **Documentation**: Complete documentation with usage examples
- **Validation**: Built-in validation for all visualization components

## Usage Examples

### Diagram Creation and Management
```python
from active_inference.visualization import VisualizationEngine, DiagramType, DiagramNode, DiagramEdge

# Initialize visualization engine
viz_engine = VisualizationEngine(config)

# Create Active Inference concept diagram
ai_diagram = viz_engine.get_concept_diagram("active_inference")
if ai_diagram:
    print(f"Created diagram with {len(ai_diagram.nodes)} nodes")

    # Export for sharing
    diagram_data = viz_engine.export_diagram("active_inference", format="json")
    print(f"Diagram exported: {len(diagram_data['nodes'])} nodes, {len(diagram_data['edges'])} edges")

# Create custom diagram
nodes = [
    DiagramNode("perception", "Perception", (100, 100), "process"),
    DiagramNode("inference", "Inference", (200, 100), "process"),
    DiagramNode("action", "Action", (150, 200), "process")
]

edges = [
    DiagramEdge("perception", "inference", "leads_to"),
    DiagramEdge("inference", "action", "guides")
]

custom_diagram = viz_engine.create_diagram(
    DiagramType.FLOWCHART,
    "Custom Process",
    nodes=nodes,
    edges=edges
)
```

### Animation System Usage
```python
from active_inference.visualization import AnimationEngine, AnimationType

# Initialize animation engine
anim_engine = AnimationEngine(config)

# Get process animation
process_anim = anim_engine.get_process_animation("perception_action_cycle")
if process_anim:
    print(f"Animation: {process_anim.name}")
    print(f"Duration: {process_anim.duration}s")
    print(f"Frames: {len(process_anim.frames)}")

    # Play animation
    playback = anim_engine.play_animation(process_anim.id, speed=1.0)
    print(f"Playback: {playback}")

    # Get specific frame
    frame = anim_engine.get_frame_at_time(process_anim.id, time=2.0)
    if frame:
        print(f"Frame components: {list(frame['components'].keys())}")

# Export animation
animation_data = anim_engine.export_animation(process_anim.id, format="json")
print(f"Animation exported: {len(animation_data['frames'])} frames")
```

## Testing Framework

### Visualization Testing Requirements
- **Rendering Testing**: Test visualization rendering and display
- **Interaction Testing**: Test user interactions and controls
- **Performance Testing**: Test visualization performance and responsiveness
- **Accessibility Testing**: Test accessibility features and compliance
- **Integration Testing**: Test integration with other platform components

### Test Structure
```python
class TestVisualizationSystem(unittest.TestCase):
    """Test visualization system functionality"""

    def setUp(self):
        """Set up test environment"""
        self.viz_engine = VisualizationEngine(test_config)
        self.anim_engine = AnimationEngine(test_config)

    def test_diagram_creation_and_export(self):
        """Test diagram creation and export functionality"""
        # Create test diagram
        nodes = [
            DiagramNode("test1", "Test Node 1", (0, 0), "concept"),
            DiagramNode("test2", "Test Node 2", (100, 100), "concept")
        ]

        edges = [
            DiagramEdge("test1", "test2", "related_to")
        ]

        diagram = self.viz_engine.create_diagram(
            DiagramType.CONCEPT_MAP,
            "Test Diagram",
            nodes=nodes,
            edges=edges
        )

        # Validate diagram structure
        self.assertEqual(len(diagram.nodes), 2)
        self.assertEqual(len(diagram.edges), 1)
        self.assertEqual(diagram.title, "Test Diagram")

        # Test export
        export_data = self.viz_engine.export_diagram("Test Diagram", format="json")
        self.assertIn("nodes", export_data)
        self.assertIn("edges", export_data)
        self.assertEqual(len(export_data["nodes"]), 2)
        self.assertEqual(len(export_data["edges"]), 1)

    def test_animation_system_functionality(self):
        """Test animation system functionality"""
        # Get process animation
        animation = self.anim_engine.get_process_animation("perception_action_cycle")
        self.assertIsNotNone(animation)
        self.assertEqual(animation.name, "Perception-Action Cycle")
        self.assertGreater(len(animation.frames), 0)

        # Test playback
        playback = self.anim_engine.play_animation(animation.id)
        self.assertIn("animation_id", playback)
        self.assertIn("total_frames", playback)
        self.assertIn("duration", playback)

        # Test frame access
        frame = self.anim_engine.get_frame_at_time(animation.id, time=1.0)
        self.assertIsNotNone(frame)
        self.assertIn("frame_id", frame)
        self.assertIn("components", frame)

    def test_dashboard_functionality(self):
        """Test dashboard functionality"""
        from active_inference.visualization.dashboards import Dashboard, DashboardConfig

        # Create dashboard
        config = DashboardConfig(
            title="Test Dashboard",
            refresh_interval=1000,
            components=[]
        )

        dashboard = Dashboard(config)

        # Test component management
        # Note: This would require mock components for complete testing
        # component = MockDashboardComponent("test_component", "test_type", {})
        # dashboard.add_component(component)
        # self.assertEqual(len(dashboard.components), 1)

        # Test state management
        state = dashboard.get_dashboard_state()
        self.assertIn("config", state)
        self.assertIn("components", state)
        self.assertEqual(state["config"]["title"], "Test Dashboard")
```

## Performance Considerations

### Rendering Performance
- **Frame Rate**: Optimize for smooth 60fps rendering
- **Memory Management**: Efficient memory usage for complex visualizations
- **GPU Utilization**: Leverage GPU acceleration where beneficial
- **Lazy Loading**: Implement lazy loading for large visualizations

### Interaction Performance
- **Response Time**: Optimize for sub-100ms interaction response
- **Smooth Animation**: Ensure smooth animation and transitions
- **Efficient Updates**: Efficient real-time data updates
- **Resource Management**: Proper cleanup of visualization resources

## Accessibility and Usability

### Accessibility Features
- **Screen Reader Support**: Full screen reader compatibility
- **Keyboard Navigation**: Complete keyboard navigation support
- **Color Accessibility**: Color schemes meeting accessibility standards
- **Font Scaling**: Support for various font sizes and scaling

### Educational Usability
- **Progressive Disclosure**: Information presented at appropriate complexity levels
- **Multiple Representations**: Visual, textual, and interactive representations
- **Learning Support**: Built-in learning support and guidance
- **Assessment Integration**: Integration with learning assessment systems

## Contributing Guidelines

When contributing to the visualization module:

1. **Visualization Design**: Design visualizations with educational value in mind
2. **Accessibility**: Ensure all visualizations are accessible to diverse users
3. **Performance**: Optimize for smooth interaction and real-time updates
4. **Testing**: Include comprehensive testing including accessibility testing
5. **Integration**: Ensure integration with knowledge and learning systems
6. **Documentation**: Update README and AGENTS files

## Related Documentation

- **[Main README](../README.md)**: Main package documentation
- **[AGENTS.md](AGENTS.md)**: Agent development guidelines for this module
- **[Diagrams Documentation](diagrams.py)**: Interactive diagram system details
- **[Animations Documentation](animations.py)**: Educational animation system details
- **[Dashboards Documentation](dashboards.py)**: Real-time dashboard details
- **[Comparative Documentation](comparative.py)**: Model comparison tools details

---

*"Active Inference for, with, by Generative AI"* - Building interactive visualizations through collaborative intelligence and comprehensive exploration tools.
