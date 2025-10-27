"""
Visualization Engine - Educational Animations

Animated visualizations for explaining Active Inference concepts and processes.
Provides step-by-step animations, process demonstrations, and interactive
learning experiences for complex theoretical concepts.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AnimationType(Enum):
    """Types of animations supported"""
    STEP_BY_STEP = "step_by_step"
    CONTINUOUS = "continuous"
    INTERACTIVE = "interactive"
    TRANSITION = "transition"
    DEMONSTRATION = "demonstration"


@dataclass
class AnimationFrame:
    """Represents a single frame in an animation"""
    frame_id: int
    timestamp: float
    components: Dict[str, Any]
    narration: Optional[str] = None
    highlights: List[str] = None

    def __post_init__(self):
        if self.highlights is None:
            self.highlights = []


@dataclass
class AnimationSequence:
    """Complete animation sequence"""
    id: str
    name: str
    description: str
    frames: List[AnimationFrame]
    duration: float  # total duration in seconds
    frame_rate: int = 30

    def get_frame_at_time(self, time: float) -> Optional[AnimationFrame]:
        """Get frame at specific time"""
        if not self.frames:
            return None

        frame_index = int((time / self.duration) * len(self.frames))
        frame_index = max(0, min(frame_index, len(self.frames) - 1))

        return self.frames[frame_index]


class ProcessAnimation:
    """Animations for Active Inference processes"""

    def __init__(self):
        self.animations: Dict[str, AnimationSequence] = {}

    def create_perception_action_cycle(self) -> AnimationSequence:
        """Create animation of the perception-action cycle"""
        frames = []

        # Frame 1: Initial state
        frames.append(AnimationFrame(
            frame_id=0,
            timestamp=0.0,
            components={
                "sensory_input": {"active": False, "intensity": 0.0},
                "generative_model": {"active": False, "beliefs": []},
                "prediction": {"active": False, "value": 0.0},
                "prediction_error": {"active": False, "value": 0.0},
                "action": {"active": False, "selection": None}
            },
            narration="Starting the perception-action cycle",
            highlights=[]
        ))

        # Frame 2: Sensory input
        frames.append(AnimationFrame(
            frame_id=1,
            timestamp=1.0,
            components={
                "sensory_input": {"active": True, "intensity": 0.8},
                "generative_model": {"active": False, "beliefs": []},
                "prediction": {"active": False, "value": 0.0},
                "prediction_error": {"active": False, "value": 0.0},
                "action": {"active": False, "selection": None}
            },
            narration="Sensory input is received",
            highlights=["sensory_input"]
        ))

        # Frame 3: Model generates prediction
        frames.append(AnimationFrame(
            frame_id=2,
            timestamp=2.0,
            components={
                "sensory_input": {"active": True, "intensity": 0.8},
                "generative_model": {"active": True, "beliefs": ["belief_1", "belief_2"]},
                "prediction": {"active": True, "value": 0.7},
                "prediction_error": {"active": False, "value": 0.0},
                "action": {"active": False, "selection": None}
            },
            narration="Generative model creates prediction",
            highlights=["generative_model", "prediction"]
        ))

        # Frame 4: Prediction error computed
        frames.append(AnimationFrame(
            frame_id=3,
            timestamp=3.0,
            components={
                "sensory_input": {"active": True, "intensity": 0.8},
                "generative_model": {"active": True, "beliefs": ["belief_1", "belief_2"]},
                "prediction": {"active": True, "value": 0.7},
                "prediction_error": {"active": True, "value": 0.1},
                "action": {"active": False, "selection": None}
            },
            narration="Prediction error is calculated",
            highlights=["prediction_error"]
        ))

        # Frame 5: Action selection
        frames.append(AnimationFrame(
            frame_id=4,
            timestamp=4.0,
            components={
                "sensory_input": {"active": True, "intensity": 0.8},
                "generative_model": {"active": True, "beliefs": ["belief_1", "belief_2"]},
                "prediction": {"active": True, "value": 0.7},
                "prediction_error": {"active": True, "value": 0.1},
                "action": {"active": True, "selection": "action_1"}
            },
            narration="Action is selected to minimize free energy",
            highlights=["action"]
        ))

        return AnimationSequence(
            id="perception_action_cycle",
            name="Perception-Action Cycle",
            description="Animation showing the complete perception-action cycle in Active Inference",
            frames=frames,
            duration=5.0,
            frame_rate=30
        )

    def create_belief_updating_animation(self, initial_belief: List[float] = None,
                                       observations: List[float] = None) -> AnimationSequence:
        """Create animation showing belief updating process"""
        if initial_belief is None:
            initial_belief = [0.5, 0.3, 0.2]  # Example belief distribution
        if observations is None:
            observations = [0.8, 0.1, 0.1]  # Example observation

        frames = []

        # Initial state
        frames.append(AnimationFrame(
            frame_id=0,
            timestamp=0.0,
            components={
                "belief_distribution": initial_belief,
                "observation": [0, 0, 0],
                "prediction_error": 0.0,
                "updated_belief": initial_belief.copy()
            },
            narration="Initial belief state before observation",
            highlights=[]
        ))

        # Observation received
        frames.append(AnimationFrame(
            frame_id=1,
            timestamp=1.0,
            components={
                "belief_distribution": initial_belief,
                "observation": observations,
                "prediction_error": 0.0,
                "updated_belief": initial_belief.copy()
            },
            narration="Observation received",
            highlights=["observation"]
        ))

        # Prediction error calculation
        prediction = initial_belief.copy()
        prediction_error = [obs - pred for obs, pred in zip(observations, prediction)]

        frames.append(AnimationFrame(
            frame_id=2,
            timestamp=2.0,
            components={
                "belief_distribution": initial_belief,
                "observation": observations,
                "prediction_error": prediction_error,
                "updated_belief": initial_belief.copy()
            },
            narration="Prediction error calculated",
            highlights=["prediction_error"]
        ))

        # Belief updating with different learning rates
        learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, alpha in enumerate(learning_rates):
            updated_belief = [
                belief + alpha * error
                for belief, error in zip(initial_belief, prediction_error)
            ]
            # Normalize to ensure valid probability distribution
            total = sum(updated_belief)
            updated_belief = [b / total for b in updated_belief]

            frames.append(AnimationFrame(
                frame_id=3 + i,
                timestamp=3.0 + i * 0.5,
                components={
                    "belief_distribution": initial_belief,
                    "observation": observations,
                    "prediction_error": prediction_error,
                    "updated_belief": updated_belief,
                    "learning_rate": alpha
                },
                narration=f"Belief update with learning rate α={alpha:.1f}",
                highlights=["updated_belief"]
            ))

        return AnimationSequence(
            id="belief_updating",
            name="Belief Updating Process",
            description="Animation showing Bayesian belief updating with different learning rates",
            frames=frames,
            duration=6.0,
            frame_rate=30
        )

    def create_multi_agent_animation(self, num_agents: int = 3) -> AnimationSequence:
        """Create animation showing multi-agent Active Inference interactions"""
        frames = []

        # Initialize agent states
        agent_states = [
            {"beliefs": [0.6, 0.3, 0.1], "action": None, "reward": 0.0}
            for _ in range(num_agents)
        ]

        # Frame 1: Initial state
        frames.append(AnimationFrame(
            frame_id=0,
            timestamp=0.0,
            components={
                "agents": agent_states,
                "environment": {"state": "neutral", "shared_info": 0.0},
                "communication": {"active": False, "messages": []}
            },
            narration="Initial state of multi-agent system",
            highlights=[]
        ))

        # Frame 2: Environment change
        frames.append(AnimationFrame(
            frame_id=1,
            timestamp=1.0,
            components={
                "agents": agent_states,
                "environment": {"state": "changed", "shared_info": 0.7},
                "communication": {"active": False, "messages": []}
            },
            narration="Environment state changes, affecting all agents",
            highlights=["environment"]
        ))

        # Frame 3: Individual agent updates
        for i in range(num_agents):
            updated_belief = agent_states[i]["beliefs"].copy()
            # Simulate belief update based on environment change
            updated_belief[0] += 0.2  # Increase belief in first state
            updated_belief[1] -= 0.1  # Decrease belief in second state
            updated_belief[2] -= 0.1  # Decrease belief in third state
            # Normalize
            total = sum(updated_belief)
            updated_belief = [b / total for b in updated_belief]

            agent_states[i]["beliefs"] = updated_belief

        frames.append(AnimationFrame(
            frame_id=2,
            timestamp=2.0,
            components={
                "agents": agent_states,
                "environment": {"state": "changed", "shared_info": 0.7},
                "communication": {"active": False, "messages": []}
            },
            narration="Agents update their beliefs individually",
            highlights=["agents"]
        ))

        # Frame 4: Communication phase
        messages = [
            {"sender": 0, "receiver": 1, "content": agent_states[0]["beliefs"]},
            {"sender": 1, "receiver": 2, "content": agent_states[1]["beliefs"]}
        ]

        frames.append(AnimationFrame(
            frame_id=3,
            timestamp=3.0,
            components={
                "agents": agent_states,
                "environment": {"state": "changed", "shared_info": 0.7},
                "communication": {"active": True, "messages": messages}
            },
            narration="Agents communicate and share information",
            highlights=["communication"]
        ))

        # Frame 5: Coordinated action
        for i in range(num_agents):
            agent_states[i]["action"] = "coordinate"

        frames.append(AnimationFrame(
            frame_id=4,
            timestamp=4.0,
            components={
                "agents": agent_states,
                "environment": {"state": "changed", "shared_info": 0.7},
                "communication": {"active": True, "messages": messages}
            },
            narration="Agents coordinate actions based on shared understanding",
            highlights=["agents"]
        ))

        return AnimationSequence(
            id="multi_agent_interaction",
            name="Multi-Agent Active Inference",
            description="Animation showing coordinated belief updating and action selection in multi-agent systems",
            frames=frames,
            duration=5.0,
            frame_rate=30
        )

    def create_neural_network_animation(self, layers: List[int] = None) -> AnimationSequence:
        """Create animation showing Active Inference in neural networks"""
        if layers is None:
            layers = [4, 8, 8, 4]  # Example network architecture

        frames = []

        # Initialize network state
        network_state = {
            "input_layer": [0.0] * layers[0],
            "hidden_layers": [[0.0] * size for size in layers[1:-1]],
            "output_layer": [0.0] * layers[-1],
            "weights": "initial",
            "activation": "none"
        }

        # Frame 1: Network initialization
        frames.append(AnimationFrame(
            frame_id=0,
            timestamp=0.0,
            components=network_state,
            narration="Neural network initialized for Active Inference",
            highlights=[]
        ))

        # Frame 2: Forward pass (perception)
        network_state["activation"] = "forward"
        frames.append(AnimationFrame(
            frame_id=1,
            timestamp=1.0,
            components=network_state,
            narration="Forward pass: Processing sensory input",
            highlights=["input_layer"]
        ))

        # Frame 3: Prediction generation
        for i in range(len(network_state["hidden_layers"])):
            network_state["hidden_layers"][i] = [0.5] * len(network_state["hidden_layers"][i])

        frames.append(AnimationFrame(
            frame_id=2,
            timestamp=2.0,
            components=network_state,
            narration="Generating predictions in hidden layers",
            highlights=["hidden_layers"]
        ))

        # Frame 4: Output prediction
        network_state["output_layer"] = [0.7, 0.2, 0.05, 0.05]
        frames.append(AnimationFrame(
            frame_id=3,
            timestamp=3.0,
            components=network_state,
            narration="Output layer generates predictions",
            highlights=["output_layer"]
        ))

        # Frame 5: Prediction error and learning
        network_state["weights"] = "updating"
        frames.append(AnimationFrame(
            frame_id=4,
            timestamp=4.0,
            components=network_state,
            narration="Updating weights to minimize prediction error",
            highlights=["weights"]
        ))

        # Frame 6: Free energy minimization
        network_state["activation"] = "minimizing"
        frames.append(AnimationFrame(
            frame_id=5,
            timestamp=5.0,
            components=network_state,
            narration="Free energy minimization through weight updates",
            highlights=["activation"]
        ))

        return AnimationSequence(
            id="neural_network_ai",
            name="Neural Network Active Inference",
            description="Animation showing Active Inference principles in neural network processing",
            frames=frames,
            duration=6.0,
            frame_rate=30
        )

    def create_information_flow_animation(self) -> AnimationSequence:
        """Create animation showing information flow in Active Inference"""
        frames = []

        # Initialize information flow components
        components = {
            "sensory_input": {"value": 0.0, "entropy": 1.0},
            "generative_model": {"complexity": 0.5, "accuracy": 0.3},
            "prediction": {"confidence": 0.2, "information": 0.1},
            "prediction_error": {"magnitude": 0.0, "information_gain": 0.0},
            "free_energy": {"value": 1.0, "trend": "decreasing"},
            "action": {"selection": None, "expected_utility": 0.0}
        }

        # Frame 1: Information at rest
        frames.append(AnimationFrame(
            frame_id=0,
            timestamp=0.0,
            components=components,
            narration="Information flow begins with high uncertainty",
            highlights=[]
        ))

        # Frame 2: Sensory input received
        components["sensory_input"]["value"] = 0.8
        components["sensory_input"]["entropy"] = 0.3
        frames.append(AnimationFrame(
            frame_id=1,
            timestamp=1.0,
            components=components,
            narration="Sensory input reduces uncertainty",
            highlights=["sensory_input"]
        ))

        # Frame 3: Model processes information
        components["generative_model"]["accuracy"] = 0.6
        components["prediction"]["confidence"] = 0.5
        frames.append(AnimationFrame(
            frame_id=2,
            timestamp=2.0,
            components=components,
            narration="Generative model processes sensory information",
            highlights=["generative_model"]
        ))

        # Frame 4: Prediction generated
        components["prediction"]["information"] = 0.4
        frames.append(AnimationFrame(
            frame_id=3,
            timestamp=3.0,
            components=components,
            narration="Prediction generated with increasing confidence",
            highlights=["prediction"]
        ))

        # Frame 5: Prediction error computed
        components["prediction_error"]["magnitude"] = 0.2
        components["prediction_error"]["information_gain"] = 0.3
        frames.append(AnimationFrame(
            frame_id=4,
            timestamp=4.0,
            components=components,
            narration="Prediction error provides information gain",
            highlights=["prediction_error"]
        ))

        # Frame 6: Free energy calculated
        components["free_energy"]["value"] = 0.7
        frames.append(AnimationFrame(
            frame_id=5,
            timestamp=5.0,
            components=components,
            narration="Free energy quantifies model fit and complexity",
            highlights=["free_energy"]
        ))

        # Frame 7: Action selected
        components["action"]["selection"] = "optimal"
        components["action"]["expected_utility"] = 0.6
        frames.append(AnimationFrame(
            frame_id=6,
            timestamp=6.0,
            components=components,
            narration="Action selected to minimize free energy",
            highlights=["action"]
        ))

        # Frame 8: Information flow complete
        components["free_energy"]["value"] = 0.4
        components["free_energy"]["trend"] = "stable"
        frames.append(AnimationFrame(
            frame_id=7,
            timestamp=7.0,
            components=components,
            narration="Information flow stabilizes with reduced free energy",
            highlights=["free_energy"]
        ))

        return AnimationSequence(
            id="information_flow",
            name="Information Flow in Active Inference",
            description="Animation showing complete information flow from sensory input to action selection",
            frames=frames,
            duration=8.0,
            frame_rate=30
        )

    def create_free_energy_minimization(self) -> AnimationSequence:
        """Create animation of free energy minimization"""
        frames = []

        # Create frames showing progressive minimization
        for i in range(11):
            free_energy_value = max(0.1, 1.0 - (i * 0.08))  # Decreasing free energy

            frames.append(AnimationFrame(
                frame_id=i,
                timestamp=i * 0.5,
                components={
                    "free_energy": {
                        "value": free_energy_value,
                        "target": 0.1,
                        "progress": (1.0 - free_energy_value) / 0.9
                    },
                    "model_updates": {"count": i, "active": i < 10},
                    "prediction_accuracy": {"value": min(0.95, 0.4 + (i * 0.05))},
                    "belief_confidence": {"value": min(0.9, 0.3 + (i * 0.06))}
                },
                narration=f"Step {i}: Free energy = {free_energy_value:.3f}",
                highlights=["free_energy"] if i % 2 == 0 else []
            ))

        return AnimationSequence(
            id="free_energy_minimization",
            name="Free Energy Minimization",
            description="Animation showing progressive free energy minimization through model updates",
            frames=frames,
            duration=5.0,
            frame_rate=30
        )


class AnimationEngine:
    """Main animation engine for educational content"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process_animation = ProcessAnimation()
        self.animations: Dict[str, AnimationSequence] = {}

        logger.info("AnimationEngine initialized")

    def create_animation(self, animation_type: AnimationType, name: str,
                        frames: List[AnimationFrame]) -> AnimationSequence:
        """Create a new animation sequence"""
        animation = AnimationSequence(
            id=f"{animation_type.value}_{name.lower().replace(' ', '_')}",
            name=name,
            description=f"{animation_type.value} animation: {name}",
            frames=frames,
            duration=frames[-1].timestamp if frames else 0.0,
            frame_rate=30
        )

        self.animations[animation.id] = animation
        logger.info(f"Created animation: {animation.id}")

        return animation

    def get_process_animation(self, process: str) -> Optional[AnimationSequence]:
        """Get a pre-built process animation"""
        if process == "perception_action_cycle":
            return self.process_animation.create_perception_action_cycle()
        elif process == "free_energy_minimization":
            return self.process_animation.create_free_energy_minimization()
        else:
            logger.warning(f"Process animation not found: {process}")
            return None

    def play_animation(self, animation_id: str, speed: float = 1.0) -> Dict[str, Any]:
        """Play an animation and return playback info"""
        if animation_id not in self.animations:
            logger.error(f"Animation not found: {animation_id}")
            return {}

        animation = self.animations[animation_id]

        return {
            "animation_id": animation_id,
            "name": animation.name,
            "total_frames": len(animation.frames),
            "duration": animation.duration / speed,
            "frame_rate": animation.frame_rate * speed,
            "is_playing": True
        }

    def get_frame_at_time(self, animation_id: str, time: float) -> Optional[Dict[str, Any]]:
        """Get animation frame at specific time"""
        if animation_id not in self.animations:
            return None

        animation = self.animations[animation_id]
        frame = animation.get_frame_at_time(time)

        if frame:
            return {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "components": frame.components,
                "narration": frame.narration,
                "highlights": frame.highlights
            }

        return None

    def export_animation(self, animation_id: str, format: str = "json") -> Optional[Dict[str, Any]]:
        """Export animation in specified format"""
        if animation_id not in self.animations:
            logger.error(f"Animation not found: {animation_id}")
            return None

        animation = self.animations[animation_id]

        if format == "json":
            return {
                "id": animation.id,
                "name": animation.name,
                "description": animation.description,
                "frames": [
                    {
                        "frame_id": frame.frame_id,
                        "timestamp": frame.timestamp,
                        "components": frame.components,
                        "narration": frame.narration,
                        "highlights": frame.highlights
                    }
                    for frame in animation.frames
                ],
                "duration": animation.duration,
                "frame_rate": animation.frame_rate
            }
        else:
            logger.warning(f"Export format not supported: {format}")
            return None
