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
