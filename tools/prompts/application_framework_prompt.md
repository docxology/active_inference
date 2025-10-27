# Application Framework and Domain Development Prompt

**"Active Inference for, with, by Generative AI"**

## 🎯 Mission: Develop Application Frameworks and Domain Implementations

You are tasked with developing comprehensive application frameworks and domain-specific implementations that make Active Inference accessible and practical across diverse fields and industries. This involves creating reusable templates, case studies, integration APIs, and best practice frameworks that enable real-world application of Active Inference principles.

## 📋 Application Framework Requirements

### Core Application Standards (MANDATORY)
1. **Domain Expertise Integration**: Deep understanding of target domain requirements and constraints
2. **Practical Implementation**: Working code examples with real-world applicability
3. **Scalability**: Applications that scale from prototypes to production systems
4. **Integration**: Seamless integration with existing domain tools and workflows
5. **Validation**: Empirical validation and performance benchmarking
6. **Documentation**: Comprehensive guides for domain practitioners

### Application Framework Architecture
```
applications/
├── templates/                  # Reusable implementation templates
│   ├── active_inference_agent_template.py  # Basic agent template
│   ├── generative_model_template.py        # Generative model template
│   ├── policy_selection_template.py        # Policy selection template
│   ├── learning_system_template.py         # Learning system template
│   └── optimization_template.py            # Optimization template
├── case_studies/               # Real-world application examples
│   ├── robotics/               # Robotics case studies
│   ├── neuroscience/           # Neuroscience applications
│   ├── ai_safety/              # AI safety implementations
│   ├── autonomous_systems/     # Autonomous system examples
│   └── clinical_applications/  # Clinical psychology cases
├── integrations/               # External system integrations
│   ├── api_clients/            # API integration clients
│   ├── data_connectors/        # Data source connectors
│   ├── model_exporters/        # Model export utilities
│   └── deployment_tools/       # Deployment automation
├── best_practices/             # Domain-specific best practices
│   ├── robotics_guidelines.py  # Robotics implementation guide
│   ├── neuroscience_practices.py # Neuroscience application practices
│   ├── safety_protocols.py     # AI safety protocols
│   ├── validation_methods.py   # Validation methodologies
│   └── performance_optimization.py # Performance optimization guide
└── domains/                    # Domain-specific implementations
    ├── artificial_intelligence/ # AI domain applications
    ├── engineering/            # Engineering applications
    ├── neuroscience/           # Neuroscience implementations
    ├── psychology/             # Psychology applications
    ├── economics/              # Economic modeling
    └── climate_science/        # Climate decision making
```

## 🏗️ Template Library Development

### Phase 1: Active Inference Agent Template

#### 1.1 Basic Agent Template
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

class AgentState(Enum):
    """Agent operational states"""
    INITIALIZING = "initializing"
    LEARNING = "learning"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    ADAPTING = "adapting"

@dataclass
class GenerativeModel:
    """Generative model specification"""
    state_space: Dict[str, Any]
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]
    transition_model: Callable[[Any, Any], np.ndarray]
    observation_model: Callable[[Any], np.ndarray]
    preference_model: Callable[[Any, Any], float]

@dataclass
class Policy:
    """Policy specification"""
    actions: List[Any]
    horizon: int
    value_function: Optional[Callable] = None
    metadata: Dict[str, Any] = None

@dataclass
class BeliefState:
    """Agent belief state"""
    state_posterior: np.ndarray
    observation_history: List[Any]
    action_history: List[Any]
    free_energy_history: List[float]
    timestamp: float

class ActiveInferenceAgent(ABC):
    """Template for Active Inference agent implementations"""

    def __init__(self, generative_model: GenerativeModel, config: Dict[str, Any]):
        """Initialize Active Inference agent"""
        self.generative_model = generative_model
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Agent state
        self.current_state = AgentState.INITIALIZING
        self.belief_state = self.initialize_beliefs()
        self.policies = self.initialize_policies()

        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.01)
        self.exploration_rate = config.get('exploration_rate', 0.1)
        self.discount_factor = config.get('discount_factor', 0.99)

        # Performance tracking
        self.performance_metrics = {
            'episodes_completed': 0,
            'average_free_energy': [],
            'policy_success_rate': [],
            'adaptation_events': []
        }

    def initialize_beliefs(self) -> BeliefState:
        """Initialize agent beliefs"""
        n_states = len(self.generative_model.state_space)

        # Uniform prior over states
        state_posterior = np.ones(n_states) / n_states

        return BeliefState(
            state_posterior=state_posterior,
            observation_history=[],
            action_history=[],
            free_energy_history=[],
            timestamp=0.0
        )

    @abstractmethod
    def initialize_policies(self) -> List[Policy]:
        """Initialize policy repertoire"""
        pass

    def perceive(self, observation: Any) -> None:
        """Update beliefs based on observation"""
        self.current_state = AgentState.LEARNING

        # Bayesian belief update
        self.belief_state = self.update_beliefs(observation, self.belief_state)

        # Record observation
        self.belief_state.observation_history.append(observation)
        self.belief_state.timestamp = self.get_current_time()

    @abstractmethod
    def update_beliefs(self, observation: Any, belief_state: BeliefState) -> BeliefState:
        """Update beliefs using variational inference"""
        pass

    def plan(self) -> Policy:
        """Select optimal policy using Active Inference"""
        self.current_state = AgentState.PLANNING

        # Calculate expected free energy for each policy
        policy_values = []
        for policy in self.policies:
            efe = self.calculate_expected_free_energy(policy, self.belief_state)
            policy_values.append((policy, efe))

        # Select policy (epsilon-greedy for exploration)
        if np.random.random() < self.exploration_rate:
            selected_policy = np.random.choice(self.policies)
        else:
            # Select policy with lowest expected free energy
            selected_policy = min(policy_values, key=lambda x: x[1])[0]

        return selected_policy

    @abstractmethod
    def calculate_expected_free_energy(self, policy: Policy, belief_state: BeliefState) -> float:
        """Calculate expected free energy for a policy"""
        pass

    def act(self, policy: Policy) -> Any:
        """Execute selected policy"""
        self.current_state = AgentState.EXECUTING

        # Execute first action of policy
        action = policy.actions[0]

        # Record action
        self.belief_state.action_history.append(action)

        # Execute action in environment (to be implemented by subclasses)
        self.execute_action(action)

        return action

    @abstractmethod
    def execute_action(self, action: Any) -> None:
        """Execute action in the environment"""
        pass

    def learn(self, reward: float, next_observation: Any) -> None:
        """Learn from experience"""
        self.current_state = AgentState.EVALUATING

        # Update generative model based on experience
        self.update_generative_model(reward, next_observation, self.belief_state)

        # Update performance metrics
        free_energy = self.calculate_expected_free_energy(
            self.policies[0], self.belief_state
        )
        self.belief_state.free_energy_history.append(free_energy)

        # Decay exploration rate
        self.exploration_rate *= 0.995

    @abstractmethod
    def update_generative_model(self, reward: float, next_observation: Any,
                              belief_state: BeliefState) -> None:
        """Update generative model parameters"""
        pass

    def adapt(self, performance_feedback: Dict[str, Any]) -> None:
        """Adapt agent behavior based on performance feedback"""
        self.current_state = AgentState.ADAPTING

        # Analyze performance
        if performance_feedback.get('poor_performance', False):
            # Increase exploration
            self.exploration_rate = min(0.5, self.exploration_rate * 1.2)

        if performance_feedback.get('overfitting', False):
            # Regularize learning
            self.learning_rate *= 0.8

        # Update policies if needed
        if performance_feedback.get('policy_update_needed', False):
            self.policies = self.update_policies(self.policies, performance_feedback)

        # Record adaptation event
        self.performance_metrics['adaptation_events'].append({
            'timestamp': self.get_current_time(),
            'reason': performance_feedback.get('reason', 'unknown'),
            'changes': performance_feedback.get('changes', {})
        })

    @abstractmethod
    def update_policies(self, policies: List[Policy], feedback: Dict[str, Any]) -> List[Policy]:
        """Update policy repertoire based on feedback"""
        pass

    def get_current_time(self) -> float:
        """Get current time (to be implemented by subclasses)"""
        import time
        return time.time()

    def get_status(self) -> Dict[str, Any]:
        """Get agent status and metrics"""
        return {
            'state': self.current_state.value,
            'belief_entropy': -np.sum(self.belief_state.state_posterior *
                                     np.log(self.belief_state.state_posterior + 1e-10)),
            'policies_count': len(self.policies),
            'observations_count': len(self.belief_state.observation_history),
            'actions_count': len(self.belief_state.action_history),
            'average_free_energy': np.mean(self.belief_state.free_energy_history[-100:]) if self.belief_state.free_energy_history else 0,
            'performance_metrics': self.performance_metrics
        }

    def save_agent(self, filepath: str) -> None:
        """Save agent state and parameters"""
        agent_state = {
            'generative_model': self.generative_model,
            'belief_state': self.belief_state,
            'policies': self.policies,
            'config': self.config,
            'performance_metrics': self.performance_metrics,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate
        }

        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(agent_state, f)

    def load_agent(self, filepath: str) -> None:
        """Load agent state and parameters"""
        import pickle
        with open(filepath, 'rb') as f:
            agent_state = pickle.load(f)

        self.generative_model = agent_state['generative_model']
        self.belief_state = agent_state['belief_state']
        self.policies = agent_state['policies']
        self.config = agent_state['config']
        self.performance_metrics = agent_state['performance_metrics']
        self.exploration_rate = agent_state['exploration_rate']
        self.learning_rate = agent_state['learning_rate']

class SimpleActiveInferenceAgent(ActiveInferenceAgent):
    """Simple implementation of Active Inference agent"""

    def initialize_policies(self) -> List[Policy]:
        """Initialize simple policy repertoire"""
        policies = []

        # Generate random policies
        n_policies = self.config.get('n_policies', 10)
        policy_horizon = self.config.get('policy_horizon', 3)

        for i in range(n_policies):
            actions = []
            for _ in range(policy_horizon):
                # Random action selection
                action_space = self.generative_model.action_space
                if isinstance(action_space, list):
                    action = np.random.choice(action_space)
                else:
                    action = np.random.randint(action_space)
                actions.append(action)

            policy = Policy(
                actions=actions,
                horizon=policy_horizon,
                metadata={'policy_id': i, 'type': 'random'}
            )
            policies.append(policy)

        return policies

    def update_beliefs(self, observation: Any, belief_state: BeliefState) -> BeliefState:
        """Simple belief update"""
        # Get observation likelihood
        obs_likelihood = self.generative_model.observation_model(observation)

        # Bayesian update: posterior ∝ likelihood × prior
        posterior = obs_likelihood * belief_state.state_posterior
        posterior /= np.sum(posterior)  # Normalize

        # Update belief state
        updated_belief_state = BeliefState(
            state_posterior=posterior,
            observation_history=belief_state.observation_history.copy(),
            action_history=belief_state.action_history.copy(),
            free_energy_history=belief_state.free_energy_history.copy(),
            timestamp=self.get_current_time()
        )

        return updated_belief_state

    def calculate_expected_free_energy(self, policy: Policy, belief_state: BeliefState) -> float:
        """Calculate expected free energy for policy"""
        efe = 0.0
        current_beliefs = belief_state.state_posterior.copy()

        # Simulate policy execution
        for action in policy.actions:
            # Epistemic affordance (exploration bonus)
            epistemic_term = 0.0

            # Extrinsic term (goal achievement)
            extrinsic_term = 0.0

            # Simulate state transitions
            for state_idx, state_prob in enumerate(current_beliefs):
                if state_prob > 0:
                    # Predict next state distribution
                    transition_probs = self.generative_model.transition_model(state_idx, action)
                    next_beliefs = transition_probs * state_prob

                    # Epistemic term: KL divergence from current beliefs
                    kl_div = np.sum(next_beliefs * np.log(next_beliefs / current_beliefs + 1e-10))
                    epistemic_term += state_prob * kl_div

                    # Extrinsic term: preference satisfaction
                    for next_state_idx, next_prob in enumerate(next_beliefs):
                        if next_prob > 0:
                            preference = self.generative_model.preference_model(next_state_idx, action)
                            extrinsic_term += state_prob * next_prob * preference

            # Update current beliefs for next step
            next_beliefs_all = np.zeros_like(current_beliefs)
            for state_idx, state_prob in enumerate(current_beliefs):
                transition_probs = self.generative_model.transition_model(state_idx, action)
                next_beliefs_all += transition_probs * state_prob
            current_beliefs = next_beliefs_all / np.sum(next_beliefs_all)

            efe += epistemic_term - extrinsic_term  # Minimize EFE

        return efe

    def execute_action(self, action: Any) -> None:
        """Execute action (placeholder for environment interaction)"""
        self.logger.debug(f"Executing action: {action}")
        # In practice, this would interact with the actual environment

    def update_generative_model(self, reward: float, next_observation: Any,
                              belief_state: BeliefState) -> None:
        """Update generative model parameters"""
        # Simple learning rule (could be more sophisticated)
        learning_rate = self.learning_rate

        # Update observation model based on prediction error
        predicted_obs = self.generative_model.observation_model(
            np.argmax(belief_state.state_posterior)
        )
        prediction_error = next_observation - predicted_obs

        # Simple gradient descent update (placeholder)
        # In practice, this would update the actual model parameters
        self.logger.debug(f"Prediction error: {np.mean(np.abs(prediction_error))}")

    def update_policies(self, policies: List[Policy], feedback: Dict[str, Any]) -> List[Policy]:
        """Update policy repertoire"""
        # Simple policy evolution (placeholder)
        # In practice, this could use evolutionary algorithms or policy gradients

        # Remove worst performing policies and add new random ones
        n_remove = max(1, len(policies) // 4)
        n_add = n_remove

        # Sort policies by some criteria (placeholder)
        sorted_policies = sorted(policies, key=lambda p: len(p.actions))

        # Remove worst policies
        updated_policies = sorted_policies[:-n_remove] if len(sorted_policies) > n_remove else sorted_policies

        # Add new random policies
        for i in range(n_add):
            new_policy = self.generate_random_policy()
            updated_policies.append(new_policy)

        return updated_policies

    def generate_random_policy(self) -> Policy:
        """Generate a random policy"""
        policy_horizon = self.config.get('policy_horizon', 3)
        actions = []

        for _ in range(policy_horizon):
            action_space = self.generative_model.action_space
            if isinstance(action_space, list):
                action = np.random.choice(action_space)
            else:
                action = np.random.randint(action_space)
            actions.append(action)

        return Policy(
            actions=actions,
            horizon=policy_horizon,
            metadata={'type': 'random', 'generated': True}
        )

class AgentFactory:
    """Factory for creating Active Inference agents"""

    @staticmethod
    def create_agent(agent_type: str, generative_model: GenerativeModel,
                   config: Dict[str, Any]) -> ActiveInferenceAgent:
        """Create agent instance based on type"""

        agent_types = {
            'simple': SimpleActiveInferenceAgent,
            'advanced': None,  # Placeholder for more sophisticated agents
            'hierarchical': None,  # Placeholder for hierarchical agents
            'multi_agent': None   # Placeholder for multi-agent systems
        }

        if agent_type not in agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_class = agent_types[agent_type]
        if agent_class is None:
            raise NotImplementedError(f"Agent type {agent_type} not yet implemented")

        return agent_class(generative_model, config)

    @staticmethod
    def create_generative_model(model_spec: Dict[str, Any]) -> GenerativeModel:
        """Create generative model from specification"""

        def default_transition_model(state, action):
            """Simple random transition model"""
            n_states = len(model_spec['state_space'])
            # Uniform transitions (could be learned)
            return np.ones(n_states) / n_states

        def default_observation_model(state):
            """Simple observation model"""
            n_observations = len(model_spec['observation_space'])
            # Random observations (could be learned)
            return np.random.rand(n_observations)

        def default_preference_model(state, action):
            """Simple preference model"""
            # No preferences by default
            return 0.0

        return GenerativeModel(
            state_space=model_spec.get('state_space', {}),
            observation_space=model_spec.get('observation_space', {}),
            action_space=model_spec.get('action_space', []),
            transition_model=model_spec.get('transition_model', default_transition_model),
            observation_model=model_spec.get('observation_model', default_observation_model),
            preference_model=model_spec.get('preference_model', default_preference_model)
        )
```

### Phase 2: Domain-Specific Application Frameworks

#### 2.1 Robotics Application Framework
```python
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from active_inference_agent import ActiveInferenceAgent, GenerativeModel, Policy

@dataclass
class RobotState:
    """Robot state representation"""
    position: np.ndarray  # (x, y, z)
    orientation: np.ndarray  # Quaternion or Euler angles
    joint_angles: np.ndarray
    velocity: np.ndarray
    sensor_readings: Dict[str, float]

@dataclass
class RobotAction:
    """Robot action specification"""
    joint_velocities: np.ndarray
    end_effector_target: Optional[np.ndarray] = None
    gripper_command: Optional[str] = None  # 'open', 'close', 'hold'

@dataclass
class TaskSpecification:
    """Robotics task specification"""
    goal_position: np.ndarray
    goal_orientation: np.ndarray
    allowed_collision_objects: List[str]
    time_limit: float
    success_criteria: Dict[str, Any]

class RoboticsActiveInferenceAgent(ActiveInferenceAgent):
    """Active Inference agent for robotics applications"""

    def __init__(self, robot_interface: Any, task_spec: TaskSpecification,
                 config: Dict[str, Any]):
        """Initialize robotics agent"""

        # Define generative model for robotics
        generative_model = self.create_robotics_generative_model(robot_interface, task_spec)

        super().__init__(generative_model, config)

        self.robot_interface = robot_interface
        self.task_spec = task_spec
        self.collision_detector = self.initialize_collision_detector()
        self.path_planner = self.initialize_path_planner()

        # Robotics-specific metrics
        self.robotics_metrics = {
            'path_efficiency': [],
            'collision_avoidance': [],
            'task_completion_time': [],
            'energy_consumption': []
        }

    def create_robotics_generative_model(self, robot_interface: Any,
                                       task_spec: TaskSpecification) -> GenerativeModel:
        """Create generative model for robotics domain"""

        # State space: robot configuration + environment state
        state_space = {
            'robot_config': robot_interface.get_configuration_space(),
            'environment_objects': ['table', 'obstacle1', 'obstacle2', 'goal'],
            'task_progress': ['not_started', 'in_progress', 'completed', 'failed']
        }

        # Observation space: sensor readings
        observation_space = {
            'joint_positions': robot_interface.get_joint_limits(),
            'end_effector_pose': {'position': [-2, 2], 'orientation': [-np.pi, np.pi]},
            'force_torque': {'range': [-100, 100]},
            'vision_features': {'dimensions': 128}
        }

        # Action space: joint velocities + gripper commands
        action_space = robot_interface.get_action_space()

        def transition_model(state, action):
            """Predict state transitions"""
            # Use robot kinematics and dynamics
            next_state = robot_interface.predict_state_transition(state, action)
            return next_state

        def observation_model(state):
            """Predict observations from state"""
            # Use sensor models
            predicted_obs = robot_interface.predict_observations(state)
            return predicted_obs

        def preference_model(state, action):
            """Preference for state-action pairs"""
            # Task-specific preferences
            distance_to_goal = np.linalg.norm(
                state['end_effector_position'] - task_spec.goal_position
            )

            # Negative preference for collisions
            collision_penalty = -10 if self.check_collision(state) else 0

            # Positive preference for progress toward goal
            progress_reward = -distance_to_goal * 0.1

            return progress_reward + collision_penalty

        return GenerativeModel(
            state_space=state_space,
            observation_space=observation_space,
            action_space=action_space,
            transition_model=transition_model,
            observation_model=observation_model,
            preference_model=preference_model
        )

    def initialize_collision_detector(self) -> Any:
        """Initialize collision detection system"""
        # Placeholder for collision detection library (e.g., FCL, Bullet)
        class CollisionDetector:
            def check_collision(self, state):
                return False  # No collision by default

        return CollisionDetector()

    def initialize_path_planner(self) -> Any:
        """Initialize path planning system"""
        # Placeholder for motion planning (e.g., OMPL, MoveIt!)
        class PathPlanner:
            def plan_path(self, start, goal):
                return [start, goal]  # Direct path

        return PathPlanner()

    def initialize_policies(self) -> List[Policy]:
        """Initialize robotics-specific policies"""
        policies = []

        # Reach-to-grasp policy
        reach_policy = Policy(
            actions=[
                RobotAction(joint_velocities=np.zeros(6), end_effector_target=self.task_spec.goal_position),
                RobotAction(joint_velocities=np.zeros(6), gripper_command='close')
            ],
            horizon=2,
            metadata={'type': 'reach_and_grasp', 'skill_level': 'basic'}
        )
        policies.append(reach_policy)

        # Obstacle avoidance policy
        avoidance_policy = Policy(
            actions=[
                RobotAction(joint_velocities=np.array([0.1, 0, 0, 0, 0, 0])),  # Move right
                RobotAction(joint_velocities=np.zeros(6), end_effector_target=self.task_spec.goal_position)
            ],
            horizon=2,
            metadata={'type': 'obstacle_avoidance', 'skill_level': 'intermediate'}
        )
        policies.append(avoidance_policy)

        return policies

    def update_beliefs(self, observation: Any, belief_state: BeliefState) -> BeliefState:
        """Update beliefs with sensor fusion"""
        # Fuse multiple sensor modalities
        joint_positions = observation.get('joint_positions', np.zeros(6))
        vision_features = observation.get('vision_features', np.zeros(128))
        force_torque = observation.get('force_torque', np.zeros(6))

        # Multi-modal belief update
        # Combine proprioception, vision, and tactile feedback
        state_posterior = self.fuse_sensor_modalities(
            joint_positions, vision_features, force_torque, belief_state.state_posterior
        )

        # Normalize
        state_posterior /= np.sum(state_posterior)

        updated_belief_state = BeliefState(
            state_posterior=state_posterior,
            observation_history=belief_state.observation_history + [observation],
            action_history=belief_state.action_history,
            free_energy_history=belief_state.free_energy_history,
            timestamp=self.get_current_time()
        )

        return updated_belief_state

    def fuse_sensor_modalities(self, joint_positions: np.ndarray, vision_features: np.ndarray,
                             force_torque: np.ndarray, prior_beliefs: np.ndarray) -> np.ndarray:
        """Fuse multiple sensor modalities for belief update"""
        # Placeholder for sensor fusion algorithm
        # In practice, could use Kalman filters, particle filters, etc.

        # Simple weighted combination
        proprioception_likelihood = self.compute_proprioception_likelihood(joint_positions)
        vision_likelihood = self.compute_vision_likelihood(vision_features)
        tactile_likelihood = self.compute_tactile_likelihood(force_torque)

        # Combine likelihoods
        combined_likelihood = (proprioception_likelihood * vision_likelihood * tactile_likelihood)

        # Bayesian update
        posterior = combined_likelihood * prior_beliefs
        return posterior

    def compute_proprioception_likelihood(self, joint_positions: np.ndarray) -> np.ndarray:
        """Compute likelihood from proprioceptive sensors"""
        # Placeholder: assume Gaussian noise model
        n_states = len(self.generative_model.state_space)
        likelihood = np.exp(-np.random.rand(n_states))  # Random for now
        return likelihood / np.sum(likelihood)

    def compute_vision_likelihood(self, vision_features: np.ndarray) -> np.ndarray:
        """Compute likelihood from vision sensors"""
        # Placeholder: use vision model
        n_states = len(self.generative_model.state_space)
        likelihood = np.exp(-np.random.rand(n_states))
        return likelihood / np.sum(likelihood)

    def compute_tactile_likelihood(self, force_torque: np.ndarray) -> np.ndarray:
        """Compute likelihood from tactile sensors"""
        # Placeholder: detect contact
        n_states = len(self.generative_model.state_space)
        contact_detected = np.linalg.norm(force_torque) > 10
        likelihood = np.ones(n_states) * 0.1
        if contact_detected:
            likelihood[0] = 0.9  # Assume contact indicates specific state
        return likelihood / np.sum(likelihood)

    def calculate_expected_free_energy(self, policy: Policy, belief_state: BeliefState) -> float:
        """Calculate EFE with robotics constraints"""
        efe = 0.0
        current_beliefs = belief_state.state_posterior.copy()

        for action in policy.actions:
            # Epistemic affordance (exploration)
            epistemic_term = self.calculate_epistemic_term(current_beliefs, action)

            # Extrinsic term (task achievement)
            extrinsic_term = self.calculate_extrinsic_term(current_beliefs, action)

            # Safety term (collision avoidance)
            safety_term = self.calculate_safety_term(current_beliefs, action)

            # Combine terms
            step_efe = epistemic_term + extrinsic_term + safety_term
            efe += step_efe

            # Update beliefs for next step
            current_beliefs = self.predict_belief_update(current_beliefs, action)

        return efe

    def calculate_epistemic_term(self, beliefs: np.ndarray, action: RobotAction) -> float:
        """Calculate epistemic affordance for action"""
        # Expected information gain
        predicted_beliefs = self.predict_belief_update(beliefs, action)
        kl_divergence = np.sum(predicted_beliefs * np.log(predicted_beliefs / beliefs + 1e-10))
        return kl_divergence

    def calculate_extrinsic_term(self, beliefs: np.ndarray, action: RobotAction) -> float:
        """Calculate extrinsic value (task progress)"""
        expected_progress = 0.0

        for state_idx, prob in enumerate(beliefs):
            if prob > 0:
                # Expected progress toward goal
                progress = self.evaluate_task_progress(state_idx, action)
                expected_progress += prob * progress

        return -expected_progress  # Negative because we minimize EFE

    def calculate_safety_term(self, beliefs: np.ndarray, action: RobotAction) -> float:
        """Calculate safety penalty"""
        collision_probability = self.estimate_collision_probability(beliefs, action)
        safety_penalty = collision_probability * 100  # High penalty for collisions
        return safety_penalty

    def predict_belief_update(self, beliefs: np.ndarray, action: RobotAction) -> np.ndarray:
        """Predict how beliefs will update after action"""
        # Use generative model to predict observation likelihoods
        predicted_likelihoods = []

        for state_idx in range(len(beliefs)):
            # Predict what observation we'd get from this state-action pair
            predicted_obs = self.generative_model.observation_model(state_idx)
            likelihood = np.exp(-np.random.rand())  # Placeholder
            predicted_likelihoods.append(likelihood)

        predicted_likelihoods = np.array(predicted_likelihoods)

        # Bayesian update
        posterior = predicted_likelihoods * beliefs
        posterior /= np.sum(posterior)

        return posterior

    def estimate_collision_probability(self, beliefs: np.ndarray, action: RobotAction) -> float:
        """Estimate probability of collision for action"""
        # Use collision detector to estimate collision risk
        collision_prob = 0.0

        for state_idx, prob in enumerate(beliefs):
            if prob > 0:
                # Simulate action execution
                predicted_state = self.generative_model.transition_model(state_idx, action)
                if self.collision_detector.check_collision(predicted_state):
                    collision_prob += prob

        return collision_prob

    def evaluate_task_progress(self, state_idx: int, action: RobotAction) -> float:
        """Evaluate how action contributes to task progress"""
        # Placeholder: distance reduction toward goal
        current_position = np.random.rand(3)  # Placeholder
        goal_position = self.task_spec.goal_position

        distance_before = np.linalg.norm(current_position - goal_position)

        # Predict position after action
        predicted_position = current_position + np.random.rand(3) * 0.1  # Placeholder
        distance_after = np.linalg.norm(predicted_position - goal_position)

        progress = distance_before - distance_after
        return max(0, progress)  # Only positive progress

    def execute_action(self, action: RobotAction) -> None:
        """Execute robot action"""
        try:
            # Send commands to robot
            if action.joint_velocities is not None:
                self.robot_interface.set_joint_velocities(action.joint_velocities)

            if action.end_effector_target is not None:
                self.robot_interface.move_end_effector_to(action.end_effector_target)

            if action.gripper_command:
                self.robot_interface.control_gripper(action.gripper_command)

            # Update robotics metrics
            self.robotics_metrics['energy_consumption'].append(
                np.sum(np.abs(action.joint_velocities)) if action.joint_velocities is not None else 0
            )

        except Exception as e:
            self.logger.error(f"Failed to execute robot action: {e}")
            raise

    def check_collision(self, state: Dict[str, Any]) -> bool:
        """Check for collisions in given state"""
        return self.collision_detector.check_collision(state)

    def get_robotics_status(self) -> Dict[str, Any]:
        """Get robotics-specific status"""
        status = self.get_status()
        status.update({
            'robotics_metrics': {
                metric: np.mean(values[-100:]) if values else 0
                for metric, values in self.robotics_metrics.items()
            },
            'task_progress': self.evaluate_overall_task_progress(),
            'safety_status': 'safe' if not self.check_collision(self.belief_state) else 'collision_detected'
        })
        return status

    def evaluate_overall_task_progress(self) -> float:
        """Evaluate overall progress toward task completion"""
        # Placeholder: distance to goal
        current_position = np.random.rand(3)  # Placeholder
        goal_position = self.task_spec.goal_position
        distance = np.linalg.norm(current_position - goal_position)

        # Normalize by initial distance (placeholder)
        initial_distance = 5.0
        progress = 1.0 - (distance / initial_distance)
        return max(0, min(1, progress))

class RoboticsApplicationFramework:
    """Framework for robotics applications of Active Inference"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize robotics application framework"""
        self.config = config
        self.logger = logging.getLogger('RoboticsApplicationFramework')
        self.agent_templates = self.initialize_agent_templates()
        self.task_templates = self.initialize_task_templates()

    def initialize_agent_templates(self) -> Dict[str, Any]:
        """Initialize agent templates for different robot types"""
        return {
            'manipulator': {
                'generative_model_config': {
                    'state_space': {'dof': 6, 'workspace': {'x': [-1, 1], 'y': [-1, 1], 'z': [0, 1]}},
                    'action_space': ['joint_control', 'cartesian_control', 'gripper_control']
                },
                'agent_config': {
                    'learning_rate': 0.01,
                    'exploration_rate': 0.1,
                    'policy_horizon': 5
                }
            },
            'mobile_robot': {
                'generative_model_config': {
                    'state_space': {'position': {'x': [-10, 10], 'y': [-10, 10]}, 'heading': [0, 2*np.pi]},
                    'action_space': ['move_forward', 'turn_left', 'turn_right', 'stop']
                },
                'agent_config': {
                    'learning_rate': 0.05,
                    'exploration_rate': 0.2,
                    'policy_horizon': 10
                }
            }
        }

    def initialize_task_templates(self) -> Dict[str, Any]:
        """Initialize task templates"""
        return {
            'pick_and_place': {
                'goal_position': np.array([0.5, 0.0, 0.2]),
                'goal_orientation': np.array([0, 0, 0, 1]),  # Quaternion
                'allowed_collision_objects': ['table'],
                'time_limit': 60.0,
                'success_criteria': {
                    'position_tolerance': 0.05,
                    'orientation_tolerance': 0.1
                }
            },
            'navigation': {
                'goal_position': np.array([5.0, 5.0, 0.0]),
                'allowed_collision_objects': [],
                'time_limit': 300.0,
                'success_criteria': {
                    'position_tolerance': 0.5,
                    'timeout_penalty': True
                }
            }
        }

    def create_robotics_agent(self, robot_type: str, task_type: str,
                             robot_interface: Any) -> RoboticsActiveInferenceAgent:
        """Create robotics agent for specific robot and task"""

        if robot_type not in self.agent_templates:
            raise ValueError(f"Unknown robot type: {robot_type}")

        if task_type not in self.task_templates:
            raise ValueError(f"Unknown task type: {task_type}")

        agent_template = self.agent_templates[robot_type]
        task_template = self.task_templates[task_type]

        # Create task specification
        task_spec = TaskSpecification(**task_template)

        # Merge configurations
        agent_config = {**agent_template['agent_config'], **self.config}

        # Create agent
        agent = RoboticsActiveInferenceAgent(robot_interface, task_spec, agent_config)

        return agent

    def run_robotics_experiment(self, agent: RoboticsActiveInferenceAgent,
                               max_episodes: int = 100) -> Dict[str, Any]:
        """Run robotics experiment"""
        results = {
            'episodes': [],
            'success_rate': 0.0,
            'average_completion_time': 0.0,
            'collision_rate': 0.0,
            'learning_curve': []
        }

        successful_episodes = 0
        completion_times = []
        collision_count = 0

        for episode in range(max_episodes):
            episode_result = self.run_episode(agent)
            results['episodes'].append(episode_result)

            if episode_result['success']:
                successful_episodes += 1
                completion_times.append(episode_result['completion_time'])

            if episode_result['collision']:
                collision_count += 1

            # Track learning progress
            results['learning_curve'].append({
                'episode': episode,
                'free_energy': episode_result.get('final_free_energy', 0),
                'success': episode_result['success']
            })

        # Calculate summary statistics
        results['success_rate'] = successful_episodes / max_episodes
        results['average_completion_time'] = np.mean(completion_times) if completion_times else 0
        results['collision_rate'] = collision_count / max_episodes

        return results

    def run_episode(self, agent: RoboticsActiveInferenceAgent) -> Dict[str, Any]:
        """Run single episode"""
        episode_start_time = self.get_current_time()
        max_steps = 1000
        collision_detected = False

        # Reset agent for new episode
        agent.belief_state = agent.initialize_beliefs()

        for step in range(max_steps):
            # Get current observation (placeholder)
            observation = self.generate_observation(agent)

            # Agent perception and planning cycle
            agent.perceive(observation)
            policy = agent.plan()
            action = agent.act(policy)

            # Check for collisions
            if agent.check_collision(agent.belief_state):
                collision_detected = True
                break

            # Check for task completion
            if agent.evaluate_overall_task_progress() >= 0.95:  # 95% complete
                break

        completion_time = self.get_current_time() - episode_start_time

        return {
            'success': agent.evaluate_overall_task_progress() >= 0.95 and not collision_detected,
            'completion_time': completion_time,
            'steps_taken': step + 1,
            'collision': collision_detected,
            'final_progress': agent.evaluate_overall_task_progress(),
            'final_free_energy': agent.belief_state.free_energy_history[-1] if agent.belief_state.free_energy_history else 0
        }

    def generate_observation(self, agent: RoboticsActiveInferenceAgent) -> Dict[str, Any]:
        """Generate observation for agent (placeholder)"""
        return {
            'joint_positions': np.random.rand(6) * 2 * np.pi,  # Random joint angles
            'end_effector_pose': {
                'position': np.random.rand(3) * 2 - 1,  # Random position in [-1, 1]
                'orientation': np.random.rand(4)  # Random quaternion
            },
            'force_torque': np.random.randn(6) * 10,  # Random forces/torques
            'vision_features': np.random.rand(128)  # Random vision features
        }

    def get_current_time(self) -> float:
        """Get current time"""
        import time
        return time.time()

class TestRoboticsApplicationFramework:
    """Tests for robotics application framework"""

    @pytest.fixture
    def framework(self):
        """Create robotics framework for testing"""
        config = {'test_mode': True}
        return RoboticsApplicationFramework(config)

    def test_create_robotics_agent(self, framework):
        """Test creating robotics agent"""
        # Mock robot interface
        class MockRobotInterface:
            def get_configuration_space(self):
                return {'joints': 6, 'limits': [-np.pi, np.pi]}

            def get_action_space(self):
                return ['joint_velocity', 'end_effector_pose', 'gripper_command']

            def predict_state_transition(self, state, action):
                return np.random.rand(10)  # Mock next state

            def predict_observations(self, state):
                return np.random.rand(5)  # Mock observations

        robot_interface = MockRobotInterface()

        agent = framework.create_robotics_agent('manipulator', 'pick_and_place', robot_interface)

        assert isinstance(agent, RoboticsActiveInferenceAgent)
        assert agent.task_spec.time_limit == 60.0
        assert agent.config['learning_rate'] == 0.01

    def test_robotics_experiment_execution(self, framework):
        """Test running robotics experiment"""
        # Create mock agent
        class MockRoboticsAgent:
            def initialize_beliefs(self):
                return None
            def evaluate_overall_task_progress(self):
                return 0.9  # 90% complete
            def check_collision(self, state):
                return False

        agent = MockRoboticsAgent()

        # Override methods for testing
        framework.generate_observation = lambda agent: {'mock': 'observation'}
        framework.run_episode = lambda agent: {
            'success': True,
            'completion_time': 10.0,
            'steps_taken': 50,
            'collision': False,
            'final_progress': 0.95,
            'final_free_energy': -5.0
        }

        results = framework.run_robotics_experiment(agent, max_episodes=5)

        assert results['success_rate'] == 1.0  # All episodes successful in mock
        assert len(results['episodes']) == 5
        assert len(results['learning_curve']) == 5
        assert results['average_completion_time'] == 10.0
```

---

**"Active Inference for, with, by Generative AI"** - Building practical application frameworks that make Active Inference accessible across domains, from robotics to neuroscience, enabling real-world deployment and validation of theoretical principles.
