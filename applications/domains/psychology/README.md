# Psychology Domain

This directory contains Active Inference implementations and interfaces specifically designed for cognitive psychology, behavioral modeling, and mental process applications.

## Overview

The psychology domain provides specialized tools and interfaces for applying Active Inference to:

- **Cognitive processes** and decision-making mechanisms
- **Learning and memory** systems and models
- **Attention and cognitive control** architectures
- **Social cognition** and theory of mind

These implementations bridge cognitive psychology with computational modeling, providing tools for understanding and simulating human cognition and behavior.

## Directory Structure

```
psychology/
├── interfaces/           # Domain-specific Active Inference interfaces
│   ├── cognitive_models.py # Cognitive process models
│   ├── learning_memory.py  # Learning and memory systems
│   ├── attention_control.py # Attention and control
│   └── social_cognition.py # Social interaction models
├── implementations/      # Complete psychological applications
│   ├── decision_making.py  # Decision-making systems
│   ├── skill_learning.py   # Motor and cognitive skill acquisition
│   ├── problem_solving.py  # Problem-solving and reasoning
│   └── emotion_modeling.py # Emotion and motivation systems
├── examples/            # Usage examples and tutorials
│   ├── basic_decision.py   # Basic decision-making
│   ├── learning_tasks.py   # Learning and memory tasks
│   ├── attention_study.py  # Attention experiments
│   └── social_scenarios.py # Social interaction scenarios
├── experiments/         # Psychological experiment implementations
│   ├── stroop_task.py     # Cognitive conflict tasks
│   ├── n_back.py          # Working memory tasks
│   └── iowa_gambling.py   # Decision-making under risk
└── tests/               # Psychology-specific tests
    ├── test_cognition.py
    ├── test_learning.py
    └── test_social.py
```

## Core Components

### 🧠 Cognitive Models
Active Inference implementations of core cognitive processes:

- **Decision Making**: Bayesian decision theory and value-based choice
- **Problem Solving**: Reasoning and planning under uncertainty
- **Cognitive Control**: Executive function and goal maintenance
- **Metacognition**: Self-monitoring and confidence judgments

### 📚 Learning and Memory
Models of learning and memory systems:

- **Associative Learning**: Classical and operant conditioning models
- **Working Memory**: Active maintenance of information
- **Long-term Memory**: Encoding, storage, and retrieval processes
- **Skill Acquisition**: Motor and cognitive skill learning

### 🎯 Attention and Control
Models of attention and cognitive control:

- **Selective Attention**: Filtering and selection of relevant information
- **Cognitive Control**: Goal-directed behavior and conflict resolution
- **Task Switching**: Switching between different cognitive tasks
- **Response Inhibition**: Stopping and control of automatic responses

### 👥 Social Cognition
Models of social interaction and understanding:

- **Theory of Mind**: Understanding others' mental states
- **Social Learning**: Learning from others' behavior and outcomes
- **Cooperation**: Cooperative behavior and coordination
- **Communication**: Language and non-verbal communication models

## Getting Started

### Decision Making Models

```python
from active_inference.applications.domains.psychology.interfaces.cognitive_models import DecisionMakingModel

# Create decision-making model
decision_config = {
    'options': ['A', 'B', 'C'],
    'reward_structure': {'A': 1.0, 'B': 0.5, 'C': -0.2},
    'uncertainty': 'risk',  # risk vs ambiguity
    'time_horizon': 10
}

decision_model = DecisionMakingModel(decision_config)

# Simulate decision-making process
context = {'previous_outcomes': [1.0, 0.5, 1.0, -0.2]}
preferences = {'maximize_reward': 1.0, 'minimize_effort': 0.3}

# Make decision
choice, confidence = decision_model.make_decision(context, preferences)
expected_value = decision_model.compute_expected_value(choice)
```

### Learning and Memory Systems

```python
from active_inference.applications.domains.psychology.interfaces.learning_memory import LearningSystem

# Set up learning system
learning_config = {
    'learning_type': 'associative',  # associative, reinforcement, supervised
    'memory_capacity': 1000,
    'learning_rate': 0.1,
    'retention_rate': 0.95
}

learning_system = LearningSystem(learning_config)

# Learning task example
stimuli = ['red_circle', 'blue_square', 'green_triangle']
responses = ['approach', 'avoid', 'neutral']
rewards = [1.0, -1.0, 0.0]

# Train the system
for trial in range(100):
    stimulus = random.choice(stimuli)
    correct_response = responses[stimuli.index(stimulus)]

    prediction = learning_system.predict(stimulus)
    reward = rewards[stimuli.index(stimulus)] if prediction == correct_response else -0.5

    learning_system.update(stimulus, prediction, reward)
```

### Attention and Cognitive Control

```python
from active_inference.applications.domains.psychology.interfaces.attention_control import AttentionSystem

# Create attention system
attention_config = {
    'attention_capacity': 4,  # items
    'task_demands': 'high',
    'distractor_resistance': 0.7,
    'goal_priority': 0.8
}

attention_system = AttentionSystem(attention_config)

# Attention task simulation
stimuli = [
    {'type': 'target', 'location': [100, 100], 'priority': 1.0},
    {'type': 'distractor', 'location': [200, 150], 'priority': 0.2},
    {'type': 'target', 'location': [150, 200], 'priority': 0.8}
]

# Allocate attention
attention_weights = attention_system.allocate_attention(stimuli)
focused_items = attention_system.focus_attention(attention_weights)
performance = attention_system.evaluate_performance(focused_items)
```

## Key Features

### Psychological Validity
- **Empirical Grounding**: Based on established psychological theories and findings
- **Behavioral Predictions**: Generate testable predictions about human behavior
- **Individual Differences**: Model individual variation in cognitive abilities
- **Developmental Changes**: Account for cognitive development across the lifespan

### Experimental Integration
- **Task Implementation**: Ready-to-use implementations of classic psychological tasks
- **Parameter Fitting**: Tools for fitting models to behavioral data
- **Statistical Testing**: Hypothesis testing and model comparison tools
- **Data Visualization**: Psychological data visualization and analysis

### Cognitive Architecture
- **Modular Design**: Separate components for different cognitive processes
- **Integration**: Seamless integration between different cognitive systems
- **Hierarchical Processing**: Multi-level cognitive processing architectures
- **Adaptive Systems**: Systems that learn and adapt to task demands

## Applications

### Research Applications
- **Cognitive Modeling**: Understanding mechanisms of human cognition
- **Individual Differences**: Modeling variation in cognitive abilities
- **Developmental Psychology**: Cognitive development across the lifespan
- **Clinical Psychology**: Modeling cognitive processes in mental health

### Clinical Applications
- **Assessment**: Cognitive assessment and diagnosis tools
- **Treatment**: Cognitive training and rehabilitation programs
- **Prognosis**: Predicting treatment outcomes and recovery
- **Prevention**: Early identification of cognitive risk factors

### Educational Applications
- **Learning Science**: Understanding how people learn and remember
- **Educational Technology**: Adaptive learning systems and intelligent tutors
- **Assessment**: Cognitive assessment for educational placement
- **Intervention**: Targeted interventions for learning difficulties

## Integration with Core Framework

```python
from active_inference.core import GenerativeModel, PolicySelection
from active_inference.applications.domains.psychology.interfaces import CognitiveInterface

# Configure core components for cognitive modeling
generative_model = GenerativeModel({
    'model_type': 'cognitive',
    'belief_dim': 50,
    'policy_dim': 20,
    'precision': 'adaptive'
})

policy_selection = PolicySelection({
    'method': 'expected_free_energy',
    'preferences': {'accuracy': 1.0, 'effort': -0.5, 'novelty': 0.3}
})

# Create psychology-specific interface
cognitive_interface = CognitiveInterface({
    'cognitive_domain': 'decision_making',
    'task_type': 'probabilistic_reversal',
    'generative_model': generative_model,
    'policy_selection': policy_selection
})

# Run cognitive experiment
experiment_data = load_experiment_data('participant_01.csv')
results = cognitive_interface.run_experiment(experiment_data)
model_fit = cognitive_interface.fit_model(results)
```

## Performance Characteristics

### Cognitive Task Requirements
- **Response Time**: Support for millisecond-precision behavioral modeling
- **Memory Load**: Efficient handling of working memory requirements
- **Task Complexity**: Scaling with cognitive task complexity
- **Individual Differences**: Modeling inter-individual variability

### Computational Efficiency
- **Real-time Simulation**: Fast enough for real-time cognitive modeling
- **Parameter Estimation**: Efficient parameter fitting algorithms
- **Model Comparison**: Statistical tools for model comparison
- **Data Handling**: Efficient processing of large behavioral datasets

### Validation Metrics
- **Behavioral Fit**: Accuracy in reproducing human behavioral patterns
- **Predictive Power**: Ability to predict behavior in new situations
- **Generalization**: Performance across different tasks and contexts
- **Individual Differences**: Accounting for individual variation

## Experimental Integration

### Standard Tasks
- **Psychophysics**: Detection, discrimination, and scaling tasks
- **Cognitive Tasks**: Working memory, attention, and executive function tasks
- **Learning Tasks**: Classical conditioning, skill learning, and concept formation
- **Social Tasks**: Theory of mind, cooperation, and social learning tasks

### Data Integration
- **Behavioral Data**: Reaction times, accuracy, confidence ratings
- **Physiological Data**: Eye movements, EEG, fMRI integration
- **Performance Metrics**: Learning curves, transfer effects, individual differences
- **Statistical Analysis**: Parameter estimation, model comparison, hypothesis testing

## Contributing

We welcome contributions to the psychology domain! Priority areas include:

### Implementation Contributions
- **New Cognitive Models**: Implementations of specific cognitive theories
- **Experimental Tasks**: New psychological experiment implementations
- **Data Analysis Tools**: Tools for analyzing behavioral and cognitive data
- **Integration Methods**: Better integration with psychological research tools

### Research Contributions
- **Model Validation**: Empirical validation studies of cognitive models
- **Clinical Applications**: Applications to mental health and clinical psychology
- **Educational Applications**: Learning and educational technology applications
- **Cross-cultural Studies**: Cultural variation in cognitive processes

### Quality Standards
- **Psychological Validity**: Grounded in established psychological theory and research
- **Empirical Testing**: Validated against behavioral or neural data where possible
- **Reproducibility**: Reproducible implementations and results
- **Documentation**: Clear documentation accessible to psychology researchers

## Learning Resources

### Tutorials and Examples
- **Cognitive Modeling**: Introduction to computational cognitive modeling
- **Experimental Design**: Designing experiments with Active Inference models
- **Data Analysis**: Analyzing behavioral data with cognitive models
- **Clinical Applications**: Applications in clinical psychology and neuroscience

### Research Literature
- **Cognitive Psychology**: Active Inference applications in cognitive psychology
- **Computational Models**: Computational modeling of cognitive processes
- **Individual Differences**: Modeling individual variation in cognition
- **Clinical Applications**: Applications in mental health research

## Related Domains

- **[Neuroscience Domain](../neuroscience/)**: Neural implementations of cognitive processes
- **[Education Domain](../education/)**: Learning and educational applications
- **[Artificial Intelligence Domain](../artificial_intelligence/)**: AI models of cognition
- **[Knowledge Repository](../../../knowledge/)**: Theoretical foundations

---

*"Active Inference for, with, by Generative AI"* - Cognitive implementations built through collaborative intelligence and comprehensive psychological research.
