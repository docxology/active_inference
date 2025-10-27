# Active Inference Knowledge Base

This directory contains the comprehensive knowledge base for Active Inference and the Free Energy Principle. The content is organized into four main sections:

## Directory Structure

```
knowledge/
├── foundations/          # Core theoretical concepts
├── mathematics/          # Mathematical formulations and derivations
├── implementations/      # Code examples and tutorials
├── applications/         # Real-world applications and use cases
└── learning_paths.json   # Structured learning paths
```

## Foundations

Core theoretical concepts including:
- **Information Theory**: Entropy, KL divergence, mutual information
- **Bayesian Fundamentals**: Bayes' theorem, probabilistic modeling, belief updating
- **Free Energy Principle**: Mathematical formulation, biological applications
- **Active Inference**: Framework overview, generative models, policy selection

## Mathematics

Rigorous mathematical treatments:
- **Variational Free Energy**: Derivation and optimization
- **Information Geometry**: Riemannian geometry of probability distributions
- **Expected Free Energy**: Policy selection and planning mathematics
- **Predictive Coding**: Hierarchical inference and neural dynamics

## Implementations

Practical code examples and tutorials:
- **Basic Active Inference**: Step-by-step agent implementation
- **Variational Inference**: Algorithms for approximate Bayesian computation
- **Expected Free Energy**: Policy evaluation and selection algorithms

## Applications

Real-world applications across domains:
- **Robotics**: Control systems and autonomous navigation
- **Neuroscience**: Brain function and perceptual processing
- **Decision Making**: Human and artificial decision processes

## Usage

Each JSON file follows a structured format with:
- **id**: Unique identifier
- **title**: Descriptive title
- **content_type**: foundation|mathematics|implementation|application
- **difficulty**: beginner|intermediate|advanced|expert
- **description**: Brief summary
- **prerequisites**: Required prior knowledge
- **content**: Detailed structured content
- **metadata**: Additional information

## Learning Paths

The `learning_paths.json` file defines structured curricula for different audiences and learning objectives. Each path includes:
- Node dependencies and prerequisites
- Estimated time requirements
- Difficulty progression
- Learning outcomes

## Contributing

When adding new content:
1. Follow the established JSON schema
2. Include comprehensive explanations and examples
3. Add interactive exercises where appropriate
4. Update relevant learning paths
5. Include references and further reading

## File Naming Convention

Files use descriptive names with underscores:
- `info_theory_entropy.json`
- `variational_free_energy.json`
- `active_inference_basic.json`
- `robotics_control.json`

This ensures consistency and makes files easy to locate and reference.
