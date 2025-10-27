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
- **Information Theory**: Entropy, KL divergence, mutual information, cross-entropy
- **Bayesian Fundamentals**: Bayes' theorem, probabilistic modeling, belief updating, conjugate priors, hierarchical models, empirical Bayes
- **Free Energy Principle**: Mathematical formulation, biological applications
- **Active Inference**: Framework overview, generative models, policy selection, multi-agent systems, continuous control

## Mathematics

Rigorous mathematical treatments:
- **Variational Methods**: Variational free energy, information geometry, expected free energy
- **Computational Methods**: Predictive coding, stochastic processes, optimal transport, Markov Chain Monte Carlo
- **Advanced Topics**: Neural dynamics, filtering, sampling methods

## Implementations

Practical code examples and tutorials:
- **Core Algorithms**: Basic Active Inference, variational inference, expected free energy calculation
- **Advanced Methods**: MCMC sampling, neural network implementations
- **Development Tools**: Frameworks, optimization, performance tuning

## Applications

Real-world applications across domains:
- **Engineering**: Robotics, control systems, autonomous technologies
- **Scientific**: Neuroscience, perceptual processing, cognitive science
- **Decision Making**: Human and artificial decision processes, game theory
- **Domain-Specific**:
  - **Artificial Intelligence**: Alignment, safety, machine learning applications
  - **Education**: Adaptive learning systems, personalized instruction
  - **Engineering**: Control systems, robust design, safety-critical systems
  - **Psychology**: Cognitive modeling, behavioral science, mental health
  - **Economics**: Market behavior, strategic interaction, decision theory
  - **Climate Science**: Environmental modeling, uncertainty quantification, policy

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
