# Deep Tree Echo State Network - Quick Reference Guide

## Installation

The Deep Tree ESN framework is built as part of the Inferno OS build process.

```sh
# Edit mkconfig to set your ROOT and SYSHOST
vi mkconfig

# Build the system (this builds the limbo compiler and libraries)
./makemk.sh
export PATH=$ROOT/$OBJTYPE/bin:$PATH

# Build and install everything
mk install

# The reservoir library will be at: $ROOT/dis/lib/reservoir.dis
# The demo command will be at: $ROOT/dis/deeptreeecho.dis
```

## Quick Start

### Load the Module

```limbo
implement YourApp;

include "sys.m";
    sys: Sys;
include "draw.m";
include "reservoir.m";
    reservoir: Reservoir;
    DeepTreeEchoSelf, PersonaTrait: import reservoir;

init(nil: ref Draw->Context, args: list of string)
{
    sys = load Sys Sys->PATH;
    reservoir = load Reservoir Reservoir->PATH;
    reservoir->init();
    
    # Your code here
}
```

### Create a Simple ESN

```limbo
# Define architecture
depth := 2;
layerSizes := array[depth] of int;
layerSizes[0] = 50;
layerSizes[1] = 25;

# Initialize model
dtes := ref DeepTreeEchoSelf;
dtes.init(depth, layerSizes);

# Process input
input := array[10] of real;
for (i := 0; i < len input; i++)
    input[i] = real i / 10.0;

output := dtes.process(input);
```

### Train with Data

```limbo
# Prepare training data
n_samples := 100;
inputs := array[n_samples] of array of real;
targets := array[n_samples] of array of real;

# Fill with your data
for (i := 0; i < n_samples; i++) {
    inputs[i] = array[10] of real;  # Your input
    targets[i] = array[5] of real;  # Your target
    # ... fill arrays ...
}

# Train
dtes.train(inputs, targets);

# Predict
prediction := dtes.generate_response(test_input);
```

### Set Personality

```limbo
persona := ref PersonaTrait;
persona.openness = 0.8;          # Creative
persona.conscientiousness = 0.9;  # Organized
persona.extraversion = 0.6;       # Moderately social
persona.agreeableness = 0.7;      # Cooperative
persona.neuroticism = 0.3;        # Emotionally stable

dtes.persona = persona;
```

### Monitor Emotions

```limbo
emotion := dtes.affectiveAgent.emotionState;
sys->print("Joy: %.3f\n", emotion.joy);
sys->print("Sadness: %.3f\n", emotion.sadness);
sys->print("Valence: %.3f\n", emotion.valence());
sys->print("Intensity: %.3f\n", emotion.intensity());
```

## Command-Line Tool

### Basic Demo

```sh
deeptreeecho -demo
```

Shows:
- System initialization
- Stimulus processing
- Emotional state
- Ricci curvature
- Persona traits

### Training Demo

```sh
deeptreeecho -train
```

Demonstrates:
- Training data generation
- Reservoir training
- Prediction testing

### Persona Comparison

```sh
deeptreeecho -persona
```

Compares responses from:
- Creative artist persona
- Analytical scientist persona  
- Empathetic counselor persona

### Emotion Processing

```sh
deeptreeecho -emotion
```

Shows emotional responses to:
- Positive stimuli
- Negative stimuli
- Neutral stimuli

## API Reference

### Core Types

```limbo
# Basic reservoir node
Node: adt {
    init: fn(n: self ref Node, id: int, size: int, activation: string);
    update: fn(n: self ref Node, input: array of real): array of real;
};

# Reservoir layer
Layer: adt {
    init: fn(l: self ref Layer, size: int, connectivity: real);
    activate: fn(l: self ref Layer, input: array of real): array of real;
};

# Deep hierarchical ESN
DeepTreeESN: adt {
    init: fn(esn: self ref DeepTreeESN, depth: int, layerSizes: array of int);
    forward: fn(esn: self ref DeepTreeESN, input: array of real): array of real;
    train: fn(esn: self ref DeepTreeESN, inputs: array of array of real, 
              targets: array of array of real);
    predict: fn(esn: self ref DeepTreeESN, input: array of real): array of real;
};

# Integrated model
DeepTreeEchoSelf: adt {
    init: fn(dtes: self ref DeepTreeEchoSelf, depth: int, layerSizes: array of int);
    process: fn(dtes: self ref DeepTreeEchoSelf, input: array of real): array of real;
    evolve: fn(dtes: self ref DeepTreeEchoSelf, timesteps: int);
    train: fn(dtes: self ref DeepTreeEchoSelf, inputs: array of array of real, 
              targets: array of array of real);
    generate_response: fn(dtes: self ref DeepTreeEchoSelf, prompt: array of real): 
                         array of real;
};
```

### Emotion Types

```limbo
EmotionState: adt {
    joy, sadness, anger, fear, disgust: real;
    surprise, interest, contempt, shame, guilt: real;
    
    init: fn(e: self ref EmotionState);
    normalize: fn(e: self ref EmotionState);
    intensity: fn(e: self ref EmotionState): real;
    valence: fn(e: self ref EmotionState): real;
};

AffectiveAgent: adt {
    init: fn(aa: self ref AffectiveAgent, reservoirDepth: int);
    process_stimulus: fn(aa: self ref AffectiveAgent, stimulus: array of real): 
                         ref EmotionState;
    modulate_response: fn(aa: self ref AffectiveAgent, response: array of real): 
                          array of real;
};
```

### Personality Types

```limbo
PersonaTrait: adt {
    openness: real;           # 0.0 - 1.0
    conscientiousness: real;  # 0.0 - 1.0
    extraversion: real;       # 0.0 - 1.0
    agreeableness: real;      # 0.0 - 1.0
    neuroticism: real;        # 0.0 - 1.0
    
    init: fn(pt: self ref PersonaTrait);
    to_reservoir_params: fn(pt: self ref PersonaTrait): array of real;
};
```

### Utility Functions

```limbo
# Activation functions
tanh: fn(x: real): real;
sigmoid: fn(x: real): real;
relu: fn(x: real): real;

# Linear algebra
matmul: fn(A: array of array of real, B: array of real): array of real;
vecadd: fn(a: array of real, b: array of real): array of real;
dotproduct: fn(a: array of real, b: array of real): real;
normalize: fn(v: array of real): array of real;
```

## Configuration Parameters

### ESN Hyperparameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| depth | 2-5 | Number of hierarchical layers |
| layerSize | 20-200 | Nodes per layer |
| spectralRadius | 0.8-0.99 | Stability parameter |
| leakingRate | 0.1-0.5 | Memory decay rate |
| connectivity | 0.01-0.2 | Connection density |

### Training Parameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| regularization | 0.0001-0.01 | Ridge regression λ |
| trainSamples | 100-10000 | Training set size |
| washout | 50-500 | Initial transient steps |

### Affective Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| emotionThreshold | 0.1-0.3 | Activation threshold |
| valenceBias | -0.2 to 0.2 | Positive/negative bias |
| modulationStrength | 0.1-0.5 | Response modulation |

## Troubleshooting

### Common Issues

**Issue**: Unstable reservoir dynamics
- **Solution**: Reduce spectral radius below 0.95
- **Check**: Ensure leaking rate > 0

**Issue**: Poor training performance
- **Solution**: Increase reservoir size or depth
- **Check**: Verify sufficient training samples
- **Adjust**: Increase regularization parameter

**Issue**: Emotional states saturate
- **Solution**: Normalize inputs to [-1, 1] range
- **Check**: Verify emotion computation in affective agent
- **Adjust**: Scale emotion thresholds

**Issue**: Memory usage too high
- **Solution**: Reduce layer sizes
- **Alternative**: Use sparse connectivity
- **Optimize**: Implement batch processing

### Debugging Tips

```limbo
# Check reservoir state
sys->print("Reservoir state norm: %.3f\n", 
    reservoir->dotproduct(state, state));

# Monitor training
sys->print("Training error: %.6f\n", training_error);

# Verify emotion distribution
emotion.normalize();
sys->print("Emotion sum: %.3f\n", emotion.intensity());

# Check persona influence
params := persona.to_reservoir_params();
sys->print("Persona params: [%.2f, %.2f, %.2f, %.2f, %.2f]\n",
    params[0], params[1], params[2], params[3], params[4]);
```

## Examples

### Time Series Prediction

```limbo
# Sine wave prediction
n := 1000;
inputs := array[n] of array of real;
targets := array[n] of array of real;

for (i := 0; i < n; i++) {
    t := real i / 10.0;
    inputs[i] = array[1] of real;
    inputs[i][0] = Math->sin(t);
    targets[i] = array[1] of real;
    targets[i][0] = Math->sin(t + 0.1);
}

dtes.train(inputs, targets);
```

### Sentiment Analysis Simulation

```limbo
# Map sentiment to emotion
process_sentiment := fn(text_embedding: array of real): string {
    emotion := dtes.affectiveAgent.process_stimulus(text_embedding);
    valence := emotion.valence();
    
    if (valence > 0.3)
        return "Positive";
    else if (valence < -0.3)
        return "Negative";
    else
        return "Neutral";
};
```

### Adaptive Behavior

```limbo
# Adjust behavior based on personality
if (persona.openness > 0.7) {
    # Explore new patterns
    dtes.esn.spectralRadius = 0.95;
} else if (persona.conscientiousness > 0.7) {
    # Maintain stability
    dtes.esn.spectralRadius = 0.85;
}
```

## Best Practices

### Architecture Design
1. Start with 2-3 layers
2. Use decreasing layer sizes (e.g., 100→50→25)
3. Keep connectivity low (5-10%)
4. Set spectral radius to 0.9-0.95

### Training
1. Normalize all inputs to [-1, 1]
2. Use washout period (skip first 50-100 steps)
3. Start with high regularization, decrease if underfitting
4. Monitor training and validation error

### Affective Computing
1. Initialize emotions to neutral state
2. Update emotions after each input
3. Apply modulation conservatively (β < 0.3)
4. Normalize emotion vectors regularly

### Personality Integration
1. Design persona profiles intentionally
2. Test extreme trait values (0.0 and 1.0)
3. Map traits to hyperparameters consistently
4. Document persona-behavior relationships

## Performance Tips

1. **Pre-allocate arrays**: Avoid dynamic allocation in loops
2. **Batch processing**: Process multiple inputs together
3. **Sparse operations**: Use sparse matrix libraries if available
4. **Cache results**: Store frequently used computations
5. **Profile code**: Identify bottlenecks with timing

## Further Reading

- See `doc/deeptreeecho-technical.md` for detailed theory
- See `appl/cmd/deeptreeecho.README` for overview
- Check module definition in `module/reservoir.m`
- Review implementation in `appl/lib/reservoir.b`

## Support & Community

This is a research implementation for Inferno OS. For issues or questions:
- Review the source code documentation
- Check the technical architecture document
- Experiment with the demo modes
- Modify and extend for your use case

## License

See repository NOTICE and LICENSE files for licensing terms.
