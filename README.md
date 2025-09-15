# Maze-Solver
Intelligent pathfinding system using RRT* algorithms enhanced with Q-learning.

An intelligent pathfinding system that combines **RRT* (Rapidly-exploring Random Tree Star)** algorithms with **Q-learning reinforcement learning** to navigate through maze environments. The agent learns optimal sampling strategies through experience, evolving from random exploration to intelligent goal-directed movement.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

## Features

- **Hybrid Algorithm**: Combines traditional RRT* pathfinding with Q-learning reinforcement learning
- **Adaptive Learning**: Agent learns from experience across multiple episodes with epsilon-decay exploration
- **Multiple Sampling Strategies**: 
  - Random sampling for exploration
  - Goal-biased sampling for directed progress
  - Exploration-focused sampling for unvisited areas
  - Path-optimised sampling near promising routes
- **Real-time Visualisation**: Live turtle graphics animation showing tree growth and pathfinding process
- **Model Persistence**: Save and load trained Q-tables for continuous learning
- **Path Optimisation**: RRT* rewiring for shorter, more efficient routes
- **Performance Tracking**: Training statistics including success rates and convergence metrics

## Usage

When you run the program, you'll be presented with three options:

```
Choose an option:
1. Train new RRT* model (100 episodes)
2. Load existing RRT* model and demonstrate
3. Train new RRT* model and demonstrate
```

### Training Mode (Option 1 or 3)
- Trains the Q-learning agent for 100 episodes
- Displays training progress with success rates and rewards
- Saves the trained model as `rrt_star_q_model.pkl`

### Demonstration Mode (Option 2)
- Loads a pre-trained model (if available)
- Visualises the pathfinding process with turtle graphics
- Shows the learned optimal path in orange

## Algorithm Overview

### RRT* (Rapidly-exploring Random Tree Star)
- **Tree Growth**: Builds a tree of connected nodes from start toward goal
- **Rewiring**: Continuously optimises paths for shorter routes
- **Collision Detection**: Ensures paths avoid obstacles

### Q-Learning Enhancement
- **State Representation**: Distance to goal, direction, and nearest node proximity
- **Action Space**: Four sampling strategies (random, goal-biased, exploration, path-optimized)
- **Reward System**: 
  - Progress toward goal: +10 × distance_improvement
  - Exploration bonus: +1 for distant nodes
  - Time penalty: -1 per iteration
  - Success reward: +10 for reaching goal

### Learning Process
1. **Exploration Phase**: High epsilon (ε=0.3) for random action selection
2. **Exploitation Phase**: Gradual epsilon decay (0.995) toward greedy policy
3. **Experience Retention**: Q-table persistence across training sessions

## Core Classes

### `QLearningAgent`
- Implements Q-learning with epsilon-greedy policy
- State feature extraction and action selection
- Q-value updates using temporal difference learning

### `Maze`
- 20×20 grid environment with walls and free spaces
- Turtle graphics visualisation
- Collision detection and coordinate transformations

### `SmartRRTStar`
- Enhanced RRT* with Q-learning integration
- Adaptive sampling based on learned policies
- Tree rewiring for path optimisation

### `RRTTrainer`
- Training orchestration and statistics tracking
- Model persistence (save/load functionality)
- Demonstration mode with visualisation

## Controls & Visualisation

- **Green**: Start position
- **Red**: Goal position
- **Black**: Walls/obstacles
- **Blue**: RRT* tree edges
- **Yellow**: Current path to goal
- **Orange**: Final learned optimal path

## Customisation

### Maze Configuration
Modify the `TILES` array to create custom maze layouts:
```python
TILES = [
    0, 0, 0, 0, 0,  # 0 = wall
    0, 1, 1, 1, 0,  # 1 = free space
    0, 1, 0, 1, 0,
    0, 1, 1, 1, 0,
    0, 0, 0, 0, 0,
]
```

### Hyperparameter Tuning
Adjust learning parameters in `QLearningAgent`:
- `learning_rate`: Q-value update rate (default: 0.1)
- `discount_factor`: Future reward importance (default: 0.95)
- `epsilon`: Exploration rate (default: 0.3)
- `epsilon_decay`: Exploration decay (default: 0.995)

### Algorithm Parameters
Modify RRT* behavior in `SmartRRTStar`:
- `search_radius`: Rewiring neighborhood size (default: 3)
- `max_iterations`: Maximum tree growth steps (default: 1000)
- `step_size`: Node expansion distance (default: 1)


## Known Issues

- Large mazes may require increased `max_iterations`
- Turtle graphics can be slow for real-time visualisation
- Model convergence depends on maze complexity

## References

- [RRT* Algorithm](https://arxiv.org/abs/1105.1186) - Karaman & Frazzoli
- [Q-Learning](https://link.springer.com/article/10.1007/BF00992698) - Watkins & Dayan
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/) - Sutton & Barto

## License

This project is licensed under the MIT License.

## Future Enhancements

- [ ] Multi-goal pathfinding
- [ ] Dynamic obstacle environments
- [ ] Deep Q-Networks (DQN) implementation
- [ ] 3D maze environments
- [ ] Comparative algorithm benchmarks
- [ ] Web-based visualization interface
