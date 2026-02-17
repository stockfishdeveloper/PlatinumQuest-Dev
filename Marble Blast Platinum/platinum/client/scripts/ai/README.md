# AI Agent System

This directory contains the autonomous marble control system for AI development.

## Files

- **agent.cs** - Main AI agent implementation with movement control

## Architecture

The AI agent is designed to be modular and extensible:

1. **Action System** - Actions are defined in an ArrayObject with configurable inputs
2. **Update Loop** - Runs at 50ms intervals for smooth control
3. **Custom Actions** - Can be overridden for ML model integration

## Usage

### Basic Control

```torquescript
// Start the agent
AIAgent::start();

// Stop the agent
AIAgent::stop();

// Toggle auto-start
$AIAgent::AutoStart = true;  // or false
```

### Adding Custom Actions

```torquescript
// Add a new action to the action list
AIAgentActions.push_back("SPIN-LEFT" TAB "Spin left" TAB 1.0 TAB 0 TAB 1.0 TAB 0 TAB 0 TAB 0);
```

### Direct Control (for ML models)

```torquescript
// Set custom movement values
// AIAgent::setCustomAction(left, right, forward, backward, jump, usePowerup)
AIAgent::setCustomAction(0.5, 0, 1.0, 0, 0, 0);
```

## Action Format

Actions are stored as tab-separated values:
```
"NAME" TAB "Description" TAB left TAB right TAB forward TAB backward TAB jump TAB usePowerup
```

Values:
- Movement: 0.0 to 1.0 (analog control)
- Jump/Powerup: 0 or 1 (digital control)

## Current Actions

1. LEFT - Move left
2. RIGHT - Move right
3. FORWARD - Move forward
4. BACKWARD - Move backward
5. JUMP - Jump
6. FORWARD-LEFT - Diagonal movement
7. FORWARD-RIGHT - Diagonal movement
8. USE-POWERUP - Use collected powerup

## Future Development

This system is designed for integration with:
- Machine Learning models
- Reinforcement Learning agents
- Computer vision systems
- Automated testing

## Integration Points

To integrate an ML model:

1. Override `AIAgent::update()` with your model's decision function
2. Call `AIAgent::setCustomAction()` with model outputs
3. Access game state through TorqueScript globals
4. Log actions/rewards for training
