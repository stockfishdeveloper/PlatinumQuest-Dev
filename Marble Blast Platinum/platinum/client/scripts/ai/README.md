# AI Agent System

This directory contains the autonomous marble control system for AI development.

## Files

- **agent.cs** - Main AI agent implementation with movement control
- **observer.cs** - Game state observation system (284-dimensional state vector)
- **recorder.cs** - Human gameplay data recorder for behavioral cloning
- **socketBridge.cs** - (Future) TCP communication bridge for Python ML models

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

## Data Collection System (Currently Disabled)

**IMPORTANT**: The system includes a fully functional human gameplay recorder for behavioral cloning training.

**Status**:
- ❌ Currently disabled (`$AIRecorder::AutoStart = false`)
- ✅ Fully tested and working
- ✅ Ready to enable when needed

**Capabilities**:
- Records 284-dimensional state observations at 20 Hz
- Captures all player inputs (WASD, jump, powerup)
- Auto-starts after "GO!" in Hunt mode
- Creates unique timestamped files per match (`.jsonl` format)
- Auto-flushes to disk every 1 second (no data loss)
- Saves to: `platinum/data/recordings/`

**To Enable**:
1. Set `$AIRecorder::AutoStart = true` in `recorder.cs` (line 21)
2. Play Hunt mode matches normally
3. Data automatically collected and saved

**What Gets Recorded**:
- Self state: position, velocity, camera angles, powerup status
- Gems: all visible gem positions and point values (up to 50)
- Opponents: positions and states (up to 3)
- Game state: time, score, gems remaining
- Actions: your WASD/jump/powerup inputs
- Camera: yaw/pitch control

**Data Format** (JSON Lines):
```json
{"state": [284 floats], "action": [6 binary], "camera": [2 floats]}
```

**Analysis Tool**:
- `analyze_game.py` - Analyzes recorded gameplay and generates statistics

**Use Case**: When ready to train behavioral cloning model, enable recorder and play 20-30 matches to generate training data.

## Future Development

This system is designed for integration with:
- Machine Learning models (via behavioral cloning or RL)
- Reinforcement Learning agents
- Computer vision systems
- Automated testing

## Integration Points

To integrate an ML model:

1. Override `AIAgent::update()` with your model's decision function
2. Call `AIAgent::setCustomAction()` with model outputs
3. Access game state through TorqueScript globals
4. Log actions/rewards for training
