# PlatinumQuest RL Training

Reinforcement learning system for training an AI agent to play Hunt mode in PlatinumQuest.

## Quick Start

### 1. Install Python Dependencies

```bash
cd ml_agent
pip install -r requirements.txt
```

### 2. Start the Python RL Server

```bash
python train_rl.py
```

You should see:
```
RL Server listening on 127.0.0.1:8888
Waiting for game to connect...
```

### 3. Start the Game and Enable ML Agent

1. Launch PlatinumQuest
2. Open the console (press `~`)
3. Type: `MLAgent::start();`

The game will connect to the Python server and start sending observations!

### 4. Test on Your Training Map

1. Go to **Multiplayer → Host Server**
2. Select **Hunt mode → Custom → Flat Gem Training**
3. Start the game
4. The AI agent will start controlling the marble!

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   Game (C++)    │  TCP    │  Python Server   │
│                 │ <-----> │                  │
│ - Observations  │         │ - Neural Network │
│ - Actions       │         │ - Training Loop  │
└─────────────────┘         └──────────────────┘
```

**Update Loop (20 Hz):**
1. Game collects 284-dim observation
2. Sends to Python via TCP socket
3. Python runs model inference
4. Returns 6 binary actions
5. Game executes actions
6. Repeat!

## Console Commands

In-game console commands:

- `MLAgent::start()` - Connect to Python server and start AI control
- `MLAgent::stop()` - Disconnect and stop AI control
- `$MLAgent::AutoStart = true` - Auto-start when entering Hunt mode
- `AIObserver::collectState()` - Test observation collection

## Current Status

**Phase 1: Infrastructure** ✅
- [x] Socket bridge (TorqueScript ↔ Python)
- [x] Observation collection (284-dim state)
- [x] Action execution (6 binary actions)
- [x] Basic random policy

**Phase 2: Training** (Next Steps)
- [ ] Implement PPO training loop
- [ ] Add reward function
- [ ] Collect training episodes
- [ ] Train model to collect gems

**Phase 3: Evaluation**
- [ ] Test trained model
- [ ] Measure gem collection rate
- [ ] Compare to human baseline

## Next Steps

Right now the agent takes **random actions**. To start actual learning, we need to:

1. **Define rewards** - How do we score the agent's performance?
   - +10 for collecting a gem
   - +score differential for beating opponents
   - -0.01 per timestep (encourages speed)

2. **Collect episodes** - Run many games and save experiences

3. **Train with PPO** - Use the collected data to improve the policy

See `train_ppo.py` (coming soon) for the full training loop!

## Troubleshooting

**"Connection refused"**
- Make sure Python server is running first
- Check that port 8888 is not in use

**"No observations received"**
- Make sure you're in Hunt mode multiplayer
- Check that `$MP::MyMarble` exists (you must be spawned in)

**Agent doesn't move**
- Check console for errors
- Verify `MLAgent::Enabled` is true
- Try `AIAgent::test()` to test the action system directly
