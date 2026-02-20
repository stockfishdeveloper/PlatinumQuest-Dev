//------------------------------------------------------------------------------
// ML Agent Controller
// Main loop that sends observations + reward to Python and executes actions
// Protocol: sends JSON with {obs: [...], reward: float, done: bool, info: {...}}
//------------------------------------------------------------------------------

$MLAgent::Enabled = false;
$MLAgent::UpdateInterval = 16; // 60 Hz (16ms) - matches game physics tick rate
$MLAgent::AutoStart = true;  // Auto-start when Hunt mode begins
$MLAgent::TrainingSpeed = 3.0;  // Game speed multiplier (1.0 = normal, 3.0 = 3x speed, etc.)

// Reward tracking
$MLAgent::LastGemScore = 0;
$MLAgent::LastNearestGemDist = 999;
$MLAgent::EpisodeReward = 0;
$MLAgent::WasOOB = false;

function MLAgent::start() {
    if ($MLAgent::Enabled) {
        echo("MLAgent: Already running");
        return;
    }

    // Connect to Python server
    if (!AIBridge::connect("", "")) {
        error("MLAgent: Failed to connect to Python server");
        return;
    }

    // Wait a moment for connection
    schedule(500, 0, "MLAgent::startLoop");
}

function MLAgent::startLoop() {
    if (!$AIBridge::Connected) {
        error("MLAgent: Not connected to Python server");
        return;
    }

    echo("MLAgent: Starting update loop at " @ (1000 / $MLAgent::UpdateInterval) @ " Hz");
    $MLAgent::Enabled = true;
    $MLAgent::StepCount = 0;
    $MLAgent::EpisodeStartTime = getRealTime();

    // Speed up game simulation for faster training
    setTimeScale($MLAgent::TrainingSpeed);
    echo("MLAgent: Set game speed to " @ $MLAgent::TrainingSpeed @ "x for faster training");

    // Initialize reward tracking
    $MLAgent::LastGemScore = PlayGui.gemCount;
    $MLAgent::LastNearestGemDist = 999;
    $MLAgent::SkipPotentialSteps = 1;  // Suppress the sentinel spike on first step
    $MLAgent::EpisodeReward = 0;
    $MLAgent::WasOOB = false;
    $MLAgent::EpisodeShouldEnd = false;
    $MLAgent::NoGemSteps = 0;

    MLAgent::update();
}

function MLAgent::stop() {
    if (!$MLAgent::Enabled) {
        return;
    }

    echo("MLAgent: Stopping (completed " @ $MLAgent::StepCount @ " steps, total reward: " @ $MLAgent::EpisodeReward @ ")");
    $MLAgent::Enabled = false;

    // Cancel scheduled update
    if ($MLAgent::UpdateSchedule !$= "") {
        cancel($MLAgent::UpdateSchedule);
        $MLAgent::UpdateSchedule = "";
    }

    // Clear inputs
    AIAgent::clearInputs();

    // Reset game speed to normal
    setTimeScale(1.0);
    echo("MLAgent: Reset game speed to normal");

    // Disconnect from server
    AIBridge::disconnect();
}

function MLAgent::update() {
    if (!$MLAgent::Enabled) {
        return;
    }

    // Enforce time scale every update (game code may reset it)
    if (getTimeScale() != $MLAgent::TrainingSpeed) {
        setTimeScale($MLAgent::TrainingSpeed);
    }

    // Check if we're in a valid game state
    if (!isObject($MP::MyMarble) || !$Game::Running) {
        // Not in game, try again later
        $MLAgent::UpdateSchedule = schedule($MLAgent::UpdateInterval, 0, "MLAgent::update");
        return;
    }

    // 1. Collect observation
    %obs = AIObserver::collectState();

    // If this is the OOB penalty step, override the position with the saved
    // edge position so the network associates the -100 penalty with the edge,
    // not the spawn point it just respawned to.
    if ($MLAgent::WasOOB && $MLAgent::OOBPosX !$= "") {
        echo("MLAgent: [DIAG-OOB] Credit assignment FIRING at step=" @ $MLAgent::StepCount
            @ " | edge_pos=" @ $MLAgent::OOBPosX SPC $MLAgent::OOBPosY SPC $MLAgent::OOBPosZ
            @ " | spawn_pos=" @ %obs.selfPosX SPC %obs.selfPosY SPC %obs.selfPosZ);
        %obs.selfPosX = $MLAgent::OOBPosX;
        %obs.selfPosY = $MLAgent::OOBPosY;
        %obs.selfPosZ = $MLAgent::OOBPosZ;
        $MLAgent::OOBPosX = "";
    } else if ($MLAgent::WasOOB && $MLAgent::OOBPosX $= "") {
        echo("MLAgent: [DIAG-OOB] WARNING: WasOOB=true but OOBPosX is EMPTY — credit assignment MISSED!");
    }

    // 2. Compute reward for this step
    %reward = MLAgent::computeReward(%obs);
    $MLAgent::EpisodeReward += %reward;

    // 3. Check if episode is done
    %done = MLAgent::checkDone();

    // 4. Build message: obs_json|reward|done|gemDelta
    %json = AIObserver::serializeToJSON(%obs);
    %msg = %json @ "|" @ %reward @ "|" @ %done @ "|" @ $MLAgent::LastGemDelta;

    // 5. Send to Python server and get action
    AIBridge::sendState(%msg);
    %actionStr = $AIBridge::LastAction;

    // 6. Parse and execute action
    if (%actionStr !$= "") {
        MLAgent::executeAction(%actionStr);
    }

    // 7. Clean up observation object
    %obs.delete();

    // Increment step counter
    $MLAgent::StepCount++;

    // Log every 200 steps
    if ($MLAgent::StepCount % 200 == 0) {
        echo("MLAgent: Step " @ $MLAgent::StepCount @ " | Episode Reward: " @ $MLAgent::EpisodeReward @ " | Gems: " @ PlayGui.gemCount);
    }

    // 8. If done, send done signal and wait for next episode
    if (%done) {
        echo("MLAgent: Episode done! Steps: " @ $MLAgent::StepCount @ " | Total Reward: " @ $MLAgent::EpisodeReward);
        // Reset for next episode (game will restart automatically in Hunt mode)
        MLAgent::resetEpisode();
    }

    // 9. Schedule next update
    $MLAgent::UpdateSchedule = schedule($MLAgent::UpdateInterval, 0, "MLAgent::update");
}

//------------------------------------------------------------------------------
// Reward Computation
//------------------------------------------------------------------------------

function MLAgent::computeReward(%obs) {
    %reward = 0;

    // 1. Gem collection reward: +200 per point scored
    //    After 0.1 reward_scale: 1pt gem = +20.0, 5pt gem = +100.0 in buffer.
    //    OOB is -10 raw (-1.0 scaled), so 1pt gem = 20 OOBs — gems are very worth pursuing.
    //    Max episode spike: ~7 gems = +1400 raw (+140 scaled) — strong but manageable for critic.
    //    History: +100 too weak, +500 caused VLoss blow-up, +200 with 0.1 scale is the sweet spot.
    %currentGemScore = PlayGui.gemCount;
    %gemDelta = %currentGemScore - $MLAgent::LastGemScore;
    $MLAgent::LastGemDelta = %gemDelta;  // Expose for protocol message
    if (%gemDelta > 0) {
        %reward += %gemDelta * 200;
        echo("MLAgent: Gem collected! +" @ (%gemDelta * 200) @ " reward");
        // Suppress potential-shaping for 20 steps (~1 sec) after gem collection.
        // Without this grace period, the nearest gem jumps from ~0 to far away,
        // and every step produces NEGATIVE shaping as the marble drifts without
        // direction. This teaches the agent "collect gem → everything is punishment"
        // which incentivizes going OOB to reset position instead of seeking the next gem.
        $MLAgent::SkipPotentialSteps = 20;
    }
    $MLAgent::LastGemScore = %currentGemScore;

    // 2. Distance-based potential shaping: smooth reward gradient toward gem
    // Formula: reward = P(new_dist) - P(old_dist)
    // Potential function: P(d) = 20 / (1 + d/50) (Reduced from 500)
    //   - At typical distances (5-30 units), moving 0.3 units/tick → shaping ~0.02-0.12/step.
    //   - After 0.1 reward_scale: 0.002-0.012 in buffer — barely audible whisper.
    //   - History: scale=500 drowned out the sparse gem reward.
    //     scale=20 provides a tiny directional hint but forces agent to rely on +200 gem signal.
    //   - The 20-step grace period after gem collection prevents sign-flip thrashing.
    %nearestDist = %obs.gem[0, "distance"];
    if (%nearestDist > 0 && %nearestDist < 900) { // Not a sentinel value
        if ($MLAgent::SkipPotentialSteps > 0) {
            $MLAgent::LastNearestGemDist = %nearestDist;
            $MLAgent::SkipPotentialSteps--;
            // [DIAG] Log skip countdown (first and last of each grace period)
            if ($MLAgent::SkipPotentialSteps == 19 || $MLAgent::SkipPotentialSteps == 0)
                echo("MLAgent: [DIAG-SHAPE] SKIP potential step=" @ $MLAgent::StepCount @ " remaining=" @ $MLAgent::SkipPotentialSteps @ " dist=" @ %nearestDist);
        } else {
            %currentPotential = 20 / (1 + %nearestDist / 50);
            %lastPotential = 20 / (1 + $MLAgent::LastNearestGemDist / 50);
            %shapingReward = %currentPotential - %lastPotential;
            %reward += %shapingReward;
            // [DIAG] Log shaping reward every 200 steps and whenever it's large
            if ($MLAgent::StepCount % 200 == 0 || %shapingReward > 1.0 || %shapingReward < -1.0)
                echo("MLAgent: [DIAG-SHAPE] step=" @ $MLAgent::StepCount @ " dist=" @ mFloor(%nearestDist) @ " lastDist=" @ mFloor($MLAgent::LastNearestGemDist) @ " shaping=" @ %shapingReward);
            $MLAgent::LastNearestGemDist = %nearestDist;
        }
    } else {
        // [DIAG] Log when gem distance is invalid/sentinel — means no gem is on the map
        if ($MLAgent::StepCount % 200 == 0)
            echo("MLAgent: [DIAG-SHAPE] NO VALID GEM dist=" @ %nearestDist @ " step=" @ $MLAgent::StepCount);
        // Track consecutive steps with no gem available
        $MLAgent::NoGemSteps++;
        if ($MLAgent::NoGemSteps == 1)
            echo("MLAgent: [NO-GEM-START] No gem on map at step=" @ $MLAgent::StepCount);
    }

    // Detect gem reappearance after a gap
    if (%nearestDist > 0 && %nearestDist < 900 && $MLAgent::NoGemSteps > 0) {
        echo("MLAgent: [NO-GEM-END] Gem returned after " @ $MLAgent::NoGemSteps @ " steps (step=" @ $MLAgent::StepCount @ ")");
        $MLAgent::NoGemSteps = 0;
    }

    // 3. Time penalty: REMOVED.
    //    Was -0.02/step = -130/episode, drowning out the shaping signal (~0.05/step).
    //    The agent couldn't distinguish "moved toward gem" from "moved away" because
    //    both felt like punishment. Let shaping be the dominant per-step signal.

    // 4. OOB penalty: -25 for going out of bounds.
    //    At -100, five OOBs = -500 which overwhelmed everything including gem rewards.
    //    At -10 the agent never learned to avoid edges (every episode had OOBs).
    //    At -25, one OOB = 1/8th of a 1-pt gem — noticeable but not catastrophic.
    if ($MLAgent::WasOOB) {
        %reward -= 25;
        $MLAgent::WasOOB = false;
        echo("MLAgent: [REWARD] OOB penalty applied! -25 (step " @ $MLAgent::StepCount @ ", total episode reward now: " @ ($MLAgent::EpisodeReward + %reward) @ ")");
    }

    return %reward;
}

//------------------------------------------------------------------------------
// Episode Done Check
//------------------------------------------------------------------------------

function MLAgent::checkDone() {
    // Episode ends when:

    // 1. Time runs out (Hunt mode: currentTime counts UP from 0)
    //    Minimum 100 steps (~1.6s real time) prevents micro-episodes at round
    //    boundaries where the timer is expired during scoreboard/restart.
    //    These "NEUTRAL rwd=-2.2" episodes waste ~50% of training signal.
    if (isObject(MissionInfo) && MissionInfo.time > 0) {
        if (PlayGui.currentTime >= MissionInfo.time && $MLAgent::StepCount > 100) {
            return 1;
        }
    }

    // 2. Hard step-count cap — backup if the time check fails.
    //    5-min round at 3x speed, 16ms update interval = ~6,250 steps.
    //    7,000 gives a small buffer above that.
    if ($MLAgent::StepCount >= 7000) {
        echo("MLAgent: Hit max step cap (7000), forcing episode end");
        return 1;
    }

    // 3. All gems collected (rare but possible)
    if (PlayGui.gemCount >= PlayGui.maxGems && PlayGui.maxGems > 0) {
        return 1;
    }

    return 0;
}

//------------------------------------------------------------------------------
// Episode Reset
//------------------------------------------------------------------------------

function MLAgent::resetEpisode() {
    echo("MLAgent: Resetting episode");
    $MLAgent::StepCount = 0;
    $MLAgent::EpisodeStartTime = getRealTime();
    // Sync to current score, not 0 — within a Hunt round the score accumulates,
    // so resetting to 0 would cause a false gem-collection reward on the next step
    // equal to however many gems were already collected this round.
    $MLAgent::LastGemScore = PlayGui.gemCount;
    $MLAgent::LastGemDelta = 0;
    $MLAgent::LastNearestGemDist = 999;
    $MLAgent::SkipPotentialSteps = 1;  // Suppress sentinel spike on first step
    $MLAgent::EpisodeReward = 0;
    $MLAgent::WasOOB = false;
    $MLAgent::NoGemSteps = 0;
}

//------------------------------------------------------------------------------
// Action Execution
//------------------------------------------------------------------------------

function MLAgent::executeAction(%actionStr) {
    // Parse comma-separated action: "0,1,0,1" (forward,backward,left,right)
    // NOTE: Jump and Powerup actions removed — flat training map, no powerups.
    // To restore jump: add %jump as 5th element, change action_dim to 5 in train_ppo.py
    // To restore powerup: add %powerup as 6th element, change action_dim to 6 in train_ppo.py
    %forward = getWord(strreplace(%actionStr, ",", " "), 0);
    %backward = getWord(strreplace(%actionStr, ",", " "), 1);
    %left = getWord(strreplace(%actionStr, ",", " "), 2);
    %right = getWord(strreplace(%actionStr, ",", " "), 3);
    // %jump = getWord(strreplace(%actionStr, ",", " "), 4);     // DISABLED: flat map, no jumping needed
    // %powerup = getWord(strreplace(%actionStr, ",", " "), 5);  // DISABLED: no powerups on map

    // Execute via existing AI agent system (jump and powerup always 0)
    AIAgent::setBinaryActions(%forward, %backward, %left, %right, 0, 0);
}

//------------------------------------------------------------------------------
// OOB Hook - called when marble goes out of bounds
//------------------------------------------------------------------------------

function MLAgent::onOOB() {
    if ($MLAgent::Enabled) {
        echo("MLAgent: [OOB EVENT] Marble went out of bounds at step " @ $MLAgent::StepCount);

        // Save the edge position NOW, before respawn moves the marble.
        // collectSelfState() will inject this position when WasOOB is true,
        // so the -10 penalty is associated with the edge — not the spawn point.
        if (isObject($MP::MyMarble)) {
            %pos = $MP::MyMarble.getPosition();
            $MLAgent::OOBPosX = getWord(%pos, 0);
            $MLAgent::OOBPosY = getWord(%pos, 1);
            $MLAgent::OOBPosZ = getWord(%pos, 2);
            echo("MLAgent: [DIAG-OOB] Saved edge position=" @ %pos @ " at step=" @ $MLAgent::StepCount);
        } else {
            echo("MLAgent: [DIAG-OOB] WARNING: No marble in onOOB — cannot save edge position!");
        }

        $MLAgent::WasOOB = true;
        $MLAgent::LastNearestGemDist = 999;
        $MLAgent::SkipPotentialSteps = 1;  // Just suppress the sentinel spike

        // Delay respawn by 2 update intervals so the next update() fires
        // while the marble is still at the edge position.
        schedule($MLAgent::UpdateInterval * 2, 0, "MLAgent::triggerQuickRespawn");
    }
}

function MLAgent::triggerQuickRespawn() {
    // Send quick respawn command to server (same as left-click after OOB)
    // Called immediately when OOB message appears
    if ($MLAgent::Enabled && isObject($MP::MyMarble)) {
        commandToServer('QuickRespawn');

        // Restore time scale after respawn (respawn resets it)
        // Small delay to let respawn complete
        schedule(50, 0, "MLAgent::restoreTimeScale");
    }
}

function MLAgent::restoreTimeScale() {
    if ($MLAgent::Enabled) {
        setTimeScale($MLAgent::TrainingSpeed);
        echo("MLAgent: Restored game speed to " @ $MLAgent::TrainingSpeed @ "x after respawn");
    }
}

//------------------------------------------------------------------------------
// Game Lifecycle Hooks
//------------------------------------------------------------------------------

function MLAgent::onGameStart() {
    // Called when entering a Hunt mode game
    if (!$MLAgent::AutoStart || !mp() || !$Game::isMode["hunt"])
        return;

    // Enable quick respawn for training (faster OOB recovery)
    $MPPref::AllowQuickRespawn = true;
    $MP::AllowQuickRespawn = true;
    $MPPref::Server::CompetitiveMode = false;
    echo("MLAgent: Enabled quick respawn for training");

    // Optimize graphics for training speed
    $pref::Video::disableVerticalSync = true;  // Unlock framerate
    echo("MLAgent: Disabled VSync for faster training");

    echo("MLAgent: Game started, waiting for GO! signal...");
    $MLAgent::ReadyToStart = true;
}

function MLAgent::onTimerStart() {
    // Called when "GO!" appears and timer starts
    if (!$MLAgent::AutoStart || !$MLAgent::ReadyToStart)
        return;

    $MLAgent::ReadyToStart = false;
    echo("MLAgent: GO! Starting ML agent...");

    schedule(100, 0, "MLAgent::start");
}

function MLAgent::onGameEnd() {
    // Send final done signal before stopping
    if ($MLAgent::Enabled && $AIBridge::Connected) {
        %msg = "[]|0|1";  // Empty obs, 0 reward, done=1
        AIBridge::sendState(%msg);
    }

    // Don't fully stop - just flag ready to restart
    if ($MLAgent::Enabled) {
        echo("MLAgent: Round ended, auto-restarting in 2 seconds...");
        $MLAgent::Enabled = false;  // Temporarily disable updates

        // Auto-restart after brief delay
        schedule(2000, 0, "MLAgent::autoRestart");
    }
}

function MLAgent::autoRestart() {
    // Auto-restart the level for continuous training
    if (mp() && $Game::isMode["hunt"]) {
        echo("MLAgent: Restarting Hunt round...");

        // Close end game dialog if open
        if (isObject(MPEndGameDlg) && MPEndGameDlg.isAwake()) {
            Canvas.popDialog(MPEndGameDlg);
        }

        // Restart the mission
        commandToServer('restartLevel');

        // Re-enable ML agent
        schedule(1000, 0, "MLAgent::start");
    }
}

// Hook into game start
function clientCmdGameStart() {
    Parent::clientCmdGameStart();
    MLAgent::onGameStart();
}

// Hook into setMessage to detect OOB (more reliable than callback system)
function clientCmdSetMessage(%message, %timeout) {
    if (%message $= "outOfBounds") {
        MLAgent::onOOB();
    }

    // Call original function
    PlayGui.setMessage(%message, %timeout);
}

// Hook into game end
function clientCmdGameEnd() {
    Parent::clientCmdGameEnd();
    MLAgent::onGameEnd();
}

// Test function to manually trigger OOB (for debugging)
function testOOB() {
    echo("=== MANUAL OOB TEST ===");
    echo("Calling MLAgent::onOOB() directly...");
    MLAgent::onOOB();
    echo("Test complete.");
}
