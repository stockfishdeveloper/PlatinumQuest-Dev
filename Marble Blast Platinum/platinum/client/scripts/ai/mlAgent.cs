//------------------------------------------------------------------------------
// ML Agent Controller
// Main loop that sends observations + reward to Python and executes actions
// Protocol: sends JSON with {obs: [...], reward: float, done: bool, info: {...}}
//------------------------------------------------------------------------------

$MLAgent::Enabled = false;
$MLAgent::UpdateInterval = 16; // 60 Hz (16ms) - matches game physics tick rate
$MLAgent::AutoStart = true;  // Auto-start when Hunt mode begins
$MLAgent::TrainingSpeed = 3.0;  // Game speed multiplier (1.0 = normal, 3.0 = 3x speed, etc.)
$MLAgent::DiagnosticMode = false; // When true: send obs but don't execute actions or change speed

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

    // Speed up game simulation for faster training (skip in diagnostic mode)
    if (!$MLAgent::DiagnosticMode) {
        setTimeScale($MLAgent::TrainingSpeed);
        echo("MLAgent: Set game speed to " @ $MLAgent::TrainingSpeed @ "x for faster training");
    } else {
        echo("MLAgent: DIAGNOSTIC MODE — normal speed, player controls marble");
    }

    // Initialize reward tracking
    $MLAgent::LastGemScore = PlayGui.gemCount;
    $MLAgent::LastNearestGemDist = 999;
    $MLAgent::SkipPotentialSteps = 1;  // Suppress the sentinel spike on first step
    $MLAgent::EpisodeReward = 0;
    $MLAgent::WasOOB = false;
    $MLAgent::EpisodeShouldEnd = false;
    $MLAgent::NoGemSteps = 0;
    $MLAgent::TimerStarted = false;

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
    // Skip in diagnostic mode — player controls game speed
    if (!$MLAgent::DiagnosticMode && getTimeScale() != $MLAgent::TrainingSpeed) {
        setTimeScale($MLAgent::TrainingSpeed);
    }

    // Check if we're in a valid game state
    if (!isObject($MP::MyMarble) || !$Game::Running) {
        // Not in game, try again later
        $MLAgent::UpdateSchedule = schedule($MLAgent::UpdateInterval, 0, "MLAgent::update");
        return;
    }

    // Wait for the timer to actually start before collecting observations.
    // After restartLevel, there's a brief window where $Game::Running is true
    // but the timer still shows 300,000ms (expired from last round). Steps
    // taken during this dead zone produce garbage data (marble at origin,
    // zero velocity, expired timer). Wait until currentTime < total time.
    // Only applies at episode start — once we've seen a valid timer, we let
    // the episode run to natural completion and send done=1 normally.
    if (!$MLAgent::TimerStarted && isObject(MissionInfo) && MissionInfo.time > 0) {
        if (PlayGui.currentTime >= MissionInfo.time) {
            $MLAgent::UpdateSchedule = schedule($MLAgent::UpdateInterval, 0, "MLAgent::update");
            return;
        }
        $MLAgent::TimerStarted = true;
    }

    // 1. Collect observation
    %obs = AIObserver::collectState();

    // If this is the OOB penalty step, override the position with the saved
    // edge position so the network associates the -100 penalty with the edge,
    // not the spawn point it just respawned to.
    if ($MLAgent::WasOOB && $MLAgent::OOBPosX !$= "") {
        %obs.selfPosX = $MLAgent::OOBPosX;
        %obs.selfPosY = $MLAgent::OOBPosY;
        %obs.selfPosZ = $MLAgent::OOBPosZ;
        $MLAgent::OOBPosX = "";
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

    // 6. Parse and execute action (skip in diagnostic mode — player controls marble)
    if (%actionStr !$= "" && !$MLAgent::DiagnosticMode) {
        MLAgent::executeAction(%actionStr);
    }

    // 7. Clean up observation object
    %obs.delete();

    // Increment step counter
    $MLAgent::StepCount++;

    // 8. If done, reset for next episode (game will restart automatically in Hunt mode)
    if (%done) {
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
        // Grace period: suppress shaping for 20 steps after gem so the jump to
        // next-nearest doesn't produce negative shaping that punishes collection.
        $MLAgent::SkipPotentialSteps = 20;
    }
    $MLAgent::LastGemScore = %currentGemScore;

    // 2. Distance-based potential shaping: smooth reward gradient toward gem
    // Formula: reward = P(new_dist) - P(old_dist)
    // Potential function: P(d) = 20 / (1 + d/50)
    //   - Gentle gradient that guides marble toward gems without overshoot fear.
    //   - At d=30, moving 0.3 closer → shaping ~0.05/step (gentle pull).
    //   - At d=2, moving 0.3 closer → shaping ~0.11/step (slightly stronger close).
    //   - Total approach shaping (30→0) ≈ 7.5 raw = 0.75 scaled. Gem = +20 scaled.
    //   - History: d/2 caused wide orbiting (agent feared overshoot penalty),
    //     d/5 still too steep. d/50 got marble within 1 marble width.
    //   - The real close-range fix is unit direction vectors in normalize_obs,
    //     not steeper potential.
    //   - Still potential-based (P(s')-P(s)), so cannot be "farmed" by oscillating.
    //   - The 20-step grace period after gem collection prevents sign-flip thrashing.
    %nearestDist = %obs.gem[0, "distance"];
    if (%nearestDist > 0 && %nearestDist < 900) { // Not a sentinel value
        if ($MLAgent::SkipPotentialSteps > 0) {
            $MLAgent::LastNearestGemDist = %nearestDist;
            $MLAgent::SkipPotentialSteps--;
        } else {
            %currentPotential = 20 / (1 + %nearestDist / 5);
            %lastPotential = 20 / (1 + $MLAgent::LastNearestGemDist / 5);
            %shapingReward = %currentPotential - %lastPotential;
            %reward += %shapingReward;
            $MLAgent::LastNearestGemDist = %nearestDist;
        }
    } else {
        $MLAgent::NoGemSteps++;
    }

    // Detect gem reappearance after a gap
    if (%nearestDist > 0 && %nearestDist < 900 && $MLAgent::NoGemSteps > 0) {
        $MLAgent::NoGemSteps = 0;
    }

    // OOB penalty: -25 for going out of bounds
    if ($MLAgent::WasOOB) {
        %reward -= 25;
        $MLAgent::WasOOB = false;
    }

    return %reward;
}

//------------------------------------------------------------------------------
// Episode Done Check
//------------------------------------------------------------------------------

function MLAgent::checkDone() {
    // Episode ends when:

    // 1. Time runs out (Hunt mode: currentTime counts UP from 0)
    //    The dead-zone guard in update() prevents observations before the timer
    //    starts, so we only need a small safety margin (10 steps) here.
    if (isObject(MissionInfo) && MissionInfo.time > 0) {
        if (PlayGui.currentTime >= MissionInfo.time && $MLAgent::StepCount > 10) {
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

    // 3. All gems collected — REMOVED for Hunt mode.
    //    In Hunt, PlayGui.gemCount is cumulative points scored (never resets mid-round)
    //    while PlayGui.maxGems is the number of gem slots on the map (~7).
    //    Once score >= 7, this was permanently true, creating an 11-step episode
    //    flood (StepCount > 10 guard = 11 steps, then done=1 fires every time).
    //    Timer expiry + 7000-step cap are sufficient episode boundaries.

    return 0;
}

//------------------------------------------------------------------------------
// Episode Reset
//------------------------------------------------------------------------------

function MLAgent::resetEpisode() {
    $MLAgent::StepCount = 0;
    $MLAgent::EpisodeStartTime = getRealTime();
    // Sync to current score, not 0 — within a Hunt round the score accumulates,
    // so resetting to 0 would cause a false gem-collection reward on the next step
    // equal to however many gems were already collected this round.
    $MLAgent::LastGemScore = PlayGui.gemCount;
    $MLAgent::LastGemDelta = 0;
    $MLAgent::LastNearestGemDist = 999;
    $MLAgent::SkipPotentialSteps = 20;  // Same grace period as post-gem-collection
    $MLAgent::EpisodeReward = 0;
    $MLAgent::WasOOB = false;
    $MLAgent::NoGemSteps = 0;
    $MLAgent::TimerStarted = false;
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
        // Save edge position before respawn so OOB penalty is associated
        // with the edge, not the spawn point.
        if (isObject($MP::MyMarble)) {
            %pos = $MP::MyMarble.getPosition();
            $MLAgent::OOBPosX = getWord(%pos, 0);
            $MLAgent::OOBPosY = getWord(%pos, 1);
            $MLAgent::OOBPosZ = getWord(%pos, 2);
        }

        $MLAgent::WasOOB = true;
        $MLAgent::LastNearestGemDist = 999;
        $MLAgent::SkipPotentialSteps = 1;

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
    }
}

//------------------------------------------------------------------------------
// Game Lifecycle Hooks
//------------------------------------------------------------------------------

function MLAgent::onGameStart() {
    // Called when entering a Hunt mode game
    if (!$MLAgent::AutoStart || !mp() || !$Game::isMode["hunt"])
        return;

    $MPPref::AllowQuickRespawn = true;
    $MP::AllowQuickRespawn = true;
    $MPPref::Server::CompetitiveMode = false;
    $pref::Video::disableVerticalSync = true;

    echo("MLAgent: Game started, waiting for GO!");
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

// Enable diagnostic mode: observations are sent to Python but you control the marble.
// Run this in the game console BEFORE the Hunt round starts.
// Then start python diagnostic.py and play normally.
function MLAgent::enableDiagnostic() {
    $MLAgent::DiagnosticMode = true;
    $MLAgent::AutoStart = true;
    setTimeScale(1.0);
    echo("=== DIAGNOSTIC MODE ENABLED ===");
    echo("Observations will be sent to Python server but YOU control the marble.");
    echo("Start: python diagnostic.py");
    echo("Then start a Hunt round normally.");
}
