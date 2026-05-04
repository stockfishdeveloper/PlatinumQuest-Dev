//------------------------------------------------------------------------------
// ML Agent Controller
// Main loop that sends observations to Python and executes actions.
// Reward computation happens in Python (train_ppo.py) for easy tuning.
// Protocol: obs_json|gem_delta|oob|done
//------------------------------------------------------------------------------

$MLAgent::Enabled = false;
$MLAgent::UpdateInterval = 16; // 60 Hz (16ms) - matches game physics tick rate
$MLAgent::AutoStart = true;  // Auto-start when Hunt mode begins
$MLAgent::TrainingSpeed = 3.0;  // Game speed multiplier (1.0 = normal, 3.0 = 3x speed, etc.)
$MLAgent::DiagnosticMode = false; // When true: send obs but don't execute actions or change speed

// State tracking (reward computation is in Python)
$MLAgent::LastGemScore = 0;
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

    // Initialize state
    $MLAgent::LastGemScore = PlayGui.gemCount;
    $MLAgent::GameStartGemScore = PlayGui.gemCount;  // Baseline for full-game gem count
    $MLAgent::WasOOB = false;
    $MLAgent::EpisodeShouldEnd = false;
    $MLAgent::TimerStarted = false;

    MLAgent::update();
}

function MLAgent::stop() {
    if (!$MLAgent::Enabled) {
        return;
    }

    echo("MLAgent: Stopping (completed " @ $MLAgent::StepCount @ " steps)");
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

    // If this is the OOB step, override the position with the saved
    // edge position so the network associates the penalty with the edge,
    // not the spawn point it just respawned to.
    if ($MLAgent::WasOOB && $MLAgent::OOBPosX !$= "") {
        %obs.selfPosX = $MLAgent::OOBPosX;
        %obs.selfPosY = $MLAgent::OOBPosY;
        %obs.selfPosZ = $MLAgent::OOBPosZ;
        $MLAgent::OOBPosX = "";
    }

    // 2. Capture OOB flag and clear it
    %oobFlag = $MLAgent::WasOOB ? 1 : 0;
    $MLAgent::WasOOB = false;

    // 3. Compute gem delta (Python computes the actual reward)
    %currentGemScore = PlayGui.gemCount;
    %gemDelta = %currentGemScore - $MLAgent::LastGemScore;
    $MLAgent::LastGemScore = %currentGemScore;

    // 4. Check if episode is done
    %done = MLAgent::checkDone();

    // 5. Build message: obs_json|gemDelta|oob|done
    %json = AIObserver::serializeToJSON(%obs);
    %msg = %json @ "|" @ %gemDelta @ "|" @ %oobFlag @ "|" @ %done;

    // 6. Send to Python server and get action
    AIBridge::sendState(%msg);
    %actionStr = $AIBridge::LastAction;

    // 7. Parse and execute action (skip in diagnostic mode — player controls marble)
    if (%actionStr !$= "" && !$MLAgent::DiagnosticMode) {
        MLAgent::executeAction(%actionStr);
    }

    // 8. Clean up observation object
    %obs.delete();

    // Increment step counter
    $MLAgent::StepCount++;

    // 9. If done, reset for next episode (game will restart automatically in Hunt mode)
    if (%done) {
        MLAgent::resetEpisode();
    }

    // 10. Schedule next update
    $MLAgent::UpdateSchedule = schedule($MLAgent::UpdateInterval, 0, "MLAgent::update");
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
    //    Actual 5-min round at 3x takes ~11,873 steps (not 6,250 as originally estimated).
    //    15,000 gives a comfortable buffer above that.
    if ($MLAgent::StepCount >= 15000) {
        echo("MLAgent: Hit max step cap (15000), forcing episode end");
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
    $MLAgent::WasOOB = false;
    $MLAgent::TimerStarted = false;
}

//------------------------------------------------------------------------------
// Action Execution
//------------------------------------------------------------------------------

function MLAgent::executeAction(%actionStr) {
    // Parse comma-separated action: "fwd,back,left,right,jump,camYaw"
    // Movement: continuous floats [0.0, 1.0]. Jump: 0 or 1. CamYaw: radians (optional).
    %words = strreplace(%actionStr, ",", " ");
    %forward = getWord(%words, 0);
    %backward = getWord(%words, 1);
    %left = getWord(%words, 2);
    %right = getWord(%words, 3);
    %jump = getWord(%words, 4);
    %camYaw = getWord(%words, 5);

    // If a camera yaw was provided, set it (must use setMarbleCamYaw to keep
    // both $cameraYaw and $MP::MyMarble camera in sync — observer reads $cameraYaw
    // while engine applies movement relative to marble's internal camera).
    // Also zero $mvYaw — the engine applies this as a per-tick delta to the
    // marble's camera, so any residual from mouse input causes drift.
    if (%camYaw !$= "") {
        setMarbleCamYaw(%camYaw + 0);
        $mvYaw = 0;
        $mvYawLeftSpeed = 0;
        $mvYawRightSpeed = 0;
    }

    // Execute via analog input (accepts float values 0.0-1.0)
    // Jump is binary (0 or 1) via $mvTriggerCount0
    AIAgent::setCustomAction(%left, %right, %forward, %backward, %jump + 0, 0);
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
    // Send game-end signal with the TOTAL gems collected this entire game.
    // Uses GameStartGemScore (set at round start, never reset by resetEpisode)
    // so the Python side gets the accurate full-game gem count.
    // The gem_delta field is repurposed here: negative = total game gems signal.
    // Protocol: []|<neg_total_game_gems>|0|1
    if ($MLAgent::Enabled && $AIBridge::Connected) {
        %totalGameGems = PlayGui.gemCount - $MLAgent::GameStartGemScore;
        // Send as negative to distinguish from normal per-step gem deltas.
        // Python checks for negative gem_delta on empty obs to record game total.
        %msg = "[]|" @ -%totalGameGems @ "|0|1";
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
