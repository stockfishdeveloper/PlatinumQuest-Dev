//------------------------------------------------------------------------------
// ML Agent Controller
// Main loop that sends observations + reward to Python and executes actions
// Protocol: sends JSON with {obs: [...], reward: float, done: bool, info: {...}}
//------------------------------------------------------------------------------

$MLAgent::Enabled = false;
$MLAgent::UpdateInterval = 50; // 20 Hz (50ms)
$MLAgent::AutoStart = true;  // Auto-start when Hunt mode begins

// Reward tracking
$MLAgent::LastGemScore = 0;
$MLAgent::LastNearestGemDist = 999;
$MLAgent::EpisodeReward = 0;
$MLAgent::WasOOB = false;
$MLAgent::EpisodeShouldEnd = false;  // Persistent flag for early termination

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

    // Initialize reward tracking
    $MLAgent::LastGemScore = PlayGui.gemCount;
    $MLAgent::LastNearestGemDist = 999;
    $MLAgent::EpisodeReward = 0;
    $MLAgent::WasOOB = false;
    $MLAgent::EpisodeShouldEnd = false;

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

    // Disconnect from server
    AIBridge::disconnect();
}

function MLAgent::update() {
    if (!$MLAgent::Enabled) {
        return;
    }

    // Check if we're in a valid game state
    if (!isObject($MP::MyMarble) || !$Game::Running) {
        // Not in game, try again later
        $MLAgent::UpdateSchedule = schedule($MLAgent::UpdateInterval, 0, "MLAgent::update");
        return;
    }

    // 1. Collect observation
    %obs = AIObserver::collectState();

    // 2. Compute reward for this step
    %reward = MLAgent::computeReward(%obs);
    $MLAgent::EpisodeReward += %reward;

    // 3. Check if episode is done
    %done = MLAgent::checkDone();

    // 4. Build message: obs_json|reward|done
    %json = AIObserver::serializeToJSON(%obs);
    %msg = %json @ "|" @ %reward @ "|" @ %done;

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

    // 1. Gem collection reward: +100 per point scored (DOMINANT SIGNAL)
    %currentGemScore = PlayGui.gemCount;
    %gemDelta = %currentGemScore - $MLAgent::LastGemScore;
    if (%gemDelta > 0) {
        %reward += %gemDelta * 100;
        echo("MLAgent: Gem collected! +" @ (%gemDelta * 100) @ " reward");
    }
    $MLAgent::LastGemScore = %currentGemScore;

    // DEBUG: Log if WasOOB flag is set
    if ($MLAgent::WasOOB) {
        echo("MLAgent: [DEBUG] WasOOB flag is TRUE before checking in reward computation");
    }

    // 2. Distance shaping: reward for getting closer to nearest gem
    %nearestDist = %obs.gem[0, "distance"];
    if (%nearestDist > 0 && %nearestDist < 900) { // Not a sentinel value
        %distDelta = $MLAgent::LastNearestGemDist - %nearestDist;
        %reward += %distDelta * 0.1; // Reward for approaching gems (10x stronger)
        $MLAgent::LastNearestGemDist = %nearestDist;
    }

    // 3. Time penalty: -0.1 per step (encourages speed)
    %reward -= 0.1;

    // 4. OOB penalty: -50 for going out of bounds (SEVERE PUNISHMENT)
    if ($MLAgent::WasOOB) {
        %reward -= 50;
        $MLAgent::WasOOB = false;
        echo("MLAgent: [REWARD] OOB penalty applied! -50 (step " @ $MLAgent::StepCount @ ", total episode reward now: " @ ($MLAgent::EpisodeReward + %reward) @ ")");
    }

    return %reward;
}

//------------------------------------------------------------------------------
// Episode Done Check
//------------------------------------------------------------------------------

function MLAgent::checkDone() {
    // Episode ends when:

    // 1. Out of bounds (early termination to prevent wasting training time)
    if ($MLAgent::EpisodeShouldEnd) {
        echo("MLAgent: Episode ending early due to OOB");
        return 1;
    }

    // 2. Time runs out (Hunt mode: currentTime counts UP from 0)
    if (isObject(MissionInfo) && MissionInfo.time > 0) {
        if (PlayGui.currentTime >= MissionInfo.time && $MLAgent::StepCount > 20) {
            return 1;
        }
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
    $MLAgent::LastGemScore = 0;
    $MLAgent::LastNearestGemDist = 999;
    $MLAgent::EpisodeReward = 0;
    $MLAgent::WasOOB = false;
    $MLAgent::EpisodeShouldEnd = false;  // Reset early termination flag
}

//------------------------------------------------------------------------------
// Action Execution
//------------------------------------------------------------------------------

function MLAgent::executeAction(%actionStr) {
    // Parse comma-separated action: "0,1,0,1,0,0" (forward,backward,left,right,jump,powerup)
    %forward = getWord(strreplace(%actionStr, ",", " "), 0);
    %backward = getWord(strreplace(%actionStr, ",", " "), 1);
    %left = getWord(strreplace(%actionStr, ",", " "), 2);
    %right = getWord(strreplace(%actionStr, ",", " "), 3);
    %jump = getWord(strreplace(%actionStr, ",", " "), 4);
    %powerup = getWord(strreplace(%actionStr, ",", " "), 5);

    // Execute via existing AI agent system
    AIAgent::setBinaryActions(%forward, %backward, %left, %right, %jump, %powerup);
}

//------------------------------------------------------------------------------
// OOB Hook - called when marble goes out of bounds
//------------------------------------------------------------------------------

function MLAgent::onOOB() {
    if ($MLAgent::Enabled) {
        echo("MLAgent: [OOB EVENT] Marble went out of bounds at step " @ $MLAgent::StepCount);
        $MLAgent::WasOOB = true;
        $MLAgent::EpisodeShouldEnd = true;  // Mark episode for early termination
        // Reset nearest gem tracking since position changed
        $MLAgent::LastNearestGemDist = 999;

        // Schedule quick respawn after OOB message appears (500ms delay)
        // This mimics the player clicking left mouse to respawn faster
        schedule(500, 0, "MLAgent::triggerQuickRespawn");
    }
}

function MLAgent::triggerQuickRespawn() {
    // Send quick respawn command to server (same as left-click after OOB)
    // Waits 500ms after OOB to ensure "Out of Bounds" message has appeared
    // This avoids accidentally using powerups
    if ($MLAgent::Enabled && isObject($MP::MyMarble)) {
        commandToServer('QuickRespawn');
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
    // Check if this is an out of bounds message
    if (%message $= "outOfBounds") {
        echo("MLAgent: [DEBUG] OOB message detected!");
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
