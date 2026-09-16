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
$MLAgent::RecordMode = false;     // When true (with DiagnosticMode): append the human's inputs to each message
// Observation frame is ALWAYS the world frame (yaw 0), for training, play and
// recording alike (2026-09-14). Before this the observer rotated everything by
// $MP::MyMarble.getCameraYaw(), which is set by the spawn trigger's rotation
// (0/90/180/-90 deg on King of the Marble) and is NOT reset by
// setMarbleCamYaw(0), so the policy's frame changed with every spawn: the
// terrain observation (sampled from the obs position as if world) was wrong in
// 3 of 4 games and human demos (recorded in the world frame) did not transfer.
// The engine still applies F/B/L/R relative to the marble camera, so
// MLAgent::executeAction rotates the world-frame command into the camera frame.
$AIObserver::ForceYaw = 0;

// State tracking (reward computation is in Python)
$MLAgent::LastGemScore = 0;
$MLAgent::WasOOB = false;

function MLAgent::start() {
    if ($MLAgent::Enabled) {
        echo("MLAgent: Already running");
        return;
    }
    // Never sleep when the window is in the background (2026-09-15). The
    // default $Pref::backgroundSleepTime = 200 ms made the engine step physics
    // in coarse chunks whenever the window lost focus: rounds took 124 s of
    // wall time instead of 64, the marble drove drunk on the changed dynamics
    // and the trainer learned from three hours of it. The trainer also guards
    // itself (real-time factor per rollout), but the game must not slow down.
    $Pref::backgroundSleepTime = 0;
    echo("MLAgent: backgroundSleepTime set to 0 (no throttling when the window is in the background)");

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

    // Recording mode: append what the human is doing this tick so Python can
    // turn it into the agent's action format (see ml_agent/record_demos.py):
    //   |forward,backward,left,right,jump,usePowerup,cameraYaw
    // Movement values are the engine's move inputs (0/1 from keys, scaled by
    // $Game::MovementSpeedMultiplier); jump/usePowerup are 1 while held. The
    // camera yaw is the human's live camera, needed to rotate their key
    // direction into the fixed frame the observations are in.
    if ($MLAgent::RecordMode) {
        %msg = %msg @ "|" @ ($mvForwardAction + 0) @ "," @ ($mvBackwardAction + 0) @ ","
                    @ ($mvLeftAction + 0) @ "," @ ($mvRightAction + 0) @ ","
                    @ ($mvTriggerCount2 + 0) @ "," @ ($mvTriggerCount0 + 0) @ ","
                    @ ($MP::MyMarble.getCameraYaw() + 0);
    }

    // 6. Send to Python server and get action
    AIBridge::sendState(%msg);
    %actionStr = $AIBridge::LastAction;

    // Control replies from the Python side (not actions; consumed once):
    //   "SPEED n"                       set the game speed (the trainer can now
    //                                   choose 6x itself; no console needed)
    //   "TELEPORT x y z [vx vy vz]"     move the marble (listen server: the same
    //                                   call cannon.cs uses); replay_demo.py uses
    //                                   it to start from a recorded position
    if (getWord(%actionStr, 0) $= "SPEED") {
        %spd = getWord(%actionStr, 1) + 0;
        if (%spd > 0) {
            $MLAgent::TrainingSpeed = %spd;
            if (!$MLAgent::DiagnosticMode)
                setTimeScale(%spd);
            echo("MLAgent: game speed set to " @ %spd @ "x by the Python server");
        }
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "TELEPORT") {
        if (isObject($MP::MyMarble)) {
            %tp = getWord(%actionStr, 1) SPC getWord(%actionStr, 2) SPC getWord(%actionStr, 3) SPC "1 0 0 0";
            $MP::MyMarble.setTransform(%tp);
            if (getWord(%actionStr, 4) !$= "")
                $MP::MyMarble.setVelocity(getWord(%actionStr, 4) SPC getWord(%actionStr, 5) SPC getWord(%actionStr, 6));
            echo("MLAgent: teleported to " @ %tp @ " by the Python server");
        }
        $AIBridge::LastAction = "";
        %actionStr = "";
    }

    // Recording handshake: record_demos.py answers every message with "RECORD".
    // The first time we see it, switch into recording mode ourselves (you
    // control the marble at 1x, observations pinned to the fixed frame, your
    // inputs appended to each message). No console command needed. If a
    // numeric action shows up instead, the trainer is connected: switch back.
    if (%actionStr $= "RECORD") {
        if (!$MLAgent::RecordMode)
            MLAgent::enableRecording();
    } else if ($MLAgent::RecordMode && %actionStr !$= "" && strPos(%actionStr, ",") != -1) {
        MLAgent::disableRecording();
    }

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
    // Parse comma-separated action: "fwd,back,left,right,jump,camYaw[,usePowerup]"
    // Movement: continuous floats [0.0, 1.0]. Jump: 0 or 1. CamYaw: radians (optional).
    // usePowerup (optional, 7th): 0 or 1, drives the use-powerup trigger.
    %words = strreplace(%actionStr, ",", " ");
    %forward = getWord(%words, 0);
    %backward = getWord(%words, 1);
    %left = getWord(%words, 2);
    %right = getWord(%words, 3);
    %jump = getWord(%words, 4);
    %camYaw = getWord(%words, 5);
    %usePow = getWord(%words, 6);

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

    // World frame -> camera frame (2026-09-14). The command (F/B/L/R) is a
    // direction in the observation frame, which is the world frame
    // ($AIObserver::ForceYaw = 0). The engine applies F/B/L/R relative to the
    // marble's camera, whose yaw comes from the spawn trigger's rotation, so
    // rotate the commanded vector into the camera frame with the observer's
    // convention: right = (cos yaw, -sin yaw), forward = (sin yaw, cos yaw).
    // (The same inverse the demo recorder uses to turn key presses into world
    // directions, which the acceleration check on the recordings confirmed.)
    if ($AIObserver::ForceYaw !$= "" && isObject($MP::MyMarble)) {
        %wx = (%right + 0) - (%left + 0);
        %wy = (%forward + 0) - (%backward + 0);
        %yaw = $MP::MyMarble.getCameraYaw() + 0;
        %cx = %wx * mCos(%yaw) - %wy * mSin(%yaw);
        %cy = %wx * mSin(%yaw) + %wy * mCos(%yaw);
        %right = (%cx > 0) ? %cx : 0;
        %left = (%cx < 0) ? -%cx : 0;
        %forward = (%cy > 0) ? %cy : 0;
        %backward = (%cy < 0) ? -%cy : 0;
    }

    // Execute via analog input (accepts float values 0.0-1.0)
    // Jump is binary (0 or 1) via $mvTriggerCount0
    AIAgent::setCustomAction(%left, %right, %forward, %backward, %jump + 0, %usePow + 0);
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
    // Never touch the time scale while a human is playing (diagnostic/recording).
    if ($MLAgent::Enabled && !$MLAgent::DiagnosticMode) {
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
// Recording mode: YOU play, the observer streams every tick to
// ml_agent/record_demos.py together with your inputs, for behaviour cloning.
// Enabled automatically by the "RECORD" handshake above when record_demos.py
// is the server (or manually via MLAgent::enableRecording() in the console).
// Observations are recorded in the agent's fixed frame (yaw 0 = world) while
// your camera stays free; your key direction is rotated into that frame on the
// Python side.
function MLAgent::enableRecording() {
    $MLAgent::DiagnosticMode = true;
    $MLAgent::RecordMode = true;
    $MLAgent::AutoStart = true;
    $AIObserver::ForceYaw = 0;
    setTimeScale(1.0);
    echo("=== RECORDING MODE ENABLED ===");
    echo("You control the marble at normal speed. Every tick is streamed to record_demos.py.");
    echo("Start: python record_demos.py   then host the map and play.");
}

function MLAgent::disableRecording() {
    $MLAgent::DiagnosticMode = false;
    $MLAgent::RecordMode = false;
    $AIObserver::ForceYaw = 0;     // world frame stays pinned (see the top of this file)
    if ($MLAgent::Enabled)
        setTimeScale($MLAgent::TrainingSpeed);
    echo("=== RECORDING MODE DISABLED ===");
}

function MLAgent::enableDiagnostic() {
    $MLAgent::DiagnosticMode = true;
    $MLAgent::AutoStart = true;
    setTimeScale(1.0);
    echo("=== DIAGNOSTIC MODE ENABLED ===");
    echo("Observations will be sent to Python server but YOU control the marble.");
    echo("Start: python diagnostic.py");
    echo("Then start a Hunt round normally.");
}

//------------------------------------------------------------------------------
// Unattended training: host and start a round automatically
//
// ml_agent/run_game_loop.ps1 launches the game with "-autotrain <MissionName>".
// (The engine's own -mission argument does nothing on the client in this build:
// the mod's argument handler is activated after the arguments were parsed.)
// This replays exactly what a person does: Multiplayer > Host, pick the map,
// Play, then Ready and Start in the pregame dialog. Every step waits for the
// game state it needs rather than a fixed delay. Once the round is running,
// the existing auto-start (onTimerStart) and round-end auto-restart take over.
//------------------------------------------------------------------------------

$MLAgent::AutoTrainMission = "";
for ($MLAgent::_argi = 1; $MLAgent::_argi < $Game::argc; $MLAgent::_argi++) {
    if ($Game::argv[$MLAgent::_argi] $= "-autotrain" && $MLAgent::_argi + 1 < $Game::argc) {
        $MLAgent::AutoTrainMission = $Game::argv[$MLAgent::_argi + 1];
    }
}
if ($MLAgent::AutoTrainMission !$= "") {
    echo("MLAgent: -autotrain " @ $MLAgent::AutoTrainMission @ " -- will host and start the round automatically");
    schedule(5000, 0, "MLAgent::autoTrainStage", 1);
}

function MLAgent::autoTrainStage(%stage) {
    %mission = $MLAgent::AutoTrainMission;
    if (%mission $= "")
        return;

    // Stage 1: wait for the main menu, then host (what Multiplayer > Host does)
    if (%stage == 1) {
        if (!$Server::Hosting) {
            if (!$Menu::Loaded) {
                schedule(1000, 0, "MLAgent::autoTrainStage", 1);
                return;
            }
            echo("MLAgent: autotrain: hosting a server");
            PlayMissionGui.startServer();
        }
        schedule(2000, 0, "MLAgent::autoTrainStage", 2);
        return;
    }

    // Stage 2: lobby is open -> select the mission and press Play
    if (%stage == 2) {
        if (!$Server::Lobby || !isObject(ServerConnection) || RootGui.getContent().getName() !$= "PlayMissionGui") {
            schedule(1000, 0, "MLAgent::autoTrainStage", 2);
            return;
        }
        %file = findNamedFile(%mission, ".m?s");
        if (%file $= "") {
            error("MLAgent: autotrain: mission not found: " @ %mission);
            $MLAgent::AutoTrainMission = "";
            return;
        }
        %info = getMissionInfo(%file);
        if (!isObject(%info)) {
            error("MLAgent: autotrain: no mission info for " @ %file);
            $MLAgent::AutoTrainMission = "";
            return;
        }
        PlayMissionGui.setSelectedMission(%info);
        echo("MLAgent: autotrain: selected " @ %file @ ", loading");
        PlayMissionGui.play();
        schedule(3000, 0, "MLAgent::autoTrainStage", 3);
        return;
    }

    // Stage 3: pregame dialog is up -> Ready
    if (%stage == 3) {
        if (!isObject(MPPreGameDlg) || !MPPreGameDlg.isAwake()) {
            schedule(1000, 0, "MLAgent::autoTrainStage", 3);
            return;
        }
        echo("MLAgent: autotrain: ready");
        commandToServer('Ready', 1);
        schedule(1500, 0, "MLAgent::autoTrainStage", 4);
        return;
    }

    // Stage 4: Start (host override, so it does not wait on anyone)
    if (%stage == 4) {
        echo("MLAgent: autotrain: starting the round");
        commandToServer('PreGamePlay', 1);
        schedule(15000, 0, "MLAgent::autoTrainStage", 5);
        return;
    }

    // Stage 5: if the dialog is still up and nothing is running, try again
    if (%stage == 5) {
        if (isObject(MPPreGameDlg) && MPPreGameDlg.isAwake() && !$Game::Running) {
            echo("MLAgent: autotrain: round did not start, retrying");
            schedule(0, 0, "MLAgent::autoTrainStage", 3);
        }
    }
}
