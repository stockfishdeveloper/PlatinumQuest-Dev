//-----------------------------------------------------------------------------
// AI Recorder - Human Gameplay Data Collection
//
// Records human gameplay for behavioral cloning training.
// Captures game state + player inputs at 20 Hz (every 50ms).
//
// Usage:
//   AIRecorder::start("recordings/session1.jsonl")  // Start recording
//   // ... play the game normally ...
//   AIRecorder::stop()                              // Stop recording
//
// Output format (one line per frame):
//   {"state": [284 floats], "action": [6 binary], "camera": [2 floats]}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Configuration
//-----------------------------------------------------------------------------

// Auto-start recording when match begins
$AIRecorder::AutoStart = true;

// Recording frequency (milliseconds between captures)
$AIRecorder::UpdateInterval = 50; // 20 Hz

// Output directory (relative to game root)
$AIRecorder::OutputDir = "platinum/data/recordings";

//-----------------------------------------------------------------------------
// State
//-----------------------------------------------------------------------------

$AIRecorder::Recording = false;
$AIRecorder::File = "";
$AIRecorder::CurrentFilename = "";
$AIRecorder::FrameCount = 0;
$AIRecorder::SessionStartTime = 0;

//-----------------------------------------------------------------------------
// Start/Stop Functions
//-----------------------------------------------------------------------------

function AIRecorder::start(%filename) {
    if ($AIRecorder::Recording) {
        echo("AIRecorder: Already recording!");
        return;
    }

    // Default filename if not provided
    if (%filename $= "") {
        %timestamp = getSimTime();
        %filename = $AIRecorder::OutputDir @ "/session_" @ %timestamp @ ".jsonl";
    }

    // Create output directory if it doesn't exist
    %dir = filePath(%filename);
    if (%dir !$= "" && !isFile(%dir)) {
        echo("AIRecorder: Creating directory" SPC %dir);
        // Note: TorqueScript doesn't have mkdir, directory must exist
    }

    // Open file for writing
    $AIRecorder::File = new FileObject();
    if (!$AIRecorder::File.openForWrite(%filename)) {
        echo("AIRecorder: ERROR - Could not open file for writing:" SPC %filename);
        $AIRecorder::File.delete();
        $AIRecorder::File = "";
        return;
    }

    $AIRecorder::Recording = true;
    $AIRecorder::CurrentFilename = %filename;
    $AIRecorder::FrameCount = 0;
    $AIRecorder::SessionStartTime = getRealTime();

    echo("===== AI RECORDER STARTED =====");
    echo("Recording to:" SPC %filename);
    echo("Capture rate:" SPC (1000 / $AIRecorder::UpdateInterval) @ " Hz");
    echo("To stop: AIRecorder::stop()");

    // Start recording loop
    AIRecorder::update();
}

function AIRecorder::stop(%silent) {
    if (!$AIRecorder::Recording) {
        if (!%silent)
            echo("AIRecorder: Not currently recording!");
        return;
    }

    $AIRecorder::Recording = false;
    cancel($AIRecorder::UpdateSchedule);

    // Get filename before closing
    %filename = $AIRecorder::CurrentFilename;

    // Close file
    if (isObject($AIRecorder::File)) {
        $AIRecorder::File.close();
        $AIRecorder::File.delete();
    }

    %duration = (getRealTime() - $AIRecorder::SessionStartTime) / 1000.0;

    if (!%silent) {
        echo("===== AI RECORDER STOPPED =====");
        echo("File saved:" SPC %filename);
        echo("Frames recorded:" SPC $AIRecorder::FrameCount);
        echo("Duration:" SPC %duration SPC "seconds");
        echo("Average FPS:" SPC mFloor($AIRecorder::FrameCount / %duration));

        // Estimate file size
        %estimatedSizeKB = mFloor($AIRecorder::FrameCount * 2.5);
        echo("Estimated size:" SPC %estimatedSizeKB SPC "KB");
    }
}

//-----------------------------------------------------------------------------
// Recording Loop
//-----------------------------------------------------------------------------

function AIRecorder::update() {
    if (!$AIRecorder::Recording)
        return;

    // Collect current game state
    %obs = AIObserver::collectState();

    // Collect current player inputs
    %actions = AIRecorder::getCurrentInputs();

    // Collect current camera inputs
    %cameraInputs = AIRecorder::getCurrentCameraInputs();

    // Build JSON line
    %json = AIRecorder::buildJSONLine(%obs, %actions, %cameraInputs);

    // Write to file
    if (isObject($AIRecorder::File)) {
        $AIRecorder::File.writeLine(%json);
        $AIRecorder::FrameCount++;
    }

    // Clean up observation object
    %obs.delete();

    // Schedule next update
    $AIRecorder::UpdateSchedule = schedule($AIRecorder::UpdateInterval, 0, AIRecorder::update);
}

//-----------------------------------------------------------------------------
// Input Capture Functions
//-----------------------------------------------------------------------------

function AIRecorder::getCurrentInputs() {
    // Capture binary inputs (6 actions)
    // Format: forward, backward, left, right, jump, use_powerup

    %forward = $mvForwardAction > 0 ? 1 : 0;
    %backward = $mvBackwardAction > 0 ? 1 : 0;
    %left = $mvLeftAction > 0 ? 1 : 0;
    %right = $mvRightAction > 0 ? 1 : 0;
    %jump = $mvTriggerCount0 > 0 ? 1 : 0;
    %usePowerup = $mvTriggerCount2 > 0 ? 1 : 0;

    return %forward TAB %backward TAB %left TAB %right TAB %jump TAB %usePowerup;
}

function AIRecorder::getCurrentCameraInputs() {
    // Capture camera rotation speeds (2 continuous values)
    // These are accumulated from mouse movement

    %yawSpeed = 0;
    %pitchSpeed = 0;

    // Camera speeds from mouse input
    if ($mvYawLeftSpeed > 0)
        %yawSpeed = -%mvYawLeftSpeed;
    else if ($mvYawRightSpeed > 0)
        %yawSpeed = $mvYawRightSpeed;

    if ($mvPitchUpSpeed > 0)
        %pitchSpeed = $mvPitchUpSpeed;
    else if ($mvPitchDownSpeed > 0)
        %pitchSpeed = -%mvPitchDownSpeed;

    return %yawSpeed TAB %pitchSpeed;
}

//-----------------------------------------------------------------------------
// JSON Serialization
//-----------------------------------------------------------------------------

function AIRecorder::buildJSONLine(%obs, %actions, %cameraInputs) {
    // Build single-line JSON: {"state": [...], "action": [...], "camera": [...]}

    // Get state JSON from observer
    %stateJSON = AIObserver::serializeToJSON(%obs);

    // Build action array
    %forward = getField(%actions, 0);
    %backward = getField(%actions, 1);
    %left = getField(%actions, 2);
    %right = getField(%actions, 3);
    %jump = getField(%actions, 4);
    %usePowerup = getField(%actions, 5);

    %actionJSON = "[" @ %forward @ "," @ %backward @ "," @ %left @ "," @
                       %right @ "," @ %jump @ "," @ %usePowerup @ "]";

    // Build camera array
    %yawSpeed = getField(%cameraInputs, 0);
    %pitchSpeed = getField(%cameraInputs, 1);
    %cameraJSON = "[" @ %yawSpeed @ "," @ %pitchSpeed @ "]";

    // Combine into single JSON line
    // Note: We're re-wrapping the state JSON to include action and camera
    %json = "{\"state\":" @ %stateJSON @ ",\"action\":" @ %actionJSON @ ",\"camera\":" @ %cameraJSON @ "}";

    return %json;
}

//-----------------------------------------------------------------------------
// Auto-Start Hook
//-----------------------------------------------------------------------------

function AIRecorder::onGameStart() {
    if (!$AIRecorder::AutoStart)
        return;

    // Generate unique filename for this match
    %mapName = fileBase($Client::MissionFile);
    %timestamp = getRealTime(); // Use real time for unique filenames
    %filename = $AIRecorder::OutputDir @ "/" @ %mapName @ "_" @ %timestamp @ ".jsonl";

    echo("AI Recorder: Auto-starting for match on" SPC %mapName);
    AIRecorder::start(%filename);
}

function AIRecorder::onGameEnd() {
    if ($AIRecorder::Recording) {
        AIRecorder::stop(true); // Silent stop
    }
}

//-----------------------------------------------------------------------------
// Helper Functions
//-----------------------------------------------------------------------------

function AIRecorder::getStats() {
    if (!$AIRecorder::Recording) {
        echo("AIRecorder: Not currently recording");
        return;
    }

    %duration = (getRealTime() - $AIRecorder::SessionStartTime) / 1000.0;
    %fps = %duration > 0 ? ($AIRecorder::FrameCount / %duration) : 0;

    echo("===== AI Recorder Stats =====");
    echo("Status: RECORDING");
    echo("Frames captured:" SPC $AIRecorder::FrameCount);
    echo("Duration:" SPC %duration SPC "seconds");
    echo("Average FPS:" SPC mFloor(%fps));
    echo("Expected file size:" SPC mFloor($AIRecorder::FrameCount * 2.5) SPC "KB");
}

function AIRecorder::toggleAutoStart() {
    $AIRecorder::AutoStart = !$AIRecorder::AutoStart;
    echo("AIRecorder auto-start:" SPC ($AIRecorder::AutoStart ? "ENABLED" : "DISABLED"));
}

//-----------------------------------------------------------------------------
// Initialization
//-----------------------------------------------------------------------------

echo("=================================================");
echo("AI Recorder System Loaded");
echo("=================================================");
echo("Commands:");
echo("  AIRecorder::start(\"filename.jsonl\")  - Start recording");
echo("  AIRecorder::stop()                     - Stop recording");
echo("  AIRecorder::getStats()                 - Show recording stats");
echo("  AIRecorder::toggleAutoStart()          - Toggle auto-record");
echo("");
echo("Auto-start:" SPC ($AIRecorder::AutoStart ? "ENABLED" : "DISABLED"));
echo("=================================================");
