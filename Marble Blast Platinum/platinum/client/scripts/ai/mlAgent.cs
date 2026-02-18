//------------------------------------------------------------------------------
// ML Agent Controller
// Main loop that sends observations to Python and executes actions
//------------------------------------------------------------------------------

$MLAgent::Enabled = false;
$MLAgent::UpdateInterval = 50; // 20 Hz (50ms)
$MLAgent::AutoStart = true;  // Auto-start when Hunt mode begins

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

    // 2. Serialize to JSON
    %json = AIObserver::serializeToJSON(%obs);

    // 3. Send to Python server and get action
    %actionJson = AIBridge::getAction(%json);

    // 4. Parse and execute action
    if (%actionJson !$= "") {
        MLAgent::executeAction(%actionJson);
    }

    // Increment step counter
    $MLAgent::StepCount++;

    // 5. Schedule next update
    $MLAgent::UpdateSchedule = schedule($MLAgent::UpdateInterval, 0, "MLAgent::update");
}

function MLAgent::executeAction(%actionJson) {
    // Parse JSON action: {"forward": 1, "backward": 0, "left": 0, "right": 1, "jump": 0, "powerup": 0}
    // For now, expect simple format: "0,1,0,1,0,0" (forward,backward,left,right,jump,powerup)

    %forward = getField(%actionJson, 0);
    %backward = getField(%actionJson, 1);
    %left = getField(%actionJson, 2);
    %right = getField(%actionJson, 3);
    %jump = getField(%actionJson, 4);
    %powerup = getField(%actionJson, 5);

    // Execute via existing AI agent system
    AIAgent::setBinaryActions(%forward, %backward, %left, %right, %jump, %powerup);
}

function MLAgent::reset() {
    echo("MLAgent: Resetting episode");
    $MLAgent::StepCount = 0;
    $MLAgent::EpisodeStartTime = getRealTime();
}

function MLAgent::onGameStart() {
    // Called when entering a Hunt mode game
    if (!$MLAgent::AutoStart || !mp() || !$Game::isMode["hunt"])
        return;

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
    // Stop agent when game ends
    if ($MLAgent::Enabled) {
        MLAgent::stop();
    }
    $MLAgent::ReadyToStart = false;
}

// Hook into game start
function clientCmdGameStart() {
    Parent::clientCmdGameStart();
    MLAgent::onGameStart();
}
