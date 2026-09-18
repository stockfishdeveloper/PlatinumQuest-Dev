//------------------------------------------------------------------------------
// AI Socket Bridge
// Provides TCP communication between game and Python ML training script
//------------------------------------------------------------------------------

$AIBridge::Connected = false;
$AIBridge::Host = "127.0.0.1";
$AIBridge::Port = 8888;
$AIBridge::LastAction = "0,0,0,0,0,0";  // Default: no movement

function AIBridge::connect(%host, %port) {
    if ($AIBridge::Connected) {
        echo("AIBridge: Already connected");
        return true;
    }

    if (%host !$= "") {
        $AIBridge::Host = %host;
    }
    if (%port !$= "") {
        $AIBridge::Port = %port;
    }

    echo("AIBridge: Connecting to " @ $AIBridge::Host @ ":" @ $AIBridge::Port);

    // Create TCP object
    if (!isObject(AIBridgeSocket)) {
        new TCPObject(AIBridgeSocket);
    }

    AIBridgeSocket.connect($AIBridge::Host @ ":" @ $AIBridge::Port);

    return true;
}

function AIBridge::disconnect() {
    if (isObject(AIBridgeSocket)) {
        AIBridgeSocket.disconnect();
        AIBridgeSocket.delete();
    }
    $AIBridge::Connected = false;
    $AI::WaitReply = false;
    echo("AIBridge: Disconnected");
}

function AIBridgeSocket::onConnected(%this) {
    $AIBridge::Connected = true;
    echo("AIBridge: Connected successfully!");
}

function AIBridgeSocket::onDisconnect(%this) {
    $AIBridge::Connected = false;
    $AI::WaitReply = false;
    echo("AIBridge: Connection lost");
    // Never leave the game frozen without a trainer
    if (getTimeScale() < 0.01)
        setTimeScale($MLAgent::TrainingSpeed > 0 ? $MLAgent::TrainingSpeed : 1);
}

function AIBridgeSocket::onLine(%this, %line) {
    // Received response from Python server.
    // A reply ending in ",t<tick>" answers the observation of that tick and is
    // applied at tick + $MLAgent::ActionDelay (fixed delay, see mlAgent.cs);
    // anything else (untagged actions, SPEED/TELEPORT/... control words) is
    // stored for the next update as before.
    %words = strreplace(%line, ",", " ");
    %n = getWordCount(%words);
    %last = getWord(%words, %n - 1);
    if (%n > 1 && getSubStr(%last, 0, 1) $= "t" && $MLAgent::ActionDelay > 0) {
        %tick = getSubStr(%last, 1, 12) + 0;
        %action = getSubStr(%line, 0, strlen(%line) - strlen(%last) - 1);
        %target = %tick + $MLAgent::ActionDelay;
        $AIBridge::DelayedMode = true;
        if (%target > $MLAgent::Tick) {
            $AIBridge::Queue[%target] = %action;
            $AIBridge::DelayedReplies++;
        } else {
            $AIBridge::LastAction = %action;
            $AIBridge::LateReplies++;
        }
    } else if (strlen(%line) > 0 && strpos("0123456789-.", getSubStr(%line, 0, 1)) == -1) {
        // Control word (SPEED, TELEPORT, SLEEPTIME, MAXFPS, RECORD, ...): its
        // own slot, so a queued action landing on the same tick cannot
        // overwrite it. Consumed once by the next update.
        // A queue, not a slot: when replies arrive in a burst (the trainer answering a
        // backlog) several control words can land between two updates and only the last
        // one survived a single slot (2026-09-18, FIXEDSTEP lost behind TELEPORT).
        $AIBridge::ControlQueue[$AIBridge::ControlTail] = %line;
        $AIBridge::ControlTail++;
        echo("AIBridge: control word received: " @ %line);
    } else {
        $AIBridge::LastAction = %line;
    }
    // Lockstep: the simulation was frozen after the observation went out;
    // the reply is here, let the next tick run.
    if ($MLAgent::Lockstep && $MLAgent::Enabled && !$MLAgent::DiagnosticMode && getTimeScale() < 0.01)
        setTimeScale($MLAgent::TrainingSpeed);
    // Engine lockstep (built engine, $AI::Lockstep): the reply is here, the sim may advance.
    $AI::WaitReply = false;
}

function AIBridge::sendState(%stateJson) {
    if (!$AIBridge::Connected) {
        return;
    }

    // Send state to Python (non-blocking)
    AIBridgeSocket.send(%stateJson @ "\n");
}

function AIBridge::getAction(%stateJson) {
    // Send current state
    AIBridge::sendState(%stateJson);

    // Return last received action (1-frame delay)
    // This allows the async onLine callback to work properly
    return $AIBridge::LastAction;
}
