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
    echo("AIBridge: Disconnected");
}

function AIBridgeSocket::onConnected(%this) {
    $AIBridge::Connected = true;
    echo("AIBridge: Connected successfully!");
}

function AIBridgeSocket::onDisconnect(%this) {
    $AIBridge::Connected = false;
    echo("AIBridge: Connection lost");
}

function AIBridgeSocket::onLine(%this, %line) {
    // Received response from Python server
    // Store it for next frame's use
    $AIBridge::LastAction = %line;
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
