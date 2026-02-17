//-----------------------------------------------------------------------------
// AI Bot - Autonomous Marble Movement
// For testing AI agents in multiplayer
//-----------------------------------------------------------------------------

// Global variables for AI control
$AI::Enabled = false;
$AI::CurrentDirection = 0; // 0 = left, 1 = right
$AI::MoveStrength = 1.0; // Movement strength (0-1)

// Start the AI bot movement
function startAIBot() {
	if ($AI::Enabled) {
		echo("AI Bot already running!");
		return;
	}

	$AI::Enabled = true;
	$AI::CurrentDirection = 0;
	$AI::StartTime = getRealTime();

	echo("===== AI BOT STARTED =====");
	echo("AI will move left and right every 1 second");
	echo("Use stopAIBot() to stop");

	// Start the movement loop
	aiMoveLoop();
}

// Stop the AI bot
function stopAIBot() {
	if (!$AI::Enabled) {
		echo("AI Bot is not running!");
		return;
	}

	$AI::Enabled = false;
	cancel($AI::MoveSchedule);

	// Clear movement
	commandToServer('setAIMove', 0, 0, 0, 0);

	echo("===== AI BOT STOPPED =====");
}

// Main AI movement loop - alternates left/right every second
function aiMoveLoop() {
	if (!$AI::Enabled) {
		return;
	}

	// Calculate elapsed time
	%elapsedMs = getRealTime() - $AI::StartTime;
	%elapsedSec = %elapsedMs / 1000;

	// Switch direction every second
	%currentSecond = mFloor(%elapsedSec);
	$AI::CurrentDirection = %currentSecond % 2;

	// Set movement: left = -1, right = 1
	%moveX = ($AI::CurrentDirection == 0) ? -1.0 : 1.0;
	%moveY = 0; // No forward/backward for now
	%jump = false;

	// Send movement command
	%dirText = ($AI::CurrentDirection == 0) ? "LEFT" : "RIGHT";
	echo("AI Move [" @ %elapsedSec @ "s]: " @ %dirText @ " (x=" @ %moveX @ ")");

	// Apply the movement
	commandToServer('setAIMove', %moveX, %moveY, 0, %jump);

	// Schedule next update (50ms for smooth movement)
	$AI::MoveSchedule = schedule(50, 0, aiMoveLoop);
}

// Server-side command to apply AI movement
function serverCmdSetAIMove(%client, %x, %y, %yaw, %jump) {
	// Safety check
	if (!isObject(%client.player)) {
		return;
	}

	// Apply movement to the marble
	// Note: This applies force/impulse to move the marble
	%marble = %client.player;

	// Get the current camera transform to determine forward direction
	%cameraYaw = %marble.getCameraYaw();
	%radYaw = %cameraYaw * $PI / 180;

	// Calculate movement vector based on camera orientation
	%forwardX = mSin(%radYaw);
	%forwardY = mCos(%radYaw);
	%rightX = mCos(%radYaw);
	%rightY = -mSin(%radYaw);

	// Combine movement inputs
	%moveVecX = %forwardX * %y + %rightX * %x;
	%moveVecY = %forwardY * %y + %rightY * %x;

	// Apply impulse to marble (strength multiplier)
	%strength = 50; // Adjust this for movement speed
	%marble.applyImpulse(%marble.getPosition(), %moveVecX * %strength SPC %moveVecY * %strength SPC "0");

	// Handle jump
	if (%jump) {
		%marble.setVelocity(VectorAdd(%marble.getVelocity(), "0 0 10"));
	}
}

echo("AI Bot system loaded. Use startAIBot() to begin autonomous movement.");
