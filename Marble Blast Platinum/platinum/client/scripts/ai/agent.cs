//-----------------------------------------------------------------------------
// AI Agent - Autonomous Marble Control System
//
// This file contains the AI agent that controls marble movement.
// It is separate from the main game code for easy development and iteration.
//
// Usage:
//   - AIAgent::start()  - Start the AI agent
//   - AIAgent::stop()   - Stop the AI agent
//   - AIAgent::setAction(action) - Set a specific action
//
// Copyright (c) 2026 - AI Development
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Configuration
//-----------------------------------------------------------------------------

// Auto-start when game begins
$AIAgent::AutoStart = false;  // Disabled - we're collecting human training data

// Action cycle time (milliseconds)
$AIAgent::ActionDuration = 1000; // 1 second per action

// Update frequency (milliseconds)
$AIAgent::UpdateInterval = 50; // 20 updates per second

//-----------------------------------------------------------------------------
// Agent State
//-----------------------------------------------------------------------------

$AIAgent::Enabled = false;
$AIAgent::StartTime = 0;
$AIAgent::CurrentActionIndex = 0;
$AIAgent::LastLoggedSecond = -1;

//-----------------------------------------------------------------------------
// Available Actions
//-----------------------------------------------------------------------------

// Action definitions: each action specifies the movement inputs to set
$AIAgent::Actions = Array(AIAgentActions);

function AIAgent::initActions() {
	// Clear existing actions
	AIAgentActions.empty();

	// Define all available actions
	// Format: "name" TAB "description" TAB mvLeft TAB mvRight TAB mvForward TAB mvBackward TAB jump TAB usePowerup

	AIAgentActions.push_back("LEFT" TAB "Move left" TAB 1.0 TAB 0 TAB 0 TAB 0 TAB 0 TAB 0);
	AIAgentActions.push_back("RIGHT" TAB "Move right" TAB 0 TAB 1.0 TAB 0 TAB 0 TAB 0 TAB 0);
	AIAgentActions.push_back("FORWARD" TAB "Move forward" TAB 0 TAB 0 TAB 1.0 TAB 0 TAB 0 TAB 0);
	AIAgentActions.push_back("BACKWARD" TAB "Move backward" TAB 0 TAB 0 TAB 0 TAB 1.0 TAB 0 TAB 0);
	AIAgentActions.push_back("JUMP" TAB "Jump" TAB 0 TAB 0 TAB 0 TAB 0 TAB 1 TAB 0);
	AIAgentActions.push_back("FORWARD-LEFT" TAB "Move forward-left" TAB 1.0 TAB 0 TAB 1.0 TAB 0 TAB 0 TAB 0);
	AIAgentActions.push_back("FORWARD-RIGHT" TAB "Move forward-right" TAB 0 TAB 1.0 TAB 1.0 TAB 0 TAB 0 TAB 0);
	AIAgentActions.push_back("USE-POWERUP" TAB "Use powerup" TAB 0 TAB 0 TAB 0 TAB 0 TAB 0 TAB 1);
}

//-----------------------------------------------------------------------------
// Movement Control Functions
//-----------------------------------------------------------------------------

// Clear all movement inputs
function AIAgent::clearInputs() {
	$mvLeftAction = 0;
	$mvRightAction = 0;
	$mvForwardAction = 0;
	$mvBackwardAction = 0;
	$mvTriggerCount0 = 0; // Jump
	$mvTriggerCount2 = 0; // Use powerup
}

// Set binary actions directly (for ML agent)
// Takes 6 binary values: forward, backward, left, right, jump, powerup
function AIAgent::setBinaryActions(%forward, %backward, %left, %right, %jump, %powerup) {
	// Clear all inputs first
	AIAgent::clearInputs();

	// Set binary inputs (convert to 0 or 1)
	$mvForwardAction = %forward ? 1.0 : 0;
	$mvBackwardAction = %backward ? 1.0 : 0;
	$mvLeftAction = %left ? 1.0 : 0;
	$mvRightAction = %right ? 1.0 : 0;
	$mvTriggerCount0 = %jump ? 1 : 0;      // Jump
	$mvTriggerCount2 = %powerup ? 1 : 0;   // Use powerup
}

// Apply a specific action by index
function AIAgent::applyAction(%actionIndex) {
	// Get action data
	%actionData = AIAgentActions.getValue(%actionIndex);

	// Parse action data
	%name = getField(%actionData, 0);
	%desc = getField(%actionData, 1);
	%mvLeft = getField(%actionData, 2);
	%mvRight = getField(%actionData, 3);
	%mvForward = getField(%actionData, 4);
	%mvBackward = getField(%actionData, 5);
	%jump = getField(%actionData, 6);
	%usePowerup = getField(%actionData, 7);

	// Clear all inputs first
	AIAgent::clearInputs();

	// Apply the action inputs
	$mvLeftAction = %mvLeft;
	$mvRightAction = %mvRight;
	$mvForwardAction = %mvForward;
	$mvBackwardAction = %mvBackward;
	$mvTriggerCount0 = %jump;
	$mvTriggerCount2 = %usePowerup;

	return %name;
}

// Set a custom action (for external control/ML models)
function AIAgent::setCustomAction(%left, %right, %forward, %backward, %jump, %usePowerup) {
	AIAgent::clearInputs();

	$mvLeftAction = %left;
	$mvRightAction = %right;
	$mvForwardAction = %forward;
	$mvBackwardAction = %backward;
	$mvTriggerCount0 = %jump;
	$mvTriggerCount2 = %usePowerup;
}

//-----------------------------------------------------------------------------
// Main Agent Loop
//-----------------------------------------------------------------------------

function AIAgent::update() {
	if (!$AIAgent::Enabled) {
		return;
	}

	// Calculate elapsed time
	%elapsedMs = getRealTime() - $AIAgent::StartTime;
	%elapsedSec = %elapsedMs / 1000;

	// Determine which action to perform based on time
	%currentSecond = mFloor(%elapsedSec);
	%actionIndex = %currentSecond % AIAgentActions.count();

	// Apply the current action
	%actionName = AIAgent::applyAction(%actionIndex);

	// Log and display when action changes
	if (%currentSecond != $AIAgent::LastLoggedSecond) {
		echo("AI Agent [" @ %currentSecond @ "s]: " @ %actionName);
		// centerprint("AI: " @ %actionName, 1);  // Disabled - causes error with fake clients
		$AIAgent::LastLoggedSecond = %currentSecond;
	}

	// Schedule next update
	$AIAgent::UpdateSchedule = schedule($AIAgent::UpdateInterval, 0, "AIAgent::update");
}

//-----------------------------------------------------------------------------
// Start/Stop Functions
//-----------------------------------------------------------------------------

function AIAgent::start() {
	if ($AIAgent::Enabled) {
		echo("AI Agent is already running!");
		return;
	}

	// Initialize actions if needed
	if (!isObject(AIAgentActions) || AIAgentActions.count() == 0) {
		AIAgent::initActions();
	}

	$AIAgent::Enabled = true;
	$AIAgent::StartTime = getRealTime();
	$AIAgent::LastLoggedSecond = -1;

	echo("===== AI AGENT STARTED =====");
	echo("Testing " @ AIAgentActions.count() @ " actions in sequence");
	echo("To stop: AIAgent::stop()");

	// centerprint("AI AGENT STARTED", 2);  // Disabled - causes error with fake clients

	// Start the update loop
	AIAgent::update();
}

function AIAgent::stop(%silent) {
	if (!$AIAgent::Enabled) {
		if (!%silent) {
			echo("AI Agent is not running!");
		}
		return;
	}

	$AIAgent::Enabled = false;
	cancel($AIAgent::UpdateSchedule);

	// Clear all movement inputs
	AIAgent::clearInputs();

	echo("===== AI AGENT STOPPED =====");

	if (!%silent) {
		centerprint("AI AGENT STOPPED", 2);
	}
}

//-----------------------------------------------------------------------------
// Auto-Start Hook (called when game begins)
//-----------------------------------------------------------------------------

function AIAgent::autoStart() {
	if (!$AIAgent::AutoStart) {
		return;
	}

	// Wait for game to fully initialize
	schedule(500, 0, "AIAgent::start");
}

//-----------------------------------------------------------------------------
// Initialization
//-----------------------------------------------------------------------------

// Initialize actions on load
AIAgent::initActions();

echo("==================================================");
echo("AI Agent System Loaded");
echo("==================================================");
echo("Available actions: " @ AIAgentActions.count());
echo("Auto-start: " @ ($AIAgent::AutoStart ? "ENABLED" : "DISABLED"));
echo("");
echo("Commands:");
echo("  AIAgent::start()  - Start the AI agent");
echo("  AIAgent::stop()   - Stop the AI agent");
echo("  $AIAgent::AutoStart = true/false - Toggle auto-start");
echo("==================================================");
