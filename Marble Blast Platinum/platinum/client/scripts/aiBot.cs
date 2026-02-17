//-----------------------------------------------------------------------------
// AI Bot - Compatibility Wrapper
// This file provides backward compatibility with the old aiBot functions.
// The actual AI agent logic is in ai/agent.cs
//-----------------------------------------------------------------------------

// Load the main AI agent system
exec("./ai/agent.cs");

// Wrapper functions for backward compatibility
function startAIBot() {
	AIAgent::start();
}

function stopAIBot(%silent) {
	AIAgent::stop(%silent);
}

function autoStartAIBot() {
	AIAgent::autoStart();
}

// Keep old variable names working
$AI::Enabled = $AIAgent::Enabled;
$AI::AutoStart = $AIAgent::AutoStart;
