//-----------------------------------------------------------------------------
// AI Observer - Game State Collection System
//
// Collects all relevant game state for ML model training and inference.
// Returns 284-dimensional observation vector:
//   - Self state: 14 dims
//   - Gems (50 slots): 250 dims (5 per gem: x, y, z, value, distance)
//   - Opponents (3 slots): 15 dims (5 per opponent: x, y, z, vel_x, vel_y)
//   - Game state: 5 dims
//
// Usage:
//   %obs = AIObserver::collectState();
//   %jsonString = AIObserver::serializeToJSON(%obs);
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Configuration
//-----------------------------------------------------------------------------

$AIObserver::MaxGems = 50;
$AIObserver::MaxOpponents = 3;

//-----------------------------------------------------------------------------
// Main State Collection
//-----------------------------------------------------------------------------

function AIObserver::collectState() {
    %obs = new ScriptObject(AIObservation);

    // Check if we're in a valid game state
    if (!isObject($MP::MyMarble)) {
        echo("AIObserver: No marble found, cannot collect state");
        return %obs;
    }

    // 1. Collect self state (14 dims)
    AIObserver::collectSelfState(%obs);

    // 2. Collect all gems (250 dims: 50 gems × 5)
    AIObserver::collectGems(%obs);

    // 3. Collect opponents (15 dims: 3 opponents × 5)
    AIObserver::collectOpponents(%obs);

    // 4. Collect game state (5 dims)
    AIObserver::collectGameState(%obs);

    return %obs;
}

//-----------------------------------------------------------------------------
// Self State Collection (14 dimensions)
//-----------------------------------------------------------------------------

function AIObserver::collectSelfState(%obs) {
    // Position (3)
    %pos = $MP::MyMarble.getPosition();
    %obs.selfPosX = getWord(%pos, 0);
    %obs.selfPosY = getWord(%pos, 1);
    %obs.selfPosZ = getWord(%pos, 2);

    // Velocity (3)
    %vel = $MP::MyMarble.getVelocity();
    %obs.selfVelX = getWord(%vel, 0);
    %obs.selfVelY = getWord(%vel, 1);
    %obs.selfVelZ = getWord(%vel, 2);

    // Camera angles (2)
    %obs.cameraYaw = getMarbleCamYaw();
    %obs.cameraPitch = getMarbleCamPitch();

    // Collision radius (1)
    %obs.collisionRadius = $MP::MyMarble.getScale();

    // Powerup state (3)
    %obs.powerupId = $MP::MyMarble.getPowerUp();
    if (%obs.powerupId $= "")
        %obs.powerupId = -1;

    // Mega marble state (2)
    %obs.megaMarbleActive = $MP::MyMarble.isMegaMarble() ? 1 : 0;
    %obs.megaMarbleTimeRemaining = AIObserver::getMegaMarbleTimeRemaining();

    // Powerup timer (2)
    %obs.powerupTimerRemaining = AIObserver::getPowerupTimerRemaining();
}

//-----------------------------------------------------------------------------
// Gem Collection (250 dimensions: 50 gems × 5)
//-----------------------------------------------------------------------------

function AIObserver::collectGems(%obs) {
    %myPos = $MP::MyMarble.getPosition();
    %myPosX = getWord(%myPos, 0);
    %myPosY = getWord(%myPos, 1);
    %myPosZ = getWord(%myPos, 2);

    // Collect all gems from ItemArray
    %gemCount = 0;
    %gems = new ArrayObject();

    if (isObject(ItemArray)) {
        %count = ItemArray.count();
        for (%i = 0; %i < %count; %i++) {
            %itemData = ItemArray.getEntryByIndex(%i);
            %objId = getField(%itemData, 0);
            %obj = nameToID(%objId);

            if (%obj != -1 && !%obj.isHidden()) {
                // Check if it's a gem (not a powerup)
                %datablock = %obj.getDatablock();
                if (%datablock.className $= "ItemData") {
                    %pos = %obj.getPosition();
                    %gemX = getWord(%pos, 0);
                    %gemY = getWord(%pos, 1);
                    %gemZ = getWord(%pos, 2);

                    // Relative position
                    %relX = %gemX - %myPosX;
                    %relY = %gemY - %myPosY;
                    %relZ = %gemZ - %myPosZ;

                    // Distance
                    %dist = mSqrt(%relX * %relX + %relY * %relY + %relZ * %relZ);

                    // Gem value
                    %value = AIObserver::getGemValue(%obj);

                    // Store gem data: x, y, z, value, distance
                    %gemData = %relX TAB %relY TAB %relZ TAB %value TAB %dist;
                    %gems.push_back(%gemData);

                    %gemCount++;
                    if (%gemCount >= $AIObserver::MaxGems)
                        break;
                }
            }
        }
    }

    // Sort gems by distance (nearest first)
    AIObserver::sortGemsByDistance(%gems);

    // Store gems in observation (pad to 50 with sentinel values)
    for (%i = 0; %i < $AIObserver::MaxGems; %i++) {
        if (%i < %gems.count()) {
            %gemData = %gems.getEntry(%i);
            %obs.gem[%i, "x"] = getField(%gemData, 0);
            %obs.gem[%i, "y"] = getField(%gemData, 1);
            %obs.gem[%i, "z"] = getField(%gemData, 2);
            %obs.gem[%i, "value"] = getField(%gemData, 3);
            %obs.gem[%i, "distance"] = getField(%gemData, 4);
        } else {
            // Padding with sentinel values
            %obs.gem[%i, "x"] = -999;
            %obs.gem[%i, "y"] = -999;
            %obs.gem[%i, "z"] = -999;
            %obs.gem[%i, "value"] = 0;
            %obs.gem[%i, "distance"] = 0;
        }
    }

    %obs.gemCount = %gemCount;
    %gems.delete();
}

//-----------------------------------------------------------------------------
// Opponent Collection (15 dimensions: 3 opponents × 5)
//-----------------------------------------------------------------------------

function AIObserver::collectOpponents(%obs) {
    %myPos = $MP::MyMarble.getPosition();
    %myPosX = getWord(%myPos, 0);
    %myPosY = getWord(%myPos, 1);
    %myPosZ = getWord(%myPos, 2);

    %oppCount = 0;

    if (isObject(PlayerListGuiList)) {
        %count = PlayerListGuiList.rowCount();
        for (%i = 0; %i < %count && %oppCount < $AIObserver::MaxOpponents; %i++) {
            %clientId = PlayerListGuiList.getRowId(%i);
            %client = ClientGroup.getObject(%clientId);

            if (!isObject(%client) || %client == $Client::MyClient)
                continue;

            %player = %client.player;
            if (!isObject(%player))
                continue;

            // Get opponent position
            %oppPos = %player.getPosition();
            %oppPosX = getWord(%oppPos, 0);
            %oppPosY = getWord(%oppPos, 1);
            %oppPosZ = getWord(%oppPos, 2);

            // Relative position
            %relX = %oppPosX - %myPosX;
            %relY = %oppPosY - %myPosY;
            %relZ = %oppPosZ - %myPosZ;

            // Relative velocity (planar only)
            %oppVel = %player.getVelocity();
            %myVel = $MP::MyMarble.getVelocity();
            %velX = getWord(%oppVel, 0) - getWord(%myVel, 0);
            %velY = getWord(%oppVel, 1) - getWord(%myVel, 1);

            // Mega marble status
            %isMega = %player.isMegaMarble() ? 1 : 0;

            // Store opponent data
            %obs.opp[%oppCount, "x"] = %relX;
            %obs.opp[%oppCount, "y"] = %relY;
            %obs.opp[%oppCount, "z"] = %relZ;
            %obs.opp[%oppCount, "velX"] = %velX;
            %obs.opp[%oppCount, "velY"] = %velY;
            %obs.opp[%oppCount, "isMega"] = %isMega;

            %oppCount++;
        }
    }

    // Pad remaining slots with sentinel values
    for (%i = %oppCount; %i < $AIObserver::MaxOpponents; %i++) {
        %obs.opp[%i, "x"] = -999;
        %obs.opp[%i, "y"] = -999;
        %obs.opp[%i, "z"] = -999;
        %obs.opp[%i, "velX"] = 0;
        %obs.opp[%i, "velY"] = 0;
        %obs.opp[%i, "isMega"] = 0;
    }

    %obs.opponentCount = %oppCount;
}

//-----------------------------------------------------------------------------
// Game State Collection (5 dimensions)
//-----------------------------------------------------------------------------

function AIObserver::collectGameState(%obs) {
    // Time elapsed (milliseconds)
    %obs.timeElapsed = PlayGui.currentTime;

    // Time remaining (milliseconds)
    if (isObject(MissionInfo) && MissionInfo.time > 0) {
        %obs.timeRemaining = MissionInfo.time - PlayGui.currentTime;
    } else {
        %obs.timeRemaining = 0;
    }

    // My gem score (total points)
    %obs.myGemScore = PlayGui.gemCount;

    // Best opponent score
    %obs.opponentBestScore = AIObserver::getBestOpponentScore();

    // Gems remaining in level
    %obs.gemsRemaining = PlayGui.maxGems - PlayGui.gemCount;
}

//-----------------------------------------------------------------------------
// Helper Functions
//-----------------------------------------------------------------------------

function AIObserver::getGemValue(%gem) {
    // Get gem point value from datablock
    if (!isObject(%gem))
        return 1;

    %datablock = %gem.getDatablock();
    if (!isObject(%datablock))
        return 1;

    // Hunt mode gems have huntExtraValue field
    if (%datablock.huntExtraValue !$= "") {
        return 1 + %datablock.huntExtraValue;
    }

    return 1;
}

function AIObserver::getMegaMarbleTimeRemaining() {
    if (!$MP::MyMarble.isMegaMarble())
        return 0;

    if ($AI::PowerupActivationTime $= "" || $AI::PowerupDuration $= "")
        return 0;

    %elapsed = PlayGui.totalTime - $AI::PowerupActivationTime;
    %remaining = $AI::PowerupDuration - %elapsed;
    return mMax(0, %remaining / 1000.0); // Convert ms to seconds
}

function AIObserver::getPowerupTimerRemaining() {
    // TODO: Track other powerup timers (super speed, etc.)
    return 0;
}

function AIObserver::getBestOpponentScore() {
    %bestScore = 0;

    if (isObject(PlayerListGuiList)) {
        %count = PlayerListGuiList.rowCount();
        for (%i = 0; %i < %count; %i++) {
            %clientId = PlayerListGuiList.getRowId(%i);
            %client = ClientGroup.getObject(%clientId);

            if (!isObject(%client) || %client == $Client::MyClient)
                continue;

            %score = %client.gemCount;
            if (%score > %bestScore)
                %bestScore = %score;
        }
    }

    return %bestScore;
}

function AIObserver::sortGemsByDistance(%gems) {
    // Simple bubble sort by distance (field index 4)
    %count = %gems.count();
    for (%i = 0; %i < %count - 1; %i++) {
        for (%j = 0; %j < %count - %i - 1; %j++) {
            %dist1 = getField(%gems.getEntry(%j), 4);
            %dist2 = getField(%gems.getEntry(%j + 1), 4);

            if (%dist1 > %dist2) {
                // Swap
                %temp = %gems.getEntry(%j);
                %gems.setEntry(%j, %gems.getEntry(%j + 1));
                %gems.setEntry(%j + 1, %temp);
            }
        }
    }
}

//-----------------------------------------------------------------------------
// JSON Serialization
//-----------------------------------------------------------------------------

function AIObserver::serializeToJSON(%obs) {
    if (!isObject(%obs))
        return "";

    %json = "{\"state\":[";

    // Self state (14 values)
    %json = %json @ %obs.selfPosX @ "," @ %obs.selfPosY @ "," @ %obs.selfPosZ @ ",";
    %json = %json @ %obs.selfVelX @ "," @ %obs.selfVelY @ "," @ %obs.selfVelZ @ ",";
    %json = %json @ %obs.cameraYaw @ "," @ %obs.cameraPitch @ ",";
    %json = %json @ %obs.collisionRadius @ ",";
    %json = %json @ %obs.powerupId @ ",";
    %json = %json @ %obs.megaMarbleActive @ "," @ %obs.megaMarbleTimeRemaining @ ",";
    %json = %json @ %obs.powerupTimerRemaining;

    // Gems (250 values: 50 × 5)
    for (%i = 0; %i < $AIObserver::MaxGems; %i++) {
        %json = %json @ "," @ %obs.gem[%i, "x"];
        %json = %json @ "," @ %obs.gem[%i, "y"];
        %json = %json @ "," @ %obs.gem[%i, "z"];
        %json = %json @ "," @ %obs.gem[%i, "value"];
        %json = %json @ "," @ %obs.gem[%i, "distance"];
    }

    // Opponents (15 values: 3 × 5)
    for (%i = 0; %i < $AIObserver::MaxOpponents; %i++) {
        %json = %json @ "," @ %obs.opp[%i, "x"];
        %json = %json @ "," @ %obs.opp[%i, "y"];
        %json = %json @ "," @ %obs.opp[%i, "z"];
        %json = %json @ "," @ %obs.opp[%i, "velX"];
        %json = %json @ "," @ %obs.opp[%i, "velY"];
    }

    // Game state (5 values)
    %json = %json @ "," @ %obs.timeElapsed;
    %json = %json @ "," @ %obs.timeRemaining;
    %json = %json @ "," @ %obs.myGemScore;
    %json = %json @ "," @ %obs.opponentBestScore;
    %json = %json @ "," @ %obs.gemsRemaining;

    %json = %json @ "]}";

    return %json;
}

//-----------------------------------------------------------------------------
// Test/Debug Functions
//-----------------------------------------------------------------------------

function AIObserver::test() {
    echo("===== AI Observer Test =====");

    %startTime = getRealTime();
    %obs = AIObserver::collectState();
    %elapsed = getRealTime() - %startTime;

    echo("Collection time:" SPC %elapsed @ "ms");
    echo("Gems found:" SPC %obs.gemCount);
    echo("Opponents found:" SPC %obs.opponentCount);

    echo("\nSelf state:");
    echo("  Position:" SPC %obs.selfPosX SPC %obs.selfPosY SPC %obs.selfPosZ);
    echo("  Velocity:" SPC %obs.selfVelX SPC %obs.selfVelY SPC %obs.selfVelZ);
    echo("  Camera:" SPC %obs.cameraYaw SPC %obs.cameraPitch);
    echo("  Powerup:" SPC %obs.powerupId);
    echo("  Mega:" SPC %obs.megaMarbleActive @ " (" @ %obs.megaMarbleTimeRemaining @ "s remaining)");

    echo("\nFirst 3 gems:");
    for (%i = 0; %i < 3 && %i < %obs.gemCount; %i++) {
        echo("  Gem" SPC %i @ ":" SPC
             %obs.gem[%i, "x"] SPC %obs.gem[%i, "y"] SPC %obs.gem[%i, "z"] SPC
             "value=" @ %obs.gem[%i, "value"] SPC
             "dist=" @ %obs.gem[%i, "distance"]);
    }

    echo("\nGame state:");
    echo("  Time elapsed:" SPC %obs.timeElapsed @ "ms");
    echo("  Time remaining:" SPC %obs.timeRemaining @ "ms");
    echo("  My score:" SPC %obs.myGemScore);
    echo("  Best opponent:" SPC %obs.opponentBestScore);

    // Test JSON serialization
    %jsonStartTime = getRealTime();
    %json = AIObserver::serializeToJSON(%obs);
    %jsonElapsed = getRealTime() - %jsonStartTime;

    echo("\nJSON serialization time:" SPC %jsonElapsed @ "ms");
    echo("JSON length:" SPC strlen(%json) SPC "characters");

    %obs.delete();

    echo("===== Test Complete =====");
}

echo("AI Observer System Loaded");
echo("Test with: AIObserver::test()");
