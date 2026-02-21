//-----------------------------------------------------------------------------
// AI Observer - Game State Collection System
//
// Collects all relevant game state for ML model training and inference.
// Returns 61-dimensional observation vector:
//   - Self state: 13 dims (pos[world], vel[camera-relative], camera, radius, powerup state)
//   - Gems (5 nearest slots): 25 dims (5 per gem: x, y, z [camera-relative], value, distance)
//   - Opponents (3 slots): 18 dims (6 per opponent: x, y, z, vel_x, vel_y [camera-relative], is_mega)
//   - Game state: 5 dims
//
// Gem/opponent/velocity observations are rotated into camera space so that
// x = camera-right, y = camera-forward. This aligns with F/B/L/R actions.
//
// Usage:
//   %obs = AIObserver::collectState();
//   %jsonString = AIObserver::serializeToJSON(%obs);
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Configuration
//-----------------------------------------------------------------------------

$AIObserver::MaxGems = 5;   // 5 nearest gems = 25 dims (was 50 = 250 dims)
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

    // 2. Collect nearest 5 gems (25 dims: 5 gems × 5)
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
    // Position (3) — kept in world space (needed for OOB credit assignment)
    %pos = $MP::MyMarble.getPosition();
    %obs.selfPosX = getWord(%pos, 0) + 0;  // +0 converts empty string to 0
    %obs.selfPosY = getWord(%pos, 1) + 0;
    %obs.selfPosZ = getWord(%pos, 2) + 0;

    // Camera angles (2)
    %obs.cameraYaw = ($cameraYaw $= "") ? 0 : $cameraYaw;
    %obs.cameraPitch = ($cameraPitch $= "") ? 0 : $cameraPitch;

    // Velocity (3) — rotated into camera space so velX = rightward, velY = forward.
    // This aligns velocity with the F/B/L/R action axes.
    %vel = $MP::MyMarble.getVelocity();
    %worldVelX = getWord(%vel, 0) + 0;
    %worldVelY = getWord(%vel, 1) + 0;
    // $cameraYaw is already in radians (wraps at ±pi), no degree conversion needed
    %yawRad = %obs.cameraYaw;
    %cosYaw = mCos(%yawRad);
    %sinYaw = mSin(%yawRad);
    // Camera right = (cos(yaw), -sin(yaw)), forward = (sin(yaw), cos(yaw))
    %obs.selfVelX = %worldVelX * %cosYaw - %worldVelY * %sinYaw;
    %obs.selfVelY = %worldVelX * %sinYaw + %worldVelY * %cosYaw;
    %obs.selfVelZ = getWord(%vel, 2) + 0;

    // Collision radius (1) - just get X component of scale vector
    %scale = $MP::MyMarble.getScale();
    %obs.collisionRadius = (%scale $= "") ? 0.2 : getWord(%scale, 0);

    // Powerup state (3)
    %obs.powerupId = $MP::MyMarble.getPowerUp();
    if (%obs.powerupId $= "")
        %obs.powerupId = -1;

    // Mega marble state (2)
    %obs.megaMarbleActive = $MP::MyMarble.isMegaMarble() ? 1 : 0;
    %obs.megaMarbleTimeRemaining = AIObserver::getMegaMarbleTimeRemaining() + 0;

    // Powerup timer (2)
    %obs.powerupTimerRemaining = AIObserver::getPowerupTimerRemaining() + 0;
}

//-----------------------------------------------------------------------------
// Gem Collection (25 dimensions: 5 nearest gems × 5)
//-----------------------------------------------------------------------------

function AIObserver::collectGems(%obs) {
    %myPos = $MP::MyMarble.getPosition();
    %myPosX = getWord(%myPos, 0);
    %myPosY = getWord(%myPos, 1);
    %myPosZ = getWord(%myPos, 2);

    // Camera yaw for rotating world-relative vectors into camera space.
    // This ensures gem relX/relY align with the L/R and F/B action axes.
    // Without this, "gem is at +X" might mean "press forward" or "press left"
    // depending on where the camera happens to be pointing.
    // $cameraYaw is already in radians (wraps at ±pi), no degree conversion needed.
    %yawRad = ($cameraYaw $= "") ? 0 : $cameraYaw;
    %cosYaw = mCos(%yawRad);
    %sinYaw = mSin(%yawRad);

    // Collect all gems from ItemArray
    %gemCount = 0;

    // Instead of creating array, use direct storage in observation
    // This avoids creating/deleting temporary array objects every frame

    // Try ItemArray first, fallback to ServerConnection if empty
    %count = 0;
    %useServerConnection = false;

    if (isObject(ItemArray)) {
        %count = ItemArray.getSize();  // Use getSize() not count()
    }

    // If ItemArray is empty, use ServerConnection directly
    if (%count == 0 && isObject(ServerConnection)) {
        %count = ServerConnection.getCount();
        %useServerConnection = true;
    }

    if (!$AIObserver::LoggedGemCollection) {
        echo("AIObserver: " @ %count @ " objects from " @ (%useServerConnection ? "ServerConnection" : "ItemArray"));
        $AIObserver::LoggedGemCollection = true;
    }

    for (%i = 0; %i < %count; %i++) {
        if (%useServerConnection) {
            %obj = ServerConnection.getObject(%i);
        } else {
            %itemData = ItemArray.getEntryByIndex(%i);
            %objId = getField(%itemData, 0);
            %obj = nameToID(%objId);
        }

        if (isObject(%obj) && !%obj.isHidden()) {
                // Only accept items whose datablock classname is "Gem"
                // This matches the check in makeGemGroup (huntGems.cs line 979)
                // and filters out powerups, BackupGems, and any other non-gem Items
                %datablock = %obj.getDatablock();
                %isGem = (isObject(%datablock) && %datablock.classname $= "Gem");

                if (%isGem) {
                    %pos = %obj.getPosition();
                    %gemX = getWord(%pos, 0);
                    %gemY = getWord(%pos, 1);
                    %gemZ = getWord(%pos, 2);

                    // Relative position (world space)
                    %relX = %gemX - %myPosX;
                    %relY = %gemY - %myPosY;
                    %relZ = %gemZ - %myPosZ;

                    // Rotate into camera space so relX = camera-right, relY = camera-forward.
                    // Camera right = (cos(yaw), -sin(yaw)), forward = (sin(yaw), cos(yaw))
                    %camRelX = %relX * %cosYaw - %relY * %sinYaw;
                    %camRelY = %relX * %sinYaw + %relY * %cosYaw;

                    // Distance (same in both frames)
                    %dist = mSqrt(%relX * %relX + %relY * %relY + %relZ * %relZ);

                    // Gem value
                    %value = AIObserver::getGemValue(%obj);

                    // Store gem data in camera-relative coordinates
                    %obs.gemTemp[%gemCount, "x"] = %camRelX;
                    %obs.gemTemp[%gemCount, "y"] = %camRelY;
                    %obs.gemTemp[%gemCount, "z"] = %relZ;
                    %obs.gemTemp[%gemCount, "value"] = %value;
                    %obs.gemTemp[%gemCount, "distance"] = %dist;

                    %gemCount++;
                    if (%gemCount >= $AIObserver::MaxGems)
                        break;
                }
            }
    }  // End of item loop

    if (!$AIObserver::FrameCount)
        $AIObserver::FrameCount = 0;
    $AIObserver::FrameCount++;

    // Sort gems by distance (nearest first) - using bubble sort on temp storage
    AIObserver::sortGemsInObs(%obs, %gemCount);

    // Copy sorted gems to final storage and pad remaining slots
    for (%i = 0; %i < $AIObserver::MaxGems; %i++) {
        if (%i < %gemCount) {
            %obs.gem[%i, "x"] = %obs.gemTemp[%i, "x"];
            %obs.gem[%i, "y"] = %obs.gemTemp[%i, "y"];
            %obs.gem[%i, "z"] = %obs.gemTemp[%i, "z"];
            %obs.gem[%i, "value"] = %obs.gemTemp[%i, "value"];
            %obs.gem[%i, "distance"] = %obs.gemTemp[%i, "distance"];
        } else {
            // Padding with sentinel values — distance must also be -999
            // so Python's sentinel check (obs[b+4] < -500) catches it.
            %obs.gem[%i, "x"] = -999;
            %obs.gem[%i, "y"] = -999;
            %obs.gem[%i, "z"] = -999;
            %obs.gem[%i, "value"] = -999;
            %obs.gem[%i, "distance"] = -999;
        }
    }

    %obs.gemCount = %gemCount;
}

//-----------------------------------------------------------------------------
// Opponent Collection (15 dimensions: 3 opponents × 5)
//-----------------------------------------------------------------------------

function AIObserver::collectOpponents(%obs) {
    %myPos = $MP::MyMarble.getPosition();
    %myPosX = getWord(%myPos, 0);
    %myPosY = getWord(%myPos, 1);
    %myPosZ = getWord(%myPos, 2);

    // Camera yaw for world-to-camera rotation (same as in collectGems)
    // $cameraYaw is already in radians
    %yawRad = ($cameraYaw $= "") ? 0 : $cameraYaw;
    %cosYaw = mCos(%yawRad);
    %sinYaw = mSin(%yawRad);

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

            // Relative position (world space)
            %relX = %oppPosX - %myPosX;
            %relY = %oppPosY - %myPosY;
            %relZ = %oppPosZ - %myPosZ;

            // Rotate into camera space
            // Camera right = (cos(yaw), -sin(yaw)), forward = (sin(yaw), cos(yaw))
            %camRelX = %relX * %cosYaw - %relY * %sinYaw;
            %camRelY = %relX * %sinYaw + %relY * %cosYaw;

            // Relative velocity (world space, planar only)
            %oppVel = %player.getVelocity();
            %myVel = $MP::MyMarble.getVelocity();
            %velX = getWord(%oppVel, 0) - getWord(%myVel, 0);
            %velY = getWord(%oppVel, 1) - getWord(%myVel, 1);

            // Rotate velocity into camera space too
            // Camera right = (cos(yaw), -sin(yaw)), forward = (sin(yaw), cos(yaw))
            %camVelX = %velX * %cosYaw - %velY * %sinYaw;
            %camVelY = %velX * %sinYaw + %velY * %cosYaw;

            // Mega marble status
            %isMega = %player.isMegaMarble() ? 1 : 0;

            // Store opponent data in camera-relative coordinates
            %obs.opp[%oppCount, "x"] = %camRelX;
            %obs.opp[%oppCount, "y"] = %camRelY;
            %obs.opp[%oppCount, "z"] = %relZ;
            %obs.opp[%oppCount, "velX"] = %camVelX;
            %obs.opp[%oppCount, "velY"] = %camVelY;
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

function AIObserver::sortGemsInObs(%obs, %count) {
    // Simple bubble sort by distance on gemTemp storage
    for (%i = 0; %i < %count - 1; %i++) {
        for (%j = 0; %j < %count - %i - 1; %j++) {
            %dist1 = %obs.gemTemp[%j, "distance"];
            %dist2 = %obs.gemTemp[%j + 1, "distance"];

            if (%dist1 > %dist2) {
                // Swap all gem fields
                %tempX = %obs.gemTemp[%j, "x"];
                %tempY = %obs.gemTemp[%j, "y"];
                %tempZ = %obs.gemTemp[%j, "z"];
                %tempVal = %obs.gemTemp[%j, "value"];
                %tempDist = %obs.gemTemp[%j, "distance"];

                %obs.gemTemp[%j, "x"] = %obs.gemTemp[%j + 1, "x"];
                %obs.gemTemp[%j, "y"] = %obs.gemTemp[%j + 1, "y"];
                %obs.gemTemp[%j, "z"] = %obs.gemTemp[%j + 1, "z"];
                %obs.gemTemp[%j, "value"] = %obs.gemTemp[%j + 1, "value"];
                %obs.gemTemp[%j, "distance"] = %obs.gemTemp[%j + 1, "distance"];

                %obs.gemTemp[%j + 1, "x"] = %tempX;
                %obs.gemTemp[%j + 1, "y"] = %tempY;
                %obs.gemTemp[%j + 1, "z"] = %tempZ;
                %obs.gemTemp[%j + 1, "value"] = %tempVal;
                %obs.gemTemp[%j + 1, "distance"] = %tempDist;
            }
        }
    }
}

//-----------------------------------------------------------------------------
// JSON Serialization
//-----------------------------------------------------------------------------

function AIObserver::safeNum(%val) {
    // Ensure numeric value - if empty string or undefined, return 0
    if (%val $= "")
        return 0;
    return %val;
}

function AIObserver::serializeToJSON(%obs) {
    if (!isObject(%obs))
        return "";

    %json = "[";

    // Helper to ensure numeric value (replace empty string with 0)
    // Self state (13 values)
    %json = %json @ AIObserver::safeNum(%obs.selfPosX) @ "," @ AIObserver::safeNum(%obs.selfPosY) @ "," @ AIObserver::safeNum(%obs.selfPosZ) @ ",";
    %json = %json @ AIObserver::safeNum(%obs.selfVelX) @ "," @ AIObserver::safeNum(%obs.selfVelY) @ "," @ AIObserver::safeNum(%obs.selfVelZ) @ ",";
    %json = %json @ AIObserver::safeNum(%obs.cameraYaw) @ "," @ AIObserver::safeNum(%obs.cameraPitch) @ ",";
    %json = %json @ AIObserver::safeNum(%obs.collisionRadius) @ ",";
    %json = %json @ AIObserver::safeNum(%obs.powerupId) @ ",";
    %json = %json @ AIObserver::safeNum(%obs.megaMarbleActive) @ "," @ AIObserver::safeNum(%obs.megaMarbleTimeRemaining) @ ",";
    %json = %json @ AIObserver::safeNum(%obs.powerupTimerRemaining);

    // Gems (25 values: 5 × 5)
    for (%i = 0; %i < $AIObserver::MaxGems; %i++) {
        %json = %json @ "," @ AIObserver::safeNum(%obs.gem[%i, "x"]);
        %json = %json @ "," @ AIObserver::safeNum(%obs.gem[%i, "y"]);
        %json = %json @ "," @ AIObserver::safeNum(%obs.gem[%i, "z"]);
        %json = %json @ "," @ AIObserver::safeNum(%obs.gem[%i, "value"]);
        %json = %json @ "," @ AIObserver::safeNum(%obs.gem[%i, "distance"]);
    }

    // Opponents (18 values: 3 × 6 - added isMega field)
    for (%i = 0; %i < $AIObserver::MaxOpponents; %i++) {
        %json = %json @ "," @ AIObserver::safeNum(%obs.opp[%i, "x"]);
        %json = %json @ "," @ AIObserver::safeNum(%obs.opp[%i, "y"]);
        %json = %json @ "," @ AIObserver::safeNum(%obs.opp[%i, "z"]);
        %json = %json @ "," @ AIObserver::safeNum(%obs.opp[%i, "velX"]);
        %json = %json @ "," @ AIObserver::safeNum(%obs.opp[%i, "velY"]);
        %json = %json @ "," @ AIObserver::safeNum(%obs.opp[%i, "isMega"]);
    }

    // Game state (5 values)
    %json = %json @ "," @ AIObserver::safeNum(%obs.timeElapsed);
    %json = %json @ "," @ AIObserver::safeNum(%obs.timeRemaining);
    %json = %json @ "," @ AIObserver::safeNum(%obs.myGemScore);
    %json = %json @ "," @ AIObserver::safeNum(%obs.opponentBestScore);
    %json = %json @ "," @ AIObserver::safeNum(%obs.gemsRemaining);

    %json = %json @ "]";

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
