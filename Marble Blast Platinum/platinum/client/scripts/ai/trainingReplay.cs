// Local, static Hunt replay fixtures. Enabled only by REPLAYCAPTURE in lockstep.
// This is a test/training bridge, never a policy or a replacement for game pickup rules.
$MLReplay::Enabled = false;

function MLReplay::array(%words) {
    return "[" @ strreplace(trim(%words), " ", ",") @ "]";
}

function MLReplay::scan(%group) {
    for (%i = 0; %i < %group.getCount(); %i++) {
        %o = %group.getObject(%i);
        %class = %o.getClassName();
        if (%class $= "SimGroup" || %class $= "SimSet") {
            MLReplay::scan(%o);
        } else if (%class $= "PathedInterior") {
            $MLReplay::Supported = false;
        } else if (%class $= "Item" && !%o.aiMarker) {
            %db = %o.getDataBlock();
            %gem = %db.className $= "Gem" && %db.getName() !$= "BackupGem";
            if (!%gem && AIObserver::powerupType(%db) == 0) continue;
            if (%o.path !$= "" || !%o.isStatic()) $MLReplay::Supported = false;
            %n = $MLReplay::Count;
            $MLReplay::Item[%n] = %o;
            %o._mlReplayIndex = %n;
            $MLReplay::Gem[%n] = %gem;
            %desc = "[" @ (%gem ? 1 : 0) @ ",\"" @ %db.getName() @ "\"," @ MLReplay::array(%o.getPosition()) @ "]";
            $MLReplay::Catalog = $MLReplay::Catalog @ (%n > 0 ? "," : "") @ %desc;
            $MLReplay::Count++;
        }
    }
}

function MLReplay::init() {
    $MLReplay::Supported = $Server::Hosting && $Game::isMode["hunt"] && ClientGroup.getCount() == 1
        && !$MPPref::Server::CompetitiveMode;
    $MLReplay::Count = 0; $MLReplay::Catalog = "";
    MLReplay::scan(MissionGroup);
    $MLReplay::Catalog = "[" @ $MLReplay::Catalog @ "]";
    $MLReplay::Mission = MissionGroup.getId();
    AIBridge::sendState("REPLAYCAT|" @ $MLReplay::Catalog);
}

function MLReplay::snapshot() {
    if (!$MLReplay::Enabled) return;
    if ($MLReplay::Mission !$= MissionGroup.getId()) MLReplay::init();
    %cl = ClientGroup.getObject(0); %pl = %cl.player;
    %ok = $MLReplay::Supported && !%cl.isOOB && !$MLAgent::WasOOB && !$MLAgent::PowYawHold
        && !%pl.megaMarble && $Game::Running && $MLAgent::TimerStarted && !MLAgent::inCountdown();
    for (%i = 1; %i <= 6; %i++) {
        // The impulse's internal active state is not restorable through script.
        if (isEventPending(%pl.powerupSchedule[%i])) %ok = false;
    }
    %items = "";
    for (%i = 0; %i < $MLReplay::Count; %i++) {
        %o = $MLReplay::Item[%i];
        %left = isEventPending(%o._respawnSchedule) ? getEventTimeLeft(%o._respawnSchedule) : -1;
        %items = %items @ (%i ? "," : "") @ "[" @ (%o.isHidden() ? 1 : 0) @ ","
            @ %left @ "," @ (%o._spawnWeight + 0) @ "," @ (%o._respawns + 0) @ "]";
    }
    %groups = "";
    if (isObject(GemGroups)) {
        for (%i = 0; %i < GemGroups.getCount(); %i++)
            %groups = %groups @ (%i ? "," : "") @ (GemGroups.getObject(%i)._spawnCount + 0);
    }
    %held = isObject(%pl.powerUpData) ? %pl.powerUpData.getName() : "none";
    %last = isObject($Game::LastGemSpawner) ? $Game::LastGemSpawner._mlReplayIndex : -1;
    if (%last $= "") %last = -1;
    %pose = $MP::MyMarble.getPosition() SPC $MP::MyMarble.getVelocity() SPC $MP::MyMarble.getAngularVelocity();
    %header = getRandomSeed() SPC ($Time::CurrentTime + 0) SPC ($Time::ElapsedTime + 0)
        SPC (%cl.gemCount + 0) SPC %last SPC ($Game::FirstSpawn ? 1 : 0)
        SPC ($MP::BlastValue + 0) SPC ($MP::SpecialBlast ? 1 : 0) SPC ($MP::MyMarble.getCameraYaw() + 0);
    AIBridge::sendState("REPLAYWORLD|{\"schema\":2,\"supported\":" @ (%ok ? "true" : "false")
        @ ",\"clock_ms\":" @ getSimTime() @ ",\"header\":" @ MLReplay::array(%header)
        @ ",\"held\":\"" @ %held @ "\",\"pose\":" @ MLReplay::array(%pose)
        @ ",\"items\":[" @ %items @ "],\"groups\":[" @ %groups @ "]}");
}

function MLReplay::queueRestore(%payload) {
    if (!$MLReplay::Enabled || !$MLReplay::Supported) {
        AIBridge::sendState("DEBUG|replay_error|replay_not_supported");
        return;
    }
    // Flush the previous trial's commands before giving the saved inventory back.
    $MLReplay::Restore = %payload;
    $MLReplay::RestoreWait = 3;
    %cl = ClientGroup.getObject(0);
    %cl.clearAllPowerups();
    %cl.player.setPowerUpId(0, true);
    $MP::MyMarble._setPowerUp("", true, 0);
    AIAgent::setCustomAction(0, 0, 0, 0, 0, 0);
    $MLAgent::PowYawHold = 0; $MLAgent::PowUseSent = false;
}

function MLReplay::beforeObservation() {
    if ($MLReplay::Restore $= "") return;
    AIAgent::setCustomAction(0, 0, 0, 0, 0, 0);
    if ($MLReplay::RestoreWait-- > 0) return;
    %fields = strreplace($MLReplay::Restore, "|", "\t");
    $MLReplay::Restore = "";
    %head = getField(%fields, 0);
    %pose = getField(%fields, 1);
    %states = strreplace(getField(%fields, 3), ";", "\t");
    if (getFieldCount(%states) != $MLReplay::Count || getWordCount(%pose) != 9 || getWordCount(%head) != 9) {
        AIBridge::sendState("DEBUG|replay_error|invalid_fixture");
        return;
    }
    %cl = ClientGroup.getObject(0); %pl = %cl.player;
    cancel(%cl.respawnSchedule); %cl.respawnSchedule = 0;
    %cl.isOOB = false; %cl.spawningBlocked = false;
    %cl.player.outOfBounds = false;
    $MLAgent::WasOOB = false; $MLAgent::OOBPosX = "";
    if (!isObject(SpawnedSet)) RootGroup.add(new SimSet(SpawnedSet));
    while (SpawnedSet.getCount()) SpawnedSet.remove(SpawnedSet.getObject(0));
    $Hunt::CurrentGemCount = 0;
    for (%i = 0; %i < $MLReplay::Count; %i++) {
        %o = $MLReplay::Item[%i]; %s = getField(%states, %i);
        cancel(%o._respawnSchedule); cancel(%o.predictionSchedule);
        %o._respawnSchedule = 0;
        %o._spawnWeight = getWord(%s, 2); %o._respawns = getWord(%s, 3);
        %o.hide(getWord(%s, 0)); %o.setFadeVal(1);
        %cl.gemPickup[%o] = false;
        if ($MLReplay::Gem[%i] && !getWord(%s, 0)) {
            SpawnedSet.add(%o); $Hunt::CurrentGemCount++;
        } else if (!$MLReplay::Gem[%i] && getWord(%s, 1) >= 0) {
            %o._respawnSchedule = %o.schedule(getWord(%s, 1), "onRespawn");
        }
    }
    %groups = getField(%fields, 4);
    if (isObject(GemGroups)) {
        for (%i = 0; %i < GemGroups.getCount(); %i++)
            GemGroups.getObject(%i)._spawnCount = getWord(%groups, %i);
    }
    %last = getWord(%head, 4);
    $Game::LastGemSpawner = %last < 0 ? 0 : $MLReplay::Item[%last];
    $Game::FirstSpawn = getWord(%head, 5);
    %cl.gemCount = getWord(%head, 3); %cl.gemPickupCount = 0;
    PlayGui.gemCount = %cl.gemCount;
    $MLAgent::LastGemScore = %cl.gemCount;
    Time::set(getWord(%head, 1));
    $Time::ElapsedTime = getWord(%head, 2);
    PlayGui.currentTime = getWord(%head, 1);
    %held = getField(%fields, 2);
    %pl.powerUpData = ""; %pl.heldPowerup = "";
    if (%held !$= "none" && isObject(%held)) {
        %pl.setPowerUp(%held.getId(), true, 0);
        $MP::MyMarble._setPowerUp(%held.getId(), true, 0);
    }
    $MP::BlastValue = getWord(%head, 6); $MP::SpecialBlast = getWord(%head, 7);
    %transform = getWords(%pose, 0, 2) SPC "1 0 0 0";
    %pl.setTransform(%transform); $MP::MyMarble.setTransform(%transform);
    %pl.setVelocity(getWords(%pose, 3, 5)); $MP::MyMarble.setVelocity(getWords(%pose, 3, 5));
    %pl.setAngularVelocity(getWords(%pose, 6, 8)); $MP::MyMarble.setAngularVelocity(getWords(%pose, 6, 8));
    setMarbleCamYaw(getWord(%head, 8)); $mvYaw = 0;
    $MLAgent::PowYawHold = 0; $MLAgent::PowUseSent = false;
    $AIBridge::LastAction = "";
    setRandomSeed(getWord(%head, 0));
    AIBridge::sendState("DEBUG|replay_restored|" @ getSimTime());
}
