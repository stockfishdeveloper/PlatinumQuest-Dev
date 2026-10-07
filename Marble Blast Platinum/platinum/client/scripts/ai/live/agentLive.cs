//------------------------------------------------------------------------------
// Live play against people (production bridge, 2026-10-04)
//
// Loaded ONLY when the game is launched with -ailive (hook in client/init.cs, right after mlAgent.cs).
// It runs after the training bridge (ai/mlAgent.cs, ai/observer.cs) and REDEFINES the functions below, so the
// training files stay exactly as training needs them. Pair with the Python side: python -m nav.live_play
// (or ml_agent/play_live.ps1, which starts both).
//
// What live play needs that training does not:
// * real time: time scale 1, no lockstep / fixed step (the Python side sends neither);
// * a joined client of a remote server: $Game::Running is set only by the hosting server's scripts, so the agent
//   acts from GO (clientCmdStartTimer, or the game state "go") and plays the Ready/Set countdown;
// * gems on a joined client: datablocks arrive without their script fields (classname reads "ItemData"), so gems
//   are recognised by their datablock name (the server sends the names) or a gem shape;
// * the round ends when the server ends it (no clock test, no step cap) and the lobby starts every round (no
//   automatic restart);
// * a fixed render view (the steering turns the marble camera every decision);
// * DEBUG|live| lines to the Python side saying why the agent waits and what the gem scan sees (the engine buffers
//   console.log), printed by nav/live_play.py.
//------------------------------------------------------------------------------

$MLAgent::Live = true;
$MLAgent::TrainingSpeed = 1.0;
$MLAgent::ViewYawOn = true;
$MLAgent::ViewYaw = 0;
$MLAgent::ViewYawApplied = "";
echo("MLAgent: -ailive: production bridge loaded (real time; drives from GO; the lobby starts each round)");

// ---- MLAgent::startLoop (live version; the training version is in ../mlAgent.cs) ----
function MLAgent::startLoop() {
    if (!$AIBridge::Connected) {
        // Python server not up yet (or restarting): keep trying every 2 s for
        // as long as the round runs. Rounds restart by themselves (onGameEnd),
        // so a game launched before its trainer connects as soon as it appears.
        $MLAgent::ConnectAttempts++;
        if ($MLAgent::ConnectAttempts == 1 || $MLAgent::ConnectAttempts % 15 == 0)
            echo("MLAgent: Python server not reachable on " @ $AIBridge::Host @ ":" @ $AIBridge::Port @ ", retrying (attempt " @ $MLAgent::ConnectAttempts @ ")");
        if ($Game::Running) {
            AIBridge::disconnect();
            AIBridge::connect("", "");
            schedule(2000, 0, "MLAgent::startLoop");
        } else {
            $MLAgent::StartPending = false;
        }
        return;
    }
    $MLAgent::ConnectAttempts = 0;
    $MLAgent::StartPending = false;
    // One loop only: a new generation invalidates every update() still scheduled
    // by an older loop, and any pending schedule is cancelled.
    $MLAgent::LoopGen++;
    if ($MLAgent::UpdateSchedule !$= "") {
        cancel($MLAgent::UpdateSchedule);
        $MLAgent::UpdateSchedule = "";
    }

    echo("MLAgent: Starting update loop at " @ (1000 / $MLAgent::UpdateInterval) @ " Hz");
    $MLAgent::Enabled = true;
    $MLAgent::StepCount = 0;
    $MLAgent::EpisodeStartTime = getRealTime();

    // Speed up game simulation for faster training (skip in diagnostic mode)
    if (!$MLAgent::DiagnosticMode) {
        setTimeScale($MLAgent::TrainingSpeed);
        echo("MLAgent: Set game speed to " @ $MLAgent::TrainingSpeed @ "x for faster training");
    } else {
        echo("MLAgent: DIAGNOSTIC MODE — normal speed, player controls marble");
    }

    // Initialize state
    $MLAgent::LastGemScore = PlayGui.gemCount;
    $MLAgent::GameStartGemScore = PlayGui.gemCount;  // Baseline for full-game gem count
    $MLAgent::WasOOB = false;
    $MLAgent::EpisodeShouldEnd = false;
    // live play: a loop that (re)starts after GO keeps the round started (onTimerStart set $MLAgent::GoSeen)
    $MLAgent::TimerStarted = $MLAgent::Live && $MLAgent::GoSeen;
    $MLAgent::LiveSending = false;
    $MLAgent::CountdownTicks = 0;       // the countdown settle restarts every round (see MLAgent::inCountdown)

    MLAgent::update($MLAgent::LoopGen);
}

// ---- MLAgent::update (live version; the training version is in ../mlAgent.cs) ----
function MLAgent::update(%gen) {
    if (%gen !$= $MLAgent::LoopGen) {
        return;                  // an update from a superseded loop
    }
    if (!$MLAgent::Enabled) {
        if ($MLAgent::Live)
            echo("MLAgent: update loop stopped (agent disabled)");
        return;
    }

    // Enforce time scale every update (game code may reset it)
    // Skip in diagnostic mode — player controls game speed
    if (!$MLAgent::DiagnosticMode && getTimeScale() != $MLAgent::TrainingSpeed) {
        setTimeScale($MLAgent::TrainingSpeed);
    }

    // Check if we're in a valid game state
    // Live play: the game may be a client of a remote server, where $Game::Running (set by the server scripts) is never
    // true; the countdown / GO gate below decides when to act instead.
    if (!isObject($MP::MyMarble) || (!$Game::Running && !$MLAgent::Live)) {
        if ($MLAgent::Live) {
            if (!isObject($MP::MyMarble))
                MPgetMyMarble();      // the marble is recreated between the lobby and the round
            MLAgent::liveWaitNote("no marble or the game is not running");
        }
        // Not in game, try again later
        $MLAgent::UpdateSchedule = schedule($MLAgent::UpdateInterval, 0, "MLAgent::update", $MLAgent::LoopGen);
        return;
    }

    // The marble object is recreated on spawn, so re-apply the fixed view yaw per marble.
    if ($MLAgent::ViewYawOn && $MLAgent::ViewYawApplied !$= $MP::MyMarble.getId()) {
        $MP::MyMarble.setViewYaw($MLAgent::ViewYaw);
        $MLAgent::ViewYawApplied = $MP::MyMarble.getId();
    }

    // Wait for the timer to actually start before collecting observations.
    // After restartLevel, there's a brief window where $Game::Running is true
    // but the timer still shows 300,000ms (expired from last round). Steps
    // taken during this dead zone produce garbage data (marble at origin,
    // zero velocity, expired timer). Wait until currentTime < total time.
    // Only applies at episode start — once we've seen a valid timer, we let
    // the episode run to natural completion and send done=1 normally.
    //
    // PRE-SPIN (2026-09-26, operator): the Ready/Set countdown is PLAYED, not skipped. During it the
    // marble sits on its pad in Start mode, where the engine zeroes friction and horizontal velocity but
    // still applies the control torque (marble.cc, mMode == 2), so input spins the marble in place and GO
    // turns that spin into speed, as a human does. MLAgent::inCountdown() tells the countdown apart from
    // the dead zone above. Recording / diagnostic mode keeps the old behaviour.
    // Live play: act from GO (onTimerStart, or the game state "go" a client receives), and spin during the Ready/Set
    // countdown; never from the clock alone (a joined client's clock and MissionInfo.time need not match).
    if ($MLAgent::Live && !$MLAgent::TimerStarted) {
        if ($Game::State $= "go" || $Game::State $= "play") {
            $MLAgent::TimerStarted = true;
            $MLAgent::GoSeen = true;
        } else if (!MLAgent::inCountdown()) {
            MLAgent::liveWaitNote("waiting for the countdown or GO");
            $MLAgent::UpdateSchedule = schedule($MLAgent::UpdateInterval, 0, "MLAgent::update", $MLAgent::LoopGen);
            return;
        }
    } else if (!$MLAgent::TimerStarted && isObject(MissionInfo) && MissionInfo.time > 0) {
        if (PlayGui.currentTime >= MissionInfo.time) {
            if ($MLAgent::DiagnosticMode || !MLAgent::inCountdown()) {
                if ($MLAgent::Live)
                    MLAgent::liveWaitNote("the round has not started");
                $MLAgent::UpdateSchedule = schedule($MLAgent::UpdateInterval, 0, "MLAgent::update", $MLAgent::LoopGen);
                return;
            }
        } else {
            $MLAgent::TimerStarted = true;
        }
    }

    // 1. Collect observation
    %obs = AIObserver::collectState();

    // If this is the OOB step, override the position with the saved
    // edge position so the network associates the penalty with the edge,
    // not the spawn point it just respawned to.
    if ($MLAgent::WasOOB && $MLAgent::OOBPosX !$= "") {
        %obs.selfPosX = $MLAgent::OOBPosX;
        %obs.selfPosY = $MLAgent::OOBPosY;
        %obs.selfPosZ = $MLAgent::OOBPosZ;
        $MLAgent::OOBPosX = "";
    }

    // 2. Capture OOB flag and clear it
    %oobFlag = $MLAgent::WasOOB ? 1 : 0;
    $MLAgent::WasOOB = false;

    // 3. Compute gem delta (Python computes the actual reward)
    %currentGemScore = PlayGui.gemCount;
    %gemDelta = %currentGemScore - $MLAgent::LastGemScore;
    $MLAgent::LastGemScore = %currentGemScore;

    // 4. Check if episode is done
    %done = MLAgent::checkDone();

    // 5. Build message: obs_json|gemDelta|oob|done
    %json = AIObserver::serializeToJSON(%obs);
    %msg = %json @ "|" @ %gemDelta @ "|" @ %oobFlag @ "|" @ %done;

    // Recording mode: append what the human is doing this tick so Python can
    // turn it into the agent's action format (see ml_agent/record_demos.py):
    //   |forward,backward,left,right,jump,usePowerup,cameraYaw
    // Movement values are the engine's move inputs (0/1 from keys, scaled by
    // $Game::MovementSpeedMultiplier); jump/usePowerup are 1 while held. The
    // camera yaw is the human's live camera, needed to rotate their key
    // direction into the fixed frame the observations are in.
    if ($MLAgent::RecordMode) {
        %msg = %msg @ "|" @ ($mvForwardAction + 0) @ "," @ ($mvBackwardAction + 0) @ ","
                    @ ($mvLeftAction + 0) @ "," @ ($mvRightAction + 0) @ ","
                    @ ($mvTriggerCount2 + 0) @ "," @ ($mvTriggerCount0 + 0) @ ","
                    @ ($MP::MyMarble.getCameraYaw() + 0);
    }

    // Tick number: always the last field, so the reply can name the tick it
    // answers (fixed action delay, see $MLAgent::ActionDelay).
    $MLAgent::Tick++;
    %msg = %msg @ "|" @ $MLAgent::Tick;

    // 6. Send to Python server and get action
    if ($MLAgent::Live && !$MLAgent::LiveSending) {
        $MLAgent::LiveSending = true;
        echo("MLAgent: live play: observations flowing (state " @ $Game::State @ ", clock " @ PlayGui.currentTime @ ")");
    }
    AIBridge::sendState(%msg);
    // Engine lockstep (built engine only, $AI::Lockstep): the simulation now stands still
    // until the reply to this observation arrives (socketBridge.cs clears the flag).
    if ($AI::Lockstep && $AI::FixedStepMs > 0 && $AIBridge::Connected)
        $AI::WaitReply = true;
    // Fixed delay: the reply queued for this tick becomes the action now;
    // with nothing queued the previous action stays in force (held).
    if ($AIBridge::Queue[$MLAgent::Tick] !$= "") {
        $AIBridge::LastAction = $AIBridge::Queue[$MLAgent::Tick];
        $AIBridge::Queue[$MLAgent::Tick] = "";
    } else if ($AIBridge::DelayedMode) {
        $AIBridge::HeldTicks++;
    }
    %actionStr = $AIBridge::LastAction;
    if ($AIBridge::ControlHead < $AIBridge::ControlTail) {
        // one queued control word per tick, in arrival order (see socketBridge.cs)
        %actionStr = $AIBridge::ControlQueue[$AIBridge::ControlHead];
        $AIBridge::ControlQueue[$AIBridge::ControlHead] = "";
        $AIBridge::ControlHead++;
    }
    if ($MLAgent::Lockstep && !$MLAgent::DiagnosticMode && $AIBridge::Connected)
        setTimeScale($MLAgent::LockstepScale);   // near-frozen until onLine() delivers the reply

    // Control replies from the Python side (not actions; consumed once):
    //   "SPEED n"                       set the game speed (the trainer can now
    //                                   choose 6x itself; no console needed)
    //   "TELEPORT x y z [vx vy vz]"     move the marble (listen server: the same
    //                                   call cannon.cs uses); replay_demo.py uses
    //                                   it to start from a recorded position
    //   "FIXEDSTEP n"    built engine: advance the sim exactly n ms per frame (0 = normal timing);
    //                    the observation interval follows it (one observation per step)
    //   "LOCKSTEP 0|1"   built engine: hold the sim until each observation is answered
    //   "RENDEREVERY n"  built engine: render one frame in n while in fixed-step mode
    if (getWord(%actionStr, 0) $= "FIXEDSTEP") {
        %n = getWord(%actionStr, 1) + 0;
        $AI::FixedStepMs = %n;
        $MLAgent::UpdateInterval = %n > 0 ? %n : 16;
        echo("MLAgent: AI::FixedStepMs = " @ $AI::FixedStepMs @ ", update interval " @ $MLAgent::UpdateInterval @ " ms (Python server)");
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "LOCKSTEP") {
        $AI::Lockstep = (getWord(%actionStr, 1) + 0) > 0;
        $AI::WaitReply = false;
        echo("MLAgent: AI::Lockstep = " @ $AI::Lockstep @ " (Python server)");
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "VIEWYAW") {
        // VIEWING ONLY (engine change 2026-09-20, Marble.setViewYaw): render the third-person
        // camera at a FIXED yaw while the marble's own camera yaw keeps rotating every decision
        // to align the input square with the commanded direction (the sqrt(2) "two keys" force).
        // Render-only in the engine; physics and observations untouched. "VIEWYAW off" restores.
        %v = getWord(%actionStr, 1);
        if (%v $= "off") {
            $MLAgent::ViewYawOn = false;
            if (isObject($MP::MyMarble)) $MP::MyMarble.clearViewYaw();
        } else {
            $MLAgent::ViewYawOn = true;
            $MLAgent::ViewYaw = %v + 0;
            $MLAgent::ViewYawApplied = "";        // force re-apply on the next tick
        }
        echo("MLAgent: ViewYaw " @ (%v $= "off" ? "off" : $MLAgent::ViewYaw) @ " (Python server)");
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "RENDEREVERY") {
        $AI::RenderEvery = getWord(%actionStr, 1) + 0;
        echo("MLAgent: AI::RenderEvery = " @ $AI::RenderEvery @ " (Python server)");
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "SPEED") {
        %spd = getWord(%actionStr, 1) + 0;
        if (%spd > 0) {
            $MLAgent::TrainingSpeed = %spd;
            if (!$MLAgent::DiagnosticMode)
                setTimeScale(%spd);
            echo("MLAgent: game speed set to " @ %spd @ "x by the Python server");
        }
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "SPINWAIT") {
        // "SPINWAIT ms" (2026-09-29): in lockstep, busy-wait this many ms for the reply before the
        // engine sleeps 1 ms at a time (built engine $AI::SpinWaitMs, default 4). Tried against the
        // locked-session stall of 2026-09-29 (log 34): not the cause; harmless, a learned_nav session sets it.
        $AI::SpinWaitMs = getWord(%actionStr, 1) + 0;
        echo("MLAgent: AI::SpinWaitMs = " @ $AI::SpinWaitMs @ " (Python server)");
        AIBridge::sendState("DEBUG|spinwait=" @ $AI::SpinWaitMs @ "|lockstep=" @ $AI::Lockstep @ "|fixedstep=" @ $AI::FixedStepMs);
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "SLOWLOG") {
        // "SLOWLOG ms" (2026-09-30, diagnostic): main-loop stages slower than this are logged by the engine
        $AI::SlowLogMs = getWord(%actionStr, 1) + 0;
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "SLEEPTIME") {
        // Probe control: background sleep pref (ms per frame when unfocused)
        $Pref::backgroundSleepTime = getWord(%actionStr, 1) + 0;
        echo("MLAgent: backgroundSleepTime set to " @ $Pref::backgroundSleepTime @ " by the Python server");
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "MAXFPS") {
        // Probe control: frame-rate cap (setMaxFPS takes a frame period in ms; 0 = uncapped)
        %fps = getWord(%actionStr, 1) + 0;
        setMaxFPS(%fps > 0 ? 1000 / %fps : 0);
        echo("MLAgent: setMaxFPS for " @ %fps @ " fps by the Python server");
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "TELEPORT") {
        if (isObject($MP::MyMarble)) {
            %tp = getWord(%actionStr, 1) SPC getWord(%actionStr, 2) SPC getWord(%actionStr, 3) SPC "1 0 0 0";
            $MP::MyMarble.setTransform(%tp);
            if (getWord(%actionStr, 4) !$= "")
                $MP::MyMarble.setVelocity(getWord(%actionStr, 4) SPC getWord(%actionStr, 5) SPC getWord(%actionStr, 6));
            // a teleported marble kept its SPIN, which turned into a roll in an arbitrary
            // direction on landing (the navigator's "wrong direction" segments, 2026-09-17).
            // Reset both the client marble and the server-side player object.
            // 2026-09-26 (physics plan): optional words 7-9 set the spin instead of zeroing it,
            // so a teleported marble can start rolling without skidding: TELEPORT x y z vx vy vz wx wy wz
            %spin = (getWord(%actionStr, 7) !$= "") ? (getWord(%actionStr, 7) SPC getWord(%actionStr, 8) SPC getWord(%actionStr, 9)) : "0 0 0";
            $MP::MyMarble.setAngularVelocity(%spin);
            if (isObject(ClientGroup) && ClientGroup.getCount() > 0) {
                %scl = ClientGroup.getObject(0);
                if (isObject(%scl) && isObject(%scl.player) && %scl.player != $MP::MyMarble) {
                    %scl.player.setTransform(%tp);
                    %scl.player.setVelocity(getWord(%actionStr, 4) !$= "" ? (getWord(%actionStr, 4) SPC getWord(%actionStr, 5) SPC getWord(%actionStr, 6)) : "0 0 0");
                    %scl.player.setAngularVelocity(%spin);
                }
            }
            echo("MLAgent: teleported to " @ %tp @ " by the Python server");
        }
        $AIBridge::LastAction = "";
        %actionStr = "";
    }

    //   "STATS"                        send "STATS|onTime,late,held,delay,tick"
    //                                   back on the socket and reset the
    //                                   fixed-delay counters (probes use it)
    //   "DELAY n"                       set $MLAgent::ActionDelay (ticks)
    //   "RESPAWN"                      force a server-side respawn of the marble
    //   "OOBCLICK"                     the legal quick respawn a human gets by clicking while OOB
    //   "MARK x y z"                   show the navigator's waypoint as a pad (cosmetic)
    //   "GEMRESET"                     drill maps: respawn the gem group if no gem is up
    //   "CONTACT 1|0"                  append the engine's contact telemetry to each observation
    if (getWord(%actionStr, 0) $= "STATS") {
        AIBridge::sendState("STATS|" @ $AIBridge::DelayedReplies @ "," @ $AIBridge::LateReplies @ ","
                            @ $AIBridge::HeldTicks @ "," @ $MLAgent::ActionDelay @ "," @ $MLAgent::Tick);
        $AIBridge::DelayedReplies = 0; $AIBridge::LateReplies = 0; $AIBridge::HeldTicks = 0;
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "INFO") {
        // "INFO" -> "INFO|<mission file>|<round ms>|<time scale>" on the socket
        // (the navigator trainer picks the terrain map from the mission file)
        %mf = isObject(MissionInfo) && MissionInfo.file !$= "" ? MissionInfo.file : $Server::MissionFile;
        AIBridge::sendState("INFO|" @ fileBase(%mf) @ "|" @ (isObject(MissionInfo) ? MissionInfo.time : 0) @ "|" @ getTimeScale());
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "RESPAWN") {
        // Force a respawn on the (listen) server: after some falls the game's own
        // OOB respawn never comes and TELEPORT is ignored while the marble is in
        // the OOB state (seen 2026-09-17 on the flat training map).
        if (isObject(ClientGroup) && ClientGroup.getCount() > 0) {
            %cl = ClientGroup.getObject(0);
            if (isObject(%cl) && isObject(%cl.player)) {
                // respawnPlayer() refuses ("Spawning blocked") for 300 ms after any
                // respawn; clear that and the OOB respawn still scheduled by the game so
                // it cannot yank the marble back to the spawn during the next segment.
                cancel(%cl.respawnSchedule);
                %cl.unblockSpawning();
                %cl.player.setOOB(false);
                %cl.isOOB = false;
                %before = %cl.player.getPosition();
                %cl.respawnPlayer();
                echo("MLAgent: forced respawn by the Python server: server marble " @ %cl.player @ " " @ %before
                     @ " -> " @ %cl.player.getPosition() @ "; client marble " @ $MP::MyMarble @ " " @ $MP::MyMarble.getPosition()
                     @ "; blocked " @ %cl.spawningBlocked @ " state " @ $Game::State);
            }
        }
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "OOBCLICK") {
        // THE LEGAL QUICK RESPAWN. A human who falls out of bounds clicks the left mouse as soon
        // as the "Out of Bounds" text appears and comes straight back, instead of waiting out the
        // game's automatic respawn. Their click runs
        //     input_mouseFire -> commandToServer('MouseFire') -> serverCmdMouseFire
        //       -> MPOutofBounds() -> if (%client.isOOB) %client.respawnFromOOB()
        // (server/scripts/mp/server.cs:144). This reproduces exactly that, and nothing else.
        //
        // WHY THE isOOB GATE IS THE WHOLE POINT. In GameConnection::outOfBounds
        // (server/scripts/game.cs) three things happen in ONE synchronous function, in order:
        //     %this.isOOB = true;                              // source comment: "used for OOB Click"
        //     %this.setMessage("outOfBounds", 2000);           // the on-screen text
        //     %this.schedule(2500, respawnFromOOB);            // the automatic respawn
        // so isOOB turns true on the same tick the text appears. Gating on it is EXACTLY the gate
        // a human's click passes, not an approximation, and it cannot fire earlier than a human
        // could legally click because the flag is false until that function runs. The operator's
        // requirement (2026-09-21) was that training must never use a respawn a real competitive
        // Hunt game would refuse.
        //
        // This is NOT the same as the RESPAWN control above, which force-clears isOOB and
        // respawns unconditionally. That one is a stuck-marble fallback and is not legal play.
        // It is also not serverCmdQuickRespawn (the respawn KEY), which the game explicitly
        // blocks in competitive Hunt: (!$MPPref::Server::CompetitiveMode || !$Game::isMode["hunt"]).
        // The mouse OOB path carries no such check, which is why it is the allowed one.
        //
        // MEASURED WORTH, KOTM 2 rounds 2026-09-21: 4.5 falls a round, dead time per fall is
        // bimodal at 0.83 s and 3.26 s, and the 2.43 s gap between the clusters IS the 2500 ms
        // schedule. Recovering it is ~7.3 s a round, 4.0 % of a 3.02 min round, ~3 gems.
        if (isObject(ClientGroup) && ClientGroup.getCount() > 0) {
            %cl = ClientGroup.getObject(0);
            if (isObject(%cl) && %cl.isOOB) {
                %cl.respawnFromOOB();
                $MLAgent::OOBClicks ++;
            }
        }
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "MARK") {
        // "MARK x y z": show the navigator's current waypoint as a start pad (cosmetic,
        // for watching training). Server-side object in the listen server; recreated
        // after every level restart.
        %mx = getWord(%actionStr, 1); %my = getWord(%actionStr, 2); %mz = getWord(%actionStr, 3);
        if (%mx $= "off") {
            // "MARK off": hide the marker. No gem on the map means nothing is drawn anywhere,
            // because a waypoint may only ever be shown inside a gem (user rule 2026-09-20).
            if (isObject($MLAgent::Marker))
                $MLAgent::Marker.hide(true);
        } else {
            if (!isObject($MLAgent::Marker)) {
                // a real black gem item (registered datablock), same size as the map's gems;
                // Gem::onPickup (server/scripts/gems.cs) ignores items flagged aiMarker
                $MLAgent::Marker = new Item() {
                    dataBlock = "AIMarkerGem";       // black gem look, className AIMarker (gems.cs)
                    position = %mx SPC %my SPC %mz;      // EXACTLY the waypoint: no cosmetic lift
                    rotation = "1 0 0 0";
                    scale = "1 1 1";
                    collideable = "0";
                    static = "1";
                    rotate = "1";
                    aiMarker = "1";
                };
                if (isObject(MissionGroup))
                    MissionGroup.add($MLAgent::Marker);
                echo("MLAgent: waypoint marker " @ $MLAgent::Marker @ " created at " @ %mx SPC %my SPC %mz);
            } else {
                $MLAgent::Marker.setTransform(%mx SPC %my SPC %mz SPC "1 0 0 0");
            }
            $MLAgent::Marker.hide(false);         // in case anything hid it
        }
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "DEBUG") {
        // "DEBUG" -> frame diagnostics on the socket: client marble yaw vs server marble yaw,
        // $cameraYaw, $mvYaw, gravity, ids (used when the trainer detects a flipped frame)
        %cl = (isObject(ClientGroup) && ClientGroup.getCount() > 0) ? ClientGroup.getObject(0) : 0;
        %sp = (isObject(%cl) && isObject(%cl.player)) ? %cl.player : 0;
        %d = "DEBUG|client=" @ $MP::MyMarble @ "|clientYaw=" @ (isObject($MP::MyMarble) ? $MP::MyMarble.getCameraYaw() : "?")
           @ "|server=" @ %sp @ "|serverYaw=" @ (isObject(%sp) ? %sp.getCameraYaw() : "?")
           @ "|cameraYaw=" @ $cameraYaw @ "|mvYaw=" @ $mvYaw
           @ "|clientPos=" @ (isObject($MP::MyMarble) ? $MP::MyMarble.getPosition() : "?")
           @ "|serverPos=" @ (isObject(%sp) ? %sp.getPosition() : "?")
           @ "|control=" @ (isObject(ServerConnection) ? ServerConnection.getControlObject() : "?")
           @ "|gravRot=" @ $Game::GravityRot @ "|oob=" @ (isObject(%cl) ? %cl.isOOB : "?")
           // 2026-09-26 (physics plan): spin and velocity of both marble objects
           @ "|clientOmega=" @ (isObject($MP::MyMarble) ? $MP::MyMarble.getAngularVelocity() : "?")
           @ "|serverOmega=" @ (isObject(%sp) ? %sp.getAngularVelocity() : "?")
           @ "|clientVel=" @ (isObject($MP::MyMarble) ? $MP::MyMarble.getVelocity() : "?")
           @ "|serverVel=" @ (isObject(%sp) ? %sp.getVelocity() : "?");
        echo("MLAgent: DEBUG requested -> " @ %d);
        AIBridge::sendState(%d);
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "RADIUS") {
        // "RADIUS" (2026-10-02, POWERUP_PLAN phase 2, read-only): the marble's collision radius and powerup
        // state as a DEBUG line, for the mega marble measurements
        %pl = (isObject(ClientGroup) && ClientGroup.getCount() > 0) ? ClientGroup.getObject(0).player : -1;
        AIBridge::sendState("DEBUG|radius=" @ $MP::MyMarble.getCollisionRadius()
            @ "|mega=" @ (isObject(%pl) ? (%pl.megaMarble ? 1 : 0) : -1)
            @ "|camyaw=" @ $MP::MyMarble.getCameraYaw()
            @ "|srvyaw=" @ (isObject(%pl) ? %pl.getCameraYaw() : -99)
            @ "|globalyaw=" @ $cameraYaw @ "|mvyaw=" @ $mvYaw
            @ "|blast=" @ $MP::BlastValue @ "|special=" @ ($MP::SpecialBlast ? 1 : 0)
            @ "|held=" @ ((isObject(%pl) && isObject(%pl.powerUpData)) ? %pl.powerUpData.getName() : "none")
            @ "|act=" @ (isObject(%pl) ? (%pl.powerupActive[1] + 0) @ (%pl.powerupActive[2] + 0) @ (%pl.powerupActive[3] + 0) @ (%pl.powerupActive[4] + 0) @ (%pl.powerupActive[5] + 0) @ (%pl.powerupActive[6] + 0) : "")
            @ "|sched=" @ (isObject(%pl) ? (isEventPending(%pl.powerupSchedule[1]) ? 1 : 0) @ (isEventPending(%pl.powerupSchedule[2]) ? 1 : 0) @ (isEventPending(%pl.powerupSchedule[3]) ? 1 : 0) @ (isEventPending(%pl.powerupSchedule[4]) ? 1 : 0) @ (isEventPending(%pl.powerupSchedule[5]) ? 1 : 0) @ (isEventPending(%pl.powerupSchedule[6]) ? 1 : 0) : ""));
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "GIVEPOW") {
        // "GIVEPOW <ItemDatablock>|none" (2026-10-04, log 40.27): the Super Speed curriculum's Stage 2 drills start
        // with the powerup held as if just picked up: the server's own pickup path (Marble::setPowerUp sets
        // powerUpData, the engine id the use key fires and the client's HUD) plus the client prediction that the mp
        // item pickup also sets; "none" clears both
        %pl = (isObject(ClientGroup) && ClientGroup.getCount() > 0) ? ClientGroup.getObject(0).player : -1;
        %db = getWord(%actionStr, 1);
        if (isObject(%pl)) {
            if (%db !$= "none" && isObject(%db)) {
                %pl.setPowerUp(%db.getId(), true, 0);
                if (isObject($MP::MyMarble))
                    $MP::MyMarble._setPowerUp(%db.getId(), true, 0);
            } else {
                %pl.powerUpData = "";
                %pl.heldPowerup = "";
                %pl.setPowerUpId(0, true);
                if (isObject($MP::MyMarble))
                    $MP::MyMarble._setPowerUp("", true, 0);
            }
        }
        AIBridge::sendState("DEBUG|givepow=" @ %db @ "|held=" @ ((isObject(%pl) && isObject(%pl.powerUpData)) ? %pl.powerUpData.getName() : "none"));
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "YAWSET") {
        // "YAWSET yaw mode" (2026-10-02, diagnostic): mode 0 = client marble setCameraYaw, 1 = server marble
        // setCameraYaw, 2 = $mvYaw delta (the move's yaw), 3 = all three; RADIUS reads the yaws back
        %y = getWord(%actionStr, 1) + 0; %mode = getWord(%actionStr, 2) + 0;
        %pl = (isObject(ClientGroup) && ClientGroup.getCount() > 0) ? ClientGroup.getObject(0).player : -1;
        if (%mode == 0 || %mode == 3) { $MP::MyMarble.setCameraYaw(%y); $cameraYaw = %y; }
        if ((%mode == 1 || %mode == 3) && isObject(%pl)) %pl.setCameraYaw(%y);
        if (%mode == 2 || %mode == 3) { $MLAgent::YawDelta = %y - $MP::MyMarble.getCameraYaw(); }
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "MARBLES") {
        // "MARBLES" (2026-09-29, diagnostic, read-only): every Marble object on the server (MissionCleanup,
        // MissionGroup) and on the client (ServerConnection ghosts), as a DEBUG line:
        // DEBUG|server=id:pos:client;...|client=id:pos;...|mine=<$MP::MyMarble>|player=<client 0's player>
        %s = "";
        for (%g = 0; %g < 2; %g ++) {
            %grp = getWord("MissionCleanup MissionGroup", %g);
            if (isObject(%grp)) {
                for (%i = 0; %i < %grp.getCount(); %i ++) {
                    %o = %grp.getObject(%i);
                    if (%o.getClassName() $= "Marble")
                        %s = %s @ %o @ ":" @ %o.getPosition() @ ":" @ %o.client @ ";";
                }
            }
        }
        %c = "";
        if (isObject(ServerConnection)) {
            for (%i = 0; %i < ServerConnection.getCount(); %i ++) {
                %o = ServerConnection.getObject(%i);
                if (%o.getClassName() $= "Marble")
                    %c = %c @ %o @ ":" @ %o.getPosition() @ ";";
            }
        }
        %cl = (isObject(ClientGroup) && ClientGroup.getCount() > 0) ? ClientGroup.getObject(0) : 0;
        AIBridge::sendState("DEBUG|server=" @ %s @ "|client=" @ %c @ "|mine=" @ $MP::MyMarble
            @ "|player=" @ (isObject(%cl) ? %cl.player : "?") @ "|clients=" @ (isObject(ClientGroup) ? ClientGroup.getCount() : 0));
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "DELAY") {
        $MLAgent::ActionDelay = getWord(%actionStr, 1) + 0;
        echo("MLAgent: action delay set to " @ $MLAgent::ActionDelay @ " ticks by the Python server");
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "GEMRESET") {
        // "GEMRESET" (2026-09-27, jump physics P0 drills): if no hunt gem is spawned, spawn the gem group
        // again. The game respawns a group EXCLUDING the gem just collected, so a one-gem drill map
        // (kotmjump_p0) would stay empty after its first pickup. Does nothing while a gem is up.
        if ($Hunt::CurrentGemCount <= 0)
            spawnHuntGemGroup();
        $AIBridge::LastAction = "";
        %actionStr = "";
    } else if (getWord(%actionStr, 0) $= "CONTACT") {
        // "CONTACT 1|0" (2026-09-27, jump physics stage 3): append the engine's contact telemetry
        // (Marble::getContactTelemetry, 13 numbers) after the 38 observation numbers. Off by default.
        $AIObserver::ContactTelemetry = getWord(%actionStr, 1) + 0;
        if ($AIObserver::ContactTelemetry && isObject($MP::MyMarble))
            $MP::MyMarble.getContactTelemetry();          // start the first summary clean
        $AIBridge::LastAction = "";
        %actionStr = "";
    }

    // Recording handshake: record_demos.py answers every message with "RECORD".
    // The first time we see it, switch into recording mode ourselves (you
    // control the marble at 1x, observations pinned to the fixed frame, your
    // inputs appended to each message). No console command needed. If a
    // numeric action shows up instead, the trainer is connected: switch back.
    if (%actionStr $= "RECORD") {
        if (!$MLAgent::RecordMode)
            MLAgent::enableRecording();
    } else if ($MLAgent::RecordMode && %actionStr !$= "" && strPos(%actionStr, ",") != -1) {
        MLAgent::disableRecording();
    }

    // 7. Parse and execute action (skip in diagnostic mode — player controls marble)
    if (%actionStr !$= "" && !$MLAgent::DiagnosticMode) {
        MLAgent::executeAction(%actionStr);
    }

    // 8. Clean up observation object
    %obs.delete();

    // Increment step counter
    $MLAgent::StepCount++;

    // 9. If done, reset for next episode (game will restart automatically in Hunt mode)
    if (%done) {
        MLAgent::resetEpisode();
    }

    // 10. Schedule next update
    $MLAgent::UpdateSchedule = schedule($MLAgent::UpdateInterval, 0, "MLAgent::update", $MLAgent::LoopGen);
}

// ---- MLAgent::checkDone (live version; the training version is in ../mlAgent.cs) ----
function MLAgent::checkDone() {
    // Episode ends when:

    // 1. Time runs out (Hunt mode: currentTime counts UP from 0)
    //    The dead-zone guard in update() prevents observations before the timer
    //    starts, so we only need a small safety margin (10 steps) here.
    //    Only once the clock has started (2026-09-26): during the pre-spin countdown the clock shows the
    //    full round length, which this test would otherwise read as "time is up" every step.
    //    Live play (-ailive): the round's end comes from the server (clientCmdGameEnd -> onGameEnd), whatever
    //    its time limit, so neither this clock test (it reads MissionInfo.time) nor the step cap applies.
    if ($MLAgent::Live)
        return 0;
    if (isObject(MissionInfo) && MissionInfo.time > 0) {
        if ($MLAgent::TimerStarted && PlayGui.currentTime >= MissionInfo.time && $MLAgent::StepCount > 10) {
            return 1;
        }
    }

    // 2. Hard step-count cap — backup if the time check fails.
    //    Actual 5-min round at 3x takes ~11,873 steps (not 6,250 as originally estimated).
    //    15,000 gives a comfortable buffer above that.
    if ($MLAgent::StepCount >= 15000) {
        echo("MLAgent: Hit max step cap (15000), forcing episode end");
        return 1;
    }

    // 3. All gems collected — REMOVED for Hunt mode.
    //    In Hunt, PlayGui.gemCount is cumulative points scored (never resets mid-round)
    //    while PlayGui.maxGems is the number of gem slots on the map (~7).
    //    Once score >= 7, this was permanently true, creating an 11-step episode
    //    flood (StepCount > 10 guard = 11 steps, then done=1 fires every time).
    //    Timer expiry + 7000-step cap are sufficient episode boundaries.

    return 0;
}

// ---- MLAgent::liveWaitNote (live play only) ----
// Live play (-ailive): say why the agent is not acting yet, at most every 3 s (read console.log; launch with -log 1)
function MLAgent::liveWaitNote(%why) {
    %now = getRealTime();
    if (%now - $MLAgent::LastWaitNote < 3000)
        return;
    $MLAgent::LastWaitNote = %now;
    %note = "waiting (" @ %why @ "): marble " @ isObject($MP::MyMarble) @ ", running " @ ($Game::Running + 0)
         @ ", lobby " @ ($Server::Lobby + 0) @ ", state " @ $Game::State @ ", timer started " @ ($MLAgent::TimerStarted + 0)
         @ ", GO seen " @ ($MLAgent::GoSeen + 0) @ ", clock " @ PlayGui.currentTime @ " of " @ MissionInfo.time
         @ ", connected " @ ($AIBridge::Connected + 0);
    echo("MLAgent: " @ %note);
    // also to the Python side, which prints DEBUG lines in live mode (the engine buffers console.log)
    if ($AIBridge::Connected)
        AIBridge::sendState("DEBUG|live|" @ %note);
}

// ---- MLAgent::onTimerStart (live version; the training version is in ../mlAgent.cs) ----
function MLAgent::onTimerStart() {
    // Called when "GO!" appears and timer starts
    // Live play (-ailive, 2026-10-04): the round counts as started at GO whatever the server's time limit. The
    // clock test in update() compares with MissionInfo.time (3:00 on KOTM), so on a server with a longer limit
    // the agent sat still until the clock fell below 3:00.
    if ($MLAgent::Live) {
        $MLAgent::GoSeen = true;
        $MLAgent::TimerStarted = true;
        echo("MLAgent: GO (live play): the agent drives from now on");
    }
    if (!$MLAgent::AutoStart || !$MLAgent::ReadyToStart)
        return;

    $MLAgent::ReadyToStart = false;
    echo("MLAgent: GO! Starting ML agent...");

    schedule(100, 0, "MLAgent::start");
}

// ---- MLAgent::onGameEnd (live version; the training version is in ../mlAgent.cs) ----
function MLAgent::onGameEnd() {
    if ($AIBridge::DelayedMode) {
        echo("MLAgent: action delay " @ $MLAgent::ActionDelay @ " ticks: " @ $AIBridge::DelayedReplies
             @ " replies on time, " @ $AIBridge::LateReplies @ " late, " @ $AIBridge::HeldTicks @ " ticks held");
        $AIBridge::DelayedReplies = 0; $AIBridge::LateReplies = 0; $AIBridge::HeldTicks = 0;
    }
    // Send game-end signal with the TOTAL gems collected this entire game.
    // Uses GameStartGemScore (set at round start, never reset by resetEpisode)
    // so the Python side gets the accurate full-game gem count.
    // The gem_delta field is repurposed here: negative = total game gems signal.
    // Protocol: []|<neg_total_game_gems>|0|1
    if ($MLAgent::Enabled && $AIBridge::Connected) {
        %totalGameGems = PlayGui.gemCount - $MLAgent::GameStartGemScore;
        // Send as negative to distinguish from normal per-step gem deltas.
        // Python checks for negative gem_delta on empty obs to record game total.
        %msg = "[]|" @ -%totalGameGems @ "|0|1";
        AIBridge::sendState(%msg);
    }

    // Live play: no restart from here; the lobby starts the next round and onGameStart starts the agent again.
    $MLAgent::GoSeen = false;
    if ($MLAgent::Live) {
        $MLAgent::Enabled = false;
        echo("MLAgent: Round ended (live play); waiting for the lobby to start the next round");
        return;
    }

    // Don't fully stop - just flag ready to restart. Restart whenever
    // auto-start is on, even if the trainer never connected this round, so an
    // unattended game keeps cycling rounds until a trainer shows up.
    if ($MLAgent::Enabled || $MLAgent::AutoStart) {
        echo("MLAgent: Round ended, auto-restarting in 2 seconds...");
        $MLAgent::Enabled = false;  // Temporarily disable updates

        // Auto-restart after brief delay
        schedule(2000, 0, "MLAgent::autoRestart");
    }
}

// ---- AIObserver::collectGems (live version; the training version is in ../observer.cs) ----
function AIObserver::collectGems(%obs) {
    %myPos = $MP::MyMarble.getPosition();
    %myPosX = getWord(%myPos, 0);
    %myPosY = getWord(%myPos, 1);
    %myPosZ = getWord(%myPos, 2);

    // Camera yaw for rotating world-relative vectors into camera space.
    // This ensures gem relX/relY align with the L/R and F/B action axes.
    // Read from marble's internal camera (same source as engine movement).
    %yawRad = ($AIObserver::ForceYaw !$= "") ? $AIObserver::ForceYaw : $MP::MyMarble.getCameraYaw();
    %cosYaw = mCos(%yawRad);
    %sinYaw = mSin(%yawRad);

    // WHERE GEMS COME FROM. This used to read ItemArray first and only fall back to
    // ServerConnection when ItemArray was EMPTY. ItemArray is a client-side cached snapshot built
    // by buildItemList() (client/scripts/mp/items.cs), which skips hidden items at build time, and
    // in single player nothing refreshes it: its only callers there are updateClientItems(), which
    // returns unless $Server::ServerType $= "MultiPlayer", and updateItemCollision(), which returns
    // on "SinglePlayer". So the snapshot went stale and, being non-empty, never triggered the
    // fallback.
    //
    // Measured 2026-09-21 on FlatGemTraining, which carries maxGemsPerSpawn = minGemsPerSpawn = 1,
    // so every pickup empties the map and the server replaces the gem synchronously inside the
    // same call (unspawnGem -> spawnHuntGemGroup -> spawnGem -> hide(false), no schedule anywhere
    // on that path). Despite the gem existing at once, 92 % of pickups were followed by a window
    // with NO gem in the observation: median 1.34 s, up to 1.92 s, 31 % of all decisions in the
    // round. The navigator has no gem to steer at during that window, falls back to a spawn-point
    // centroid a median 35 deg off the real gem, and 11 % of the round's travel closed on nothing.
    //
    // ServerConnection holds the LIVE ghosted objects, so a gem the server just unhid is visible
    // immediately. The per-object !isHidden() and classname $= "Gem" filters below are unchanged,
    // which matters because those filters are what keeps powerups and BackupGems out.
    // Set $AIObserver::GemSource = "itemarray" to restore the old behaviour.
    %gemCount = 0;

    // Instead of creating array, use direct storage in observation
    // This avoids creating/deleting temporary array objects every frame
    %count = 0;
    %useServerConnection = false;

    if ($AIObserver::GemSource !$= "itemarray" && isObject(ServerConnection)) {
        %count = ServerConnection.getCount();
        %useServerConnection = true;
    } else if (isObject(ItemArray)) {
        %count = ItemArray.getSize();  // Use getSize() not count()
    }

    // Last resort: whichever source we picked came back empty
    if (%count == 0 && isObject(ServerConnection)) {
        %count = ServerConnection.getCount();
        %useServerConnection = true;
    }

    if (!$AIObserver::LoggedGemCollection) {
        echo("AIObserver: " @ %count @ " objects from " @ (%useServerConnection ? "ServerConnection" : "ItemArray"));
        $AIObserver::LoggedGemCollection = true;
    }

    // Live play diagnostics (-ailive, 2026-10-04): what the gem scan sees on this client, every 5 s, to the Python side
    // (the engine buffers console.log)
    if ($MLAgent::Live && isObject(ServerConnection) && getRealTime() - $AIObserver::LastGemDiag > 5000) {
        $AIObserver::LastGemDiag = getRealTime();
        %nItems = 0; %nHidden = 0; %sample = "";
        %scn = ServerConnection.getCount();
        for (%k = 0; %k < %scn; %k++) {
            %o = ServerConnection.getObject(%k);
            if (!(%o.getType() & $TypeMasks::ItemObjectType))
                continue;
            %nItems++;
            if (%o.isHidden())
                %nHidden++;
            if (%nItems <= 4) {
                %d = %o.getDatablock();
                %sample = %sample @ " [" @ %o.getClassName() @ " db=" @ (isObject(%d) ? %d.getName() : "none")
                    @ " cls=" @ (isObject(%d) ? %d.classname : "") @ " shape=" @ (isObject(%d) ? fileBase(%d.shapeFile) : "")
                    @ " hid=" @ %o.isHidden() @ "]";
            }
        }
        if ($AIBridge::Connected)
            AIBridge::sendState("DEBUG|live|gems: sc " @ %scn @ ", items " @ %nItems @ ", hidden " @ %nHidden
                @ ", source " @ (%useServerConnection ? "SC" : "ItemArray") @ ", count " @ %count
                // 2026-10-06 (physics step check): the real game's physics step is its frame time
                @ ", fps " @ $fps::modded @ ", drawfps " @ $fps::draw @ ", maxfps pref " @ $pref::Video::MaxFPS @ %sample);
    }

    for (%i = 0; %i < %count; %i++) {
        if (%useServerConnection) {
            %obj = ServerConnection.getObject(%i);
            // ServerConnection carries every ghosted object, not just items, so reject non-items
            // on the cheap bitmask before touching the datablock or comparing any strings. Same
            // pre-filter buildItemList() uses, and it keeps this loop's cost near the old one.
            if (!(%obj.getType() & $TypeMasks::ItemObjectType))
                continue;
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
                // Live play on a joined client (-ailive, 2026-10-04): datablocks arrive without their script fields, so
                // classname is empty there and no gem matched. Accept the gem datablocks by name (the server sends the
                // names) or by a gem shape, excluding editor-only items (BackupGem shares the shape; renderEditor is
                // sent to clients).
                if (!%isGem && $MLAgent::Live && isObject(%datablock)) {
                    %dbn = %datablock.getName();
                    %isGem = (getSubStr(%dbn, 0, 7) $= "GemItem")
                        || (strstr(strlwr(%datablock.shapeFile), "gem") != -1 && !%datablock.renderEditor && %dbn !$= "BackupGem");
                }

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

    // DIAGNOSTIC, temporary. While $AIObserver::GemDiag > 0, also count what the OLD ItemArray
    // path would have found and echo every disagreement, spending one budget line each time. A
    // run of lines reading "server 1 itemarray 0" is the blind window being manufactured by the
    // stale cache, and is the direct proof that reading ServerConnection removes it. Set
    // $AIObserver::GemDiag = 0 once that is confirmed, so this second loop stops costing anything.
    if ($AIObserver::GemDiag > 0 && %useServerConnection && isObject(ItemArray)) {
        %altCount = 0;
        %altSize = ItemArray.getSize();
        for (%k = 0; %k < %altSize; %k++) {
            %altObj = nameToID(getField(ItemArray.getEntryByIndex(%k), 0));
            if (isObject(%altObj) && !%altObj.isHidden()) {
                %altDb = %altObj.getDatablock();
                if (isObject(%altDb) && %altDb.classname $= "Gem")
                    %altCount++;
            }
        }
        if (%altCount != %gemCount) {
            echo("AIObserver GEMDIAG: server " @ %gemCount @ " itemarray " @ %altCount @ " size " @ %altSize @ " t " @ PlayGui.currentTime);
            $AIObserver::GemDiag--;
        }
    }

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
