//------------------------------------------------------------------------------
// Free camera for watching the agent at 1x (2026-10-04, operator: "rotate the camera without affecting the
// model", "only unlock camera in the 1x speed version"; then "add the free camera fix to the live server version
// so we can move the camera in lobby games")
//
// Loaded ONLY when the game is launched with -aifreecam (hook in client/init.cs, right after mlAgent.cs), which
// ml_agent/watch_model.ps1 and ml_agent/play_live.ps1 pass. Training launchers never pass it, and the training
// bridge (ai/mlAgent.cs, ai/observer.cs) is not changed by this file.
//
// Why the mouse used to disturb the agent: the agent steers through the marble's own camera yaw. Every decision
// MLAgent::executeAction sets that yaw (setMarbleCamYaw) and the engine pushes F/B/L/R relative to it; Super
// Speed also fires along it. The picture is drawn at a separate render-only yaw (engine Marble.setViewYaw,
// pinned at 0 by mlAgent.cs). The game's mouse handler adds to $mvYaw, which turns the STEERING yaw between
// decisions, so moving the mouse bent the agent's moves while the picture stayed put.
//
// While the agent runs at 1x with every frame drawn (watch mode: SPEED 1 and RENDEREVERY 1 from nav.real_run
// with NAV_WATCH), and in live play (-ailive: always real time, the view pinned by ai/live/agentLive.cs) whenever
// the marble exists, this file sends the mouse and the turn keys (left/right arrows) to the render-only yaw
// instead. Pitch keeps the game's own handler: it only tilts the picture (getMarbleAxis and Super Speed use the
// yaw alone, and the model never reads pitch). The mouse buttons do nothing meanwhile, so the click that focuses
// the window cannot fire the agent's powerup (left button) or blast (right button). The keyboard otherwise works
// as usual: W/A/S/D and space still act on the marble, Esc pauses the round. Outside watch mode (menus,
// training speeds) every handler below behaves exactly as in default.bind.cs.
//------------------------------------------------------------------------------

$MLWatchCam::Yaw = "";        // the viewer's yaw, radians; "" until the viewer first turns (starts from the pinned view)
$MLWatchCam::TurnLeft = 0;    // held arrow keys, radians per move (the engine's $mvYawLeftSpeed units)
$MLWatchCam::TurnRight = 0;
echo("MLAgent: -aifreecam: the mouse and arrow keys turn the view only, while the agent plays at 1x");

function MLWatchCam::active() {
    if (!isObject($MP::MyMarble) || $MLAgent::TrainingSpeed != 1)
        return false;
    if ($MLAgent::Live)
        return true;        // live play: the agent drives from GO and the view stays pinned between rounds too
    return $MLAgent::Enabled && $AI::RenderEvery <= 1;
}

// Show the viewer's yaw. The bridge's own copy is set too, so its per-marble re-apply (MLAgent::update,
// MLAgent::viewYawWatch) keeps the viewer's yaw after a respawn instead of going back to 0.
function MLWatchCam::apply() {
    $MLAgent::ViewYawOn = true;
    $MLAgent::ViewYaw = $MLWatchCam::Yaw;
    $MP::MyMarble.setViewYaw($MLWatchCam::Yaw);
    $MLAgent::ViewYawApplied = $MP::MyMarble.getId();
}

function MLWatchCam::turn(%delta) {
    if ($MLWatchCam::Yaw $= "")
        $MLWatchCam::Yaw = $MLAgent::ViewYaw + 0;
    %y = $MLWatchCam::Yaw + %delta;
    $MLWatchCam::Yaw = %y - 6.283185307179586 * mFloor(%y / 6.283185307179586);   // [0, 2 pi) like the engine
    MLWatchCam::apply();
}

// Every 16 ms (one move at 1x): turn while an arrow key is held, and take the view back when the bridge re-pins
// it (Python sends VIEWYAW 0 at every new round; a respawn makes a new marble).
function MLWatchCam::tick() {
    if (MLWatchCam::active()) {
        %turn = $MLWatchCam::TurnRight - $MLWatchCam::TurnLeft;
        if (%turn != 0)
            MLWatchCam::turn(%turn);
        else if ($MLWatchCam::Yaw !$= "" && (!$MLAgent::ViewYawOn || $MLAgent::ViewYaw != $MLWatchCam::Yaw
                 || $MLAgent::ViewYawApplied !$= $MP::MyMarble.getId()))
            MLWatchCam::apply();
    }
    schedule(16, 0, "MLWatchCam::tick");
}
schedule(16, 0, "MLWatchCam::tick");

// ---- input handlers (the defaults are in ../../default.bind.cs; config.cs binds the mouse and arrows to them) ----

// Mouse x: the same amount input_yaw would add to $mvYaw, applied to the view instead.
function yaw(%val) {
    %d = getMouseAdjustAmount(%val);
    if ($pref::Input::InvertXAxis)
        %d = -%d;
    if (MLWatchCam::active())
        MLWatchCam::turn(%d * $Game::CameraSpeedMultiplier);
    else
        input_yaw(%d);
}

// Left arrow. The engine's move yaw is $mvYawLeftSpeed - $mvYawRightSpeed and input_turnLeft sets the RIGHT speed,
// so turning left lowers the yaw.
function turnLeft(%val) {
    if (MLWatchCam::active()) {
        $MLWatchCam::TurnLeft = %val ? $Pref::Input::KeyboardTurnSpeed * $Game::CameraSpeedMultiplier : 0;
        return;
    }
    $MLWatchCam::TurnLeft = 0;
    input_turnLeft(%val ? $Pref::Input::KeyboardTurnSpeed : 0);
}

function turnRight(%val) {
    if (MLWatchCam::active()) {
        $MLWatchCam::TurnRight = %val ? $Pref::Input::KeyboardTurnSpeed * $Game::CameraSpeedMultiplier : 0;
        return;
    }
    $MLWatchCam::TurnRight = 0;
    input_turnRight(%val ? $Pref::Input::KeyboardTurnSpeed : 0);
}

// Left mouse button (bound only to the mouse). Press and release are both dropped: a release would zero
// $mvTriggerCount0 under the agent's own use key.
function mouseFire(%val) {
    if (MLWatchCam::active())
        return;
    input_mouseFire(%val);
}

// Right mouse button (bound only to the mouse; the E key is useBlast1). The bridge blasts through input_useBlast.
function useBlast(%val) {
    if (MLWatchCam::active())
        return;
    input_useBlast(%val);
}
