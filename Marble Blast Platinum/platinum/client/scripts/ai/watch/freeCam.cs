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
// instead; mouse up/down tilts the view only in marbleblast_realtime.exe and is locked in the other exes (see pitch
// below). Since 2026-10-19 the mouse is locked to the
// window as in normal play (MLWatchCam::holdMouse), so the pointer never leaves the window and the view keeps
// turning however far the mouse moves (this PC's own mouse; remote desktop's absolute pointer is not supported).
// The mouse buttons do nothing meanwhile, so the click that focuses
// the window cannot fire the agent's powerup (left button) or blast (right button). The keyboard otherwise works
// as usual: W/A/S/D and space still act on the marble, Esc pauses the round. Outside watch mode (menus,
// training speeds) every handler below behaves exactly as in default.bind.cs.
//------------------------------------------------------------------------------

$MLWatchCam::Yaw = "";        // the viewer's yaw, radians; "" until the viewer first turns (starts from the pinned view)
$MLWatchCam::Pitch = "";      // the viewer's pitch, radians; "" until the viewer first tilts (starts from the marble's)
$MLWatchCam::PitchApplied = "";   // the marble the view pitch was last set on
$MLWatchCam::TurnLeft = 0;    // held arrow keys, radians per move (the engine's $mvYawLeftSpeed units)
$MLWatchCam::TurnRight = 0;
$MLWatchCam::HoldsMouse = false;   // this file turned the cursor off (MLWatchCam::holdMouse)
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

// View-only pitch (2026-10-19, operator: "can we not lock the up/down camera movement?"). marbleblast_realtime.exe has
// Marble.setViewPitch, a render-only pitch like setViewYaw: the picture tilts while the camera pitch, which is part of
// the movement frame, stays where the agent keeps it. Other exes do not have it; there the pitch stays locked.
function MLWatchCam::canPitch() {
    return isObject($MP::MyMarble) && $MP::MyMarble.isMethod("setViewPitch");
}

function MLWatchCam::applyPitch() {
    $MP::MyMarble.setViewPitch($MLWatchCam::Pitch);
    $MLWatchCam::PitchApplied = $MP::MyMarble.getId();
}

function MLWatchCam::tilt(%delta) {
    if ($MLWatchCam::Pitch $= "")
        $MLWatchCam::Pitch = $MP::MyMarble.getCameraPitch();
    %p = $MLWatchCam::Pitch + %delta;
    $MLWatchCam::Pitch = %p > 1.5 ? 1.5 : (%p < -0.95 ? -0.95 : %p);   // the engine's camera pitch range
    MLWatchCam::applyPitch();
}

// The pause menu (Esc) needs the cursor.
function MLWatchCam::menuOpen() {
    return (isObject(ExitGameDlg) && ExitGameDlg.isAwake()) || (isObject(MPExitGameDlg) && MPExitGameDlg.isAwake());
}

// 2026-10-19 (operator: "when i go far to the right or left the mouse leaves the screen and it just stops moving the
// camera. make it so that it never loses focus from the window"; viewing is always on this PC, never through remote
// desktop): while the agent plays at 1x and no pause menu is open, the mouse is LOCKED to the window exactly as in
// normal play (core/client/cursor.cs cursorOff: lockMouse(true) + no canvas cursor). The engine then confines the
// pointer to the window (winWindow.cc setMouseClipping: ClipCursor) and reads the mouse as raw relative motion
// (WM_INPUT), which the yaw handler below sends to the view, so the view turns however far the mouse moves and the
// pointer never leaves the window. A GUI change can turn the cursor back on (Canvas.checkCursor -> cursorOn ->
// unlock); the next tick locks it again. The pause menu (Esc) and the menus after the round get the cursor as usual.
// (2026-10-18 22:50 - 10-19: the view followed the unlocked pointer's position instead, for Chrome Remote Desktop's
// absolute pointer; the pointer then stopped at the window edge and the turning stopped with it.)
function MLWatchCam::holdMouse() {
    %want = MLWatchCam::active() && !MLWatchCam::menuOpen();
    if (!%want) {
        if ($MLWatchCam::HoldsMouse) {
            $MLWatchCam::HoldsMouse = false;
            Canvas.checkCursor();   // whatever the open GUIs want now (the pause menu and the end-of-round menus: the cursor)
        }
        return;
    }
    if (!$MLWatchCam::HoldsMouse || Canvas.isCursorOn()) {
        $MLWatchCam::HoldsMouse = true;
        cursorOff();               // lockMouse(true) + Canvas.cursorOff(), as in normal play
    }
}

// Every 16 ms (one move at 1x): turn while an arrow key is held, and take the view back when the bridge re-pins
// it (Python sends VIEWYAW 0 at every new round; a respawn makes a new marble).
function MLWatchCam::tick() {
    MLWatchCam::holdMouse();
    if (MLWatchCam::active()) {
        %turn = $MLWatchCam::TurnRight - $MLWatchCam::TurnLeft;
        if (%turn != 0)
            MLWatchCam::turn(%turn);
        else if ($MLWatchCam::Yaw !$= "" && (!$MLAgent::ViewYawOn || $MLAgent::ViewYaw != $MLWatchCam::Yaw
                 || $MLAgent::ViewYawApplied !$= $MP::MyMarble.getId()))
            MLWatchCam::apply();
        if ($MLWatchCam::Pitch !$= "" && $MLWatchCam::PitchApplied !$= $MP::MyMarble.getId() && MLWatchCam::canPitch())
            MLWatchCam::applyPitch();   // a respawn makes a new marble
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

// Mouse y, while the agent plays at 1x: the view-only pitch where the exe has it (marbleblast_realtime.exe), else
// locked. The marble's own camera pitch is not render-only: the engine builds the movement frame from the camera
// offset (marble.cc getMarbleAxis), so tilting it moved the marble a little (2026-10-19: a replay left its recording at
// decision 250) and real-time play needs the physics to match the planner's forecast exactly.
// Otherwise the game's own handler (free look key / preference).
function pitch(%val) {
    if (MLWatchCam::active()) {
        if (MLWatchCam::canPitch()) {
            %d = getMouseAdjustAmount(%val);
            if ($pref::input::InvertYAxis)
                %d = -%d;
            MLWatchCam::tilt(%d * $Game::CameraSpeedMultiplier);
        }
        return;
    }
    if ($freelooking || $pref::Input::AlwaysFreeLook || isCannonActive()) {
        if ($pref::input::InvertYAxis)
            input_pitch(-getMouseAdjustAmount(%val));
        else
            input_pitch(getMouseAdjustAmount(%val));
    }
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
