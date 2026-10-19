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
// instead. Mouse up/down is locked meanwhile (operator 2026-10-18). Since 2026-10-18 the view turns from the
// pointer's left/right motion (MLWatchCam::holdMouse), which works through remote desktop (absolute pointer
// positions) as well as with the PC's own mouse.
// The mouse buttons do nothing meanwhile, so the click that focuses
// the window cannot fire the agent's powerup (left button) or blast (right button). The keyboard otherwise works
// as usual: W/A/S/D and space still act on the marble, Esc pauses the round. Outside watch mode (menus,
// training speeds) every handler below behaves exactly as in default.bind.cs.
//------------------------------------------------------------------------------

$MLWatchCam::Yaw = "";        // the viewer's yaw, radians; "" until the viewer first turns (starts from the pinned view)
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

// The pause menu (Esc) needs the cursor.
function MLWatchCam::menuOpen() {
    return (isObject(ExitGameDlg) && ExitGameDlg.isAwake()) || (isObject(MPExitGameDlg) && MPExitGameDlg.isAwake());
}

// 2026-10-18 (operator: "the camera doesn't move in any direction when I move the mouse"): the operator watches
// through Chrome Remote Desktop, which sends the mouse as ABSOLUTE pointer positions. The engine's mouse look only
// takes RELATIVE raw-input motion while the mouse is locked (platformWin32/winWindow.cc WM_INPUT; an absolute move
// becomes a pointer-position event and no look), so the mouse never turned the view, while the arrow keys did (a
// local test with injected relative motion passed and missed this). The view therefore looks from the pointer's
// position: while the agent plays at 1x and no pause menu is open, the mouse is left unlocked and the canvas cursor
// on (GuiCanvas::processMouseMoveEvent tracks the pointer only while it is on) but not drawn, and every 16 ms the
// pointer's left/right motion since the last tick turns the view at the game's mouse sensitivity (up/down is
// locked, operator 2026-10-18). Local
// relative mice work the same way (WM_MOUSEMOVE positions). At the window edge the pointer stops; move back, or use
// the arrow keys. Otherwise the cursor goes back to what the open GUIs want (pause menu, menus after the round).
$MLWatchCam::LastX = "";
$MLWatchCam::LastY = "";
function MLWatchCam::holdMouse() {
    %want = MLWatchCam::active() && !MLWatchCam::menuOpen();
    if (!%want) {
        if ($MLWatchCam::HoldsMouse) {
            $MLWatchCam::HoldsMouse = false;
            $MLWatchCam::LastX = "";
            Canvas.showCursor();
            Canvas.checkCursor();
        }
        return;
    }
    if (!$MLWatchCam::HoldsMouse) {
        $MLWatchCam::HoldsMouse = true;
        $MLWatchCam::LastX = "";
        lockMouse(false);
    }
    if (!Canvas.isCursorOn()) {   // a GUI change ran Canvas.checkCursor -> cursorOff (which also locks the mouse)
        lockMouse(false);
        Canvas.cursorOn();
    }
    Canvas.hideCursor();
    %p = Canvas.getCursorPos();
    %x = getWord(%p, 0);
    %y = getWord(%p, 1);
    // "0 0" = no pointer event yet (the first real position would read as a jump)
    if ($MLWatchCam::LastX !$= "" && !($MLWatchCam::LastX == 0 && $MLWatchCam::LastY == 0)) {
        %dx = %x - $MLWatchCam::LastX;
        %dy = %y - $MLWatchCam::LastY;
        if (mAbs(%dx) < 250 && mAbs(%dy) < 250) {   // a larger jump in 16 ms = the pointer came back into the window elsewhere
            if (%dx != 0) {   // left / right only (operator 2026-10-18: "lock up and down")
                %d = getMouseAdjustAmount(%dx);
                if ($pref::Input::InvertXAxis)
                    %d = -%d;
                MLWatchCam::turn(%d * $Game::CameraSpeedMultiplier);
            }
        }
    }
    $MLWatchCam::LastX = %x;
    $MLWatchCam::LastY = %y;
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

// Mouse y: locked while the agent plays at 1x (operator 2026-10-18: "lock up and down, I only need left to right").
// Otherwise the game's own handler (free look key / preference).
function pitch(%val) {
    if (MLWatchCam::active())
        return;
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
