param(
    [string]$EngineRoot = 'C:/Users/doug/src/OpenPQ-TGEMIT-mbx',
    [string]$OutputName = 'marbleblast_geometry_probe.exe',
    [switch]$FixtureProbe
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$taskNative = (Resolve-Path -LiteralPath $EngineRoot).Path
$taskMarble = Join-Path $taskNative 'engine/game/marble/marble.cc'
if ($OutputName -notmatch '^marbleblast_[a-z0-9_]+_probe\.exe$') { throw 'OutputName must be a separate probe filename.' }
if ($FixtureProbe -and $OutputName -ne 'marbleblast_fixture_probe.exe') { throw 'Fixture commands require their own executable.' }
if (Get-Process -Name ([IO.Path]::GetFileNameWithoutExtension($OutputName)) -ErrorAction SilentlyContinue) { throw 'An owned probe must be closed before rebuilding.' }
$taskDynamics = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'engine_dynamics.inc'))
if ($FixtureProbe) {
    # An inherited protected physical field needs a typed read accessor. This
    # nonvirtual method adds no data/layout and changes no simulation behavior.
    $taskHeader = Join-Path $taskNative 'engine/game/marble/marble.h'
    $taskHeaderCode = [IO.File]::ReadAllText($taskHeader).Replace("`r`n", "`n")
    if (-not $taskHeaderCode.Contains('aiFixtureRuntimeMass()')) {
        $taskHeaderAnchor = "public:`n    enum UpdateMaskBits"
        if (-not $taskHeaderCode.Contains($taskHeaderAnchor)) { throw 'Typed fixture mass accessor anchor changed.' }
        $taskHeaderCode = $taskHeaderCode.Replace($taskHeaderAnchor,"public:`n    F32 aiFixtureRuntimeMass() const { return mMass; } // Read-only fixture observation.`n    enum UpdateMaskBits")
        [IO.File]::WriteAllText($taskHeader, $taskHeaderCode, [Text.UTF8Encoding]::new($false))
    }
    # Only this generated native include gets setup hooks. The maintained recorder
    # source, sealed datasets and both existing executable bytes remain intact.
    $taskEntry = 'void aiBeginMarbleDynamics(Marble* m,const Move* move,U32 dt) {'
    if (-not $taskDynamics.Contains($taskEntry)) { throw 'Fixture bootstrap anchor changed.' }
    $taskStateAnchor = "params(f,m->mDataBlock); std::fputc('}',f);"
    if (-not $taskDynamics.Contains($taskStateAnchor)) { throw 'Fixture script-state export anchor changed.' }
    $taskDynamics = $taskDynamics.Replace($taskStateAnchor,"params(f,m->mDataBlock); aiFixtureScriptState(f,m); std::fputc('}',f);")
    $taskDynamics = "void aiFixtureBootstrap();`nvoid aiFixtureScriptState(FILE*,Marble*);`n" + $taskDynamics.Replace($taskEntry, $taskEntry + "`n    aiFixtureBootstrap();")
    $taskDynamics += "`n#include `"aiMarbleFixture.inc`"`n"
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'engine_fixture.inc') -Destination (Join-Path $taskNative 'engine/game/marble/aiMarbleFixture.inc')
}
[IO.File]::WriteAllText((Join-Path $taskNative 'engine/game/marble/aiMarbleDynamics.inc'), $taskDynamics, [Text.UTF8Encoding]::new($false))
$taskCode = [IO.File]::ReadAllText($taskMarble).Replace("`r`n", "`n")
if (-not $taskCode.Contains('#include "aiMarbleDynamics.inc"')) {
    $taskAnchor = 'void Marble::advancePhysics(const Move* move, U32 timeDelta)'
    if (-not $taskCode.Contains($taskAnchor)) { throw 'Physics entry anchor changed.' }
    $taskCode = $taskCode.Replace($taskAnchor, "// Recording-only instrumentation, separate probe.`nvoid aiBeginMarbleDynamics(Marble*, const Move*, U32);`nvoid aiEndMarbleDynamics(Marble*, const Move*, U32);`n" + $taskAnchor)
    $taskBefore = "move = &NullMove;`n    advanceCamera(move, timeDelta);"
    if (-not $taskCode.Contains($taskBefore)) { throw 'Applied-move anchor changed.' }
    $taskCode = $taskCode.Replace($taskBefore, "move = &NullMove;`n    aiBeginMarbleDynamics(this, move, timeDelta);`n    advanceCamera(move, timeDelta);")
    $taskAfter = "if (!clientsidePrediction && gInterpolateMarbles)`n        interpolateMarble(timeDelta);`n}"
    if (-not $taskCode.Contains($taskAfter)) { throw 'Physics exit anchor changed.' }
    $taskCode = $taskCode.Replace($taskAfter, "if (!clientsidePrediction && gInterpolateMarbles)`n        interpolateMarble(timeDelta);`n    aiEndMarbleDynamics(this, move, timeDelta);`n}")
    $taskCode += "`n#include `"aiMarbleDynamics.inc`"`n"
    [IO.File]::WriteAllText($taskMarble, $taskCode, [Text.UTF8Encoding]::new($false))
}
if (-not $taskCode.Contains('aiDynamicsContactEntry(this)')) {
    $taskCode = $taskCode.Replace('void aiBeginMarbleDynamics(Marble*, const Move*, U32);', "void aiBeginMarbleDynamics(Marble*, const Move*, U32);`nvoid aiDynamicsContactEntry(Marble*);`nvoid aiDynamicsForces(Marble*, const Point3D&, const Point3D&, F64);")
    $taskContact = 'findContacts(StaticObjectType | TerrainObjectType | InteriorObjectType | WaterObjectType | StaticShapeObjectType | StaticTSObjectType, timePassed * 1000.0f);'
    if (-not $taskCode.Contains($taskContact)) { throw 'Contact diagnostic anchor changed.' }
    $taskCode = $taskCode.Replace($taskContact, $taskContact + "`n            aiDynamicsContactEntry(this);")
    $taskForces = 'mVelocity = A * timeStep + mVelocity;'
    if (-not $taskCode.Contains($taskForces)) { throw 'Force diagnostic anchor changed.' }
    $taskCode = $taskCode.Replace($taskForces, "aiDynamicsForces(this, A, a, timeStep);`n            " + $taskForces)
    [IO.File]::WriteAllText($taskMarble, $taskCode, [Text.UTF8Encoding]::new($false))
}
# Existing helper preserves the geometry instrumentation and builds only the separate executable.
if (-not $taskCode.Contains('aiDynamicsCameraCommand(object,')) {
    $taskCode = $taskCode.Replace('void aiBeginMarbleDynamics(Marble*, const Move*, U32);', "void aiBeginMarbleDynamics(Marble*, const Move*, U32);`nvoid aiDynamicsCameraCommand(Marble*, const char*, F32);")
    $taskYaw = 'object->mCameraYaw = yaw;'
    $taskPitch = 'object->mCameraPitch = pitch;'
    if (-not $taskCode.Contains($taskYaw) -or -not $taskCode.Contains($taskPitch)) { throw 'Camera-command anchors changed.' }
    $taskCode = $taskCode.Replace($taskYaw,$taskYaw + "`n    aiDynamicsCameraCommand(object, `"camera_yaw`", yaw);")
    $taskCode = $taskCode.Replace($taskPitch,$taskPitch + "`n    aiDynamicsCameraCommand(object, `"camera_pitch`", pitch);")
    [IO.File]::WriteAllText($taskMarble, $taskCode, [Text.UTF8Encoding]::new($false))
}
& (Join-Path $PSScriptRoot '../collision/build_probe.ps1') -EngineRoot $taskNative -OutputName $OutputName
