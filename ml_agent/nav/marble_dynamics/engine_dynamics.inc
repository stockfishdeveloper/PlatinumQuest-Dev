// Recording-only hooks in the separate probe binary. Included after aiGeometryExport.inc.
// No forces, velocities, timers, controls, collision decisions or scripts are changed.
namespace AIMarbleDynamics {
static FILE* stream = NULL;
static U32 target = 0;
static U32 samples = 0;
static Marble::ContactTelemetry beforeTelemetry;
static bool rowOpen = false;
static bool detailed = false;
static U32 substepRows = 0;
static Point3D contactEntryVelocity,contactEntrySpin;
struct CameraCommand { const char* kind; F32 value; U32 clock; };
static Vector<CameraCommand> cameraCommands;
template<typename P> static void vec(FILE* f, const P& p) {
    std::fprintf(f, "[%.17g,%.17g,%.17g]", (double)p.x, (double)p.y, (double)p.z);
}
static void params(FILE* f, MarbleData* d) {
    // MarbleData members are private. Registered typed fields preserve the actual
    // values without widening access or changing the native physics class.
    const char* native[] = {"maxRollVelocity","angularAcceleration","brakingAcceleration","staticFriction",
        "kineticFriction","bounceKineticFriction","gravity","maxDotSlide","bounceRestitution","airAcceleration","jumpImpulse","minBounceVel","mass"};
    const char* keys[] = {"max_roll_velocity","angular_acceleration","braking_acceleration","static_friction",
        "kinetic_friction","bounce_kinetic_friction","gravity","max_dot_slide","bounce_restitution","air_acceleration","jump_impulse","min_bounce_velocity","mass"};
    std::fputc('{',f);
    for(U32 i=0;i<13;++i){
        if(i)std::fputc(',',f); AIGeometry::quoted(f,keys[i]);std::fputc(':',f);
        const AbstractClassRep::Field* field=d->findField(StringTable->insert(native[i]));
        if(field && field->type==TypeF32){
            const F32 value=*(const F32*)(((const char*)d)+field->offset);
            std::fprintf(f,"%.9g",value);
        }else std::fprintf(f,"null");
    }
    std::fprintf(f,",\"effect_duration_ms\":[");
    const AbstractClassRep::Field* durations=d->findField(StringTable->insert("powerUpTime"));
    for(U32 i=0;i<6;++i){
        if(i)std::fputc(',',f);
        if(durations && durations->type==TypeS32)
            std::fprintf(f,"%u",((const U32*)(((const char*)d)+durations->offset))[i]);
        else std::fprintf(f,"null");
    }
    std::fprintf(f,"]}");
}
static void state(FILE* f, Marble* m) {
    std::fprintf(f,"{\"clock_ms\":%u,\"position\":",Sim::getCurrentTime()); vec(f,m->mPosition);
    std::fprintf(f,",\"velocity\":"); vec(f,m->mVelocity);
    std::fprintf(f,",\"spin\":"); vec(f,m->mOmega);
    std::fprintf(f,",\"camera_offset\":"); vec(f,m->mCameraOffset);
    std::fprintf(f,",\"gravity_direction\":"); vec(f,gGlobalGravityDir);
    std::fprintf(f,",\"best_contact_normal\":"); vec(f,m->mBestContact.normal);
    std::fprintf(f,",\"smooth_camera\":%s,\"next_yaw\":%.9g,\"next_pitch\":%.9g,\"pitch\":%.9g",
        smoothCamera?"true":"false",m->mNextCameraYaw,m->mNextCameraPitch,m->mCameraPitch);
    std::fprintf(f,",\"yaw\":%.9g,\"radius\":%.9g,\"held\":%u,\"mode\":%u,\"oob\":%s,\"ground_ms\":%u,\"effects\":[",
        m->mCameraYaw,m->mRadius,m->mPowerUpId,m->mMode,m->mOOB?"true":"false",m->mGroundTime);
    for(U32 i=0;i<6;++i) {
        if(i)std::fputc(',',f);
        const S32 left=m->mPowerUpState[i].endTime-(S32)Sim::getCurrentTime();
        std::fprintf(f,"[%s,%d]",m->mPowerUpState[i].active?"true":"false",m->mPowerUpState[i].active?getMax(0,left):0);
    }
    std::fprintf(f,"],\"effect_end_clock_ms\":[");
    for(U32 i=0;i<6;++i){if(i)std::fputc(',',f);std::fprintf(f,"%u",m->mPowerUpState[i].endTime);}
    std::fprintf(f,"],\"datablock\":"); AIGeometry::quoted(f,m->mDataBlock->getName());
    std::fprintf(f,",\"parameters\":"); params(f,m->mDataBlock); std::fputc('}',f);
}
static bool close() {
    if(!stream)return true;
    if(rowOpen){ std::fprintf(stream,"],\"aborted\":true}\n"); rowOpen=false; }
    std::fprintf(stream,"{\"kind\":\"complete\",\"samples\":%u}\n",samples);
    bool good=std::ferror(stream)==0;
    good=std::fclose(stream)==0 && good; stream=NULL; target=0; cameraCommands.clear();return good;
}
}
void aiDynamicsCameraCommand(Marble* m,const char* kind,F32 value) {
    using namespace AIMarbleDynamics;
    if(!stream || target!=m->getId() || runningPredictions)return;
    CameraCommand command;command.kind=kind;command.value=value;command.clock=Sim::getCurrentTime();
    cameraCommands.push_back(command);
}
void aiBeginMarbleDynamics(Marble* m,const Move* move,U32 dt) {
    using namespace AIMarbleDynamics;
    if(!stream || target!=m->getId() || runningPredictions)return;
    beforeTelemetry=m->mTelem; rowOpen=true;
    std::fprintf(stream,"{\"kind\":\"transition\",\"index\":%u,\"sim_ms\":%u,\"dt_ms\":%u,\"before\":",samples,Sim::getCurrentTime(),dt);
    state(stream,m);
    std::fprintf(stream,",\"control_events\":[");
    for(U32 i=0;i<cameraCommands.size();++i){
        if(i)std::fputc(',',stream);std::fprintf(stream,"{\"kind\":");AIGeometry::quoted(stream,cameraCommands[i].kind);
        std::fprintf(stream,",\"value\":%.9g,\"sim_ms\":%u}",cameraCommands[i].value,cameraCommands[i].clock);
    }
    std::fputc(']',stream);cameraCommands.clear();
    std::fprintf(stream,",\"move\":{\"x\":%.9g,\"y\":%.9g,\"z\":%.9g,\"yaw\":%.9g,\"pitch\":%.9g,\"jump\":%s,\"use\":%s,\"trigger1\":%s}",
        move->x,move->y,move->z,move->yaw,move->pitch,move->trigger[2]?"true":"false",move->trigger[0]?"true":"false",move->trigger[1]?"true":"false");
    std::fprintf(stream,",\"substep_trace\":[");substepRows=0;
}
void aiDynamicsContactEntry(Marble* m) {
    using namespace AIMarbleDynamics;
    if(!stream || !detailed || target!=m->getId() || runningPredictions || !rowOpen)return;
    contactEntryVelocity=m->mVelocity;contactEntrySpin=m->mOmega;
}
void aiDynamicsForces(Marble* m,const Point3D& A,const Point3D& angular,F64 dt) {
    using namespace AIMarbleDynamics;
    if(!stream || !detailed || target!=m->getId() || runningPredictions || !rowOpen)return;
    if(substepRows++)std::fputc(',',stream);
    std::fprintf(stream,"{\"dt_s\":%.17g,\"contact_entry_velocity\":",dt);vec(stream,contactEntryVelocity);
    std::fprintf(stream,",\"contact_entry_spin\":");vec(stream,contactEntrySpin);
    std::fprintf(stream,",\"pre_integration\":");state(stream,m);
    std::fprintf(stream,",\"linear_accel\":");vec(stream,A);
    std::fprintf(stream,",\"angular_accel\":");vec(stream,angular);
    std::fprintf(stream,",\"contacts\":[");
    for(U32 i=0;i<m->mContacts.size();++i){
        if(i)std::fputc(',',stream);const Marble::Contact& c=m->mContacts[i];
        std::fprintf(stream,"{\"normal\":");vec(stream,c.normal);
        std::fprintf(stream,",\"position\":");vec(stream,c.position);
        std::fprintf(stream,",\"friction\":%.9g,\"normal_force\":%.17g,\"object_id\":%u,\"material\":%u}",
                     c.friction,c.normalForce,c.object?c.object->getId():0,c.material);
    }
    std::fprintf(stream,"]}");
}
void aiEndMarbleDynamics(Marble* m,const Move*,U32) {
    using namespace AIMarbleDynamics;
    if(!stream || target!=m->getId() || runningPredictions || !rowOpen)return;
    std::fprintf(stream,"],\"after\":"); state(stream,m);
    std::fprintf(stream,",\"telemetry\":{\"substeps\":%u,\"contact_steps\":%u,\"support_steps\":%u,\"collisions\":%u,\"max_contacts\":%u,\"max_approach\":%.9g},\"contacts\":[",
        m->mTelem.subSteps-beforeTelemetry.subSteps,m->mTelem.contactSteps-beforeTelemetry.contactSteps,
        m->mTelem.supportSteps-beforeTelemetry.supportSteps,m->mTelem.collisions-beforeTelemetry.collisions,
        m->mTelem.maxContacts,m->mTelem.maxApproach);
    for(U32 i=0;i<m->mContacts.size();++i) {
        if(i)std::fputc(',',stream);
        const Marble::Contact& c=m->mContacts[i];
        std::fprintf(stream,"{\"normal\":"); vec(stream,c.normal);
        std::fprintf(stream,",\"position\":"); vec(stream,c.position);
        std::fprintf(stream,",\"surface_velocity\":");vec(stream,c.surfaceVelocity);
        std::fprintf(stream,",\"friction\":%.9g,\"restitution\":%.9g,\"force\":%.9g,\"distance\":%.17g,\"material\":%u,\"object_id\":%u}",
            c.friction,c.restitution,c.force,c.contactDistance,c.material,c.object?c.object->getId():0);
    }
    std::fprintf(stream,"]}\n"); ++samples;rowOpen=false;
}
ConsoleFunction(aiDynamicsProbeAvailable,bool,1,1,"Recording hooks are present only in the separate probe executable") {return true;}
ConsoleFunction(aiDynamicsDetail,bool,2,2,"Toggle read-only per-substep force/contact diagnostics") {AIMarbleDynamics::detailed=dAtoi(argv[1])!=0;return true;}
ConsoleMethod(Marble,aiRecordDynamics,bool,3,3,"aiRecordDynamics(path|off): actual applied moves and native before/after states") {
    using namespace AIMarbleDynamics;
    if(!close())return false;
    if(!std::strcmp(argv[2],"off"))return true;
    stream=std::fopen(argv[2],"wb");if(!stream)return false;
    target=object->getId();samples=0;rowOpen=false;cameraCommands.clear();
    std::fprintf(stream,"{\"kind\":\"header\",\"schema\":\"marble_native_transition_v1\",\"control_contract\":\"native_move_and_absolute_camera_commands_v1\",\"object_id\":%u,\"physics_substep_cap_ms\":8,\"client\":%s}\n",target,object->isClientObject()?"true":"false");
    std::fflush(stream);return true;
}
ConsoleMethod(Marble,aiWriteDynamicsState,bool,3,3,"Read-only full precision physical state snapshot") {
    FILE* f=std::fopen(argv[2],"wb");if(!f)return false;
    AIMarbleDynamics::state(f,object);std::fputc('\n',f);bool good=std::ferror(f)==0;
    return std::fclose(f)==0 && good;
}
ConsoleMethod(Marble,aiLiveDynamicsJSON,const char*,2,2,"Read-only state at the bridge observation; separate probe only") {
    // Reuse the exact recorder serializer. A private temporary stream avoids
    // rounding through script getters and does not advance or alter physics.
    FILE* f=std::tmpfile();if(!f)return "";
    extern Move gFirstMove;
    std::fprintf(f,"{\"object_id\":%u,\"frame_clock_ms\":%u,\"state\":",object->getId(),Platform::getVirtualMilliseconds());
    AIMarbleDynamics::state(f,object);
    const Move* moves[]={&object->delta.move,&gFirstMove};
    const char* names[]={"last_move","pending_move"};
    for(U32 i=0;i<2;++i){const Move& move=*moves[i];
        std::fprintf(f,",\"%s\":{\"x\":%.9g,\"y\":%.9g,\"z\":%.9g,\"yaw\":%.9g,\"pitch\":%.9g,\"jump\":%s,\"use\":%s,\"trigger1\":%s}",names[i],
            move.x,move.y,move.z,move.yaw,move.pitch,move.trigger[2]?"true":"false",move.trigger[0]?"true":"false",move.trigger[1]?"true":"false");
    }
    std::fputc('}',f);
    if(std::ferror(f)){std::fclose(f);return "";}
    long size=std::ftell(f);if(size<0 || size>16000){std::fclose(f);return "";}
    std::rewind(f);char* result=Con::getReturnBuffer((U32)size+1);
    size_t count=std::fread(result,1,size,f);result[count]=0;std::fclose(f);
    return count==(size_t)size?result:"";
}
