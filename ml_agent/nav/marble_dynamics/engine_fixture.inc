// Setup-only physical copy prototype; compiled ONLY into marbleblast_fixture_probe.
// This atomic command never advances physics or rewinds a clock/world. Contact
// padding/unused bytes are preserved for roundtrip checks, not interpreted features.
extern Move gFirstMove,gNextMove;
namespace AIObservedScriptState {
static Marble* authority() {
    SimSet* clients=dynamic_cast<SimSet*>(Sim::findObject("ClientGroup"));
    if(!clients || clients->size()!=1)return NULL;
    GameConnection* connection=dynamic_cast<GameConnection*>((*clients)[0]);
    return connection?dynamic_cast<Marble*>(connection->getControlObject()):NULL;
}
static const char* field(SimObject* object,const char* key,const char* array=NULL) {
    return object?object->getDataField(StringTable->insert(key),array):"";
}
static void schedule(FILE* f,SimObject* object,const char* key,const char* array=NULL) {
    const U32 id=(U32)dAtoi(field(object,key,array));const bool pending=id && Sim::isEventPending(id);
    std::fprintf(f,"{\"id\":%u,\"pending\":%s,\"remaining_ms\":%u}",id,pending?"true":"false",pending?Sim::getEventTimeLeft(id):0);
}
static void body(FILE* f,Marble* marble) {
    if(!marble){std::fprintf(f,"null");return;}
    const char* raw=field(marble,marble->isClientObject()?"_powerUpData":"powerUpData");
    SimObject* item=raw && raw[0]?Sim::findObject(raw):NULL;
    std::fprintf(f,"{\"object_id\":%u,\"native_held\":%u,\"runtime_mass\":%.9g,\"held_item\":",marble->getId(),marble->mPowerUpId,marble->aiFixtureRuntimeMass());
    AIGeometry::quoted(f,item && item->getName()?item->getName():"");
    std::fprintf(f,",\"held_item_power_id\":%d,\"raw_held_datablock\":",dAtoi(field(item,"powerUpId")));
    AIGeometry::quoted(f,raw);std::fprintf(f,",\"dynamic_fields\":{");
    const char* names[]={"_powerUpId","powerUpId","heldPowerup","lockPowerup","hasClientPowerup","powerUpObj"};
    for(U32 i=0;i<6;++i){if(i)std::fputc(',',f);AIGeometry::quoted(f,names[i]);std::fputc(':',f);AIGeometry::quoted(f,field(marble,names[i]));}
    std::fprintf(f,"},\"active_flags\":[");
    for(U32 i=0;i<7;++i){char index[16];dSprintf(index,sizeof(index),"%u",i);if(i)std::fputc(',',f);std::fprintf(f,"%s",dAtob(field(marble,"powerupActive",index))?"true":"false");}
    std::fprintf(f,"],\"effect_schedules\":[");
    for(U32 i=0;i<7;++i){char index[16];dSprintf(index,sizeof(index),"%u",i);if(i)std::fputc(',',f);schedule(f,marble,"powerupSchedule",index);}
    std::fprintf(f,"],\"mega_schedule\":");schedule(f,marble,"megaSchedule");std::fputc('}',f);
}
}
namespace AIObservedNativeContext {
template<typename P> static void vec(FILE* f,const P& p) {
    std::fprintf(f,"[%.17g,%.17g,%.17g]",(double)p.x,(double)p.y,(double)p.z);
}
static void matrix(FILE* f,const MatrixF& m) {
    std::fputc('[',f);for(U32 i=0;i<16;++i){if(i)std::fputc(',',f);std::fprintf(f,"%.9g",m[i]);}std::fputc(']',f);
}
static void contact(FILE* f,const Marble::Contact& c) {
    // findContacts value-initializes current records. Best is emitted only after
    // a nonzero initialized normal; historical mLastContact is never decoded.
    std::fprintf(f,"{\"position\":");vec(f,c.position);std::fprintf(f,",\"normal\":");vec(f,c.normal);
    std::fprintf(f,",\"surface_velocity\":");vec(f,c.surfaceVelocity);
    std::fprintf(f,",\"distance\":%.17g,\"normal_force\":%.17g,\"friction\":%.9g,\"restitution\":%.9g,\"force\":%.9g,\"material\":%u}",
        c.contactDistance,c.normalForce,c.friction,c.restitution,c.force,c.material);
}
static void state(FILE* f,Marble* m) {
    std::fprintf(f,",\"native_context\":{\"schema\":\"observed_native_context_v2\",\"runtime_mass\":%.9g,\"gravity_up\":",m->aiFixtureRuntimeMass());vec(f,m->mGravityUp);
    std::fprintf(f,",\"previous_gravity_direction\":");vec(f,gPrevGravityDir);
    std::fprintf(f,",\"global_gravity_matrix\":");matrix(f,gGlobalGravityMatrix);
    std::fprintf(f,",\"previous_gravity_matrix\":");matrix(f,gPrevGravityMatrix);
    std::fprintf(f,",\"default_gravity_transform\":");matrix(f,gDefaultGravityTransform);
    std::fprintf(f,",\"gravity_interpolation_active\":%s,\"gravity_interpolation\":",gInterpGravityDir?"true":"false");
    if(gInterpGravityDir) {
        std::fprintf(f,"{\"elapsed_ms\":%u,\"duration_ms\":%u,\"start_quaternion\":[%.9g,%.9g,%.9g,%.9g],\"end_quaternion\":[%.9g,%.9g,%.9g,%.9g]}",
            gInterpCurrentTime,gInterpTotalTime,gStartGravityQuat.x,gStartGravityQuat.y,gStartGravityQuat.z,gStartGravityQuat.w,
            gEndGravityQuat.x,gEndGravityQuat.y,gEndGravityQuat.z,gEndGravityQuat.w);
    }else std::fprintf(f,"null");
    const MarbleUpdateInfo& u=m->mUpdateInfo;
    std::fprintf(f,",\"pending_update\":{\"types\":%d,\"velocity\":",u.types);
    if(u.types & VelocityUpdateFlag)vec(f,u.velocity);else std::fprintf(f,"null");
    std::fprintf(f,",\"spin\":");if(u.types & AngularVelocityUpdateFlag)vec(f,u.angularVelocity);else std::fprintf(f,"null");
    std::fprintf(f,",\"transform\":");if(u.types & TransformUpdateFlag)matrix(f,u.transform);else std::fprintf(f,"null");
    std::fprintf(f,",\"camera\":");if(u.types & CameraUpdateFlag)vec(f,u.camera);else std::fprintf(f,"null");
    std::fprintf(f,",\"gravity\":");
    if(u.types & GravityUpdateFlag) {
        std::fprintf(f,"{\"right\":");vec(f,u.gravity.right);std::fprintf(f,",\"back\":");vec(f,u.gravity.back);
        std::fprintf(f,",\"down\":");vec(f,u.gravity.down);std::fprintf(f,",\"instant\":%s}",u.gravityInstant?"true":"false");
    }else std::fprintf(f,"null");
    std::fprintf(f,",\"radius\":");if(u.types & SizeUpdateFlag)std::fprintf(f,"%.9g",u.size);else std::fprintf(f,"null");
    std::fprintf(f,"},\"current_contacts\":[");
    for(U32 i=0;i<m->mContacts.size();++i){if(i)std::fputc(',',f);contact(f,m->mContacts[i]);}
    std::fprintf(f,"],\"best_contact\":");
    if(m->mBestContact.normal.lenSquared()>0)contact(f,m->mBestContact);else std::fprintf(f,"null");
    // Typed registered datablock field, distinct from protected runtime mass.
    std::fprintf(f,",\"max_force_radius\":");
    const AbstractClassRep::Field* radius=m->mDataBlock->findField(StringTable->insert("maxForceRadius"));
    if(radius && radius->type==TypeF32)std::fprintf(f,"%.9g",*(const F32*)(((const char*)m->mDataBlock)+radius->offset));else std::fprintf(f,"null");
    std::fprintf(f,",\"runtime_mass_observed\":true,\"last_contact_decoded\":false}");
}
}
void aiFixtureScriptState(FILE* f,Marble* marble) {
    // Read-only, bounded context; never restore scripts, timers or the world.
    Marble* authority=AIObservedScriptState::authority();
    std::fprintf(f,",\"script_context\":{\"schema\":\"observed_local_script_v1\",\"target\":");
    AIObservedScriptState::body(f,marble);std::fprintf(f,",\"authority\":");AIObservedScriptState::body(f,authority);
    std::fprintf(f,",\"bridge_control\":{\"schema\":\"observed_bridge_control_v1\",\"power_use_latched\":%s,\"power_yaw_hold_ticks\":%d,\"power_yaw\":%.9g,\"configured_yaw_hold_ticks\":%d,\"ss_yaw_fix\":%s,\"fast_powerups\":%s,\"live\":%s}}",
        Con::getBoolVariable("$MLAgent::PowUseSent")?"true":"false",Con::getIntVariable("$MLAgent::PowYawHold"),
        Con::getFloatVariable("$MLAgent::PowYaw"),Con::getIntVariable("$MLAgent::PowYawHoldTicks"),
        Con::getBoolVariable("$MLAgent::SSYawFix")?"true":"false",Con::getBoolVariable("$MP::FastPowerups")?"true":"false",
        Con::getBoolVariable("$MLAgent::Live")?"true":"false");
    AIObservedNativeContext::state(f,marble);
}
namespace AIMarbleFixture {
#define AI_FIXTURE_FIELDS(X) \
    X(mPosition) X(mVelocity) X(mOmega) X(mGravityUp) X(mCameraOffset) \
    X(mBestContact) X(mLastContact) X(mGroundTime) X(mRadius) X(mDataBlock) \
    X(mMode) X(mControllable) X(mOOB) X(mOOBCamPos) X(mPowerUpId) \
    X(mPowerUpTimer) X(mPrevPowerUpTimer) X(mPositionKey) X(mBounceEmitDelay) \
    X(delta) X(mTelem) X(mCameraYaw) X(mCameraPitch) X(mNextCameraYaw) \
    X(mNextCameraPitch) X(mMouseZ) X(mPowerUpState) X(mExtraData)

static void fingerprint(FILE* f,const void* bytes,size_t length) {
    // FNV-1a verifies this synchronous in-process roundtrip only. Source/artifact
    // integrity uses SHA256 in the caller; this is not a cross-process fixture ID.
    const unsigned char* p=(const unsigned char*)bytes;
    unsigned long long value=14695981039346656037ull;
    for(size_t i=0;i<length;++i){value^=p[i];value*=1099511628211ull;}
    std::fprintf(f,"\"%016llx\"",value);
}
struct Snapshot {
#define AI_DECLARE(name) decltype(((Marble*)0)->name) name;
    AI_FIXTURE_FIELDS(AI_DECLARE)
#undef AI_DECLARE
    Vector<Marble::Contact> contacts;
    Move firstMove,nextMove;
    MatrixF transform,renderTransform;
    Point3D gravity;
    bool cameraSmoothing;
    U32 clock;
    void capture(Marble* m) {
#define AI_CAPTURE(name) std::memcpy(&name,&m->name,sizeof(name));
        AI_FIXTURE_FIELDS(AI_CAPTURE)
#undef AI_CAPTURE
        contacts.setSize(m->mContacts.size());
        for(U32 i=0;i<contacts.size();++i)std::memcpy(&contacts[i],&m->mContacts[i],sizeof(Marble::Contact));
        std::memcpy(&firstMove,&::gFirstMove,sizeof(Move));
        std::memcpy(&nextMove,&::gNextMove,sizeof(Move));
        gravity=gGlobalGravityDir;cameraSmoothing=smoothCamera;clock=Sim::getCurrentTime();
        transform=m->getTransform();renderTransform=m->getRenderTransform();
    }
    void restore(Marble* m,bool restorePlacement=false) const {
        if(restorePlacement){m->SceneObject::setTransform(transform);m->setRenderTransform(renderTransform);}
#define AI_RESTORE(name) std::memcpy(&m->name,&name,sizeof(name));
        AI_FIXTURE_FIELDS(AI_RESTORE)
#undef AI_RESTORE
        m->mContacts.setSize(contacts.size());
        for(U32 i=0;i<contacts.size();++i)std::memcpy(&m->mContacts[i],&contacts[i],sizeof(Marble::Contact));
        std::memcpy(&::gFirstMove,&firstMove,sizeof(Move));
        std::memcpy(&::gNextMove,&nextMove,sizeof(Move));
    }
    bool equal(Marble* m,bool requireClock=true) const {
#define AI_EQUAL(name) if(std::memcmp(&name,&m->name,sizeof(name)))return false;
        AI_FIXTURE_FIELDS(AI_EQUAL)
#undef AI_EQUAL
        if(contacts.size()!=m->mContacts.size())return false;
        for(U32 i=0;i<contacts.size();++i)if(std::memcmp(&contacts[i],&m->mContacts[i],sizeof(Marble::Contact)))return false;
        return !std::memcmp(&firstMove,&::gFirstMove,sizeof(Move)) && !std::memcmp(&nextMove,&::gNextMove,sizeof(Move)) &&
            gravity==gGlobalGravityDir && cameraSmoothing==smoothCamera && (!requireClock || clock==Sim::getCurrentTime());
    }
};
static void fields(FILE* f,Marble* m) {
    bool first=true;std::fputc('{',f);
#define AI_FIELD(name) if(!first)std::fputc(',',f);first=false;AIGeometry::quoted(f,#name);std::fputc(':',f);fingerprint(f,&m->name,sizeof(m->name));
    AI_FIXTURE_FIELDS(AI_FIELD)
#undef AI_FIELD
    std::fprintf(f,",\"current_contacts\":[");
    for(U32 i=0;i<m->mContacts.size();++i){if(i)std::fputc(',',f);fingerprint(f,&m->mContacts[i],sizeof(Marble::Contact));}
    std::fprintf(f,"],\"first_move\":");fingerprint(f,&::gFirstMove,sizeof(Move));
    std::fprintf(f,",\"next_move\":");fingerprint(f,&::gNextMove,sizeof(Move));std::fputc('}',f);
}
static const char* refusal(Marble* m) {
    if(!Con::getBoolVariable("$AI::Lockstep") || !Con::getBoolVariable("$AI::WaitReply") ||
       !Con::getBoolVariable("$Server::Hosting") || Con::getBoolVariable("$MLAgent::Live"))return "not_frozen_owned_local_setup";
    if(!m || m->mMode!=0 || m->mOOB || !m->mControllable)return "unsupported_body_mode";
    if(!m->mActiveEvents.empty())return "active_native_events";
    for(U32 i=0;i<6;++i)if(m->mPowerUpState[i].active)return "active_effect_not_supported";
    return NULL;
}
struct ContactReference { U32 id;SimObject* pointer; };
struct CapturedPair {
    Snapshot client,server;
    U32 clientId,serverId,missionId;
    bool ready;
    Vector<ContactReference> references;
    void rememberContacts(Marble* m) {
        for(U32 i=0;i<m->mContacts.size();++i){
            SimObject* p=m->mContacts[i].object;if(!p)continue;
            ContactReference r;r.id=p->getId();r.pointer=p;references.push_back(r);
        }
        // Constructor initializes only cached normal, not its object pointer.
        // Read the pointer only after a best contact has actually been assigned.
        if(m->mBestContact.normal.lenSquared()>0 && m->mBestContact.object){
            ContactReference r;r.pointer=m->mBestContact.object;r.id=r.pointer->getId();references.push_back(r);
        }
    }
};
static CapturedPair captured;
static const char* pairRefusal(Marble* client,Marble* server,bool restoring) {
    const char* r=refusal(client);if(r)return r;r=refusal(server);if(r)return r;
    if(client==server || !client->isClientObject() || server->isClientObject())return "invalid_local_server_pair";
    if(Con::getIntVariable("$MLAgent::ActionDelay")!=0 || Con::getBoolVariable("$AIBridge::DelayedMode"))return "delayed_bridge_queue_not_supported";
    if(PathedInterior::getClientPathedInteriors())return "pathed_interiors_not_supported";
    if(client->mUpdateInfo.types || server->mUpdateInfo.types)return "pending_native_update_not_supported";
    if(AIMarbleDynamics::stream)return "close_recording_before_capture_restore";
    SimObject* mission=Sim::findObject("MissionGroup");if(!mission)return "missing_mission";
    if(restoring){
        if(!captured.ready || client->getId()!=captured.clientId || server->getId()!=captured.serverId ||
           mission->getId()!=captured.missionId)return "captured_body_or_mission_changed";
        if(client->mDataBlock!=captured.client.mDataBlock || server->mDataBlock!=captured.server.mDataBlock ||
           client->mRadius!=captured.client.mRadius || server->mRadius!=captured.server.mRadius)return "body_parameters_changed";
        if(Sim::getCurrentTime()-captured.client.clock>1024)return "capture_expired";
        if(Sim::getCurrentTime()%32!=captured.client.clock%32)return "native_tick_phase_changed";
        if(gGlobalGravityDir!=captured.client.gravity || smoothCamera!=captured.client.cameraSmoothing)return "global_context_changed";
        for(U32 i=0;i<captured.references.size();++i)
            if(Sim::findObject(captured.references[i].id)!=captured.references[i].pointer)return "contact_reference_expired";
    }
    return NULL;
}
static void pairReport(FILE* f,Marble* client,Marble* server,const char* operation,const char* reason,bool passed) {
    std::fprintf(f,"{\"schema\":\"inprocess_marble_physical_pair_v1\",\"operation\":");AIGeometry::quoted(f,operation);
    std::fprintf(f,",\"refused\":");if(reason)AIGeometry::quoted(f,reason);else std::fprintf(f,"null");
    std::fprintf(f,",\"passed\":%s,\"clock_ms\":%u,\"capture_clock_ms\":%u,\"world_restored\":false,\"clock_rewound\":false",
        passed?"true":"false",Sim::getCurrentTime(),captured.ready?captured.client.clock:0);
    if(client){std::fprintf(f,",\"client\":");AIMarbleDynamics::state(f,client);std::fprintf(f,",\"client_fields\":");fields(f,client);}
    if(server){std::fprintf(f,",\"server\":");AIMarbleDynamics::state(f,server);std::fprintf(f,",\"server_fields\":");fields(f,server);}
    std::fprintf(f,"}\n");
}
static void setPose(Marble* m,const Point3D& p,const Point3D& v,const Point3D& w) {
    MatrixF placement=m->getTransform();placement.setPosition(p.toPoint3F());
    m->SceneObject::setTransform(placement);m->setRenderTransform(placement);
    m->mPosition=p;m->mVelocity=v;m->mOmega=w;
    // Synthetic setup has no claimed preexisting contact history. Actual native
    // neutral steps subsequently establish it; never label this as natural play.
    std::memset(&m->mBestContact,0,sizeof(Marble::Contact));
    std::memset(&m->mLastContact,0,sizeof(Marble::Contact));m->mContacts.clear();
    m->mMaterialCollisions.clear();m->mGroundTime=0;m->delta.move=NullMove;m->delta.pos=p;
    m->delta.posVec.set(0,0,0);m->resetContactTelemetry();++m->mPositionKey;
}
}

ConsoleMethod(Marble,aiFixtureRoundTrip,bool,3,3,"Atomic setup-only copy/perturb/restore; no physical step") {
    using namespace AIMarbleFixture;
    FILE* f=std::fopen(argv[2],"wb");if(!f)return false;
    const char* reason=refusal(object);
    std::fprintf(f,"{\"schema\":\"atomic_marble_physical_roundtrip_v1\",\"object_id\":%u,\"client\":%s,\"refused\":",
        object->getId(),object->isClientObject()?"true":"false");
    if(reason){AIGeometry::quoted(f,reason);std::fprintf(f,",\"passed\":false}\n");std::fclose(f);return false;}
    std::fprintf(f,"null,\"clock_before\":%u,\"before\":",Sim::getCurrentTime());AIMarbleDynamics::state(f,object);
    std::fprintf(f,",\"fields_before\":");fields(f,object);
    Snapshot saved;saved.capture(object);
    // Only direct copied fields change; no transform setter, network mask, world
    // callback, power activation or native physics routine is called here.
    object->mPosition+=Point3D(1,2,3);object->mVelocity+=Point3D(4,5,6);object->mOmega+=Point3D(7,8,9);
    object->mGroundTime=123;object->mRadius*=2;object->mPowerUpId=0;
    object->mCameraYaw+=1;object->mCameraPitch+=.25f;object->mCameraOffset.set(1,0,0);
    object->mBestContact.normal.set(1,0,0);object->mLastContact.position.set(3,4,5);
    object->mContacts.clear();object->delta.move=NullMove;::gFirstMove=NullMove;::gNextMove=NullMove;
    const bool changed=!saved.equal(object);
    saved.restore(object);
    const bool passed=changed && saved.equal(object);
    std::fprintf(f,",\"after\":");AIMarbleDynamics::state(f,object);
    std::fprintf(f,",\"fields_after\":");fields(f,object);
    std::fprintf(f,",\"clock_after\":%u,\"perturbation_observed\":%s,\"passed\":%s,\"physics_steps\":0,\"world_restored\":false,\"whole_state_certified\":false}\n",
        Sim::getCurrentTime(),changed?"true":"false",passed?"true":"false");
    const bool good=std::ferror(f)==0;return std::fclose(f)==0 && good && passed;
}

ConsoleMethod(Marble,aiFixtureCapture,bool,4,4,"Setup-only same-process physical client/server capture") {
    using namespace AIMarbleFixture;
    Marble* server=dynamic_cast<Marble*>(Sim::findObject(dAtoi(argv[3])));
    FILE* f=std::fopen(argv[2],"wb");if(!f)return false;
    captured.ready=false;const char* reason=pairRefusal(object,server,false);
    if(!reason){
        captured.client.capture(object);captured.server.capture(server);
        captured.clientId=object->getId();captured.serverId=server->getId();captured.missionId=Sim::findObject("MissionGroup")->getId();
        captured.references.clear();captured.rememberContacts(object);captured.rememberContacts(server);captured.ready=true;
    }
    pairReport(f,object,server,"capture",reason,!reason);const bool good=std::ferror(f)==0;
    return std::fclose(f)==0 && good && !reason;
}
ConsoleMethod(Marble,aiFixtureRestore,bool,3,3,"Restore captured physical bodies without world/clock rewind") {
    using namespace AIMarbleFixture;
    Marble* server=dynamic_cast<Marble*>(Sim::findObject(captured.serverId));
    FILE* f=std::fopen(argv[2],"wb");if(!f)return false;
    const char* reason=pairRefusal(object,server,true);bool passed=false;
    if(!reason){
        captured.server.restore(server,true);captured.client.restore(object,true);
        passed=captured.client.equal(object,false) && captured.server.equal(server,false);
        if(!passed)reason="physical_restore_mismatch";
    }
    pairReport(f,object,server,"restore",reason,passed);const bool good=std::ferror(f)==0;
    return std::fclose(f)==0 && good && passed;
}
ConsoleMethod(Marble,aiFixturePose,bool,13,13,"Setup-only atomic pose/velocity/spin; no pending-update bypass") {
    using namespace AIMarbleFixture;
    Marble* server=dynamic_cast<Marble*>(Sim::findObject(dAtoi(argv[3])));
    FILE* f=std::fopen(argv[2],"wb");if(!f)return false;
    const char* reason=pairRefusal(object,server,false);F64 values[9];
    if(!reason)for(U32 i=0;i<9;++i){
        char* end=NULL;values[i]=std::strtod(argv[4+i],&end);
        if(!end || *end || !std::isfinite(values[i]) || std::fabs(values[i])>(i<3?100000:i<6?100:500)){
            reason="fixture_coordinate_not_supported";break;
        }
    }
    if(!reason){
        const Point3D p(values[0],values[1],values[2]),v(values[3],values[4],values[5]),w(values[6],values[7],values[8]);
        setPose(server,p,v,w);setPose(object,p,v,w);::gFirstMove=NullMove;::gNextMove=NullMove;captured.ready=false;
    }
    pairReport(f,object,server,"synthetic_pose",reason,!reason);const bool good=std::ferror(f)==0;
    return std::fclose(f)==0 && good && !reason;
}

namespace AIObservedScene {
struct Writer { FILE* f;U32 count;Marble* owned;bool players; };
static void matrix(FILE* f,const MatrixF& m) {
    std::fputc('[',f);for(U32 row=0;row<4;++row)for(U32 col=0;col<4;++col)
        std::fprintf(f,"%s%.9g",row || col?",":"",m(row,col));std::fputc(']',f);
}
static void collect(SceneObject* object,void* value) {
    Writer* writer=(Writer*)value;FILE* f=writer->f;if(writer->count++)std::fputc(',',f);
    std::fprintf(f,"{\"id\":%u,\"type_mask\":%u,\"class\":",object->getId(),object->getType());AIGeometry::quoted(f,object->getClassName());
    std::fprintf(f,",\"name\":");AIGeometry::quoted(f,object->getName());
    std::fprintf(f,",\"position\":");AIMarbleDynamics::vec(f,object->getPosition());std::fprintf(f,",\"scale\":");AIMarbleDynamics::vec(f,object->getScale());
    std::fprintf(f,",\"transform\":");matrix(f,object->getTransform());const Box3F& box=object->getWorldBox();
    std::fprintf(f,",\"bounds\":[");AIMarbleDynamics::vec(f,box.min);std::fputc(',',f);AIMarbleDynamics::vec(f,box.max);std::fputc(']',f);
    GameBase* game=dynamic_cast<GameBase*>(object);GameBaseData* data=game?game->getDataBlock():NULL;
    std::fprintf(f,",\"datablock\":");AIGeometry::quoted(f,data && data->getName()?data->getName():"");
    if(writer->players){Marble* marble=dynamic_cast<Marble*>(object);
        std::fprintf(f,",\"owned_body\":%s,\"partial_body_state\":true",marble==writer->owned?"true":"false");
        if(marble){std::fprintf(f,",\"native_position\":");AIMarbleDynamics::vec(f,marble->mPosition);std::fprintf(f,",\"velocity\":");AIMarbleDynamics::vec(f,marble->mVelocity);std::fprintf(f,",\"spin\":");AIMarbleDynamics::vec(f,marble->mOmega);
            std::fprintf(f,",\"radius\":%.9g,\"mode\":%u,\"oob\":%s",marble->mRadius,marble->mMode,marble->mOOB?"true":"false");}
    }else{StaticShape* shape=dynamic_cast<StaticShape*>(object);StaticShapeData* forces=dynamic_cast<StaticShapeData*>(data);
        std::fprintf(f,",\"force_definitions_supported\":%s",shape && forces?"true":"false");
        if(shape && forces){std::fprintf(f,",\"powered\":%s,\"forces\":[",shape->isPowered()?"true":"false");
            for(U32 i=0;i<4;++i){if(i)std::fputc(',',f);MatrixF node(true);shape->getMountTransform(forces->forceNode[i],&node);
                std::fprintf(f,"{\"type\":%d,\"node\":%d,\"radius\":%.9g,\"strength\":%.9g,\"arc\":%.9g,\"vector\":",
                    forces->forceType[i],forces->forceNode[i],forces->forceRadius[i],forces->forceStrength[i],forces->forceArc[i]);
                AIMarbleDynamics::vec(f,forces->forceVector[i]);std::fprintf(f,",\"mount_transform\":");matrix(f,node);std::fputc('}',f);}
            std::fputc(']',f);}
    }
    std::fputc('}',f);
}
static void container(FILE* f,Container& container,Marble* owned) {
    std::fprintf(f,"{\"force_objects\":[");Writer force={f,0,owned,false};container.findObjects(ForceObjectType,collect,&force);
    std::fprintf(f,"],\"player_bodies\":[");Writer player={f,0,owned,true};container.findObjects(PlayerObjectType,collect,&player);
    std::fprintf(f,"],\"force_count\":%u,\"player_count\":%u}",force.count,player.count);
}
}
ConsoleMethod(Marble,aiFixtureScene,bool,3,3,"Read-only owned frozen scene inventory; no force evaluation or state restore") {
    if(!Con::getBoolVariable("$AI::Lockstep") || !Con::getBoolVariable("$AI::WaitReply") ||
        !Con::getBoolVariable("$Server::Hosting") || Con::getBoolVariable("$MLAgent::Live") ||
        !object->isClientObject() || object->getId()!=(U32)Con::getIntVariable("$MP::MyMarble"))return false;
    FILE* f=std::fopen(argv[2],"wb");if(!f)return false;
    AIMarbleFixture::Snapshot before;before.capture(object);const U32 clock=Sim::getCurrentTime(),frame=Platform::getVirtualMilliseconds();
    std::fprintf(f,"{\"schema\":\"observed_scene_context_v1\",\"before_clock_ms\":%u,\"before_frame_ms\":%u,\"client\":",clock,frame);
    AIObservedScene::container(f,gClientContainer,object);std::fprintf(f,",\"server\":");AIObservedScene::container(f,gServerContainer,AIObservedScriptState::authority());
    const bool unchanged=before.equal(object,false) && clock==Sim::getCurrentTime() && frame==Platform::getVirtualMilliseconds();
    std::fprintf(f,",\"after_clock_ms\":%u,\"after_frame_ms\":%u,\"physical_fields_unchanged\":%s,\"complete_force_forecast\":false,\"world_restored\":false}",
        Sim::getCurrentTime(),Platform::getVirtualMilliseconds(),unchanged?"true":"false");
    const bool good=std::ferror(f)==0;const bool closed=std::fclose(f)==0;return unchanged && good && closed;
}

void aiFixtureBootstrap() {
    // Install an isolated command extension after ordinary bridge scripts load.
    // Existing bridge files are untouched. A fixture command does NOT call
    // Parent::onLine, so it cannot release the engine's waiting physics step.
    static bool installed=false;
    if(installed || !Con::getBoolVariable("$MLAgent::Enabled") || !Con::getBoolVariable("$AI::Lockstep"))return;
    installed=true;
    Con::evaluate(
        "package AIPhysicalFixtureProbe { function AIBridgeSocket::onLine(%this,%line) { "
        "if (getWord(%line,0) $= \"FIXTURECHECK\") { "
        "%ok=false; if(isObject($MP::MyMarble)) %ok=$MP::MyMarble.aiFixtureRoundTrip(getWord(%line,1)); "
        "AIBridge::sendState(\"DEBUG|fixture_roundtrip|\" @ (%ok+0)); return; } "
        "if(getWord(%line,0) $= \"FIXTURECAPTURE\") { %ok=false; "
        "if(isObject($MP::MyMarble) && isObject(ClientGroup) && ClientGroup.getCount()==1) "
        "%ok=$MP::MyMarble.aiFixtureCapture(getWord(%line,1),ClientGroup.getObject(0).player); "
        "AIBridge::sendState(\"DEBUG|fixture_capture|\" @ (%ok+0));return; } "
        "if(getWord(%line,0) $= \"FIXTURERESTORE\") { %ok=false; "
        "if(isObject($MP::MyMarble)) %ok=$MP::MyMarble.aiFixtureRestore(getWord(%line,1)); "
        "AIBridge::sendState(\"DEBUG|fixture_restore|\" @ (%ok+0));return; } "
        "if(getWord(%line,0) $= \"FIXTUREPOSE\") { %ok=false; "
        "if(isObject($MP::MyMarble) && isObject(ClientGroup) && ClientGroup.getCount()==1) "
        "%ok=$MP::MyMarble.aiFixturePose(getWord(%line,1),ClientGroup.getObject(0).player,"
        "getWord(%line,2),getWord(%line,3),getWord(%line,4),getWord(%line,5),getWord(%line,6),"
        "getWord(%line,7),getWord(%line,8),getWord(%line,9),getWord(%line,10)); "
        "AIBridge::sendState(\"DEBUG|fixture_pose|\" @ (%ok+0));return; } "
        "if(getWord(%line,0) $= \"FIXTURERECORD\") { %ok=false; "
        "if(isObject($MP::MyMarble) && $AI::WaitReply && $AI::Lockstep && $Server::Hosting && !$MLAgent::Live) "
        "%ok=$MP::MyMarble.aiRecordDynamics(getWord(%line,1)); "
        "AIBridge::sendState(\"DEBUG|fixture_record|\" @ (%ok+0));return; } "
        "if(getWord(%line,0) $= \"FIXTURESCENE\") { %ok=false; "
        "if(isObject($MP::MyMarble)) %ok=$MP::MyMarble.aiFixtureScene(getWord(%line,1)); "
        "AIBridge::sendState(\"DEBUG|fixture_scene|\" @ (%ok+0));return; } "
        "Parent::onLine(%this,%line); } }; activatePackage(AIPhysicalFixtureProbe);",false);
}
#undef AI_FIXTURE_FIELDS
