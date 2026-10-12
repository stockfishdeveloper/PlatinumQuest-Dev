// Read-only geometry instrumentation for OpenPQ-TGEMIT-mbx.
// Include from marble.cc; build a separate probe executable. No physics changes.
#include "collision/concretePolyList.h"
#include "sim/sceneObject.h"
#include <cstdio>
#include <cstring>

namespace AIGeometry {
static const U32 Mask = StaticObjectType | TerrainObjectType | InteriorObjectType |
    WaterObjectType | StaticShapeObjectType | StaticTSObjectType;

static void quoted(FILE* f, const char* s) {
    std::fputc('"', f);
    if (s) for (; *s; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { std::fputc('\\', f); std::fputc(c, f); }
        else if (c < 32) std::fprintf(f, "\\u%04x", c);
        else std::fputc(c, f);
    }
    std::fputc('"', f);
}
static void point(FILE* f, const Point3F& p) {
    std::fprintf(f, "[%.9g,%.9g,%.9g]", p.x, p.y, p.z);
}
struct Writer { FILE* f; U32 objects; U32 polygons; };
static const char* field(SceneObject* o, const char* name) {
    // Ghosts disable ModStaticFields, making getDataField return an empty
    // string. Read the registered typed field without changing those flags.
    const AbstractClassRep::Field* f = o->findField(StringTable->insert(name));
    if (!f || f->elementCount != 1) return "";
    const char* value = Con::getData(f->type, (void*)(((const char*)o)+f->offset), 0, f->table, f->flag);
    return (*f->getDataFn)(o, value);
}
static void collect(SceneObject* o, void* data) {
    Writer* w = (Writer*)data;
    const Box3F box = o->getWorldBox();
    Point3F center = (box.min + box.max) * 0.5f;
    // A rotated, nonuniformly scaled interior's oriented-box hull culling can
    // omit hulls even when passed its own world AABB. Use a circumscribed cube
    // with ample scale-ratio padding; this is export coverage, not collision
    // broadphase during gameplay. DTS collision builders retain all details.
    const Point3F scale = o->getScale();
    const F32 smallest = getMin(getMin(mFabs(scale.x),mFabs(scale.y)),mFabs(scale.z));
    const F32 largest = getMax(getMax(mFabs(scale.x),mFabs(scale.y)),mFabs(scale.z));
    const F32 radius = ((box.max-box.min).len()+1.f)*4.f*getMax(1.f,largest/getMax(.00001f,smallest));
    const Point3F pad(radius,radius,radius);
    Box3F queryBox(center-pad,center+pad);
    SphereF sphere(center, radius*2.f);
    ConcretePolyList list;
    const bool returned = o->buildPolyList(PLC_Collision, &list, queryBox, sphere);
    std::fprintf(w->f, "{\"kind\":\"object\",\"id\":%u,\"type_mask\":%u,\"class\":", o->getId(), o->getType());
    quoted(w->f, o->getClassName());
    std::fprintf(w->f, ",\"name\":"); quoted(w->f, o->getName());
    std::fprintf(w->f, ",\"interior_file\":"); quoted(w->f, field(o, "interiorFile"));
    std::fprintf(w->f, ",\"shape_name\":"); quoted(w->f, field(o, "shapeName"));
    GameBase* game = dynamic_cast<GameBase*>(o);
    GameBaseData* db = game ? game->getDataBlock() : NULL;
    std::fprintf(w->f, ",\"datablock\":"); quoted(w->f, db ? db->getName() : "");
    ShapeBaseData* shape = dynamic_cast<ShapeBaseData*>(db);
    std::fprintf(w->f, ",\"shape_file\":"); quoted(w->f, shape ? shape->shapeName : "");
    std::fprintf(w->f, ",\"position\":"); point(w->f, o->getPosition());
    std::fprintf(w->f, ",\"scale\":"); point(w->f, o->getScale());
    std::fprintf(w->f, ",\"transform\":[");
    for (U32 row = 0; row < 4; ++row) for (U32 col = 0; col < 4; ++col)
        std::fprintf(w->f, "%s%.9g", row || col ? "," : "", o->getTransform()(row, col));
    std::fprintf(w->f, "],\"bounds\":["); point(w->f, box.min); std::fputc(',', w->f); point(w->f, box.max);
    std::fprintf(w->f, "],\"build_returned\":%s,\"polygons\":%u}\n", returned ? "true" : "false", list.mPolyList.size());
    ++w->objects;
    for (U32 i = 0; i < list.mPolyList.size(); ++i) {
        const ConcretePolyList::Poly& p = list.mPolyList[i];
        MaterialProperty* prop = o->getMaterialProperty(p.material);
        std::fprintf(w->f, "{\"kind\":\"polygon\",\"object_id\":%u,\"surface_key\":%u,\"material_index\":%u,\"material\":",
            o->getId(), p.surfaceKey, p.material);
        quoted(w->f, prop ? prop->getName() : "DefaultMaterial");
        std::fprintf(w->f, ",\"properties_resolved\":%s,\"friction\":%.9g,\"restitution\":%.9g,\"force\":%.9g,\"normal\":[%.9g,%.9g,%.9g],\"plane_d\":%.9g,\"vertices\":[",
            prop ? "true" : "false", prop ? prop->friction : 1.f, prop ? prop->restitution : 1.f,
            prop ? prop->force : 0.f, p.plane.x, p.plane.y, p.plane.z, p.plane.d);
        for (U32 j = 0; j < p.vertexCount; ++j) {
            if (j) std::fputc(',', w->f);
            point(w->f, list.mVertexList[list.mIndexList[p.vertexStart + j]]);
        }
        std::fprintf(w->f, "]}\n");
        ++w->polygons;
    }
}
}

ConsoleFunction(aiExportCollisionGeometry, bool, 3, 3, "aiExportCollisionGeometry(file, client): read-only collision snapshot") {
    FILE* f = std::fopen(argv[1], "wb");
    if (!f) return false;
    const bool client = dAtob(argv[2]);
    std::fprintf(f, "{\"kind\":\"header\",\"schema\":\"engine_collision_v1\",\"client\":%s,\"sim_time_ms\":%u,\"gravity\":[%.9g,%.9g,%.9g],\"collision_mask\":%u}\n",
        client ? "true" : "false", Sim::getCurrentTime(), gGlobalGravityDir.x, gGlobalGravityDir.y, gGlobalGravityDir.z, AIGeometry::Mask);
    AIGeometry::Writer writer = { f, 0, 0 };
    (client ? gClientContainer : gServerContainer).findObjects(AIGeometry::Mask, AIGeometry::collect, &writer);
    std::fprintf(f, "{\"kind\":\"complete\",\"objects\":%u,\"polygons\":%u}\n", writer.objects, writer.polygons);
    const bool good = std::ferror(f) == 0;
    return std::fclose(f) == 0 && good;
}

ConsoleFunction(aiCheckGeometryRays, bool, 4, 4, "aiCheckGeometryRays(input, output, client): independent engine ray checks") {
    FILE* in = std::fopen(argv[1], "rb");
    if (!in) return false;
    FILE* out = std::fopen(argv[2], "wb");
    if (!out) { std::fclose(in); return false; }
    Container& container = dAtob(argv[3]) ? gClientContainer : gServerContainer;
    double x,y,z,a,b,c;
    U32 id = 0;
    while (std::fscanf(in, "%lf %lf %lf %lf %lf %lf", &x,&y,&z,&a,&b,&c) == 6) {
        RayInfo hit;
        const bool yes = container.castRay(Point3F((F32)x,(F32)y,(F32)z), Point3F((F32)a,(F32)b,(F32)c), AIGeometry::Mask, &hit);
        std::fprintf(out, "{\"id\":%u,\"hit\":%s", id++, yes ? "true" : "false");
        if (yes) {
            std::fprintf(out, ",\"object_id\":%u,\"t\":%.9g,\"point\":", hit.object->getId(), hit.t);
            AIGeometry::point(out, hit.point);
            std::fprintf(out, ",\"normal\":"); AIGeometry::point(out, hit.normal);
        }
        std::fprintf(out, "}\n");
    }
    const bool good = std::ferror(in) == 0 && std::ferror(out) == 0;
    std::fclose(in);
    return std::fclose(out) == 0 && good;
}

// Explicit offline probe only: use the real swept-sphere collision routine.
// Restore position/contact/poly cache after the batch. testMove may queue shape
// collision notifications; run on disposable probe sessions, never a live policy.
ConsoleMethod(Marble, aiCheckGeometrySweeps, bool, 4, 5, "marble.aiCheckGeometrySweeps(input, output, optionalMaxSegmentLength)") {
    FILE* in = std::fopen(argv[2], "rb");
    if (!in) return false;
    FILE* out = std::fopen(argv[3], "wb");
    if (!out) { std::fclose(in); return false; }
    const Point3D savedPosition = object->mPosition;
    const Marble::Contact savedContact = object->mLastContact;
    U32 id = 0;
    double x,y,z,a,b,c,r;
    const F64 maxSegment = argc > 4 ? dAtof(argv[4]) : 0.0;
    while (std::fscanf(in, "%lf %lf %lf %lf %lf %lf %lf", &x,&y,&z,&a,&b,&c,&r) == 7) {
        Point3D start(x,y,z), velocity(a-x,b-y,c-z);
        const U32 segments = maxSegment > 0 ? (U32)mCeil(velocity.len()/maxSegment) + 1 : 1;
        velocity *= 1.0/segments;
        bool hit = false;
        F64 fraction = 1.0;
        for (U32 part=0; part<segments; ++part) {
            F64 dt = 1.0;
            object->mPosition = start;
            hit = object->testMove(velocity, start, dt, r, AIGeometry::Mask, false);
            if (hit) { fraction = (part+dt)/segments; break; }
        }
        std::fprintf(out, "{\"id\":%u,\"hit\":%s,\"t\":%.12g,\"client_container\":%s,\"max_segment_u\":%.9g,\"center\":[%.12g,%.12g,%.12g]",
            id++, hit ? "true" : "false", fraction, object->getContainer()==&gClientContainer ? "true" : "false", maxSegment, start.x, start.y, start.z);
        if (hit) {
            const Marble::Contact& contact = object->mLastContact;
            std::fprintf(out, ",\"normal\":[%.12g,%.12g,%.12g],\"point\":[%.12g,%.12g,%.12g],\"friction\":%.9g,\"restitution\":%.9g,\"force\":%.9g",
                contact.normal.x,contact.normal.y,contact.normal.z,
                contact.position.x,contact.position.y,contact.position.z,
                contact.friction,contact.restitution,contact.force);
        }
        std::fprintf(out, "}\n");
    }
    object->mPosition = savedPosition;
    object->mLastContact = savedContact;
    object->mPolyList.clear(); // normal physics rebuilds this on the next findContacts
    const bool good = std::ferror(in) == 0 && std::ferror(out) == 0;
    std::fclose(in);
    return std::fclose(out) == 0 && good;
}
