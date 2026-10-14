// Isolated analytical sweep prototype. No game calls, fast math, surface removal
// or modified physics. Compile with /fp:strict; validate before any live use.
#include <cmath>
#include <cstdint>
#include <limits>

struct V3 { double x,y,z; };
static V3 load(const double* p) { return {p[0],p[1],p[2]}; }
static V3 add(V3 a,V3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
static V3 sub(V3 a,V3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
static V3 mul(V3 a,double b) { return {a.x*b,a.y*b,a.z*b}; }
static double dot(V3 a,V3 b) { return (a.x*b.x+a.y*b.y)+a.z*b.z; }
static V3 cross(V3 a,V3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
static double positive(double x) { return x>0 ? x : 0; }
static double f32root(double x) { return double(std::sqrt(float(x))); }

extern "C" __declspec(dllexport)
void sweep_candidates(std::int64_t count,const std::int64_t* ids,const double* faces,const double* source,
    const double* planes,const std::uint8_t* boundary,const double* ab,const double* ac,
    const double* face_cross,const double* nn,const double* start_data,const double* velocity_data,
    double speed2,double radius,double native_dt,double* times,double* points)
{
    const V3 start=load(start_data),velocity=load(velocity_data);
    const double inf=std::numeric_limits<double>::infinity();
    const double approach_limit=-.001*std::sqrt(speed2);
    for(std::int64_t k=0;k<count;k++) {
        const auto id=ids[k]; const V3 normal=load(planes+id*4);
        const double approach=dot(normal,velocity);
        const bool valid=approach<approach_limit;
        double best=inf; V3 closest={0,0,0};
        if(valid) {
            const double distance=dot(normal,start)+planes[id*4+3];
            const double t=(radius-distance)/approach;
            const V3 point=sub(add(start,mul(velocity,t)),mul(normal,radius));
            const V3 rel=sub(point,load(faces+id*9));
            const double u=nn[id]>0 ? dot(cross(rel,load(ac+id*3)),load(face_cross+id*3))/nn[id] : 0;
            const double v=nn[id]>0 ? dot(cross(load(ab+id*3),rel),load(face_cross+id*3))/nn[id] : 0;
            if(nn[id]>0 && t>0 && t<1 && u>=-1e-7 && v>=-1e-7 && u+v<=1+1e-7) { best=t;closest=point; }
        }
        if(valid) for(int j=0;j<3;j++) {
            const V3 a=load(source+id*9+j*3),b=load(source+id*9+((j+1)%3)*3);
            const V3 edge=sub(b,a),pv=sub(start,a);
            const double length2=dot(edge,edge);
            const double ve=length2>0 ? dot(edge,velocity)/length2 : 0;
            const double pe=length2>0 ? dot(pv,edge)/length2 : 0;
            const V3 vr=sub(velocity,mul(edge,ve)),pr=sub(pv,mul(edge,pe));
            const double qa=dot(vr,vr),qb=2*dot(pr,vr),qc=dot(pr,pr)-radius*radius;
            const double disc=qb*qb-4*qa*qc;
            double root=std::sqrt(positive(disc));
            if(native_dt>0) {
                const double scaled=positive(disc)*length2*length2/(native_dt*native_dt);
                root=length2>0 ? f32root(scaled)*native_dt/length2 : 0;
            }
            const double t=qa>1e-18 ? (-qb-root)/(2*qa) : inf;
            const double exit_t=qa>1e-18 ? (-qb+root)/(2*qa) : -inf;
            const double along=pe+(std::isfinite(t) ? t : 0)*ve;
            if(boundary[id*3+j] && length2>0 && disc>=0 && t>0 && t<1 && t<best && along>=0 && along<=1 &&
                (native_dt<=0 || exit_t*native_dt>.0001)) {
                best=t;closest=add(a,mul(edge,along));
            }
            const double vqb=2*dot(pv,velocity),vqc=dot(pv,pv)-radius*radius;
            const double vdisc=vqb*vqb-4*speed2*vqc;
            double vroot=std::sqrt(positive(vdisc));
            if(native_dt>0)vroot=f32root(positive(vdisc)/(native_dt*native_dt))*native_dt;
            const double vt=(-vqb-vroot)/(2*speed2),vexit=(-vqb+vroot)/(2*speed2);
            if(vdisc>=0 && vt>0 && vt<1 && vt<best && (native_dt<=0 || vexit*native_dt>.0001)) { best=vt;closest=a; }
        }
        times[k]=best;points[k*3]=closest.x;points[k*3+1]=closest.y;points[k*3+2]=closest.z;
    }
}
