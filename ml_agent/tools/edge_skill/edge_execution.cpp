// Standalone skill forecast. Reuses the existing physics and bridge without
// changing or registering a family in the production planner.
#include "../../nav/native/marble_sim.cpp"

namespace edge_skill {
constexpr double radius=.189750001;
inline double clip(double x,double lo,double hi) { return pymin(hi,pymax(lo,x)); }
inline V3 local(const double* g,V3 p,bool position=true) {
    if(position)p=sub(p,load(g));
    return {p.x*g[3]+p.y*g[4],p.x*(-g[4])+p.y*g[3],p.z};
}
inline bool inside(const double* g,V3 p) {
    double x=p.x-g[7],y=p.y-g[8];
    return std::abs(x*g[27]+y*g[28])<=g[9]/2-.29 && std::abs(x*g[29]+y*g[30])<=g[10]/2-.29;
}
inline double floor_height(const double* g,V3 p){return g[15]+((p.x-g[13])*g[25]+(p.y-g[14])*g[26]);}
inline double floor_vz(const double* g,V3 v){return v.z-(v.x*g[25]+v.y*g[26]);}
struct Controller {
    const double* g;const double* q;
    bool jumped=false,landing=false,contact=false,entered=false,finished=false;
    int exit_count=0,done_at=-1;
    Controller(const double* gg,const double* qq):g(gg),q(qq),jumped(gg[21]!=0) {}
    Action action(const double* state,int index,const double* guide=nullptr,bool future_contact=false) {
        V3 p=local(g,load(state)),v=local(g,load(state+3),false);
        double ex=g[16],ey=g[17],es=std::sqrt(ex*ex+ey*ey),ux=ex/pymax(es,1e-6),uy=ey/pymax(es,1e-6);
        double dx=g[13]-p.x,dy=g[14]-p.y,ve=std::sqrt((v.x-ex)*(v.x-ex)+(v.y-ey)*(v.y-ey));
        bool supported=std::abs(p.z-floor_height(g,p))<.045 && std::abs(floor_vz(g,v))<.7 && inside(g,p);
        if(p.x>.22)landing=true;
        if(supported && landing) {
            if(std::sqrt(dx*dx+dy*dy)<g[19] && ve<2.5)entered=true;
            if(entered && ve<3.)exit_count++;else exit_count=0;
            if(exit_count>=9){finished=true;if(done_at<0)done_at=index;}
        } else if(!finished)exit_count=0;
        V3 cp=guide?local(g,load(guide)):p,cv=guide?local(g,load(guide+3),false):v;
        bool control_landing=landing || (guide && (future_contact || cp.x>.22));
        bool control_supported=std::abs(cp.z-floor_height(g,cp))<.045 && std::abs(floor_vz(g,cv))<.7 && inside(g,cp);
        double ax=0,ay=0;int jump=0;
        if(finished){ax=-v.x*5.;ay=-v.y*5.;}
        else if(!control_landing) {
            double center_y=q[1]-radius*g[24]*q[2];
            ax=(q[0]-v.x)*2.5;ay=(clip((center_y-p.y)*2.,-q[0],q[0])-v.y)*2.5;
            bool on_floor=std::abs(p.z-g[20]-radius)<.05 && std::abs(v.z)<.7;
            bool future_on_floor=std::abs(cp.z-g[20]-radius)<.05 && std::abs(cv.z)<.7;
            if(q[9]>.5 && !future_on_floor) {
                double height=g[5]+g[24]*q[1]+radius*q[2],disc=cv.z*cv.z-40*(height-cp.z);
                if(disc>=0) {
                    double a=(cv.z-std::sqrt(disc))/20,b=(cv.z+std::sqrt(disc))/20,t=-1;
                    if(q[4]>.5){if(b>.035)t=b;else if(a>.035)t=a;}
                    else {if(a>.035)t=a;else if(b>.035)t=b;}
                    if(t>0) {
                        double tx=-radius*std::sqrt(pymax(0.,1-q[2]*q[2]*(1+g[24]*g[24])));
                        ax=2.*(tx-cp.x-cv.x*t)/pymax(t*t,.025);ay=2.*(center_y-cp.y-cv.y*t)/pymax(t*t,.025);
                    }
                }
            }
            if(!jumped && g[22]==0) {
                double height=g[5]+g[24]*q[1]+radius*q[2]-g[20]-radius,disc=7.5*7.5-40*height;
                if(disc>=0) {
                    double flight=(7.5+(q[4]>.5?1:-1)*std::sqrt(disc))/20;
                    double trigger=pymax(v.x,1.)*(flight+.080)+q[3];
                    if(-p.x<=trigger && on_floor){jump=1;jumped=true;}
                }
            }
        } else if(control_supported) {
            double cdx=g[13]-cp.x,cdy=g[14]-cp.y;
            double along=cdx*ux+cdy*uy,lx=cdx-ux*along,ly=cdy-uy*along;
            double extra=pymax(0.,pymin(2.,along*.35));
            ax=(ex+ux*extra+lx*2.-cv.x)*3.;ay=(ey+uy*extra+ly*2.-cv.y)*3.;
        } else {
            double vv=floor_vz(g,cv),disc=vv*vv+40*(cp.z-floor_height(g,cp)),t=(vv+std::sqrt(pymax(0.,disc)))/20;
            if(t>.10) {
                double tx=g[13]-ex*.30+ux*q[5]-uy*q[8],ty=g[14]-ey*.30+uy*q[5]+ux*q[8];
                t=pymax(t,.18);
                ax=q[6]*(tx-cp.x-cv.x*t)/(t*t)+q[7]*(ex-cv.x)/t;
                ay=q[6]*(ty-cp.y-cv.y*t)/(t*t)+q[7]*(ey-cv.y)/t;
            }else {ax=(ex-cv.x)*2.5+(g[13]-cp.x)*.5;ay=(ey-cv.y)*2.5+(g[14]-cp.y)*.5;}
        }
        double length=std::sqrt(ax*ax+ay*ay);
        bool ground=supported || (std::abs(v.z)<.7 && std::abs(p.z-radius-g[20])<.05);
        double maximum=ground?8.05:5.*std::sqrt(2.),throttle=pymin(1.,length/maximum);
        if(g[23]!=0)throttle=pymin(1.0002,(std::floor(throttle*16)+.5)/16);
        double wx=g[3]*ax+(-g[4])*ay,wy=g[4]*ax+g[3]*ay;
        Action act{};joystick(wx,wy,throttle,false,jump,0,0.,0.,act.a);
        // bridge.reply: ordinary movement, no powerup or Blast.
        act.a[6]=0;act.a[7]=act.a[5];act.a[8]=0;return act;
    }
    void observe(const double* rr) {
        for(int i=0;i<4;i++) {
            const double* a=rr+(size_t)i*S_SIZE;const double* b=a+S_SIZE;
            V3 p=local(g,load(a)),n=local(g,load(b));
            double x=p.x,z=(p.z-g[5]-g[24]*p.y)/std::sqrt(1+g[24]*g[24]);
            double dx=n.x-p.x,dz=(n.z-g[5]-g[24]*n.y)/std::sqrt(1+g[24]*g[24])-z;
            double f=clip(-(x*dx+z*dz)/(dx*dx+dz*dz+1e-15),0.,1.);
            double qx=x+dx*f,qz=z+dz*f;
            if(std::sqrt(qx*qx+qz*qz)<radius+.10 && qz>.01 && qz<radius+.02 && qx<.015 &&
               std::abs(p.y)<g[6]/2-.1 && b[5]-a[5]+.32>.6){contact=true;landing=true;return;}
        }
    }
    void save(double* out){out[0]=jumped;out[1]=landing;out[2]=contact;out[3]=entered;out[4]=exit_count;out[5]=finished;out[6]=done_at;}
};
}

DLL int ms_edge_skill_version(){return 7;}
// Independent replay with contact events. It consumes the checked command table,
// never the trajectory's predicted future states. Classification stays in Python.
DLL int ms_edge_contacts(void* h,const double* state_in,const double* bridge,const double* replies,
                         int ticks,double* events_out,int capacity,int* event_count,double* states_out) {
    Map* map=(Map*)h;State s;unpack(state_in,s);std::vector<Event> events;int n=0;
    for(int i=0;i<ticks;i++) {
        int k=i%4,j=i/4;const double* r=replies+(size_t)j*R_SIZE;
        const double* mv=k==0?(j==0?bridge:replies+(size_t)(j-1)*R_SIZE+R_MOVE1):r+R_MOVE1;
        // This entry point is exclusively the unpowered skill contract.
        if(r[22]!=0 || mv[4]!=0 || mv[5]!=0)return 100;
        V3 before=s.v,before_p=s.p;s.clock_ms=(long long)bridge[17]+16LL*i;
        if(k==0){s.yaw=f32(r[21]);s.next_yaw=s.yaw;s.has_next_yaw=true;}
        Move move{mv[0],mv[1],mv[2],mv[3]!=0,false,false};events.clear();
        int code=step(*map,s,move,16,events);if(code)return code;
        pack(s,states_out+(size_t)i*S_SIZE);
        for(const auto& e:events) {
            if((e.kind!=2 && e.kind!=4) || std::abs(e.a.z)>=.9999)continue;
            if(n>=capacity)return 101;
            double* out=events_out+(size_t)n++*20;
            out[0]=e.kind;out[1]=e.triangle;out[2]=i+1;store(out+3,e.a);store(out+6,e.b);
            store(out+9,before_p);store(out+12,before);store(out+15,s.v);out[18]=e.sub;out[19]=e.value;
        }
    }
    *event_count=n;return 0;
}

DLL int ms_edge_many(void* h,const double* state_in,const double* bridge_in,const double* blast_in,const double* inv_in,
                    const double* consts,const double* geometry,const double* params,int n,int count,int early_stop,double* rows,
                    double* replies,double* bridge_out,int* statuses,double* memories) {
    for(int j=0;j<n;j++) {
        edge_skill::Controller policy(geometry,params+(size_t)j*10);
        double s[S_SIZE],bridge[18],blast[13],blast_next[13],q[Q_SIZE]{};
        std::memcpy(s,state_in,sizeof s);std::memcpy(bridge,bridge_in,sizeof bridge);std::memcpy(blast,blast_in,sizeof blast);
        double* rr=rows+(size_t)j*(count*4+1)*S_SIZE;double* reps=replies+(size_t)j*count*R_SIZE;
        std::memcpy(rr,s,sizeof s);int* status=statuses+j*8;
        int done=0,fail=-1,code=0;
        for(int i=0;i<count;i++) {
            bool lookahead=policy.q[9]>1.5;
            Action a{};if(!lookahead)a=policy.action(s,i);
            double frame_rows[5*S_SIZE],bo[1];int st[8];double* r=reps+(size_t)i*R_SIZE;
            forecast_impl(h,nullptr,s,bridge,blast,inv_in,1,0,1,a.a,0,q,nullptr,nullptr,consts,0.,-1,
                          frame_rows,r,bo,blast_next,st,nullptr);
            if(st[2]>=0){fail=i;code=st[3];break;}
            if(lookahead) {
                edge_skill::Controller preview=policy;preview.observe(frame_rows);
                a=policy.action(s,i,frame_rows+4*S_SIZE,preview.contact);
                std::memcpy(r+R_ACTION,a.a,sizeof a.a);
            }
            std::memcpy(rr+(size_t)(4*i+1)*S_SIZE,frame_rows+S_SIZE,4*S_SIZE*sizeof(double));
            policy.observe(frame_rows);std::memcpy(s,frame_rows+4*S_SIZE,sizeof s);
            std::memcpy(bridge,r+R_MOVE1,6*sizeof(double));std::memcpy(bridge+6,a.a,9*sizeof(double));
            bridge[15]=st[7];bridge[16]=bo[0];bridge[17]+=64;
            std::memcpy(blast,blast_next,sizeof blast);done++;
            if(early_stop && policy.finished && i>=policy.done_at+10)break;
            // This primitive is the first traversal of the nominated lip. Once
            // it has passed without contact, a later unrelated recovery is not
            // a successful execution of this opportunity.
            if(early_stop && !policy.contact && edge_skill::local(geometry,load(s)).x>.35)break;
        }
        status[0]=done;status[1]=4*done+1;status[2]=fail;status[3]=code;status[4]=0;status[5]=0;status[6]=4*done;status[7]=(int)bridge[15];
        bridge_out[j]=bridge[16];policy.save(memories+j*7);
    }
    return 0;
}
