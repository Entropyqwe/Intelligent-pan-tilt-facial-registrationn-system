"""Small CPU motion/assignment primitives. Predictions never drive control.

The XYWH constant-velocity model follows the tracking-by-detection family
(ByteTrack/BoT-SORT); this implementation uses seconds, not assumed 30 Hz.
"""
import numpy as np


def measurement(box):
    x,y,w,h=box
    return np.array([x+w/2,y+h/2,w,h],dtype=np.float64)


class BoxMotion:
    def __init__(self,box):
        self.mean=np.r_[measurement(box),np.zeros(4)]
        size=max(box[3],40.)
        self.cov=np.diag(np.square([size*.1]*4+[size]*4))

    @property
    def box(self):
        cx,cy,w,h=self.mean[:4];w=max(w,4.);h=max(h,8.)
        return (float(cx-w/2),float(cy-h/2),float(w),float(h))

    def predict(self,dt):
        dt=max(0.,min(float(dt),2.))
        transition=np.eye(8);transition[:4,4:]=np.eye(4)*dt
        scale=max(self.mean[3],40.)
        # Continuous white acceleration; size and position use the same units.
        acceleration=np.diag(np.square([scale*1.2]*2+[scale*.5]*2))
        noise=np.block([[acceleration*dt**4/4,acceleration*dt**3/2],
                        [acceleration*dt**3/2,acceleration*dt**2]])
        noise+=np.eye(8)*1e-6
        self.mean=transition@self.mean
        self.cov=transition@self.cov@transition.T+noise
        self.mean[2:4]=np.maximum(self.mean[2:4],[4.,8.])

    def project(self):
        scale=max(self.mean[3],40.)
        return self.mean[:4],self.cov[:4,:4]+np.eye(4)*(scale*.05)**2

    def distance(self,box):
        projected,cov=self.project();delta=measurement(box)-projected
        return float(delta@np.linalg.solve(cov,delta))

    def update(self,box):
        projected,cov=self.project()
        gain=np.linalg.solve(cov,self.cov[:,:4].T).T
        self.mean+=gain@(measurement(box)-projected)
        # Joseph form keeps covariance positive under long variable-dt runs.
        residual=np.eye(8);residual[:,:4]-=gain
        noise=cov-self.cov[:4,:4]
        self.cov=residual@self.cov@residual.T+gain@noise@gain.T
        self.cov=(self.cov+self.cov.T)*.5


def _hungarian(cost):
    """Shortest augmenting path, rows <= columns; finite double costs."""
    n,m=cost.shape
    u=np.zeros(n+1);v=np.zeros(m+1)
    p=np.zeros(m+1,dtype=int);way=np.zeros(m+1,dtype=int)
    for row in range(1,n+1):
        p[0]=row;j0=0
        minimum=np.full(m+1,np.inf);used=np.zeros(m+1,dtype=bool)
        while True:
            used[j0]=True;i0=p[j0]
            free=np.flatnonzero(~used[1:])+1
            current=cost[i0-1,free-1]-u[i0]-v[free]
            better=current<minimum[free]
            minimum[free[better]]=current[better];way[free[better]]=j0
            j1=free[np.argmin(minimum[free])];delta=minimum[j1]
            u[p[used]]+=delta;v[used]-=delta;minimum[~used]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if not j0:break
    return [(int(p[j]-1),j-1) for j in range(1,m+1) if p[j]]


def assign(cost,limit):
    """Global partial assignment with private unmatched columns per track.

    Forbidden edges cannot displace valid matches. Prefer maximum cardinality,
    then minimum cost; using a plain assignment and dropping edges is wrong.
    """
    matrix=np.asarray(cost,dtype=np.float64)
    if matrix.ndim!=2:raise ValueError('Expected two-dimensional costs')
    n,m=matrix.shape
    if not n or not m:return []
    allowed=np.isfinite(matrix)&(matrix<=limit)
    penalty=(n+1)*(limit+1.)
    padded=np.full((n,m+n),penalty,dtype=np.float64)
    padded[:,:m]=np.where(allowed,matrix,penalty*(n+2))
    return [(i,j) for i,j in _hungarian(padded) if j<m and allowed[i,j]]
