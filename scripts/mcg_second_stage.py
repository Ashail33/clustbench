"""Second stages after MCG's micro-cluster graph (round 22).

Compares, on the same micro-cluster graph (mcg_cf):

* lmm_mcg (in the library): LMM's EM on the graph's eigenvector basis;
* SVD of the point-to-micro-cluster bipartite graph (anchor graph), with and
  without CF-kernel weighting of the co-membership matrix;
* support vector clustering (Ben-Hur et al., 2001) on the micro-clusters,
  weighted by their counts: a one-class SVM boundary, centres joined when
  the segment between them stays inside, kernel width bisected until there
  are k significant components.

Evaluates on the 140 in-distribution tasks, the 35 unseen tasks (task lists
from docs/data_mcg) and 8 families at n = 20k (two seeds):

    python scripts/mcg_second_stage.py
"""
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from clustbench.algorithms.base import ALGO_REGISTRY
from clustbench.algorithms.mcg import Mcg_cf
from clustbench.datasets import DATASETS, DataSpec

def topk(X, C, s, chunk=4096):
    X32=X.astype(np.float32); C32=C.astype(np.float32); cn=(C32**2).sum(1); xn=(X32**2).sum(1)
    I=np.empty((len(X),s),np.int64); D=np.empty((len(X),s))
    for st in range(0,len(X),chunk):
        Dm=cn[None]-2*X32[st:st+chunk]@C32.T
        p=np.argpartition(Dm,s-1,axis=1)[:,:s]; d=np.take_along_axis(Dm,p,1); o=np.argsort(d,1)
        I[st:st+chunk]=np.take_along_axis(p,o,1); D[st:st+chunk]=np.maximum(np.take_along_axis(d,o,1)+xn[st:st+chunk,None],0)
    return I,D

# ---------- B: SVD of the point-to-micro-cluster bipartite graph ----------
def svd_bipartite(X,k,s=3,gamma=1.0,seed=42):
    g=Mcg_cf().build_graph(X,k); C=g["C"]; m=len(C); n=len(X); s=min(s,m)
    I,D=topk(X,C,s); w=np.exp(-D/(gamma*g["R2"][I]+1e-12)); w=w/(w.sum(1,keepdims=True)+1e-300)
    Z=sp.csr_matrix((w.ravel(),I.ravel(),np.arange(0,n*s+1,s)),shape=(n,m))
    lam=np.asarray(Z.sum(0)).ravel()+1e-12; Li=1/np.sqrt(lam)
    G=(Z.T@Z).toarray()*Li[:,None]*Li[None,:]
    ev,V=np.linalg.eigh(G); V=V[:,::-1][:,:k]; sv=np.sqrt(np.maximum(ev[::-1][:k],1e-12))
    U=(Z@(V*Li[:,None]))/sv
    U=U/np.maximum(np.linalg.norm(U,axis=1,keepdims=True),1e-12)
    return KMeans(k,n_init=10,random_state=seed).fit_predict(U)

# ---------- C: support vector clustering on micro-clusters ----------
def svc_mcg(X,k,nu=0.05,n_seg=8,seed=42):
    from sklearn.svm import OneClassSVM
    g=Mcg_cf().build_graph(X,k); C=g["C"]; N=g["N"]; W=g["W"]; a=g["a"]; m=len(C)
    alive=N>0; iu,ju=np.nonzero(np.triu(W>0,1))
    sq=(C**2).sum(1); D2=np.maximum(sq[:,None]+sq[None]-2*C@C.T,0); base=np.median(D2[iu,ju]) if len(iu) else 1.0
    ts=np.linspace(0,1,n_seg+2)[1:-1]
    def comps(qmult):
        svm=OneClassSVM(kernel="rbf",gamma=qmult/base,nu=nu).fit(C[alive],sample_weight=N[alive])
        P=(C[iu][:,None,:]*(1-ts)[None,:,None]+C[ju][:,None,:]*ts[None,:,None]).reshape(-1,C.shape[1])
        inside=(svm.decision_function(P).reshape(len(iu),n_seg)>=-1e-3).all(1)
        A=sp.coo_matrix((np.ones(inside.sum()),(iu[inside],ju[inside])),shape=(m,m))
        nc,lab=connected_components(A,directed=False)
        sizes=np.bincount(lab,weights=N,minlength=nc)
        big=np.sum(sizes>=0.02*N.sum())
        return big,lab,sizes
    lo,hi=np.log(1e-2),np.log(1e3); best=None
    for _ in range(18):
        mid=(lo+hi)/2; big,lab,sizes=comps(np.exp(mid))
        if best is None or abs(big-k)<abs(best[0]-k) or (big==k): best=(big,lab,sizes)
        if big==k: break
        if big<k: lo=mid
        else: hi=mid
    _,lab,sizes=best
    keep=np.argsort(-sizes)[:k]; cl=-np.ones(m,int)
    for i,c in enumerate(keep): cl[lab==c]=i
    # unassigned centres: nearest assigned centre
    un=np.where(cl<0)[0]; asg=np.where(cl>=0)[0]
    if len(un) and len(asg):
        cl[un]=cl[asg[np.argmin(D2[np.ix_(un,asg)],axis=1)]]
    cl[cl<0]=0
    return cl[a]

def svd_bipartite_cf(X,k,s=3,gamma=1.0,seed=42):
    """Bipartite SVD, but the m x m co-membership matrix is CF-kernel weighted like MCG's edges."""
    g=Mcg_cf().build_graph(X,k); C=g["C"]; m=len(C); n=len(X); s=min(s,m)
    I,D=topk(X,C,s); w=np.exp(-D/(gamma*g["R2"][I]+1e-12)); w=w/(w.sum(1,keepdims=True)+1e-300)
    Z=sp.csr_matrix((w.ravel(),I.ravel(),np.arange(0,n*s+1,s)),shape=(n,m))
    sq=(C**2).sum(1); D2=np.maximum(sq[:,None]+sq[None]-2*C@C.T,0)
    K=np.exp(-D2/(g["R2"][:,None]+g["R2"][None,:]+1e-12))
    G=(Z.T@Z).toarray()*K; deg=G.sum(1)+1e-12; Li=1/np.sqrt(deg)
    G=G*Li[:,None]*Li[None,:]
    ev,V=np.linalg.eigh(G); V=V[:,::-1][:,:k]
    U=Z@(V*Li[:,None]); U=U/np.maximum(np.linalg.norm(U,axis=1,keepdims=True),1e-12)
    return KMeans(k,n_init=10,random_state=seed).fit_predict(U)


def suites():
    out = {}
    for s in ("paper", "unseen"):
        r = pd.read_json(f"docs/data_mcg/results_{s}.json")
        tasks = []
        for _, t in r.iterrows():
            spec = DataSpec(n_samples=int(t.n_samples), n_features=int(t.n_features), centers=int(t.k_target),
                            compactness=float(t.compactness), seed=int(t.seed), outliers=int(t.outliers),
                            noise=int(t.noise), density=float(t.density), outlier_extremity=float(t.outlier_extremity))
            X, y = DATASETS[t.dataset_id](spec)
            tasks.append((t.dataset_id, int(t.k_target), np.asarray(X, float), np.asarray(y)))
        out[s] = tasks
    fams = [("moons", 2), ("rings", 10), ("heavy_tailed_mixture", 10), ("hierarchical_nested", 10),
            ("VariableDensityBridges", 10), ("mdcgen", 10), ("circles", 2), ("mixed_shapes", 5)]
    out["big20k"] = [(f, k, *map(np.asarray, DATASETS[f](DataSpec(n_samples=20000, n_features=20, centers=k,
                                                                    compactness=1.0, seed=s))))
                     for f, k in fams for s in (1, 2)]
    return out


def main():
    methods = {
        "mcg_cf": lambda X, k: ALGO_REGISTRY["mcg_cf"]().fit_predict(X, k=k).labels,
        "lmm_mcg (hard)": lambda X, k: ALGO_REGISTRY["lmm_mcg"]().fit_predict(X, k=k).labels,
        "lmm_mcg (soft)": lambda X, k: ALGO_REGISTRY["lmm_mcg"](interp="soft").fit_predict(X, k=k).labels,
        "svd s=5 g=4": lambda X, k: svd_bipartite(X, k, s=5, gamma=4.0),
        "svd_cf s=5": lambda X, k: svd_bipartite_cf(X, k, s=5),
        "svc nu=0.05": lambda X, k: svc_mcg(X, k, nu=0.05),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = suites()
        with threadpool_limits(1):
            for name, fn in methods.items():
                row = []
                for s, tasks in S.items():
                    aris, tt = [], []
                    for _, k, X, y in tasks:
                        t0 = time.perf_counter()
                        aris.append(adjusted_rand_score(y, fn(X, k)))
                        tt.append(time.perf_counter() - t0)
                    row.append(f"{s} {np.mean(aris):.3f} ({np.median(tt):.2f}s)")
                print(f"{name:16s}", " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
