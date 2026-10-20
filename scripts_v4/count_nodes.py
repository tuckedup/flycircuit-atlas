import numpy as np, json
from flycircuit.eval import learning_v2 as V2, learning_v3 as V3
p=V3.load_protocol(); b=V2.load_base(p,"flywire"); g=b.graph
n=g.n; s=set(map(int,b.sensory_idx)); r=list(map(int,b.readout_idx))
indeg=np.bincount(g.post,minlength=n); outdeg=np.bincount(g.pre,minlength=n)
w=np.bincount(g.post,weights=g.count,minlength=n)
print("n",n,"edges",g.n_edges,"sensory",len(s),"readout",len(r),"overlap",len(s&set(r)))
elig=[i for i in range(n) if i not in s and i not in set(r)]
print("non-sensory non-readout",len(elig))
print("elig with indeg>0",sum(indeg[i]>0 for i in elig),"outdeg>0",sum(outdeg[i]>0 for i in elig))
print("readout indeg",[int(indeg[i]) for i in r][:25])
print("sign counts",{int(k):int(v) for k,v in zip(*np.unique(b.sign,return_counts=True))})
print([a for a in dir(b) if not a.startswith('_')])
