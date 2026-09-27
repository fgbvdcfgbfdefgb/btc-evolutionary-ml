import copy, random, numpy as np, torch

def candidate(base, lr, wd): return {"model":copy.deepcopy(base),"optimizer":None,"lr":lr,"weight_decay":wd,"fitness":-1e30,"metrics":{}}
def make_optimizer(c):
    c["optimizer"]=torch.optim.AdamW(c["model"].parameters(),lr=c["lr"],weight_decay=c["weight_decay"])
    state=c.pop("optimizer_state",None)
    if state: c["optimizer"].load_state_dict(state)
def breed(elites,size,mutation=.05,crossover=.10):
    out=elites
    while len(out)<size:
        a,b=random.sample(elites,2) if len(elites)>1 else (elites[0],elites[0]); child=copy.deepcopy(a)
        if random.random()<crossover:
            with torch.no_grad():
                for p,q in zip(child["model"].parameters(),b["model"].parameters()): p.lerp_(q,.15)
        child["lr"]*=float(np.exp(np.random.normal(0,mutation))); child["weight_decay"]*=float(np.exp(np.random.normal(0,mutation)))
        with torch.no_grad():
            for p in child["model"].parameters():
                if random.random()<mutation: p.add_(torch.randn_like(p),alpha=mutation*0.002)
        child["optimizer"]=None; child["fitness"]=-1e30; out.append(child)
    return out
