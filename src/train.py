import argparse, json, os, random, signal, time
import numpy as np, torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from src.data import download_year, prepare, WindowDataset
from src.model import Forecaster
from src.evolution import candidate, make_optimizer, breed
from src.evaluate import evaluate
from src.hardware import detect, tune

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def unwrap(m): return m.module if hasattr(m,"module") else m

def args():
 p=argparse.ArgumentParser(); p.add_argument("--data",default="data/features.npz"); p.add_argument("--raw",default="data/btcusdt_1m.parquet")
 p.add_argument("--horizons",default="1,5,15,60,240,720"); p.add_argument("--hours",type=float,default=5); p.add_argument("--population",type=int,default=8)
 p.add_argument("--mutation-rate",type=float,default=.03); p.add_argument("--crossover-rate",type=float,default=.08); p.add_argument("--elite-fraction",type=float,default=.5)
 p.add_argument("--epochs-per-generation",type=int,default=1); p.add_argument("--steps-per-candidate",type=int,default=500); p.add_argument("--batch-size",type=int,default=128)
 p.add_argument("--accumulate",type=int,default=1); p.add_argument("--sequence",type=int,default=240); p.add_argument("--lr",type=float,default=3e-4)
 p.add_argument("--checkpoint-dir",default="checkpoints"); p.add_argument("--model-dir",default="models"); p.add_argument("--checkpoint-minutes",type=float,default=10)
 p.add_argument("--seed",type=int,default=42); p.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--download",action=argparse.BooleanOptionalAction,default=True)
 p.add_argument("--island-id",type=int,default=0); return p.parse_args()

def main(a=None):
 a=a or args(); rank=int(os.environ.get("RANK",0)); local=int(os.environ.get("LOCAL_RANK",0)); world=int(os.environ.get("WORLD_SIZE",1)); ddp=world>1
 if ddp: torch.cuda.set_device(local); torch.distributed.init_process_group("nccl"); device=torch.device("cuda",local)
 else: device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 seed_all(a.seed+a.island_id); workers=tune(); os.makedirs(a.checkpoint_dir,exist_ok=True); os.makedirs(a.model_dir,exist_ok=True)
 if rank==0: print(json.dumps(detect("data"),indent=2),flush=True)
 if a.download and rank==0: prepare(download_year(a.raw),a.data,tuple(map(int,a.horizons.split(","))))
 if ddp: torch.distributed.barrier()
 tr=WindowDataset(a.data,"train",a.sequence); va=WindowDataset(a.data,"val",a.sequence)
 sampler=DistributedSampler(tr,shuffle=True) if ddp else None
 kw={"batch_size":a.batch_size,"num_workers":workers,"pin_memory":device.type=="cuda","persistent_workers":workers>0}
 if workers: kw["prefetch_factor"]=4
 tl=DataLoader(tr,shuffle=sampler is None,sampler=sampler,drop_last=True,**kw); vl=DataLoader(va,shuffle=False,**kw)
 z=np.load(a.data,mmap_mode="r"); base=Forecaster(z["X"].shape[1],z["y"].shape[1]).to(device)
 pop=[candidate(base,a.lr*float(np.exp(np.random.normal(0,.15))),1e-4) for _ in range(a.population)]
 ck=os.path.join(a.checkpoint_dir,f"evolution_island{a.island_id}.pt"); generation=0; best=-1e30
 if a.resume and os.path.exists(ck):
  s=torch.load(ck,map_location=device,weights_only=False); generation=s["generation"]; best=s["best"]; pop=s["population"]
  random.setstate(s["py_rng"]); np.random.set_state(s["np_rng"]); torch.set_rng_state(s["torch_rng"])
  for c in pop: c["model"].to(device)
  print(f"Resumed generation {generation}",flush=True)
 start=time.time(); deadline=start+a.hours*3600; last_ck=0; stop=False
 def save():
  if rank: return
  serial=[]
  for c in pop:
   m=unwrap(c["model"]); serial.append({**{k:v for k,v in c.items() if k not in ("model","optimizer")},"model":m.cpu(),"optimizer_state":c["optimizer"].state_dict() if c.get("optimizer") else None}); m.to(device)
  torch.save({"generation":generation,"best":best,"population":serial,"py_rng":random.getstate(),"np_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,"args":vars(a)},ck)
 signal.signal(signal.SIGTERM,lambda *_:(save(),exit(0)))
 while time.time()<deadline:
  if sampler: sampler.set_epoch(generation)
  for ci,c in enumerate(pop):
   m=c["model"].to(device); c["model"]=DDP(m,device_ids=[local]) if ddp else m
   make_optimizer(c); opt=c["optimizer"]; scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda"); c["model"].train(); opt.zero_grad(set_to_none=True)
   for step,(x,y) in enumerate(tl):
    if step>=a.steps_per_candidate or time.time()>=deadline: break
    x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
    dtype=torch.bfloat16 if device.type=="cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=="cuda"): loss=torch.nn.functional.smooth_l1_loss(c["model"](x),y)/a.accumulate
    scaler.scale(loss).backward()
    if (step+1)%a.accumulate==0: scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
   c["metrics"]=evaluate(c["model"],vl,device,ddp); c["fitness"]=c["metrics"]["fitness"]
   c["model"]=unwrap(c["model"])
  pop.sort(key=lambda c:c["fitness"],reverse=True); best=max(best,pop[0]["fitness"])
  if rank==0:
   print(json.dumps({"generation":generation,"best":pop[0]["metrics"],"elapsed_hours":(time.time()-start)/3600}),flush=True)
   torch.save({"state_dict":pop[0]["model"].state_dict(),"metrics":pop[0]["metrics"],"horizons":a.horizons},os.path.join(a.model_dir,f"best_island{a.island_id}.pt"))
  n=max(1,int(a.population*a.elite_fraction)); pop=breed(pop[:n],a.population,a.mutation_rate,a.crossover_rate); generation+=1
  if time.time()-last_ck>a.checkpoint_minutes*60: save(); last_ck=time.time()
 save()
 if ddp: torch.distributed.destroy_process_group()
if __name__=="__main__": main()
