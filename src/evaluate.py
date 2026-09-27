import numpy as np, torch
@torch.inference_mode()
def evaluate(model,loader,device,distributed=False):
    model.eval(); se=ae=direction=n=0.0
    for x,y in loader:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); p=model(x)
        se+=((p-y)**2).sum().item(); ae+=(p-y).abs().sum().item(); direction+=(torch.sign(p)==torch.sign(y)).sum().item(); n+=y.numel()
    t=torch.tensor([se,ae,direction,n],device=device,dtype=torch.float64)
    if distributed: torch.distributed.all_reduce(t)
    se,ae,direction,n=t.cpu().tolist(); return {"rmse":(se/n)**.5,"mae":ae/n,"directional_accuracy":direction/n,"fitness":-(se/n)**.5}
