import glob, os, shutil, torch
files=glob.glob("models/best_island*.pt")
if not files: raise SystemExit("No island models found")
best=max(files,key=lambda f:torch.load(f,map_location="cpu",weights_only=False)["metrics"]["fitness"])
shutil.copy2(best,"models/best.pt"); print(f"Selected {best} -> models/best.pt")
