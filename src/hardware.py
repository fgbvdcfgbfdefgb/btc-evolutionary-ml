import os, platform, shutil, subprocess, tempfile, time
import psutil, torch

def _cmd(cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: return ""

def topology():
    return _cmd(["nvidia-smi", "topo", "-m"])

def has_nvlink(topo=None):
    t = topo if topo is not None else topology()
    return any(f"NV{i}" in t for i in range(1, 19))

def disk_hint(path="."):
    os.makedirs(path, exist_ok=True)
    p = os.path.join(path, ".io_test")
    block = b"0" * (16 * 1024 * 1024); t=time.perf_counter()
    try:
        with open(p,"wb",buffering=0) as f: f.write(block); os.fsync(f.fileno())
        return round(16/(time.perf_counter()-t),1)
    finally:
        try: os.remove(p)
        except OSError: pass

def detect(path="data"):
    topo=topology(); g=[]
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        g.append({"id":i,"name":p.name,"vram_gib":round(p.total_memory/2**30,2),"capability":f"{p.major}.{p.minor}"})
    return {"platform":platform.platform(),"physical_cores":psutil.cpu_count(False),
      "logical_cores":psutil.cpu_count(True),"ram_gib":round(psutil.virtual_memory().total/2**30,2),
      "cuda":torch.version.cuda,"torch":torch.__version__,"gpus":g,"nvlink":has_nvlink(topo),
      "topology":topo or "nvidia-smi unavailable","disk_write_mib_s_hint":disk_hint(path)}

def tune():
    n=max(1,psutil.cpu_count(True) or 1); torch.set_num_threads(max(1,n//2))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark=True
        torch.set_float32_matmul_precision("high")
    return max(1,min(16,n//max(1,torch.cuda.device_count() or 1)-1))
