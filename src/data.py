import os, time, requests, numpy as np, pandas as pd
from torch.utils.data import Dataset

URL="https://api.binance.com/api/v3/klines"
COLS=["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_base","taker_quote","ignore"]

def download_year(path="data/btcusdt_1m.parquet", symbol="BTCUSDT", days=365):
    if os.path.exists(path): return path
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
    end=int(time.time()*1000); start=end-days*86400_000; rows=[]; s=requests.Session()
    while start < end:
        r=s.get(URL,params={"symbol":symbol,"interval":"1m","startTime":start,"endTime":end,"limit":1000},timeout=30)
        r.raise_for_status(); batch=r.json()
        if not batch: break
        rows.extend(batch); start=int(batch[-1][0])+60_000; time.sleep(.04)
    df=pd.DataFrame(rows,columns=COLS).drop_duplicates("open_time")
    for c in COLS[1:6]: df[c]=pd.to_numeric(df[c])
    df.to_parquet(path,index=False); return path

def prepare(raw="data/btcusdt_1m.parquet", out="data/features.npz", horizons=(1,5,15,60,240,720)):
    if os.path.exists(out): return out
    df=pd.read_parquet(raw); c=df.close.astype("float64"); v=df.volume.astype("float64")
    x=pd.DataFrame(index=df.index)
    x["ret1"]=np.log(c).diff(); x["range"]=(df.high-df.low)/c; x["body"]=(df.close-df.open)/df.open
    x["logvol"]=np.log1p(v); x["volchg"]=np.log1p(v).diff()
    for w in (5,15,60,240,720):
        x[f"ret{w}"]=np.log(c/c.shift(w)); x[f"vol{w}"]=x.ret1.rolling(w).std()
        x[f"vrel{w}"]=v/(v.rolling(w).mean()+1e-9)
    minute=(df.open_time//60000)%1440
    x["sin_t"]=np.sin(2*np.pi*minute/1440); x["cos_t"]=np.cos(2*np.pi*minute/1440)
    y=np.stack([np.log(c.shift(-h)/c).to_numpy() for h in horizons],1)
    valid=np.isfinite(x.to_numpy()).all(1)&np.isfinite(y).all(1)
    X=x.to_numpy(dtype=np.float32)[valid]; Y=y.astype(np.float32)[valid]
    cut=int(.8*len(X)); mu=X[:cut].mean(0); sd=X[:cut].std(0)+1e-6; X=(X-mu)/sd
    os.makedirs(os.path.dirname(out) or ".",exist_ok=True)
    np.savez(out,X=X.astype(np.float32),y=Y,mean=mu,std=sd,horizons=np.array(horizons),features=np.array(x.columns))
    return out

class WindowDataset(Dataset):
    def __init__(self,path,split="train",seq=240):
        z=np.load(path,mmap_mode="r"); self.X=z["X"]; self.y=z["y"]; self.seq=seq
        cut=int(.8*len(self.X)); self.a=seq if split=="train" else cut; self.b=cut if split=="train" else len(self.X)
    def __len__(self): return self.b-self.a
    def __getitem__(self,i):
        j=self.a+i; return self.X[j-self.seq:j],self.y[j]
