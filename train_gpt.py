from __future__ import annotations
import copy,glob,io,math,os,random,subprocess,sys,time,uuid,zlib
from pathlib import Path
try:
    import zstandard as _zstd
    _C="zstd"
except:
    _C="zlib"
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor,nn
from torch.nn.parallel import DistributedDataParallel as DDP

class H:
    dp=os.environ.get("DATA_PATH","./data/datasets/fineweb10B_sp1024")
    tf=os.path.join(dp,"fineweb_train_*.bin")
    vf=os.path.join(dp,"fineweb_val_*.bin")
    tp=os.environ.get("TOKENIZER_PATH","./data/tokenizers/fineweb_1024_bpe.model")
    rid=os.environ.get("RUN_ID",str(uuid.uuid4()))
    sd=int(os.environ.get("SEED",42))
    vbs=int(os.environ.get("VAL_BATCH_SIZE",524288))
    vle=int(os.environ.get("VAL_LOSS_EVERY",500))
    tle=int(os.environ.get("TRAIN_LOG_EVERY",100))
    it=int(os.environ.get("ITERATIONS",20000))
    wdi=int(os.environ.get("WARMDOWN_ITERS",3000))
    wus=int(os.environ.get("WARMUP_STEPS",20))
    tbt=int(os.environ.get("TRAIN_BATCH_TOKENS",786432))
    tsl=int(os.environ.get("TRAIN_SEQ_LEN",2048))
    mws=float(os.environ.get("MAX_WALLCLOCK_SECONDS",600.0))
    qgi=float(os.environ.get("QK_GAIN_INIT",1.5))
    vs=int(os.environ.get("VOCAB_SIZE",1024))
    nl=int(os.environ.get("NUM_LAYERS",10))
    nkh=int(os.environ.get("NUM_KV_HEADS",4))
    md=int(os.environ.get("MODEL_DIM",512))
    nh=int(os.environ.get("NUM_HEADS",8))
    mm=float(os.environ.get("MLP_MULT",3.0))
    te=bool(int(os.environ.get("TIE_EMBEDDINGS","1")))
    rb=float(os.environ.get("ROPE_BASE",10000.0))
    lsc=float(os.environ.get("LOGIT_SOFTCAP",30.0))
    elr=float(os.environ.get("EMBED_LR",0.6))
    hlr=float(os.environ.get("HEAD_LR",0.008))
    telr=float(os.environ.get("TIED_EMBED_LR",0.03))
    teis=float(os.environ.get("TIED_EMBED_INIT_STD",0.005))
    mlr=float(os.environ.get("MATRIX_LR",0.02))
    slr=float(os.environ.get("SCALAR_LR",0.02))
    mm_=float(os.environ.get("MUON_MOMENTUM",0.99))
    mbs=int(os.environ.get("MUON_BACKEND_STEPS",5))
    mmws=float(os.environ.get("MUON_MOMENTUM_WARMUP_START",0.92))
    mmwu=int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS",1500))
    b1=float(os.environ.get("BETA1",0.9))
    b2=float(os.environ.get("BETA2",0.95))
    ae=float(os.environ.get("ADAM_EPS",1e-8))
    gcn=float(os.environ.get("GRAD_CLIP_NORM",0.3))
    wd=float(os.environ.get("WEIGHT_DECAY",0.04))
    es=int(os.environ.get("EVAL_STRIDE",64))
    ebs=int(os.environ.get("EVAL_BATCH_SEQS",32))
    bvs=int(os.environ.get("BIGRAM_VOCAB_SIZE",10240))
    bd=int(os.environ.get("BIGRAM_DIM",128))
    swae=bool(int(os.environ.get("SWA_ENABLED","1")))
    swasf=float(os.environ.get("SWA_START_FRAC",0.4))
    swaev=int(os.environ.get("SWA_EVERY",50))

CTRL=tuple(p for p in os.environ.get("CTRL_PATTERNS","attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,smear,bigram.scale").split(",")if p)
FP16P=tuple(p for p in os.environ.get("FP16_PATTERNS","tok_emb,blocks.8.attn.c_k").split(",")if p)

def tns(t:Tensor)->int:return int(t.numel())*int(t.element_size())

def zpns(G:Tensor,st:int=10,eps:float=1e-7)->Tensor:
    a,b,c=(3.4445,-4.7750,2.0315)
    X=G.bfloat16()
    X/=X.norm()+eps
    tr=G.size(0)>G.size(1)
    if tr:X=X.T
    for _ in range(st):
        A=X@X.T
        X=a*X+(b*A+c*A@A)@X
    return X.T if tr else X

class MUON(torch.optim.Optimizer):
    def __init__(s,ps,lr,mm,bs,ne=True,wd=0.0):
        super().__init__(ps,dict(lr=lr,momentum=mm,backend_steps=bs,nesterov=ne,weight_decay=wd))
    @torch.no_grad()
    def step(s,cl=None):
        l=None
        if cl:
            with torch.enable_grad():l=cl()
        d=dist.is_available() and dist.is_initialized()
        ws=dist.get_world_size()if d else 1
        rk=dist.get_rank()if d else 0
        for g in s.param_groups:
            ps=g["params"]
            if not ps:continue
            lr,mm,bs,ne=g["lr"],g["momentum"],g["backend_steps"],g["nesterov"]
            tp=sum(int(p.numel())for p in ps)
            uf=torch.zeros(tp,device=ps[0].device,dtype=torch.bfloat16)
            cr=0
            for i,p in enumerate(ps):
                if i%ws==rk and p.grad is not None:
                    gr=p.grad
                    st=s.state[p]
                    if"mb"not in st:st["mb"]=torch.zeros_like(gr)
                    bu=st["mb"]
                    bu.mul_(mm).add_(gr)
                    if ne:gr=gr.add(bu,alpha=mm)
                    gr=zpns(gr,st=bs)
                    gr*=max(1,gr.size(0)/gr.size(1))**0.5
                    uf[cr:cr+p.numel()]=gr.reshape(-1)
                cr+=p.numel()
            if d:dist.all_reduce(uf,op=dist.ReduceOp.SUM)
            wd=g.get("weight_decay",0.0)
            cr=0
            for p in ps:
                gr=uf[cr:cr+p.numel()].view_as(p).to(dtype=p.dtype)
                if wd>0:p.data.mul_(1.0-lr*wd)
                p.add_(gr,alpha=-lr)
                cr+=p.numel()
        return l

def bsl(sp:spm.SentencePieceProcessor,vs:int,dev:torch.device)->tuple:
    svs=int(sp.vocab_size())
    ts=max(svs,vs)
    bb=np.zeros((ts,),dtype=np.int16)
    hls=np.zeros((ts,),dtype=np.bool_)
    ibt=np.ones((ts,),dtype=np.bool_)
    for ti in range(svs):
        if sp.is_control(ti)or sp.is_unknown(ti)or sp.is_unused(ti):continue
        ibt[ti]=False
        if sp.is_byte(ti):
            bb[ti]=1
            continue
        pc=sp.id_to_piece(ti)
        if pc.startswith("\u2581"):
            hls[ti]=True
            pc=pc[1:]
        bb[ti]=len(pc.encode("utf-8"))
    return(torch.tensor(bb,dtype=torch.int16,device=dev),torch.tensor(hls,dtype=torch.bool,device=dev),torch.tensor(ibt,dtype=torch.bool,device=dev))

def lvt(pt:str,sl:int)->Tensor:
    fs=[Path(p)for p in sorted(glob.glob(pt))]
    if not fs:raise FileNotFoundError(pt)
    tks=torch.cat([lsh(f)for f in fs]).contiguous()
    u=((tks.numel()-1)//sl)*sl
    if u<=0:raise ValueError(f"Val too short for seq_len={sl}")
    return tks[:u+1]

def lsh(f:Path)->Tensor:
    hb=256*np.dtype("<i4").itemsize
    tb=np.dtype("<u2").itemsize
    hd=np.fromfile(f,dtype="<i4",count=256)
    if hd.size!=256 or int(hd[0])!=20240520 or int(hd[1])!=1:raise ValueError(f"Bad shard {f}")
    nt=int(hd[2])
    es=hb+nt*tb
    if f.stat().st_size!=es:raise ValueError(f"Size mismatch {f}")
    tn=np.fromfile(f,dtype="<u2",count=nt,offset=hb)
    return torch.from_numpy(tn.astype(np.uint16,copy=False))

class TS:
    def __init__(s,pt:str):
        s.fs=[Path(p)for p in sorted(glob.glob(pt))]
        if not s.fs:raise FileNotFoundError(pt)
        s.fi=0
        s.tks=lsh(s.fs[0])
        s.pos=0
    def _adv(s):
        s.fi=(s.fi+1)%len(s.fs)
        s.tks=lsh(s.fs[s.fi])
        s.pos=0
    def take(s,n)->Tensor:
        cs=[]
        r=n
        while r>0:
            av=s.tks.numel()-s.pos
            if av<=0:
                s._adv()
                continue
            k=min(r,av)
            cs.append(s.tks[s.pos:s.pos+k])
            s.pos+=k
            r-=k
        return cs[0]if len(cs)==1 else torch.cat(cs)

class DL:
    def __init__(s,pt,rk,ws,dev):
        s.rk=rk
        s.ws=ws
        s.dev=dev
        s.ts=TS(pt)
    def next(s,gt,sl,ga)->tuple:
        lt=gt//(s.ws*ga)
        ps=lt+1
        ch=s.ts.take(ps*s.ws)
        st=s.rk*ps
        lc=ch[st:st+ps].to(dtype=torch.int64)
        x=lc[:-1].reshape(-1,sl)
        y=lc[1:].reshape(-1,sl)
        return x.to(s.dev,non_blocking=True),y.to(s.dev,non_blocking=True)

class RN(nn.Module):
    def __init__(s,eps=None):super().__init__();s.eps=eps
    def forward(s,x):return F.rms_norm(x,(x.size(-1),),eps=s.eps)

class CL(nn.Linear):
    def forward(s,x):
        w=s.weight.to(x.dtype)
        b=s.bias.to(x.dtype)if s.bias is not None else None
        return F.linear(x,w,b)

def rfp32(m):
    with torch.no_grad():
        for n,p in m.named_parameters():
            if(p.ndim<2 or any(cp in n for cp in CTRL))and p.dtype!=torch.float32:
                p.data=p.data.float()

class RT(nn.Module):
    def __init__(s,dm,bs=10000.0):
        super().__init__()
        inv=1.0/(bs**(torch.arange(0,dm,2,dtype=torch.float32)/dm))
        s.register_buffer("inv",inv,persistent=False)
        s._sl=0
        s._c=None
        s._s=None
    def forward(s,sl,dev,dty):
        if s._c is None or s._s is None or s._sl!=sl or s._c.device!=dev:
            t=torch.arange(sl,device=dev,dtype=s.inv.dtype)
            fr=torch.outer(t,s.inv.to(dev))
            s._c=fr.cos()[None,None,:,:]
            s._s=fr.sin()[None,None,:,:]
            s._sl=sl
        return s._c.to(dtype=dty),s._s.to(dtype=dty)

def ar(x,c,s):
    h=x.size(-1)//2
    x1,x2=x[...,:h],x[...,h:]
    return torch.cat((x1*c+x2*s,x1*(-s)+x2*c),dim=-1)

class AT(nn.Module):
    def __init__(s,dm,nh,nkh,rb,qgi):
        super().__init__()
        if dm%nh!=0:raise ValueError("dm%nh")
        if nh%nkh!=0:raise ValueError("nh%nkh")
        s.nh=nh
        s.nkh=nkh
        s.hd=dm//nh
        if s.hd%2!=0:raise ValueError("hd even")
        kdm=s.nkh*s.hd
        s.cq=CL(dm,dm,bias=False)
        s.ck=CL(dm,kdm,bias=False)
        s.cv=CL(dm,kdm,bias=False)
        s.pj=CL(dm,dm,bias=False)
        s.pj._zi=True
        s.qg=nn.Parameter(torch.full((nh,),qgi,dtype=torch.float32))
        s.ro=RT(s.hd,bs=rb)
    def forward(s,x):
        bz,sl,dm=x.shape
        q=s.cq(x).reshape(bz,sl,s.nh,s.hd).transpose(1,2)
        k=s.ck(x).reshape(bz,sl,s.nkh,s.hd).transpose(1,2)
        v=s.cv(x).reshape(bz,sl,s.nkh,s.hd).transpose(1,2)
        q=F.rms_norm(q,(q.size(-1),))
        k=F.rms_norm(k,(k.size(-1),))
        c,sn=s.ro(sl,x.device,q.dtype)
        q=ar(q,c,sn)
        k=ar(k,c,sn)
        q=q*s.qg.to(dtype=q.dtype)[None,:,None,None]
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=None,is_causal=True,enable_gqa=(s.nkh!=s.nh))
        return s.pj(y.transpose(1,2).contiguous().reshape(bz,sl,dm))

class MP(nn.Module):
    def __init__(s,dm,mm):
        super().__init__()
        hd=int(mm*dm)
        s.fc=CL(dm,hd,bias=False)
        s.pj=CL(hd,dm,bias=False)
        s.pj._zi=True
    def forward(s,x):return s.pj(torch.relu(s.fc(x)).square())

class SG(nn.Module):
    def __init__(s,dm):super().__init__();s.g=nn.Parameter(torch.zeros(dm,dtype=torch.float32))
    def forward(s,x):
        sg=torch.sigmoid(s.g.to(dtype=x.dtype))[None,None,:]
        xp=torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],dim=1)
        return(1-sg)*x+sg*xp

class BH(nn.Module):
    def __init__(s,bvs,bd,md):
        super().__init__()
        s.bvs=bvs
        s.em=nn.Embedding(bvs,bd)
        nn.init.zeros_(s.em.weight)
        s.pj=CL(bd,md,bias=False)if bd!=md else None
        if s.pj:nn.init.zeros_(s.pj.weight)
        s.sc=nn.Parameter(torch.tensor(0.05,dtype=torch.float32))
    def _hash(s,tk):
        t=tk.to(torch.int32)
        md=s.bvs-1
        ot=torch.empty_like(t)
        ot[...,0]=md
        ot[...,1:]=torch.bitwise_xor(36313*t[...,1:],27191*t[...,:-1])%md
        return ot.long()
    def forward(s,tk):
        h=s.em(s._hash(tk))
        if s.pj:h=s.pj(h)
        return h*s.sc.to(dtype=h.dtype)

class BK(nn.Module):
    def __init__(s,dm,nh,nkh,mm,rb,qgi):
        super().__init__()
        s.an=RN()
        s.mn=RN()
        s.at=AT(dm,nh,nkh,rb,qgi)
        s.mp=MP(dm,mm)
        s.as_=nn.Parameter(torch.ones(dm,dtype=torch.float32))
        s.ms=nn.Parameter(torch.ones(dm,dtype=torch.float32))
        s.rm=nn.Parameter(torch.stack((torch.ones(dm),torch.zeros(dm))).float())
    def forward(s,x,x0):
        mx=s.rm.to(dtype=x.dtype)
        x=mx[0][None,None,:]*x+mx[1][None,None,:]*x0
        x=x+s.as_.to(dtype=x.dtype)[None,None,:]*s.at(s.an(x))
        x=x+s.ms.to(dtype=x.dtype)[None,None,:]*s.mp(s.mn(x))
        return x

class GPT(nn.Module):
    def __init__(s,vs,nl,md,nh,nkh,mm,te,teis,lsc,rb,qgi,bvs=0,bd=128):
        super().__init__()
        if lsc<=0:raise ValueError("lsc>0")
        s.te=te
        s.teis=teis
        s.lsc=lsc
        s.tok=nn.Embedding(vs,md)
        s.bh=BH(bvs,bd,md)if bvs>0 else None
        s.nel=nl//2
        s.ndl=nl-s.nel
        s.nsw=min(s.nel,s.ndl)
        s.sw=nn.Parameter(torch.ones(s.nsw,md,dtype=torch.float32))
        s.sm=SG(md)
        s.bks=nn.ModuleList([BK(md,nh,nkh,mm,rb,qgi)for _ in range(nl)])
        s.fn=RN()
        s.lh=None if te else CL(md,vs,bias=False)
        if s.lh:s.lh._zi=True
        s._iw()
    def _iw(s):
        if s.te:nn.init.normal_(s.tok.weight,mean=0.0,std=s.teis)
        nl=len(s.bks)
        for nm,md in s.named_modules():
            if isinstance(md,nn.Linear):
                if getattr(md,"_zi",False):nn.init.zeros_(md.weight)
                elif md.weight.ndim==2 and md.weight.shape[0]>=64 and md.weight.shape[1]>=64:
                    nn.init.orthogonal_(md.weight,gain=1.0)
                    if".pj."in nm or nm.endswith(".pj"):
                        with torch.no_grad():md.weight.mul_(1.0/math.sqrt(2*nl))
    def forward(s,ii,ti):
        x=s.tok(ii)
        if s.bh:x=x+s.bh(ii)
        x=F.rms_norm(x,(x.size(-1),))
        x=s.sm(x)
        x0=x
        sk=[]
        for i in range(s.nel):
            x=s.bks[i](x,x0)
            sk.append(x)
        for i in range(s.ndl):
            if sk:x=x+s.sw[i].to(dtype=x.dtype)[None,None,:]*sk.pop()
            x=s.bks[s.nel+i](x,x0)
        x=s.fn(x).reshape(-1,x.size(-1))
        tg=ti.reshape(-1)
        if s.te:lp=F.linear(x,s.tok.weight)
        else:
            if s.lh is None:raise RuntimeError("lh required")
            lp=s.lh(x)
        lg=s.lsc*torch.tanh(lp/s.lsc)
        return F.cross_entropy(lg.float(),tg,reduction="mean")
    def fwd_logits(s,ii):
        x=s.tok(ii)
        if s.bh:x=x+s.bh(ii)
        x=F.rms_norm(x,(x.size(-1),))
        x=s.sm(x)
        x0=x
        sk=[]
        for i in range(s.nel):
            x=s.bks[i](x,x0)
            sk.append(x)
        for i in range(s.ndl):
            if sk:x=x+s.sw[i].to(dtype=x.dtype)[None,None,:]*sk.pop()
            x=s.bks[s.nel+i](x,x0)
        x=s.fn(x)
        if s.te:lp=F.linear(x,s.tok.weight)
        else:lp=s.lh(x)
        return s.lsc*torch.tanh(lp/s.lsc)

def q_i5(t,cr=15):
    t32=t.float()
    if t32.ndim==2:
        rm=t32.abs().amax(dim=1)
        sc=(rm/cr).clamp_min(1e-12).to(torch.float16).clamp_min(torch.finfo(torch.float16).tiny)
        q=torch.clamp(torch.round(t32/sc.float()[:,None]),-(cr+1),cr).to(torch.int8)
        return q,sc
    am=t32.abs().max().item()
    sc=torch.tensor(max(am/cr,1e-12),dtype=torch.float16)
    q=torch.clamp(torch.round(t32/sc.float()),-(cr+1),cr).to(torch.int8)
    return q,sc

def q_i6(t,cr=31):
    return q_i5(t,cr=cr)

def q_i8(t):
    t32=t.float()
    if t32.ndim==2:
        ca=torch.quantile(t32.abs(),0.9999984,dim=1)if t32.numel()else torch.empty((t32.shape[0],),dtype=torch.float32)
        cl=torch.maximum(torch.minimum(t32,ca[:,None]),-ca[:,None])
        sc=(ca/127.0).clamp_min(1.0/127.0)
        q=torch.clamp(torch.round(cl/sc[:,None]),-127,127).to(torch.int8).contiguous()
        return q,sc.to(dtype=torch.float16).contiguous()
    ca=float(torch.quantile(t32.abs().flatten(),0.9999984).item())if t32.numel()else 0.0
    sc=torch.tensor(ca/127.0 if ca>0 else 1.0,dtype=torch.float32)
    q=torch.clamp(torch.round(torch.clamp(t32,-ca,ca)/sc),-127,127).to(torch.int8).contiguous()
    return q,sc

def _clp(nm):
    if"tok"in nm or"lh"in nm:return"emb"
    if".mp."in nm or".fc."in nm:return"mlp"
    if"bh"in nm:return"bh"
    if".at."in nm or".cq."in nm or".ck."in nm or".cv."in nm or".pj."in nm:return"attn"
    return"oth"

def mq(sd,ic):
    rs={}
    mt={}
    for nm,t in sd.items():
        tt=t.detach().cpu().contiguous()
        ct=_clp(nm)
        if not tt.is_floating_point() or tt.numel()<=8192:
            rs[nm]=tt.to(torch.float16)if tt.is_floating_point()else tt
            mt[nm]="pt"
            continue
        if any(cp in nm for cp in CTRL):
            rs[nm]=tt.float()
            mt[nm]="ptc"
            continue
        if any(fp in nm for fp in FP16P):
            rs[nm]=tt.to(dtype=torch.float16).contiguous()
            mt[nm]="ptf"
            continue
        if ct in ic and tt.ndim>=1:
            if ct=="mlp":
                q,sc=q_i5(tt,cr=15)
                mt[nm]={"t":"int5"}
            else:
                q,sc=q_i6(tt,cr=31)
                mt[nm]={"t":"int6"}
            rs[nm+".q"]=q
            rs[nm+".s"]=sc
        else:
            q,sc=q_i8(tt)
            rs[nm+".q"]=q
            rs[nm+".s"]=sc
            mt[nm]={"t":"int8"}
    return rs,mt

def dq(rs,mt,tsd):
    ot={}
    for nm,og in tsd.items():
        inf=mt[nm]
        odt=og.dtype
        if inf in("pt","ptc","ptf"):
            t=rs[nm]
            if t.dtype==torch.float16 and odt in(torch.float32,torch.bfloat16):t=t.to(odt)
            ot[nm]=t
            continue
        q,sc=rs[nm+".q"],rs[nm+".s"]
        if sc.ndim>0:ot[nm]=(q.float()*sc.float().view(q.shape[0],*([1]*(q.ndim-1)))).to(odt)
        else:ot[nm]=(q.float()*float(sc.item())).to(odt)
    return ot

def ev(args,m,rk,ws,dev,ga,vt,bb,hl,ib):
    lbt=args.vbs//(ws*ga)
    if lbt<args.tsl:raise ValueError("vbs too small")
    lbs=lbt//args.tsl
    tss=(vt.numel()-1)//args.tsl
    sst=(tss*rk)//ws
    sen=(tss*(rk+1))//ws
    vls=torch.zeros((),device=dev,dtype=torch.float64)
    vtc=torch.zeros((),device=dev,dtype=torch.float64)
    vbc=torch.zeros((),device=dev,dtype=torch.float64)
    m.eval()
    with torch.inference_mode():
        for bs in range(sst,sen,lbs):
            be=min(bs+lbs,sen)
            rs=bs*args.tsl
            re=be*args.tsl+1
            lc=vt[rs:re].to(device=dev,dtype=torch.int64,non_blocking=True)
            x=lc[:-1].reshape(-1,args.tsl)
            y=lc[1:].reshape(-1,args.tsl)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
                bl=m(x,y).detach()
            btc=float(y.numel())
            vls+=bl.to(torch.float64)*btc
            vtc+=btc
            pi=x.reshape(-1)
            ti=y.reshape(-1)
            tb=bb[ti].to(dtype=torch.int16)
            tb+=(hl[ti]&~ib[pi]).to(dtype=torch.int16)
            vbc+=tb.to(torch.float64).sum()
    if dist.is_available()and dist.is_initialized():
        dist.all_reduce(vls,op=dist.ReduceOp.SUM)
        dist.all_reduce(vtc,op=dist.ReduceOp.SUM)
        dist.all_reduce(vbc,op=dist.ReduceOp.SUM)
    vl=vls/vtc
    bpt=vl.item()/math.log(2.0)
    tpb=vtc.item()/vbc.item()
    m.train()
    return float(vl.item()),float(bpt*tpb)

def evs(args,bm,rk,ws,dev,vt,bb,hl,ib,st,bsz=32):
    sl=args.tsl
    tt=vt.numel()-1
    ws_=[ws for ws in range(0,tt,st)if min(ws+sl,tt)-ws>=st or ws==0]
    tw=len(ws_)
    ms=(tw*rk)//ws
    me=(tw*(rk+1))//ws
    mw=ws_[ms:me]
    ls=torch.zeros((),device=dev,dtype=torch.float64)
    tc=torch.zeros((),device=dev,dtype=torch.float64)
    bc=torch.zeros((),device=dev,dtype=torch.float64)
    bm.eval()
    with torch.inference_mode():
        for bi in range(0,len(mw),bsz):
            bw=mw[bi:bi+bsz]
            bz=len(bw)
            xb=torch.zeros(bz,sl,dtype=torch.int64,device=dev)
            yb=torch.zeros(bz,sl,dtype=torch.int64,device=dev)
            wls=[]
            for i,w in enumerate(bw):
                en=min(w+sl,tt)
                wl=en-w
                wls.append(wl)
                ch=vt[w:en+1].to(dtype=torch.int64,device=dev)
                xb[i,:wl]=ch[:-1]
                yb[i,:wl]=ch[1:]
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                lg=bm.fwd_logits(xb)
            nl=F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),yb.reshape(-1),reduction="none").reshape(bz,sl)
            for i,w in enumerate(bw):
                wl=wls[i]
                s=0 if w==0 else max(wl-st,0)
                sn=nl[i,s:wl].to(torch.float64)
                ls+=sn.sum()
                tc+=float(wl-s)
                tg=yb[i,s:wl]
                pv=xb[i,s:wl]
                tbb=bb[tg].to(torch.float64)
                tbb+=(hl[tg]&~ib[pv]).to(torch.float64)
                bc+=tbb.sum()
    if dist.is_available()and dist.is_initialized():
        dist.all_reduce(ls,op=dist.ReduceOp.SUM)
        dist.all_reduce(tc,op=dist.ReduceOp.SUM)
        dist.all_reduce(bc,op=dist.ReduceOp.SUM)
    vl=(ls/tc).item()
    bpt=vl/math.log(2.0)
    tpb=tc.item()/bc.item()
    bm.train()
    return vl,bpt*tpb

def main():
    global zpns
    code=Path(__file__).read_text(encoding="utf-8")
    args=H()
    zpns=torch.compile(zpns)
    d="RANK"in os.environ and "WORLD_SIZE"in os.environ
    rk=int(os.environ.get("RANK","0"))
    ws=int(os.environ.get("WORLD_SIZE","1"))
    lr=int(os.environ.get("LOCAL_RANK","0"))
    if ws<=0:raise ValueError(f"ws>0 got {ws}")
    if 8%ws!=0:raise ValueError(f"8%ws==0")
    ga=8//ws
    gs=1.0/ga
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    dev=torch.device("cuda",lr)
    torch.cuda.set_device(dev)
    if d:
        dist.init_process_group(backend="nccl",device_id=dev)
        dist.barrier()
    mp=rk==0
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    lf=None
    if mp:
        os.makedirs("logs",exist_ok=True)
        lf=f"logs/{args.rid}.txt"
        print(lf)
    def lg(m,c=True):
        if not mp:return
        if c:print(m)
        if lf:
            with open(lf,"a",encoding="utf-8")as f:print(m,file=f)
    lg(code,console=False)
    lg("="*100,console=False)
    lg(f"Python {sys.version}",console=False)
    lg(f"PyTorch {torch.__version__}",console=False)
    lg(subprocess.run(["nvidia-smi"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False).stdout,console=False)
    lg("="*100,console=False)
    random.seed(args.sd)
    np.random.seed(args.sd)
    torch.manual_seed(args.sd)
    torch.cuda.manual_seed_all(args.sd)
    if not args.tp.endswith(".model"):raise ValueError(f"SP .model required: {args.tp}")
    sp=spm.SentencePieceProcessor(model_file=args.tp)
    if int(sp.vocab_size())!=args.vs:raise ValueError(f"vs mismatch {args.vs}!={int(sp.vocab_size())}")
    dd=Path(args.dp).resolve()
    atf=len(list(dd.glob("fineweb_train_*.bin")))
    vt=lvt(args.vf,args.tsl)
    bb,hl,ib=bsl(sp,args.vs,dev)
    lg(f"val_bpb:enabled tokenizer=sentencepiece path={args.tp}")
    lg(f"train:dataset:{dd.name} shards:{atf}")
    lg(f"val:pattern={args.vf} tokens:{vt.numel()-1}")
    bm=GPT(vs=args.vs,nl=args.nl,md=args.md,nh=args.nh,nkh=args.nkh,mm=args.mm,te=args.te,teis=args.teis,lsc=args.lsc,rb=args.rb,qgi=args.qgi,bvs=args.bvs,bd=args.bd).to(dev).bfloat16()
    for md in bm.modules():
        if isinstance(md,CL):md.float()
    rfp32(bm)
    cm=torch.compile(bm,dynamic=False,fullgraph=True)
    m=DDP(cm,device_ids=[lr],broadcast_buffers=False)if d else cm
    bnp=list(bm.bks.named_parameters())
    mp_=[p for n,p in bnp if p.ndim==2 and not any(cp in n for cp in CTRL)]
    sp_=[p for n,p in bnp if p.ndim<2 or any(cp in n for cp in CTRL)]
    if bm.sw.numel()>0:sp_.append(bm.sw)
    sp_.append(bm.sm.g)
    if bm.bh is not None:
        sp_.append(bm.bh.sc)
    tlr=args.telr if args.te else args.elr
    tp=[{"params":[bm.tok.weight],"lr":tlr,"base_lr":tlr}]
    if bm.bh is not None:
        tp.append({"params":[bm.bh.em.weight],"lr":tlr,"base_lr":tlr})
        if bm.bh.pj is not None:mp_.append(bm.bh.pj.weight)
    ot=torch.optim.AdamW(tp,betas=(args.b1,args.b2),eps=args.ae,weight_decay=args.wd,fused=True)
    om=MUON(mp_,lr=args.mlr,mm=args.mm_,bs=args.mbs,wd=0.04)
    for g in om.param_groups:g["base_lr"]=args.mlr
    os_=torch.optim.AdamW([{"params":sp_,"lr":args.slr,"base_lr":args.slr}],betas=(args.b1,args.b2),eps=args.ae,weight_decay=args.wd,fused=True)
    opts=[ot,om,os_]
    if bm.lh is not None:
        oh=torch.optim.Adam([{"params":[bm.lh.weight],"lr":args.hlr,"base_lr":args.hlr}],betas=(args.b1,args.b2),eps=args.ae,fused=True)
        opts.insert(1,oh)
    np_=sum(p.numel()for p in bm.parameters())
    lg(f"params:{np_}")
    lg(f"ws:{ws} ga:{ga}")
    lg(f"attn:gqa nh:{args.nh} nkh:{args.nkh}")
    lg(f"te:{args.te} elr:{tlr} mlr:{args.mlr} slr:{args.slr}")
    lg(f"tbt:{args.tbt} tsl:{args.tsl} it:{args.it} wus:{args.wus} mws:{args.mws:.3f}")
    lg(f"seed:{args.sd}")
    lg(f"swa:{args.swae}")
    tl=DL(args.tf,rk,ws,dev)
    def zg():
        for o in opts:o.zero_grad(set_to_none=True)
    mwms=1000.0*args.mws if args.mws>0 else None
    def lrm(st,ems):
        if args.wdi<=0:return 1.0
        if mwms is None:
            wds=max(args.it-args.wdi,0)
            return max((args.it-st)/max(args.wdi,1),0.0)if wds<=st<args.it else 1.0
        stm=ems/max(st,1)
        wdm=args.wdi*stm
        rms=max(mwms-ems,0.0)
        return rms/max(wdm,1e-9)if rms<=wdm else 1.0
    if args.wus>0:
        ims={n:t.detach().cpu().clone() for n,t in bm.state_dict().items()}
        ios=[copy.deepcopy(o.state_dict())for o in opts]
        m.train()
        for wu in range(args.wus):
            zg()
            for ms in range(ga):
                if d:m.require_backward_grad_sync=ms==ga-1
                x,y=tl.next(args.tbt,args.tsl,ga)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
                    wl=m(x,y)
                (wl*gs).backward()
            for o in opts:o.step()
            zg()
            if args.wus<=20 or(wu+1)%10==0 or wu+1==args.wus:
                lg(f"warmup:{wu+1}/{args.wus}")
        bm.load_state_dict(ims,strict=True)
        for o,s in zip(opts,ios,strict=True):o.load_state_dict(s)
        zg()
        if d:m.require_backward_grad_sync=True
        tl=DL(args.tf,rk,ws,dev)
    ttm=0.0
    sas=None
    swst=None
    swc=0
    torch.cuda.synchronize()
    t0=time.perf_counter()
    st=0
    while True:
        ls=st==args.it or(sas is not None and st>=sas)
        sv=ls or(args.vle>0 and st%args.vle==0)
        if sv:
            torch.cuda.synchronize()
            ttm+=1000.0*(time.perf_counter()-t0)
            vl,vb=ev(args,m,rk,ws,dev,ga,vt,bb,hl,ib)
            lg(f"step:{st}/{args.it} val_loss:{vl:.4f} val_bpb:{vb:.4f} time:{ttm:.0f}ms avg:{ttm/max(st,1):.2f}ms")
            torch.cuda.synchronize()
            t0=time.perf_counter()
        if ls:
            if sas is not None and st<args.it:
                lg(f"early_stop: cap time:{ttm:.0f}ms step:{st}/{args.it}")
            break
        ems=ttm+1000.0*(time.perf_counter()-t0)
        sc=lrm(st,ems)
        zg()
        trl=torch.zeros((),device=dev)
        for ms in range(ga):
            if d:m.require_backward_grad_sync=ms==ga-1
            x,y=tl.next(args.tbt,args.tsl,ga)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=True):
                loss=m(x,y)
            trl+=loss.detach()
            (loss*gs).backward()
        trl/=ga
        fc=min(st/args.mmwu,1.0)if args.mmwu>0 else 1.0
        mmv=(1-fc)*args.mmws+fc*args.mm_
        for g in om.param_groups:g["momentum"]=mmv
        for o in opts:
            for g in o.param_groups:g["lr"]=g["base_lr"]*sc
        if args.gcn>0:torch.nn.utils.clip_grad_norm_(bm.parameters(),args.gcn)
        for o in opts:o.step()
        zg()
        st+=1
        atm=ttm+1000.0*(time.perf_counter()-t0)
        if args.swae and sc<args.swasf and st%args.swaev==0:
            if swst is None:
                swst={n:t.detach().cpu().clone() for n,t in bm.state_dict().items()}
                swc=1
                lg(f"swa:start step:{st}")
            else:
                for n,t in bm.state_dict().items():swst[n]+=t.detach().cpu()
                swc+=1
        slt=args.tle>0 and(st<=10 or st%args.tle==0 or sas is not None)
        if slt:
            lg(f"step:{st}/{args.it} train_loss:{trl.item():.4f} time:{atm:.0f}ms avg:{atm/st:.2f}ms")
        rc=mwms is not None and atm>=mwms
        if d and mwms is not None:
            rct=torch.tensor(int(rc),device=dev)
            dist.all_reduce(rct,op=dist.ReduceOp.MAX)
            rc=bool(rct.item())
        if sas is None and rc:sas=st
    lg(f"peak_mem: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")
    if swst is not None:
        for n in swst:swst[n]/=swc
        bm.load_state_dict(swst,strict=True)
        lg(f"swa:applied {swc} checkpoints")
    if mp:
        torch.save(bm.state_dict(),"final_model.pt")
        mb=os.path.getsize("final_model.pt")
        cb=len(code.encode("utf-8"))
        lg(f"model_raw:{mb} bytes code:{cb} total:{mb+cb}")
    qo,qmt=mq(bm.state_dict(),{"mlp","attn"})
    qb=io.BytesIO()
    torch.save({"q":qo,"m":qmt},qb)
    qr=qb.getvalue()
    if _C=="zstd":
        cctx=_zstd.ZstdCompressor(level=22)
        qbl=cctx.compress(qr)
    else:
        qbl=zlib.compress(qr,level=9)
    qrb=len(qr)
    pb=sum(tns(t)for t in qo.values())
    if mp:
        with open("final_model.int8.ptz","wb")as f:f.write(qbl)
        qfb=os.path.getsize("final_model.int8.ptz")
        cb=len(code.encode("utf-8"))
        lg(f"model_quant:{qfb} bytes payload:{pb} raw:{qrb} code:{cb} total:{qfb+cb}")
    if d:dist.barrier()
    with open("final_model.int8.ptz","rb")as f:qbd=f.read()
    if _C=="zstd":
        dctx=_zstd.ZstdDecompressor()
        qrd=dctx.decompress(qbd)
    else:
        qrd=zlib.decompress(qbd)
    ql=torch.load(io.BytesIO(qrd),map_location="cpu")
    bm.load_state_dict(dq(ql["q"],ql["m"],bm.state_dict()),strict=True)
    torch.cuda.synchronize()
    tq=time.perf_counter()
    qv,qb=evs(args,bm,rk,ws,dev,vt,bb,hl,ib,args.es,args.ebs)
    torch.cuda.synchronize()
    lg(f"final_quant val_loss:{qv:.4f} val_bpb:{qb:.4f} eval_time:{1000.0*(time.perf_counter()-tq):.0f}ms")
    lg(f"FINAL_RESULT val_loss:{qv:.8f} val_bpb:{qb:.8f}")
    if d:dist.destroy_process_group()

if __name__=="__main__":main()
