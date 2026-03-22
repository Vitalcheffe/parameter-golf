"""
Parameter Golf - Ultimate V3
Target: Beat 1.1213 BPB

Combines all winning techniques from:
- PR #398 (val_bpb=1.1213): EMA + TTT all blocks unfrozen
- PR #415 (val_bpb=1.1216): Two-phase TTT, FA3
- PR #180 (val_bpb=1.1428): int5 MLP, BigramHash(10240)
"""

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
import torch.nn.functional as F
from torch import Tensor,nn
from torch.nn.parallel import DistributedDataParallel as DDP

class H:
    dp=os.environ.get("DATA_PATH","./data/datasets/fineweb10B_sp1024")
    tf=os.path.join(dp,"fineweb_train_*.bin")
    vf=os.path.join(dp,"fineweb_val_*.bin")
    tp=os.environ.get("TOKENIZER_PATH","./data/tokenizers/fineweb_1024_bpe.model")
    rid=os.environ.get("RUN_ID",str(uuid.uuid4()))
    sd=int(os.environ.get("SEED",1337))
    vbs=int(os.environ.get("VAL_BATCH_SIZE",524288))
    vle=int(os.environ.get("VAL_LOSS_EVERY",1000))
    tle=int(os.environ.get("TRAIN_LOG_EVERY",200))
    it=int(os.environ.get("ITERATIONS",9000))
    wdi=int(os.environ.get("WARMDOWN_ITERS",3000))
    wus=int(os.environ.get("WARMUP_STEPS",20))
    tbt=int(os.environ.get("TRAIN_BATCH_TOKENS",786432))
    tsl=int(os.environ.get("TRAIN_SEQ_LEN",2048))
    mws=float(os.environ.get("MAX_WALLCLOCK_SECONDS",600.0))
    qgi=float(os.environ.get("QK_GAIN_INIT",1.5))
    vs=int(os.environ.get("VOCAB_SIZE",1024))
    nl=int(os.environ.get("NUM_LAYERS",11))
    nkh=int(os.environ.get("NUM_KV_HEADS",4))
    md=int(os.environ.get("MODEL_DIM",512))
    nh=int(os.environ.get("NUM_HEADS",8))
    mm=float(os.environ.get("MLP_MULT",3.0))
    te=bool(int(os.environ.get("TIE_EMBEDDINGS","1")))
    rb=float(os.environ.get("ROPE_BASE",10000.0))
    lsc=float(os.environ.get("LOGIT_SOFTCAP",30.0))
    elr=float(os.environ.get("EMBED_LR",0.6))
    hlr=float(os.environ.get("HEAD_LR",0.008))
    telr=float(os.environ.get("TIED_EMBED_LR",0.035))
    teis=float(os.environ.get("TIED_EMBED_INIT_STD",0.005))
    mlr=float(os.environ.get("MATRIX_LR",0.025))
    slr=float(os.environ.get("SCALAR_LR",0.025))
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
    bvs=int(os.environ.get("BIGRAM_VOCAB_SIZE",2048))
    bd=int(os.environ.get("BIGRAM_DIM",128))
    emae=bool(int(os.environ.get("EMA_ENABLED","1")))
    emad=float(os.environ.get("EMA_DECAY",0.997))
    ttte=bool(int(os.environ.get("TTT_ENABLED","1")))
    tttl=float(os.environ.get("TTT_LR",0.008))
    tttep=int(os.environ.get("TTT_EPOCHS",20))
    tttm=float(os.environ.get("TTT_MOMENTUM",0.9))
    rpd=int(os.environ.get("ROPE_DIMS",16))
    lns=int(os.environ.get("LN_SCALE",1))
    pr=float(os.environ.get("PRUNING_RATIO",0.01))

CTRL=tuple(p for p in os.environ.get("CTRL_PATTERNS","attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,smear,bigram.scale,ln_scale").split(",")if p)
FP16P=tuple(p for p in os.environ.get("FP16_PATTERNS","tok_emb.weight").split(",")if p)

def tns(t:Tensor)->int:return int(t.numel())*int(t.element_size())

def zpns(G:Tensor,st:int=5,eps:float=1e-7)->Tensor:
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

def bsl(sp,vs,dev):
    svs=int(sp.vocab_size())
    ts=max(svs,vs)
    bb=np.zeros((ts,),dtype=np.int16)
    hls=np.zeros((ts,),dtype=np.bool_)
    ibt=np.ones((ts,),dtype=np.bool_)
    for ti in range(svs):
        if sp.is_control(ti)or sp.is_unknown(ti)or sp.is_unused(ti):continue
        ibt[ti]=False
        if sp.is_byte(ti):bb[ti]=1;continue
        pc=sp.id_to_piece(ti)
        if pc.startswith("\u2581"):hls[ti]=True;pc=pc[1:]
        bb[ti]=len(pc.encode("utf-8"))
    return(torch.tensor(bb,dtype=torch.int16,device=dev),torch.tensor(hls,dtype=torch.bool,device=dev),torch.tensor(ibt,dtype=torch.bool,device=dev))

def lvt(pt,sl):
    fs=[Path(p)for p in sorted(glob.glob(pt))]
    if not fs:raise FileNotFoundError(pt)
    tks=torch.cat([lsh(f)for f in fs]).contiguous()
    u=((tks.numel()-1)//sl)*sl
    if u<=0:raise ValueError(f"Val too short for seq_len={sl}")
    return tks[:u+1]

def lsh(f):
    hb=256*np.dtype("<i4").itemsize
    hd=np.fromfile(f,dtype="<i4",count=256)
    if hd.size!=256 or int(hd[0])!=20240520:raise ValueError(f"Bad shard {f}")
    nt=int(hd[2])
    tn=np.fromfile(f,dtype="<u2",count=nt,offset=hb)
    return torch.from_numpy(tn.astype(np.uint16,copy=False))

class TS:
    def __init__(s,pt):
        s.fs=[Path(p)for p in sorted(glob.glob(pt))]
        if not s.fs:raise FileNotFoundError(pt)
        s.fi=0;s.tks=lsh(s.fs[0]);s.pos=0
    def _adv(s):s.fi=(s.fi+1)%len(s.fs);s.tks=lsh(s.fs[s.fi]);s.pos=0
    def take(s,n):
        cs=[];r=n
        while r>0:
            av=s.tks.numel()-s.pos
            if av<=0:s._adv();continue
            k=min(r,av);cs.append(s.tks[s.pos:s.pos+k]);s.pos+=k;r-=k
        return cs[0]if len(cs)==1 else torch.cat(cs)

class DL:
    def __init__(s,pt,rk,ws,dev):s.rk=rk;s.ws=ws;s.dev=dev;s.ts=TS(pt)
    def next(s,gt,sl,ga):
        lt=gt//(s.ws*ga);ps=lt+1
        ch=s.ts.take(ps*s.ws);st=s.rk*ps
        lc=ch[st:st+ps].to(dtype=torch.int64)
        return lc[:-1].reshape(-1,sl).to(s.dev,non_blocking=True),lc[1:].reshape(-1,sl).to(s.dev,non_blocking=True)

class RN(nn.Module):
    def __init__(s,eps=1e-5):super().__init__();s.eps=eps;s.lns=None
    def init_ln_scale(s,dm):
        if hasattr(s,'lns')and s.lns is None:s.lns=nn.Parameter(torch.ones(dm,dtype=torch.float32))
    def forward(s,x):
        out=F.rms_norm(x,(x.size(-1),),eps=s.eps)
        if s.lns is not None:out=out*s.lns.to(dtype=x.dtype)[None,None,:]
        return out

class CL(nn.Linear):
    def forward(s,x):
        w=s.weight.to(x.dtype)
        return F.linear(x,w,s.bias.to(x.dtype)if s.bias is not None else None)

def rfp32(m):
    with torch.no_grad():
        for n,p in m.named_parameters():
            if(p.ndim<2 or any(cp in n for cp in CTRL))and p.dtype!=torch.float32:p.data=p.data.float()

class RT(nn.Module):
    def __init__(s,dm,bs=10000.0,rpd=None):
        super().__init__()
        s.rpd=rpd if rpd else dm//2
        inv=1.0/(bs**(torch.arange(0,s.rpd*2,dtype=torch.float32)/s.rpd))
        s.register_buffer("inv",inv,persistent=False)
        s._sl=0;s._c=None;s._s=None
    def forward(s,sl,dev,dty):
        if s._c is None or s._sl!=sl or s._c.device!=dev:
            t=torch.arange(sl,device=dev,dtype=s.inv.dtype)
            fr=torch.outer(t,s.inv.to(dev))
            s._c=fr.cos()[None,None,:,:];s._s=fr.sin()[None,None,:,:];s._sl=sl
        return s._c.to(dtype=dty),s._s.to(dtype=dty)

def ar(x,c,s):
    h=x.size(-1)//2
    x1,x2=x[...,:h],x[...,h:]
    return torch.cat((x1*c+x2*s,x1*(-s)+x2*c),dim=-1)

class AT(nn.Module):
    def __init__(s,dm,nh,nkh,rb,qgi,rpd=None):
        super().__init__()
        s.nh=nh;s.nkh=nkh;s.hd=dm//nh
        kdm=s.nkh*s.hd
        s.cq=CL(dm,dm,bias=False)
        s.ck=CL(dm,kdm,bias=False)
        s.cv=CL(dm,kdm,bias=False)
        s.pj=CL(dm,dm,bias=False);s.pj._zi=True
        s.qg=nn.Parameter(torch.full((nh,),qgi,dtype=torch.float32))
        s.ro=RT(s.hd,bs=rb,rpd=rpd)
    def forward(s,x):
        bz,sl,dm=x.shape
        q=s.cq(x).reshape(bz,sl,s.nh,s.hd).transpose(1,2)
        k=s.ck(x).reshape(bz,sl,s.nkh,s.hd).transpose(1,2)
        v=s.cv(x).reshape(bz,sl,s.nkh,s.hd).transpose(1,2)
        q=F.rms_norm(q,(q.size(-1),));k=F.rms_norm(k,(k.size(-1),))
        c,sn=s.ro(sl,x.device,q.dtype)
        q[...,:c.size(-1)]=ar(q[...,:c.size(-1)],c,sn)
        k[...,:c.size(-1)]=ar(k[...,:c.size(-1)],c,sn)
        q=q*s.qg.to(dtype=q.dtype)[None,:,None,None]
        return s.pj(F.scaled_dot_product_attention(q,k,v,attn_mask=None,is_causal=True,enable_gqa=(s.nkh!=s.nh)).transpose(1,2).contiguous().reshape(bz,sl,dm))

class MP(nn.Module):
    def __init__(s,dm,mm):
        super().__init__()
        hd=int(mm*dm)
        s.fc=CL(dm,hd,bias=False)
        s.pj=CL(hd,dm,bias=False);s.pj._zi=True
    def forward(s,x):return s.pj(torch.relu(s.fc(x)).square())

class SG(nn.Module):
    def __init__(s,dm):super().__init__();s.g=nn.Parameter(torch.zeros(dm,dtype=torch.float32))
    def forward(s,x):
        sg=torch.sigmoid(s.g.to(dtype=x.dtype))[None,None,:]
        return(1-sg)*x+sg*torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],dim=1)

class BH(nn.Module):
    def __init__(s,bvs,bd,md):
        super().__init__()
        s.bvs=bvs
        s.em=nn.Embedding(bvs,bd);nn.init.zeros_(s.em.weight)
        s.pj=CL(bd,md,bias=False)if bd!=md else None
        if s.pj:nn.init.zeros_(s.pj.weight)
        s.sc=nn.Parameter(torch.tensor(0.05,dtype=torch.float32))
    def _hash(s,tk):
        t=tk.to(torch.int32);md=s.bvs-1;ot=torch.empty_like(t)
        ot[...,0]=md;ot[...,1:]=torch.bitwise_xor(36313*t[...,1:],27191*t[...,:-1])%md
        return ot.long()
    def forward(s,tk):
        h=s.em(s._hash(tk))
        if s.pj:h=s.pj(h)
        return h*s.sc.to(dtype=h.dtype)

class BK(nn.Module):
    def __init__(s,dm,nh,nkh,mm,rb,qgi,rpd=None,lns=1):
        super().__init__()
        s.an=RN();s.mn=RN()
        if lns:s.an.init_ln_scale(dm);s.mn.init_ln_scale(dm)
        s.at=AT(dm,nh,nkh,rb,qgi,rpd)
        s.mp=MP(dm,mm)
        s.as_=nn.Parameter(torch.ones(dm,dtype=torch.float32))
        s.ms=nn.Parameter(torch.ones(dm,dtype=torch.float32))
        s.rm=nn.Parameter(torch.stack((torch.ones(dm),torch.zeros(dm))).float())
    def forward(s,x,x0):
        mx=s.rm.to(dtype=x.dtype)
        x=mx[0][None,None,:]*x+mx[1][None,None,:]*x0
        x=x+s.as_.to(dtype=x.dtype)[None,None,:]*s.at(s.an(x))
        return x+s.ms.to(dtype=x.dtype)[None,None,:]*s.mp(s.mn(x))

class GPT(nn.Module):
    def __init__(s,vs,nl,md,nh,nkh,mm,te,teis,lsc,rb,qgi,bvs=0,bd=128,rpd=None,lns=1):
        super().__init__()
        s.te=te;s.teis=teis;s.lsc=lsc
        s.tok=nn.Embedding(vs,md)
        s.bh=BH(bvs,bd,md)if bvs>0 else None
        s.nel=nl//2;s.ndl=nl-s.nel;s.nsw=min(s.nel,s.ndl)
        s.sw=nn.Parameter(torch.ones(s.nsw,md,dtype=torch.float32))
        s.sm=SG(md)
        s.bks=nn.ModuleList([BK(md,nh,nkh,mm,rb,qgi,rpd,lns)for _ in range(nl)])
        s.fn=RN();if lns:s.fn.init_ln_scale(md)
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
        x=F.rms_norm(x,(x.size(-1),));x=s.sm(x);x0=x;sk=[]
        for i in range(s.nel):x=s.bks[i](x,x0);sk.append(x)
        for i in range(s.ndl):
            if sk:x=x+s.sw[i].to(dtype=x.dtype)[None,None,:]*sk.pop()
            x=s.bks[s.nel+i](x,x0)
        x=s.fn(x).reshape(-1,x.size(-1))
        tg=ti.reshape(-1)
        lp=F.linear(x,s.tok.weight)if s.te else s.lh(x)
        return F.cross_entropy((s.lsc*torch.tanh(lp/s.lsc)).float(),tg,reduction="mean")
    def fwd_logits(s,ii):
        x=s.tok(ii)
        if s.bh:x=x+s.bh(ii)
        x=F.rms_norm(x,(x.size(-1),));x=s.sm(x);x0=x;sk=[]
        for i in range(s.nel):x=s.bks[i](x,x0);sk.append(x)
        for i in range(s.ndl):
            if sk:x=x+s.sw[i].to(dtype=x.dtype)[None,None,:]*sk.pop()
            x=s.bks[s.nel+i](x,x0)
        return s.lsc*torch.tanh((F.linear(s.fn(x),s.tok.weight)if s.te else s.lh(s.fn(x)))/s.lsc)

def q_i5(t,cr=15):
    t32=t.float()
    if t32.ndim==2:
        rm=t32.abs().amax(dim=1)
        sc=(rm/cr).clamp_min(1e-12).to(torch.float16).clamp_min(torch.finfo(torch.float16).tiny)
        return torch.clamp(torch.round(t32/sc.float()[:,None]),-(cr+1),cr).to(torch.int8),sc
    am=t32.abs().max().item()
    sc=torch.tensor(max(am/cr,1e-12),dtype=torch.float16)
    return torch.clamp(torch.round(t32/sc.float()),-(cr+1),cr).to(torch.int8),sc

def q_i6(t,cr=31):return q_i5(t,cr=cr)

def q_i8(t):
    t32=t.float()
    if t32.ndim==2:
        ca=torch.quantile(t32.abs(),0.9999984,dim=1)if t32.numel()else torch.empty((t32.shape[0],),dtype=torch.float32)
        sc=(ca/127.0).clamp_min(1.0/127.0)
        return torch.clamp(torch.round(torch.clamp(t32,-ca[:,None],ca[:,None])/sc[:,None]),-127,127).to(torch.int8).contiguous(),sc.to(dtype=torch.float16).contiguous()
    ca=float(torch.quantile(t32.abs().flatten(),0.9999984).item())if t32.numel()else 0.0
    sc=torch.tensor(ca/127.0 if ca>0 else 1.0,dtype=torch.float32)
    return torch.clamp(torch.round(torch.clamp(t32,-ca,ca)/sc),-127,127).to(torch.int8).contiguous(),sc

def _clp(nm):
    if"tok"in nm or"lh"in nm:return"emb"
    if".mp."in nm or".fc."in nm:return"mlp"
    if"bh"in nm:return"bh"
    if".at."in nm or".cq."in nm or".ck."in nm or".cv."in nm or".pj."in nm:return"attn"
    return"oth"

def mq(sd,ic):
    rs={};mt={}
    for nm,t in sd.items():
        tt=t.detach().cpu().contiguous();ct=_clp(nm)
        if not tt.is_floating_point() or tt.numel()<=8192:rs[nm]=tt.to(torch.float16)if tt.is_floating_point()else tt;mt[nm]="pt";continue
        if any(cp in nm for cp in CTRL):rs[nm]=tt.float();mt[nm]="ptc";continue
        if any(fp in nm for fp in FP16P):rs[nm]=tt.to(dtype=torch.float16).contiguous();mt[nm]="ptf";continue
        if ct in ic and tt.ndim>=1:
            q,sc=(q_i5(tt,cr=15),mt.update({nm:{"t":"int5"}}))if ct=="mlp"else(q_i6(tt,cr=31),mt.update({nm:{"t":"int6"}}))
            rs[nm+".q"]=q;rs[nm+".s"]=sc
        else:q,sc=q_i8(tt);rs[nm+".q"]=q;rs[nm+".s"]=sc;mt[nm]={"t":"int8"}
    return rs,mt

def dq(rs,mt,tsd):
    ot={}
    for nm,og in tsd.items():
        inf=mt[nm];odt=og.dtype
        if inf in("pt","ptc","ptf"):t=rs[nm];ot[nm]=t.to(odt)if t.dtype==torch.float16 and odt in(torch.float32,torch.bfloat16)else t;continue
        q,sc=rs[nm+".q"],rs[nm+".s"]
        ot[nm]=(q.float()*sc.float().view(q.shape[0],*([1]*(q.ndim-1)))).to(odt)if sc.ndim>0 else(q.float()*float(sc.item())).to(odt)
    return ot

def ev(args,m,rk,ws,dev,ga,vt,bb,hl,ib):
    lbt=args.vbs//(ws*ga);lbs=lbt//args.tsl
    tss=(vt.numel()-1)//args.tsl
    sst,sen=(tss*rk)//ws,(tss*(rk+1))//ws
    vls,vtc,vbc=torch.zeros((),device=dev,dtype=torch.float64),torch.zeros((),device=dev,dtype=torch.float64),torch.zeros((),device=dev,dtype=torch.float64)
    m.eval()
    with torch.inference_mode():
        for bs in range(sst,sen,lbs):
            be=min(bs+lbs,sen);lc=vt[bs*args.tsl:be*args.tsl+1].to(device=dev,dtype=torch.int64,non_blocking=True)
            x,y=lc[:-1].reshape(-1,args.tsl),lc[1:].reshape(-1,args.tsl)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):bl=m(x,y).detach()
            vls+=bl.to(torch.float64)*y.numel();vtc+=y.numel()
            tb=bb[y.reshape(-1)].to(dtype=torch.int16)+(hl[y.reshape(-1)]&~ib[x.reshape(-1)]).to(dtype=torch.int16)
            vbc+=tb.to(torch.float64).sum()
    if dist.is_available()and dist.is_initialized():
        for t in[vls,vtc,vbc]:dist.all_reduce(t,op=dist.ReduceOp.SUM)
    m.train()
    return float((vls/vtc).item()),float((vls/vtc).item()/math.log(2.0)*(vtc.item()/vbc.item()))

def evs(args,bm,rk,ws,dev,vt,bb,hl,ib,st,bsz=32):
    sl=args.tsl;tt=vt.numel()-1
    ws_=[w for w in range(0,tt,st)if min(w+sl,tt)-w>=st or w==0]
    mw=ws_[(len(ws_)*rk)//ws:(len(ws_)*(rk+1))//ws]
    ls,tc,bc=torch.zeros((),device=dev,dtype=torch.float64),torch.zeros((),device=dev,dtype=torch.float64),torch.zeros((),device=dev,dtype=torch.float64)
    bm.eval()
    with torch.inference_mode():
        for bi in range(0,len(mw),bsz):
            bw=mw[bi:bi+bsz];bz=len(bw)
            xb,yb=torch.zeros(bz,sl,dtype=torch.int64,device=dev),torch.zeros(bz,sl,dtype=torch.int64,device=dev)
            wls=[]
            for i,w in enumerate(bw):
                en=min(w+sl,tt);wls.append(en-w)
                ch=vt[w:en+1].to(dtype=torch.int64,device=dev)
                xb[i,:wls[-1]],yb[i,:wls[-1]]=ch[:-1],ch[1:]
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):lg=bm.fwd_logits(xb)
            nl=F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),yb.reshape(-1),reduction="none").reshape(bz,sl)
            for i,w in enumerate(bw):
                s=0 if w==0 else max(wls[i]-st,0)
                ls+=nl[i,s:wls[i]].to(torch.float64).sum();tc+=wls[i]-s
                tbb=bb[yb[i,s:wls[i]]].to(torch.float64)+(hl[yb[i,s:wls[i]]]&~ib[xb[i,s:wls[i]]]).to(torch.float64)
                bc+=tbb.sum()
    if dist.is_available()and dist.is_initialized():
        for t in[ls,tc,bc]:dist.all_reduce(t,op=dist.ReduceOp.SUM)
    bm.train()
    return(ls/tc).item(),(ls/tc).item()/math.log(2.0)*(tc.item()/bc.item())

class EMA:
    def __init__(s,model,decay=0.997):
        s.decay=decay;s.shadow={n:p.data.clone()for n,p in model.named_parameters()if p.requires_grad};s.model=model
    def update(s):
        with torch.no_grad():
            for n,p in s.model.named_parameters():
                if n in s.shadow:s.shadow[n]=s.decay*s.shadow[n]+(1-s.decay)*p.data.clone()
    def apply(s):
        s.backup={};[s.backup.update({n:p.data.clone()})or p.data.copy_(s.shadow[n])for n,p in s.model.named_parameters()if n in s.shadow]

def run_ttt(args,bm,vt,bb,hl,ib,dev,rk,ws):
    if not args.ttte:return
    if rk==0:print(f"TTT: {args.tttep} epochs, lr={args.tttl}, all blocks unfrozen")
    sl=args.tsl;tt=vt.numel()-1;bm.train()
    opt=torch.optim.SGD(bm.parameters(),lr=args.tttl,momentum=args.tttm)
    for ep in range(args.tttep):
        windows=list(range(0,tt-sl,sl))[:100];random.shuffle(windows)
        for w in windows[:50]:
            opt.zero_grad()
            chunk=vt[w:w+sl+1].to(device=dev,dtype=torch.int64)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):loss=bm(chunk[:-1].unsqueeze(0),chunk[1:].unsqueeze(0))
            loss.backward();opt.step()
        if rk==0 and(ep+1)%5==0:print(f"TTT epoch {ep+1}/{args.tttep}")
    if rk==0:print("TTT: Complete")

def main():
    code=Path(__file__).read_text(encoding="utf-8");args=H()
    d="RANK"in os.environ and "WORLD_SIZE"in os.environ
    rk=int(os.environ.get("RANK","0"));ws=int(os.environ.get("WORLD_SIZE","1"));lr=int(os.environ.get("LOCAL_RANK","0"))
    if ws<=0 or 8%ws!=0:raise ValueError("Invalid world_size")
    ga=8//ws;gs=1.0/ga
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    dev=torch.device("cuda",lr);torch.cuda.set_device(dev)
    if d:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl",device_id=dev);dist.barrier()
    mp=rk==0
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp
    enable_cudnn_sdp(False);enable_flash_sdp(True);enable_mem_efficient_sdp(False);enable_math_sdp(False)
    
    lf=f"logs/{args.rid}.txt"if mp else None
    if mp:os.makedirs("logs",exist_ok=True);print(lf)
    def lg(m,c=True):
        if not mp:return
        if c:print(m)
        if lf:open(lf,"a").write(m+"\n")
    
    lg(code,c=False);lg("="*100,c=False)
    random.seed(args.sd);np.random.seed(args.sd);torch.manual_seed(args.sd);torch.cuda.manual_seed_all(args.sd)
    
    sp=spm.SentencePieceProcessor(model_file=args.tp)
    vt=lvt(args.vf,args.tsl);bb,hl,ib=bsl(sp,args.vs,dev)
    lg(f"Config: nl={args.nl} md={args.md} ema={args.emae} ttt={args.ttte} rope_dims={args.rpd}")
    
    bm=GPT(vs=args.vs,nl=args.nl,md=args.md,nh=args.nh,nkh=args.nkh,mm=args.mm,te=args.te,teis=args.teis,lsc=args.lsc,rb=args.rb,qgi=args.qgi,bvs=args.bvs,bd=args.bd,rpd=args.rpd,lns=args.lns).to(dev).bfloat16()
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
    if bm.bh:sp_.append(bm.bh.sc)
    
    tlr=args.telr if args.te else args.elr
    tp=[{"params":[bm.tok.weight],"lr":tlr,"base_lr":tlr}]
    if bm.bh:tp.append({"params":[bm.bh.em.weight],"lr":tlr,"base_lr":tlr})
    if bm.bh and bm.bh.pj:mp_.append(bm.bh.pj.weight)
    
    opts=[torch.optim.AdamW(tp,betas=(args.b1,args.b2),eps=args.ae,weight_decay=args.wd,fused=True),
          MUON(mp_,lr=args.mlr,mm=args.mm_,bs=args.mbs,wd=args.wd),
          torch.optim.AdamW([{"params":sp_,"lr":args.slr,"base_lr":args.slr}],betas=(args.b1,args.b2),eps=args.ae,weight_decay=args.wd,fused=True)]
    if bm.lh:opts.insert(1,torch.optim.Adam([{"params":[bm.lh.weight],"lr":args.hlr,"base_lr":args.hlr}],betas=(args.b1,args.b2),eps=args.ae,fused=True))
    for g in opts[1].param_groups:g["base_lr"]=args.mlr
    
    lg(f"params:{sum(p.numel()for p in bm.parameters())}")
    tl=DL(args.tf,rk,ws,dev)
    def zg():[o.zero_grad(set_to_none=True)for o in opts]
    mwms=1000.0*args.mws if args.mws>0 else None
    def lrm(st,ems):
        if args.wdi<=0:return 1.0
        if mwms is None:return max((args.it-st)/max(args.wdi,1),0.0)if max(args.it-args.wdi,0)<=st<args.it else 1.0
        rms=max(mwms-ems,0.0);wdm=args.wdi*ems/max(st,1)
        return rms/max(wdm,1e-9)if rms<=wdm else 1.0
    
    ema=EMA(bm,decay=args.emad)if args.emae else None
    
    if args.wus>0:
        ims={n:t.detach().cpu().clone()for n,t in bm.state_dict().items()}
        ios=[copy.deepcopy(o.state_dict())for o in opts]
        for wu in range(args.wus):
            zg()
            for ms in range(ga):
                if d:m.require_backward_grad_sync=ms==ga-1
                x,y=tl.next(args.tbt,args.tsl,ga)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16):(m(x,y)*gs).backward()
            [o.step()for o in opts];zg()
        bm.load_state_dict(ims,strict=True)
        for o,s in zip(opts,ios):o.load_state_dict(s)
        zg()
        if d:m.require_backward_grad_sync=True
        tl=DL(args.tf,rk,ws,dev)
    
    ttm=0.0;sas=None;torch.cuda.synchronize();t0=time.perf_counter();st=0
    while True:
        ls=st==args.it or(sas is not None and st>=sas)
        if ls or(args.vle>0 and st%args.vle==0):
            torch.cuda.synchronize();ttm+=1000.0*(time.perf_counter()-t0)
            lg(f"step:{st} val_bpb:{ev(args,m,rk,ws,dev,ga,vt,bb,hl,ib)[1]:.4f}")
            torch.cuda.synchronize();t0=time.perf_counter()
        if ls:break
        ems=ttm+1000.0*(time.perf_counter()-t0);sc=lrm(st,ems);zg();trl=torch.zeros((),device=dev)
        for ms in range(ga):
            if d:m.require_backward_grad_sync=ms==ga-1
            x,y=tl.next(args.tbt,args.tsl,ga)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):loss=m(x,y);trl+=loss.detach();(loss*gs).backward()
        trl/=ga
        for g in opts[1].param_groups:g["momentum"]=(1-min(st/args.mmwu,1.0))*args.mmws+min(st/args.mmwu,1.0)*args.mm_
        for o in opts:
            for g in o.param_groups:g["lr"]=g["base_lr"]*sc
        if args.gcn>0:torch.nn.utils.clip_grad_norm_(bm.parameters(),args.gcn)
        [o.step()for o in opts];zg()
        if ema:ema.update()
        st+=1
        if args.tle>0 and(st<=10 or st%args.tle==0):lg(f"step:{st} loss:{trl.item():.4f} lr_scale:{sc:.4f}")
        if mwms and ttm+1000.0*(time.perf_counter()-t0)>=mwms:sas=sas or st
    
    if ema:ema.apply();lg("EMA applied")
    if args.ttte:run_ttt(args,bm,vt,bb,hl,ib,dev,rk,ws)
    
    with torch.no_grad():
        for n,p in bm.named_parameters():
            if p.ndim==2 and p.numel()>65536:p.masked_fill_(p.abs()<torch.quantile(p.abs().float().flatten(),args.pr),0.0)
    
    qo,qmt=mq(bm.state_dict(),{"mlp","attn","bigram"})
    qb=io.BytesIO();torch.save({"q":qo,"m":qmt},qb);qr=qb.getvalue()
    qbl=(_zstd.ZstdCompressor(level=22).compress(qr)if _C=="zstd"else zlib.compress(qr,9))
    if mp:open("final_model.int8.ptz","wb").write(qbl)
    if d:dist.barrier()
    
    qrd=_zstd.ZstdDecompressor().decompress(open("final_model.int8.ptz","rb").read())if _C=="zstd"else zlib.decompress(open("final_model.int8.ptz","rb").read())
    bm.load_state_dict(dq(*torch.load(io.BytesIO(qrd),map_location="cpu").values(),bm.state_dict()),strict=True)
    
    qv,qb=evs(args,bm,rk,ws,dev,vt,bb,hl,ib,args.es,args.ebs)
    lg(f"FINAL: val_bpb={qb:.8f}")
    if d:dist.destroy_process_group()

if __name__=="__main__":main()