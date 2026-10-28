# Per-kernel roofline of one TRM (mlp_t) training step at B=256, calibrated to a fast RTX 5080 (5.14 steps/s).
# Prints the share of memory-bound time on the 5080 and the speed-up of other GPUs over a fast 5080 (lo/mid/hi),
# from assumed achieved bf16 (fp32-accumulate) TFLOP/s and memory bandwidth (spec-derived, not measured).
# docs/COST_ESTIMATE.md section 1.2 sets its throughput factors S at or below these values (except the RTX PRO 6000
# high end, re-derived from Vast-measured bandwidth).  Standard library only.  Run from the repository root:
#     python3 docs/cost/roofline.py
B=256; T=97; H=512; CI=1536; TI=512
R=B*T; RT=B*H   # rows for channel MLP / token MLP
def gemm(M,K,N): return (2*M*K*N, 2*(M*K+K*N+M*N))
def ew(nbytes): return (0, nbytes)
blk=[('ch_gateup',)+gemm(R,H,2*CI), ('ch_silumul',)+ew(2*R*2*CI+2*R*CI), ('ch_down',)+gemm(R,CI,H),
     ('res_norm1',)+ew(3*2*R*H), ('transpose',)+ew(2*2*R*H), ('tok_gateup',)+gemm(RT,T,2*TI),
     ('tok_silumul',)+ew(2*RT*2*TI+2*RT*TI), ('tok_down',)+gemm(RT,TI,T), ('res_norm2',)+ew(3*2*R*H)]
leval=blk*2+[('inject',)+ew(4*2*R*H)]
flop_leval=sum(k[1] for k in leval); byte_leval=sum(k[2] for k in leval)
print('GFLOP per L-eval per puzzle: %.3f'%(flop_leval/B/1e9))
print('GB per L-eval (B=256): %.2f'%(byte_leval/1e9))
NEQ=35  # L-eval equivalents per training step (flops and, approx., bytes)
def t_leval(P,BW):  # P TFLOP/s achieved on GEMMs, BW GB/s achieved
    tc=tm=0.0; tot=0.0
    for name,f,b in leval:
        a=f/(P*1e12); m=b/(BW*1e9); t=max(a,m); tot+=t
        if a>=m: tc+=t
        else: tm+=t
    return tot,tc,tm
# calibrate 5080
for P5,BW5 in [(95,820),(100,845),(105,870)]:
    tot,tc,tm=t_leval(P5,BW5)
    print('5080 P=%d BW=%d: step(gpu) %.1f ms; compute-bound share %.2f, memory-bound share %.2f'%(P5,BW5,tot*NEQ*1e3,tc/tot,tm/tot))

STEP5080=1/5.14*1e3
g={ # name: (P lo,mid,hi TFLOP/s achieved bf16 fp32-acc), (BW lo,mid,hi GB/s achieved)
 'RTX 5090':((170,186,200),(1500,1580,1650)),
 'RTX 4090':((140,150,160),(850,890,920)),
 'PRO 6000 WS 600W':((370,400,440),(1450,1550,1620)),
 'PRO 6000 Server':((360,386,420),(1400,1460,1500)),
 'PRO 6000 Max-Q 300W':((280,320,370),(1400,1500,1600)),
 'RTX 6000 Ada':((220,255,290),(800,845,880)),
 'L40S':((230,270,300),(720,760,790)),
 'A100 80GB PCIe':((240,260,280),(1650,1720,1800)),
 'A100 80GB SXM':((250,270,285),(1700,1800,1880)),
 'H100 PCIe':((450,520,580),(1700,1800,1900)),
 'H100 SXM':((650,710,760),(2900,3040,3150)),
 'H200 SXM':((650,700,760),(4000,4300,4450)),
 'B200':((1300,1500,1750),(6000,6600,7200)),
}
scen=[('lo',0,6.0,15.0),('mid',1,3.0,10.0),('hi',2,1.5,6.0)]  # (label, idx, t_o ms non-overlapped host, CPU dispatch floor ms)
t5,_,_=t_leval(100,845); t5*=NEQ*1e3
print('\n%-22s %6s %6s %6s   (pure roofline mid, no overhead)'%('GPU','lo','mid','hi'))
for name,(P,BW) in g.items():
    out=[]
    for lab,i,to,tcpu in scen:
        k=(STEP5080-to)/t5   # efficiency calibration from 5080
        tg,_,_=t_leval(P[i],BW[i]); tg*=NEQ*1e3*k
        out.append(STEP5080/(max(tg,tcpu)+to))
    tg,tc,tm=t_leval(P[1],BW[1])
    print('%-22s %6.2f %6.2f %6.2f   roofline %.2f, step %.0f ms, mem-bound share %.2f'%(name,*out,t5/(tg*NEQ*1e3),tg*NEQ*1e3,tm/tg))
