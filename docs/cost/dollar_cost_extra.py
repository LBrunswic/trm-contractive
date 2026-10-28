# Extra tables for docs/COST_ESTIMATE.md: $ per fast-5080 G-h for every offer (section 2), node break-even,
# continuous-rental cost, marginal $ per optional row, cut order, recommended package, tranches, stress case,
# node gate, per-arm sums and wall-clock.  Standard library only.  Run from the repository root:
#     python3 docs/cost/dollar_cost_extra.py
import csv
import os

from dollar_cost_model import *

# ---------------- $ per fast-5080 G-h across GPU offers ----------------
# PRO 6000: one S range for all editions (Vast-measured gpu_mem_bw 1,391-1,406 GB/s for S, WS and Max-Q alike);
# 300 W hosts (Max-Q, capped WS) lean to the low end.
S = {'5080':(0.75,1.0,1.0),'4090':(1.0,1.15,1.25),'5090':(1.5,1.75,1.9),'P6K-WS':(1.8,2.1,2.6),'P6K-SE':(1.8,2.1,2.6),
     'P6K-MQ':(1.8,2.1,2.6),'Ada':(0.95,1.2,1.4),'L40S':(0.9,1.1,1.3),'A100SXM':(1.5,1.9,2.3),'A100PCIe':(1.4,1.8,2.2),
     'H100PCIe':(1.9,2.4,2.9),'H100SXM':(2.8,3.6,4.4),'H200':(3.2,4.2,5.2),'B200':(4.0,6.0,8.5)}
# Offers: docs/cost/prices_2026-10-27.csv (27 Oct 2026 snapshot; one row per offer with its public source URL).
# Vast.ai values are medians (or min-bid medians) of the live offer search at 22:30-22:40 UTC; they move by about
# +-15% within a day.
_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prices_2026-10-27.csv')
with open(_CSV, newline='') as _f:
    OFF = [(r['gpu'], r['offer'], r['s_key'], float(r['usd_per_gpu_hour']),
            'od' if r['tier'] == 'on-demand' else 'spot') for r in csv.DictReader(_f)]
print('| GPU | Offer | $/GPU-h | S (low-central-high) | $/G-h central | $/G-h range |')
rows=[]
for g,o,k,p,t in OFF:
    lo,c,hi=S[k]
    rows.append((p/c,g,o,p,lo,c,hi))
for v,g,o,p,lo,c,hi in sorted(rows):
    print(f'| {g} | {o} | {p:.3f} | {lo}-{c}-{hi} | {v:.3f} | {p/hi:.3f}-{p/lo:.3f} |')

# ---------------- break-even node price vs the 5080 pool ----------------
fl = SC['S4']
for name,sv in [('PRO 6000',2.1),('PRO 6000 at S high',2.6),('RTX 6000 Ada',1.2)]:
    be = fl['p'][1]/fl['S'][1]/fl['U'][1]*sv*0.80
    print(f'break-even {name}: ${be:.3f}/GPU-h = ${4*be:.2f}/node-h (U=0.80; VAT-free like Vast); with 20% VAT ${4*be/1.2:.2f}/node-h')

# ---------------- continuous rental trap ----------------
for lab,ph in [('Vast 4x PRO 6000 (unverified)',5.34),('Vast 4x PRO 6000 SE (verified)',9.389),('UpCloud 4x',7.52),('Massed 4x',8.76),('RunPod Secure 4x',9.96),('Massed 4x Ada',3.16)]:
    print(f'continuous 15 d: {lab}: ${360*ph:,.0f} ex VAT, ${360*ph*1.2:,.0f} incl VAT')
for sc in ['T0R','T0L']:
    W = work(scope_rows(sc,pairing=True),1)
    print(sc,'utilisation if node kept 15 d (S=2.1):', round(W/(360*4*2.1),3))

# ---------------- marginal $ per row ----------------
def eff(scn,j,cls):
    s=SC[scn]
    return s['p'][j]/s['S'][j]/s['U'][j]*(1+F_FAIL[cls][j])*(1+s['spot'][j])*(1+s['vat'][j])*(1+s['disk'][j]*s['rate'][j]/730/s['p'][j])
# T4 gate = 2k-step pilot (0.4-0.8) + one TRM-Att A0 run (15-30) + one EV-Maze (1.28-2.37) = 16.68-33.17 lo-hi G-h
T4_GATE = (0.4+15+1.28, 0.8+30+2.37)
OPT = [
 ('Seeds 6-8 at L for X1, B3, A3, B* (n=9 on the H5/H1 arms; pre-register by 29 Oct)', 174.1, 289.9,'su'),
 ('Seeds 6-8 at L for all 6 family arms (n=9; Holm becomes feasible, 7.6)', 174.1+ (3*11.4+3*11.5)+ 6*1.58, 289.9+(3*18.7+3*19.0)+6*2.93,'su'),
 ('T3+ ungrounded visual Sudoku (Tier 1)', 34.5, 68.0,'nt'),
 ('T5 restoration incl. B9/JR (Tier 1)', 27.6, 108.0,'nt'),
 ('T6 RAFT fine-tune (Tier 1)', 18.0, 48.0,'nt'),
 ('X17 V1 caps with seeds (Tier 1)', 16.4, 28.3,'su'),
 ('X19 batch fidelity (Tier 1)', 13.4, 22.6,'su'),
 ('X15 on-trajectory pairs (Tier 1)', 11.4, 21.5,'su'),
 ('X12 local Jacobian hinge (Tier 1)', 4.2, 8.3,'su'),
 ('B6 STARS (Tier 1)', 4.1, 7.8,'su'),
 ('D6 quantisation (Tier 1)', 3.9, 7.2,'su'),
 ('Tier-1 eval extras on X1 and B* (EV-G/X/TT)', 17.5, 32.6,'su'),
 ('T4 Maze-Hard gate only (pilot + first A0 run + EV-Maze, DP4)', T4_GATE[0], T4_GATE[1],'nt'),
 ('T4 Maze-Hard full (9 runs, gated)', 159.4, 316.8,'nt'),
 ('T7 ARC-1 fine-tunes', 32.2, 71.2,'nt'),
 ('T8 Mini-ARC', 9.9, 29.7,'nt'),
 ('B8 exact monotone reference', 5.2, 12.4,'su'),
 ('X13 adversary strength', 11.7, 20.2,'su'),
 ('X16 metric variants', 14.9, 27.8,'su'),
]
print('\n| Optional row | G-h lo-hi | S4 pool $ cen | S4 $ pess | S2 node $ cen |')
for n,lo,hi,c in OPT:
    print(f'| {n} | {lo:.1f}-{hi:.1f} | {lo*eff("S4",1,c):.0f} | {lo*eff("S4",2,c):.0f} | {lo*eff("S2",1,c):.0f} |')

# Cuts: (name, [(lo G-h, cls), ...]) priced per class; scenario-difference rows priced by scenario totals
CUTS = [
 ('T4 beyond its gate', [(159.4-T4_GATE[0],'nt')]),
 ('T4 full (gate skipped too)', [(159.4,'nt')]),
 ('Tier 2 group 1: T7, B8, X13, X16, X20, T8, B10, D7', [(32.2+9.9,'nt'), (5.2+11.7+14.9+5.6+4.3+0.9,'su')]),
 ('Option +P', [(174.1,'su')]),
 ('Tier 1: X6, X7, X9', [(3.2+11.2+25.9,'su')]),
 ('Tier 1: X12, X14, X15', [(4.2+7.0+11.4,'su')]),
 ('Tier 1: T6, then T5', [(45.6,'nt')]),
 ('Tier 1: T3+ (instead of T5/T6 if DP3 failed)', [(34.5,'nt')]),
 ('L-core -> L for seeds 0-2 only (confirmatory family back at R)', [(189.0,'su')]),
 ('Plan 8.6 Tier-0 cuts 1-6', [(48.3,'su')]),
 ('L-core removed, X18a back in Tier 1, X18b/c not restored', [(443.9-72.8,'su')]),
]
print('\n| Cut | lo G-h saved | S4 $ cen | S4 $ pess | S2 $ cen | S2 $ pess |')
for n,parts in CUTS:
    lo = sum(x for x,_ in parts)
    v = [sum(x*eff(sc,j,c) for x,c in parts) for sc,j in (('S4',1),('S4',2),('S2',1),('S2',2))]
    print(f'| {n} | {lo:.1f} | {v[0]:.1f} | {v[1]:.1f} | {v[2]:.1f} | {v[3]:.1f} |')
lo = sum(tot(r) for r in scope_rows('AllL')) - sum(tot(r) for r in scope_rows('AllR'))
v = [cost(sc,'AllL',j)-cost(sc,'AllR',j) for sc,j in (('S4',1),('S4',2),('S2',1),('S2',2))]
print(f'| Revert to plan v2 as written (AllL - AllR, scenario totals) | {lo:.1f} | {v[0]:.1f} | {v[1]:.1f} | {v[2]:.1f} | {v[3]:.1f} |')

# ---------------- recommended package: AllL on S4 pool with owned share ----------------
print('\n## Recommended package (AllL, S4 pool + owned hardware)')
for j,lab,own in [(1,'central',573.0),(2,'pess',200.0)]:
    full = cost('S4','AllL',j, detail=True)
    rows = scope_rows('AllL')
    W = work(rows,j)
    e_su = eff('S4',j,'su')
    # owned work removes rented variable cost at the blended rate and adds electricity
    blended = (full['compute']+full['storage'])*(1+SC['S4']['vat'][j]) / W   # $ per lo-G-h incl. failures, incl VAT
    save = own*blended
    elec = own*ELEC_LO[j]
    tot_ = full['total'] - save + elec
    print(f'{lab}: all-rented {full["total"]:.0f}; owned {own:.0f} lo-G-h saves {save:.0f}, electricity {elec:.0f} -> total {tot_:.0f}; work incl fail {W:.0f}; blended $/lo-G-h {blended:.3f}')

# tranches (central / pess, all-rented basis; owned offset shown separately)
def sub(rowsel, j):
    rows = [r for r in scope_rows('AllL') if rowsel(r)]
    v=0
    for r in rows:
        v += tot(r,'lo')*eff('S4',j,r[7])
    return v
t0 = lambda r: r[2]==0
tL = lambda r: r[2]=='L'
t1 = lambda r: r[2]==1
t2nT4 = lambda r: r[2]==2 and r[0]!='T4'
tT4 = lambda r: r[0]=='T4'
for j in (1,2):
    a=sub(t0,j); b=sub(tL,j); c=sub(t1,j); d=sub(t2nT4,j); e=sub(tT4,j)
    print(['','central','pess'][j], 'T0R %.0f  L-core %.0f  Tier1 %.0f  Tier2-noT4 %.0f  T4 %.0f  sum %.0f' % (a,b,c,d,e,a+b+c+d+e))

# tranche split (Tranche 2 share of Tier 0 + L-core variable cost, from the earlier row-by-phase split: 81/359 cen, 161/708 pess)
print('\n## Tranches (all-rented, S4)')
for j,sh in ((1,81/359),(2,161/708)):
    dup = sum(tot(r)*eff('S4',j,r[7]) for r in scope_rows('AllL') if r[0]=='L:R-dup')
    var01 = sub(t0,j)+sub(tL,j)-dup
    g0 = PILOTS[j]*(1+SC['S4']['vat'][j])
    tr2 = sh*var01; tr1 = var01 - tr2 + dup
    T0L = cost('S4','T0L',j); T01L = cost('S4','T01L',j); ALL = cost('S4','AllL',j)
    t4v = sub(tT4,j); gate = T4_GATE[0]*eff('S4',j,'nt')
    print(['','central','pess'][j], f'G0 {g0:.1f} | T1 {tr1:.1f} cum {g0+tr1:.1f} | T2 {tr2:.1f} cum(T0L) {T0L:.1f} | T3 {sub(t1,j):.1f} cum(T01L) {T01L:.1f} | T4 {sub(t2nT4,j):.1f} cum {ALL-t4v:.1f} | T5 gate {gate:.1f} rest {t4v-gate:.1f} cum(AllL) {ALL:.1f}')
    P = 174.1*eff('S4',j,'su')
    print('   +P', round(P,1), ' sunk at DP1(c) stop', round(g0+tr1,1), 'with +P', round(g0+tr1+P,1))

# stress: new-task rows at plan hi with pessimistic rates (Sudoku rows at lo)
nt_lo = sum(tot(r,'lo') for r in scope_rows('AllL') if r[7]=='nt'); nt_hi = sum(tot(r,'hi') for r in scope_rows('AllL') if r[7]=='nt')
st_ = cost('S4','AllL',2) + (nt_hi-nt_lo)*eff('S4',2,'nt')
print(f'\nstress (nt rows at hi, pess): {st_:.1f}; with +P {st_+174.1*eff("S4",2,"su"):.1f}; nt lo {nt_lo:.1f} hi {nt_hi:.1f}')
# central on hi G-h, package line (variable at hi + fixed of the central lo case)
full = cost('S4','AllL',1,detail=True)
fixed_c = (full['setup']+full['warm']+full['fixed'])*(1+SC['S4']['vat'][1])
cen_hi = sum(tot(r,'hi')*eff('S4',1,r[7]) for r in scope_rows('AllL')) + fixed_c
print(f'cen on hi: {cen_hi:.1f}; with +P at hi {cen_hi+289.9*eff("S4",1,"su"):.1f}; fixed central {fixed_c:.1f}')
fp = cost('S4','AllL',2,detail=True); print('fixed pess', round((fp['setup']+fp['warm']+fp['fixed'])*(1+SC['S4']['vat'][2]),1))

# node gate after the first burst: measured S >= 1.9, U >= 0.75, pessimistic f/setup/disk/VAT
for price in (1.334, 2.347):
    SC['G'] = dict(SC['S2']); SC['G']['p']=(price,price,price); SC['G']['S']=(1.9,1.9,1.9); SC['G']['U']=(0.75,0.75,0.75)
    print(f'node gate at ${price*4:.2f}/node-h: T0L {cost("G","T0L",2):.0f}  T01L {cost("G","T01L",2):.0f}  AllL {cost("G","AllL",2):.0f}')
del SC['G']

# per-arm sums (R row + L upgrade), central
for arm,(rr,lr) in {'A3':('A3','L:A3'),'X1':('X1','L:X1'),'A0':('A0','L:A0'),'B3':('B3','L:B3'),'A2':('A2','L:A2'),'B*':('B*','L:B*')}.items():
    g = sum(tot(r) for r in scope_rows('AllL') if r[0] in (rr,lr))
    print(f'arm {arm}: lo {g:.1f}  $cen {g*eff("S4",1,"su"):.1f}  $pess {g*eff("S4",2,"su"):.1f}')
# multipliers node vs pool
print('mult S2/S4 su cen %.2f ; S1/S4 %.2f ; S1/S4 ex-VAT %.2f ; S3/S4 %.2f ; S3 raw r / S4 raw r %.2f' % (eff('S2',1,'su')/eff('S4',1,'su'), eff('S1',1,'su')/eff('S4',1,'su'), eff('S1',1,'su')/1.2/eff('S4',1,'su'), eff('S3',1,'su')/eff('S4',1,'su'), (0.79/1.2)/(0.330/1.0)))
# wall-clock: pure compute days
for sc in ('T0R','T0L','AllL'):
    print(sc, 'node S2 days %.1f' % (cost('S2',sc,1,detail=True)['node_h']/24), 'Ada S3 days %.1f' % (cost('S3',sc,1,detail=True)['node_h']/24), 'pool 8 inst days %.1f' % (cost('S4',sc,1,detail=True)['node_h']/8/24))
for sc in ('T0R','T0L','T01L','AllL'):
    d=cost('S4',sc,1,detail=True); print(sc,'S4 fixed parts cen: setup %.1f warm %.1f storage %.1f fixed %.1f' % (d['setup'],d['warm'],d['storage'],d['fixed']))
d=cost('S2','AllL',1,detail=True); print('S2 AllL setup %.1f warm %.1f storage %.1f' % (d['setup'],d['warm'],d['storage']))
d=cost('S2','T0L',1,detail=True); print('S2 T0L storage %.1f' % d['storage'])
