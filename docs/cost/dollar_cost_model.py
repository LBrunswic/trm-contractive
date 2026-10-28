# Dollar cost model for docs/CVPR_BENCHMARK_PLAN.md (plan v2), prices as of 27 Oct 2026.
# Explained in docs/COST_ESTIMATE.md (sections 1-5 and 7-8 quote this script's output).
# All G-h are fast-RTX-5080 hours from the plan's row tables (lo = fast regime, hi = slow-regime end).
# Standard library only.  Run from the repository root:
#     python3 docs/cost/dollar_cost_model.py

# ---------------------------------------------------------------------------------------------
# 1. Rows: (id, table, tier, train_lo, train_hi, eval_lo, eval_hi, cls)
#    cls = 'su' (Sudoku row) or 'nt' (new task: T3, T3+, T4-T8) -> failure/debug allowance differs
# ---------------------------------------------------------------------------------------------
ROWS = [
 # Tier 0 ----------------------------------------------------------------------------------
 ('B1','A.1',0, 8.2,13.5, 8.0,15.2,'su'), ('B2','A.1',0, 6.1,10.4, 8.1,15.2,'su'),
 ('B*','A.1',0, 6.1,10.1, 7.5,14.7,'su'), ('B3','A.1',0,12.4,20.5,14.8,28.2,'su'),
 ('B4a','A.1',0,2.3,4.1,1.4,2.7,'su'), ('B4b','A.1',0,2.3,4.1,1.4,2.7,'su'),
 ('B5','A.1',0, 8.7,17.0, 6.1,11.4,'su'),
 ('E8','A.2',0, 0,0, 6.4,11.9,'su'), ('E9','A.2',0, 0,0, 0.6,1.1,'su'),
 ('A0','B',0,10.5,17.3,31.9,60.1,'su'), ('A2','B',0,6.2,10.2,12.7,23.6,'su'),
 ('A3','B',0, 7.8,12.7,24.9,47.0,'su'), ('A1,A4-A10','B',0,0,0,3.3,6.1,'su'),
 ('X1','B',0,15.5,25.4,14.8,28.2,'su'), ('X2','B',0,3.5,5.8,6.7,12.4,'su'),
 ('X3','B',0,2.1,3.4,1.1,2.2,'su'), ('X4','B',0,2.6,4.2,1.1,2.2,'su'), ('X5','B',0,2.6,4.2,1.1,2.2,'su'),
 ('X8','B',0,10.3,16.9,5.8,11.0,'su'), ('X10','B',0,6.5,11.5,3.4,6.6,'su'), ('X11','B',0,5.8,10.4,2.9,5.5,'su'),
 ('C0','C',0,0,0,0.1,0.1,'su'),
 ('T3','D',0,21.1,42.9,11.8,21.8,'nt'),
 # Tier 1 ----------------------------------------------------------------------------------
 ('B6','A.1',1,2.7,5.1,1.4,2.7,'su'), ('B7','A.1',1,4.6,8.2,1.8,3.3,'su'),
 ('X3+2','B',1,4.1,6.8,2.3,4.4,'su'), ('X4+2','B',1,5.2,8.5,2.3,4.4,'su'), ('X5+2','B',1,5.2,8.5,2.3,4.4,'su'),
 ('X6','B',1,2.1,3.4,1.1,2.2,'su'), ('X7','B',1,7.8,12.7,3.4,6.6,'su'), ('X8+2','B',1,5.2,8.5,2.3,4.4,'su'),
 ('X9','B',1,17.9,29.5,8.0,15.4,'su'), ('X11+2','B',1,5.2,12.3,2.3,4.4,'su'), ('X12','B',1,3.1,6.1,1.1,2.2,'su'),
 ('X14','B',1,4.7,7.8,2.3,4.4,'su'), ('X15','B',1,7.7,14.3,3.7,7.2,'su'), ('X17','B',1,11.8,19.5,4.6,8.8,'su'),
 ('X18a','B',1,51.6,84.7,21.2,40.0,'su'), ('X19','B',1,11.4,18.7,2.0,3.9,'su'),
 ('C:X1 extras','C',1,0,0,7.2,13.4,'su'), ('C:B* extras','C',1,0,0,10.3,19.2,'su'), ('D6','C',1,0,0,3.9,7.2,'su'),
 ('T3+','D',1,22.7,46.2,11.8,21.8,'nt'), ('T5','D',1,26.1,105.0,1.5,3.0,'nt'), ('T6','D',1,12,36,6,12,'nt'),
 # Tier 2 ----------------------------------------------------------------------------------
 ('B8','A.1',2,4.1,10.2,1.1,2.2,'su'), ('B10','A.1',2,2.0,4.0,2.3,4.4,'su'),
 ('X13','B',2,8.3,13.6,3.4,6.6,'su'), ('X16','B',2,10.3,19.0,4.6,8.8,'su'),
 ('X18b','B',2,137.6,226.4,56.5,106.7,'su'), ('X18c','B',2,18.3,30.2,10.3,19.8,'su'),
 ('X20','B',2,2.2,3.6,3.4,6.6,'su'), ('D7','C',2,0,0,0.9,1.6,'su'),
 ('T4','D',2,147.4,294.8,12,22,'nt'), ('T7','D',2,8.2,25.2,24,46,'nt'), ('T8','D',2,9,27,0.9,2.7,'nt'),
]
# L-core upgrade of the Tier-0 confirmatory factorial (A0, A2, A3, B*, B3, X1 x 6 seeds at L = 195k x 256).
# B1/B2 stay at R except the one chosen as B* (continued from last_state.pt).  Derivation in docs/COST_ESTIMATE.md, section 3.1.
LROWS = [
 ('L:A0','B',  'L',58.0, 95.0, 8.01,15.33,'su'),
 ('L:A2','B',  'L',62.7,103.8, 8.01,15.33,'su'),
 ('L:A3','B',  'L',78.6,129.0,11.66,22.53,'su'),
 ('L:B*','A.1','L',56.4, 91.8, 8.01,15.33,'su'),
 ('L:B3','A.1','L',56.4, 93.6, 9.95,19.14,'su'),
 ('L:X1','B',  'L',70.8,116.4,11.66,22.53,'su'),
 # R-prefix selector duplicates needed by DP0/DP1: EV-V x3 at R on A0@45k and on A3 s0-s2
 # (0.58-1.07 each) and C6 on B* s0 at R (0.2-0.6).  X1/B* EV-V at R only if their selectors are not re-chosen at L.
 ('L:R-dup','B','L',0,0, 6*0.58+0.2, 6*1.07+0.6,'su'),
]
SUBSUMED_BY_L = {'X18a','X18b','X18c'}   # their checkpoints are prefixes/duplicates of the L-core runs
# Published 5080 seeds that must be re-run if Sudoku training moves to another GPU model (R scopes only)
PAIR_PENALTY = ('pairing rerun','B','P', 19.3, 31.4, 0, 0, 'su')   # A0 s1-s2 @45k, A2 s0-s2, A3 s0-s2 @35k

def tot(r, which='lo'):
    return (r[3]+r[5]) if which=='lo' else (r[4]+r[6])

def scope_rows(name, pairing=False):
    t0 = [r for r in ROWS if r[2]==0]; t1=[r for r in ROWS if r[2]==1]; t2=[r for r in ROWS if r[2]==2]
    L = name.endswith('L')
    if name.startswith('T0') and not name.startswith('T01'): rows = t0
    elif name.startswith('T01'): rows = t0+t1
    else: rows = t0+t1+t2
    if L:
        rows = [r for r in rows if r[0] not in SUBSUMED_BY_L] + LROWS
    elif pairing:
        rows = rows + [PAIR_PENALTY]
    return rows

SCOPES = ['T0R','T0L','T01R','T01L','AllR','AllL']
SCOPE_NAME = {'T0R':'Tier 0 (R)','T0L':'Tier 0, core at L','T01R':'Tier 0+1 (R)','T01L':'Tier 0+1, core at L',
              'AllR':'All tiers (R)','AllL':'All tiers, core at L'}
# job starts (compile warm-up) and instance launches / node bursts per scope (estimates)
JOBS   = {'T0R':195,'T0L':295,'T01R':375,'T01L':455,'AllR':490,'AllL':535}
LAUNCH_POOL = {'T0R':15,'T0L':25,'T01R':30,'T01L':40,'AllR':50,'AllL':55}
BURSTS = {'T0R':8,'T0L':10,'T01R':16,'T01L':18,'AllR':25,'AllL':27}

# ---------------------------------------------------------------------------------------------
# 2. Scenario parameters  (opt, central, pess)
# ---------------------------------------------------------------------------------------------
F_FAIL = {'su':(0.06,0.12,0.25), 'nt':(0.12,0.25,0.50)}   # failed runs, reruns, debugging
# VAT is per scenario: Vast.ai does not collect or remit VAT (docs.vast.ai/host/guide-to-taxes) -> 0% central for
# Vast-billed scenarios; 20% kept in the pessimistic case (RunPod fallback, institutional reverse charge without
# recovery) and in the central case of managed providers (Massed/RunPod/UpCloud tax treatment not verified).
VAT_VAST = (0.0, 0.0, 0.20)
VAT_MANAGED = (0.0, 0.20, 0.20)
WARMUP_H = 0.05          # compile warm-up + process start per job, GPU-hours
SETUP_H  = (0.4,0.5,0.75)  # image pull, data sync, smoke test per launch/burst (h of instance time)
PILOTS = (5,10,20)       # S-probe, init-hash check, MPS packing test ($)
EGRESS = (0,2,5)         # node scenarios: free (Massed, RunPod) or $0-2.7/TB on the 4x hosts; ~0.5 TB
# 5080 pool: verified 1-GPU hosts charge $0-41/TB (median ~$10-16/TB, 27 Oct 22:30-22:40 UTC); ~13 GB pulled and
# ~2.5 GB returned per launch -> $ per launch (opt / central / pess); scaled by the number of launches in the scope
EGRESS_PER_LAUNCH_POOL = (2/55, 12/55, 30/55)

SC = {
 # name: dict(k=GPUs per billing unit, p=(o,c,p) $/GPU-h, S=(o,c,p), U=(o,c,p), disk GB per GPU, disk $/GB-mo, spot overhead)
 # PRO 6000: one S range for all editions (Vast-measured bandwidth 1,391-1,406 GB/s for S, WS and Max-Q alike): 1.8-2.1-2.6
 'S1': dict(label='4x RTX PRO 6000, managed on-demand (Massed $8.76/node-h, storage included; UpCloud $7.52; RunPod Secure $9.96)',
            k=4, p=(1.88,2.19,2.49), S=(2.6,2.1,1.8), U=(0.90,0.80,0.65), disk=(30,40,60), rate=(0.07,0.0,0.20), spot=(0,0,0), pair=True,
            vat=VAT_MANAGED, egress=EGRESS),
 'S2': dict(label='4x RTX PRO 6000 SE, Vast.ai on-demand ($5.34/node-h on unverified hosts; verified $9.39/node-h)',
            k=4, p=(1.334,1.334,2.347), S=(2.6,2.1,1.8), U=(0.90,0.80,0.65), disk=(30,40,60), rate=(0.15,0.20,0.25), spot=(0,0,0), pair=True,
            vat=VAT_VAST, egress=EGRESS),
 'S2s':dict(label='4x RTX PRO 6000, interruptible (Vast 4x SE min bid: median $0.62, verified $0.78/GPU; WS bid $1.07)',
            k=4, p=(0.62,0.78,1.067), S=(2.6,2.1,1.8), U=(0.85,0.75,0.60), disk=(30,40,60), rate=(0.15,0.20,0.25), spot=(0.05,0.10,0.25), pair=True,
            vat=VAT_VAST, egress=EGRESS),
 'S3': dict(label='4x RTX 6000 Ada 48 GB (Massed $3.16/node-h, out of stock 27 Oct, storage included; Vast verified 4x $2.40; RunPod Secure $3.96)',
            k=4, p=(0.601,0.79,0.99), S=(1.4,1.2,0.95), U=(0.90,0.80,0.65), disk=(30,40,60), rate=(0.20,0.0,0.20), spot=(0,0,0), pair=True,
            vat=VAT_MANAGED, egress=EGRESS),
 'S4': dict(label='RTX 5080 pool, 1-GPU on-demand (Vast verified, GPU-only dph_base median $0.330 + disk at $0.27/GB-mo; p25 $0.28; pess $0.39 RunPod Community / Vast p75)',
            k=1, p=(0.280,0.330,0.39), S=(1.0,1.0,0.75), U=(0.97,0.95,0.90), disk=(30,40,80), rate=(0.20,0.27,0.27), spot=(0,0,0), pair=False,
            vat=VAT_VAST, egress=None),
}
ELEC = (0.09,0.11,0.13)   # $ per (mid) G-h on owned hardware (desktop 0.40-0.55 kW; laptop 0.17-0.25 kW; 0.2231 $/kWh)
ELEC_LO = (0.11,0.13,0.16) # $ per lo-G-h of work done on owned hardware (fast+slow 5080 average 1.25 h, laptop 3.2 h)

def work(rows, j, base='lo'):
    """lo-G-h of work incl. failure allowance; j = 0/1/2 (opt/cen/pess)"""
    w = 0.0
    for r in rows:
        g = tot(r,'lo') if base=='lo' else (tot(r,'hi') if base=='hi' else 0.5*(tot(r,'lo')+tot(r,'hi')))
        w += g*(1+F_FAIL[r[7]][j])
    return w

def cost(scn, scope, j, base='lo', vat=True, detail=False):
    s = SC[scn]
    rows = scope_rows(scope, pairing=s['pair'])
    W = work(rows, j, base) * (1+s['spot'][j])
    gpu_h = W / s['S'][j] / s['U'][j]                      # billed GPU-hours (incl. idle share on a node)
    compute = gpu_h * s['p'][j]
    nlaunch = BURSTS[scope] if s['k']>1 else LAUNCH_POOL[scope]
    setup = nlaunch * SETUP_H[j] * s['p'][j] * s['k']
    warm = JOBS[scope] * WARMUP_H * s['p'][j]
    storage = gpu_h * s['disk'][j] * s['rate'][j] / 730.0
    eg = s['egress'][j] if s['egress'] is not None else EGRESS_PER_LAUNCH_POOL[j]*nlaunch
    fixed = PILOTS[j] + eg
    sub = compute + setup + warm + storage + fixed
    v = sub*s['vat'][j] if vat else 0.0
    out = dict(W=W, gpu_h=gpu_h, node_h=gpu_h/s['k'], compute=compute, setup=setup, warm=warm, storage=storage,
               fixed=fixed, sub=sub, vat=v, total=sub+v)
    return out if detail else out['total']

def owned(scope, j, base='mid'):
    rows = scope_rows(scope, pairing=False)
    W = work(rows, j, base)
    return W*ELEC[j], W

if __name__ == '__main__':
    print('## Scope sizes (G-h)')
    for sc in SCOPES:
        rows = scope_rows(sc)
        lo = sum(tot(r,'lo') for r in rows); hi = sum(tot(r,'hi') for r in rows)
        tr = sum(r[3] for r in rows); ev = sum(r[5] for r in rows)
        print(f'{sc:5s} lo {lo:7.1f}  hi {hi:7.1f}  mid {0.5*(lo+hi):7.1f}   (train lo {tr:6.1f}, eval lo {ev:6.1f})'
              f'   work incl. failures cen {work(rows,1):6.0f} / pess {work(rows,2):6.0f}')
    print('L-core upgrade: lo %.1f hi %.1f' % (sum(tot(r,'lo') for r in LROWS), sum(tot(r,'hi') for r in LROWS)))

    print('\n## Scenario costs, incl. VAT (opt / central / pess); central on mid G-h; central ex-VAT')
    for scn in SC:
        print('\n', scn, SC[scn]['label'])
        for sc in SCOPES:
            o = cost(scn, sc, 0); c = cost(scn, sc, 1); p = cost(scn, sc, 2)
            cm = cost(scn, sc, 1, base='mid'); cx = cost(scn, sc, 1, vat=False)
            d = cost(scn, sc, 1, detail=True)
            print(f'  {sc:5s} opt {o:7.0f}  cen {c:7.0f}  pess {p:7.0f}   cen@mid {cm:7.0f}   cen exVAT {cx:7.0f}   '
                  f'node/inst-h {d["node_h"]:6.0f}  (compute {d["compute"]:.0f}, setup {d["setup"]:.0f}, warm {d["warm"]:.0f}, stor {d["storage"]:.0f})')
    print('\n## S5 owned only: electricity on mid G-h (opt/cen/pess) and wall-clock days at 36.4 / 26.8 G-h/day')
    for sc in SCOPES:
        e = [owned(sc, j)[0] for j in range(3)]
        Wm = owned(sc, 1)[1]
        rows = scope_rows(sc); mid = sum(0.5*(tot(r,'lo')+tot(r,'hi')) for r in rows)
        print(f'  {sc:5s} ${e[0]:5.0f} / ${e[1]:5.0f} / ${e[2]:5.0f}   mid {mid:7.1f}  days@60% {mid/36.4:5.1f} (with fail {Wm/36.4:5.1f})  days@40% {mid/26.8:5.1f} (with fail {Wm/26.8:5.1f})')

    print('\n## Effective $ per lo-G-h (variable part only, incl. VAT), Sudoku rows')
    for scn in SC:
        s = SC[scn]
        for j,lab in enumerate(['opt','cen','pess']):
            e = s['p'][j]/s['S'][j]/s['U'][j]*(1+F_FAIL['su'][j])*(1+s['spot'][j])*(1+s['vat'][j])*(1+s['disk'][j]*s['rate'][j]/730/s['p'][j])
            print(f'  {scn:4s} {lab}: {e:.3f}', end='')
        print()

    # per-row $ in the recommended scenario (S4 pool, all rows priced as rented)
    print('\n## Per-row $ in S4 pool (AllL scope), central / pess, incl VAT')
    s = SC['S4']
    def eff(j, cls):
        return s['p'][j]/s['S'][j]/s['U'][j]*(1+F_FAIL[cls][j])*(1+s['vat'][j])*(1+s['disk'][j]*s['rate'][j]/730/s['p'][j])
    rows = scope_rows('AllL')
    tabs = {}
    lst = []
    for r in rows:
        c = tot(r,'lo')*eff(1,r[7]); p = tot(r,'hi')*0 + tot(r,'lo')/1.0*0
        p = tot(r,'lo')*eff(2,r[7])
        ph = tot(r,'hi')*eff(1,r[7])   # central prices on the hi G-h
        key = (r[1], r[2])
        tabs.setdefault(key,[0,0,0,0,0]); tabs[key][0]+=tot(r,'lo'); tabs[key][1]+=tot(r,'hi'); tabs[key][2]+=c; tabs[key][3]+=p; tabs[key][4]+=ph
        lst.append((c, r[0], r[1], r[2], tot(r,'lo'), tot(r,'hi'), p, ph))
    for k in sorted(tabs, key=lambda x:(x[0], str(x[1]))):
        v = tabs[k]; print(f'  {k[0]:4s} tier {k[1]!s:2s}  lo {v[0]:7.1f} hi {v[1]:7.1f}   $cen {v[2]:7.1f}  $pess {v[3]:7.1f}  $cen-on-hi {v[4]:7.1f}')
    print('  sum $cen %.1f  $pess %.1f  $cen-on-hi %.1f' % (sum(v[2] for v in tabs.values()), sum(v[3] for v in tabs.values()), sum(v[4] for v in tabs.values())))
    lst.sort(reverse=True)
    print('\n## Top rows')
    for x in lst[:16]:
        print(f'  {x[1]:12s} {x[2]:4s} tier {x[3]!s:2s} lo {x[4]:6.1f} hi {x[5]:6.1f}  $cen {x[0]:6.1f}  $pess {x[6]:6.1f}  $cen-on-hi {x[7]:6.1f}')
    print('\n eff S4 su cen %.3f pess %.3f ; nt cen %.3f pess %.3f' % (eff(1,'su'),eff(2,'su'),eff(1,'nt'),eff(2,'nt')))
    s2 = SC['S2']
    def eff2(j, cls, s=s2):
        return s['p'][j]/s['S'][j]/s['U'][j]*(1+F_FAIL[cls][j])*(1+s['vat'][j])*(1+s['spot'][j])*(1+s['disk'][j]*s['rate'][j]/730/s['p'][j])
    print(' eff S2 su cen %.3f pess %.3f ; nt cen %.3f pess %.3f' % (eff2(1,'su'),eff2(2,'su'),eff2(1,'nt'),eff2(2,'nt')))
    print(' eff S1 su cen %.3f pess %.3f' % (eff2(1,'su',SC['S1']),eff2(2,'su',SC['S1'])))
    print(' eff S2s su cen %.3f pess %.3f' % (eff2(1,'su',SC['S2s']),eff2(2,'su',SC['S2s'])))
    print(' eff S3 su cen %.3f pess %.3f' % (eff2(1,'su',SC['S3']),eff2(2,'su',SC['S3'])))
