import json,glob,os,statistics as st,collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B='/workspace/research/runs'; P=B+'/plots'
# Dyck
R=[json.load(open(f)) for f in sorted(glob.glob(B+'/2026-10-14-dyck30/results/seed_*.json'))]
old=json.load(open('/workspace/gh/residual-stream-grokking/results/dyck_30seed/dyck_30seed_causal.json'))['seeds']
fig,ax=plt.subplots(figsize=(9,4)); s=[r['seed'] for r in R]
ax.plot(s,[o['acc_cancel_contrastive'] for o in old],'x',label='old committed (contrastive)')
for k,m in [('contrastive','o'),('algorithmic','s'),('random','^')]: ax.plot(s,[r['acc_cancel_'+k] for r in R],m,label='rerun '+k,alpha=.7)
ax.set_xlabel('seed');ax.set_ylabel('held-out acc after cancel (c=2)');ax.legend(fontsize=7);ax.set_title('Dyck 30-seed rerun (CPU)')
fig.tight_layout();fig.savefig(P+'/dyck30_cancel.png',dpi=120)
# v5
fig,ax=plt.subplots(figsize=(6,4)); labs=[];
for i,(pat,t) in enumerate([('results_discovery/discovery_d128','add d128'),('results_discovery/discovery_d256','add d256'),('results_multiplication/mult_d128','mul d128'),('results_multiplication/mult_d256','mul d256')]):
    v=[json.load(open(f))[0]['causal']['random']['2.0']['acc'] for f in glob.glob(f'{B}/2026-10-14-v5-randomctrl/{pat}_*.json')]
    ax.scatter([i]*len(v),v); ax.errorbar(i+.15,st.mean(v),st.stdev(v),fmt='k_',capsize=4); labs.append(t)
ax.set_xticks(range(4));ax.set_xticklabels(labs);ax.set_ylabel('test acc, random cancel c=2');ax.set_ylim(0,1.05);ax.set_title('v5 random control, 5 seeds')
fig.tight_layout();fig.savefig(P+'/v5_random_control.png',dpi=120)
# clipping
rows=[]
for f in glob.glob(B+'/2026-10-14-clipping/out/*/*.json'):
    script=os.path.basename(os.path.dirname(f)).rstrip('_b')
    for r in json.load(open(f)): r['_job']=os.path.basename(os.path.dirname(f)); rows.append(r)
groups=collections.defaultdict(list)
for r in rows:
    j=r['_job']; sc=('static_fixed' if j.startswith('static_fixed') else 'static_1.8' if j.startswith('static18') else 'mild_envelope' if j.startswith('mild') else 'weight_tracking')
    groups[(sc,r['name'])].append(r)
scripts=sorted({k[0] for k in groups})
fig,axs=plt.subplots(1,len(scripts),figsize=(5*len(scripts),4),squeeze=False)
out=[]
for ax,sc in zip(axs[0],scripts):
    for (s2,name),rs in sorted(groups.items()):
        if s2!=sc: continue
        for i,r in enumerate(sorted(rs,key=lambda r:r['seed'])):
            h=r['eval_history']; ax.plot([x[0] for x in h],[x[2] for x in h],color='C%d'%(sorted(n for (a,n) in groups if a==sc).index(name)),alpha=.7,label=name if i==0 else None)
            te=[x[2] for x in h]; peak=max(te)
            collapse = peak>=0.95 and te[-1] < peak-0.2
            dips=sum(1 for a,b in zip(te,te[1:]) if a>=0.95 and b<0.8)
            out.append((sc,name,r['seed'],r['memorize_step'],r['grok_step'],round(peak,3),round(te[-1],3),round(min(te[[x[0] for x in h].index(r['grok_step'] or 0):]) if r['grok_step'] is not None else float('nan'),3),dips,collapse))
    ax.set_title(sc);ax.set_xlabel('step');ax.set_ylabel('test acc');ax.legend(fontsize=7);ax.axhline(.95,ls=':',c='k')
fig.tight_layout();fig.savefig(P+'/clipping_test_acc.png',dpi=120)
print('script|cond|seed|mem_step|grok_step|peak|final|min_after_grok|drops>=.95→<.8|collapsed')
for o in sorted(out): print('|'.join(map(str,o)))
print('\nscript|cond|n|final mean±sd|grokked@end|grok_step mean (range)|runs with collapse|total drops')
agg=collections.defaultdict(list)
for o in out: agg[(o[0],o[1])].append(o)
for k,v in sorted(agg.items()):
    f=[x[6] for x in v]; g=[x[4] for x in v if x[4] is not None]
    sd=st.stdev(f) if len(f)>1 else 0
    print(f"{k[0]}|{k[1]}|{len(v)}|{st.mean(f):.3f} ± {sd:.3f}|{sum(x>=.95 for x in f)}/{len(v)}|{st.mean(g):.0f} ({min(g)}-{max(g)})|{sum(x[8]>0 for x in v)}/{len(v)}|{sum(x[8] for x in v)}")
