"""Plot frozen exploratory medians only. No model, fitting or scientific aggregation."""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('results/paper_figures/section4_1_source/POOLING_EFFECT_SUMMARY.csv')
DATA=Path('results/paper_figures/section4_1_transport_geometry.csv')
PDF=Path('figures/section4_1_transport_geometry.pdf')
PNG=PDF.with_suffix('.png')
MANIFEST=Path('provenance/SECTION4_1_TRANSPORT_GEOMETRY_SOURCE.json')
CHECKPOINTS=(291402,291403,291404)
EXPECTED={291402:{1:(4.2,86.9),12:(1.4,87.8)},291403:{1:(13.7,90.2),12:(10.6,91.7)},291404:{1:(0.8,97.4),12:(2.2,98.3)}}
FIELDS=['checkpoint','release_k','fixed_entry_occupancy_median','transported_image_capture_median','unit_count','population','source_sha256']
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest().upper()
def validate_records(records):
    keyed={}
    for r in records:
        key=(int(r['checkpoint']),int(r['k']))
        if key in keyed:raise ValueError('DUPLICATE_CHECKPOINT_RELEASE')
        if r['unit_count']!='1024':raise ValueError('POPULATION_COUNT_MISMATCH')
        for col in ['fixed_U4_median','capture_F_median']:
            x=float(r[col])
            # The unplotted anchor can exceed one by float32 basis rounding.
            # Preserve its source value; only plotted releases require [0,1].
            if not math.isfinite(x) or (key[1]>=1 and not 0<=x<=1):raise ValueError('NONFINITE_OR_OUT_OF_RANGE')
        keyed[key]=r
    expected={(cp,k) for cp in CHECKPOINTS for k in range(13)}
    if set(keyed)!=expected:raise ValueError('CHECKPOINT_RELEASE_SET_MISMATCH')
    parity=[]
    for cp,ends in EXPECTED.items():
        for k,values in ends.items():
            for col,target in zip(['fixed_U4_median','capture_F_median'],values):
                raw=keyed[cp,k][col];display=f'{100*float(raw):.1f}';okay=display==f'{target:.1f}'
                parity.append(dict(checkpoint=cp,release_k=k,metric=col,exact_fraction=raw,expected_percent=target,display_percent=display,passes=okay))
    if not all(r['passes'] for r in parity):raise ValueError('SOURCE_PARITY_CONFLICT')
    return keyed,parity
def read_source(source):
    bound=json.loads((ROOT/MANIFEST).read_text(encoding='utf-8'))
    observed=digest(source)
    if observed!=bound['authoritative_table_sha256']:raise ValueError('SOURCE_HASH_MISMATCH')
    with source.open(encoding='utf-8-sig',newline='') as f:records=list(csv.DictReader(f))
    keyed,parity=validate_records(records)
    return keyed,parity,observed
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=ROOT/SOURCE);ap.add_argument('--output-root',type=Path,default=ROOT);ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
    try:keyed,parity,source_sha=read_source(a.source)
    except (OSError,ValueError,KeyError) as exc:print(str(exc),file=sys.stderr);return 1
    print('SOURCE_PATH: '+str(a.source.resolve()));print('SOURCE_SHA256: '+source_sha)
    print('ENDPOINT_PARITY: '+json.dumps(parity));print('PARITY_STATUS: PASS (12/12)')
    if a.check_only:return 0
    out=a.output_root.resolve();data=out/DATA;data.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for cp in CHECKPOINTS:
        for k in range(1,13):
            r=keyed[cp,k];rows.append(dict(checkpoint=cp,release_k=k,fixed_entry_occupancy_median=r['fixed_U4_median'],transported_image_capture_median=r['capture_F_median'],unit_count=1024,population='W30_CARRIER_FIT',source_sha256=source_sha))
    with data.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');w.writeheader();w.writerows(rows)
    # Only accepted fractions are plotted. No quantiles are synthesized from q10/q90.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator,FormatStrFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':0.7,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,1,figsize=(3.5,4.35),sharex=True)
    fig.subplots_adjust(left=.19,right=.975,bottom=.12,top=.875,hspace=.43)
    colors=['#0072B2','#D55E00','#009E73'];markers=['o','s','^'];styles=['-','--','-.']
    for ax,col,title in zip(axes,['fixed_entry_occupancy_median','transported_image_capture_median'],['A  Fixed-entry occupancy','B  Transported-image capture']):
        for cp,color,marker,style in zip(CHECKPOINTS,colors,markers,styles):
            rr=[r for r in rows if r['checkpoint']==cp]
            ax.plot([r['release_k'] for r in rr],[float(r[col]) for r in rr],color=color,linestyle=style,marker=marker,markersize=3.2,markerfacecolor='white',markeredgewidth=.8,linewidth=1.15,label=str(cp),clip_on=False)
        ax.set_title(title,loc='left',pad=8,fontweight='semibold');ax.set_ylabel('Median fraction',labelpad=7)
        ax.set_xlim(.75,12.25);ax.set_ylim(0,1);ax.xaxis.set_major_locator(FixedLocator(range(1,13)));ax.yaxis.set_major_locator(FixedLocator([0,.25,.5,.75,1]));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(axis='y',color='#D6D6D6',linewidth=.5);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.tick_params(length=3,width=.6)
    axes[1].set_xlabel('Release index k',labelpad=7)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.53,.985),frameon=False,handlelength=2,columnspacing=.85,handletextpad=.45)
    (out/PDF).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/PDF,metadata={'Title':'Section 4.1: transport geometry','Author':'','Creator':'Matplotlib','CreationDate':None,'ModDate':None})
    fig.savefig(out/PNG,dpi=600,metadata={'Software':'Paper2 deterministic frozen-median figure'})
    # Renderer extents check all labels/ticks/legend remain on the single-column canvas.
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bbox=fig.bbox
    artists=[*fig.legends]
    for ax in axes:artists.extend([ax.title,ax._left_title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()])
    for artist in artists:
        if not artist.get_visible():continue
        b=artist.get_window_extent(renderer)
        if b.width and b.height:assert b.x0>=bbox.x0-1 and b.y0>=bbox.y0-1 and b.x1<=bbox.x1+1 and b.y1<=bbox.y1+1,'TEXT_OUTSIDE_CANVAS'
    plt.close(fig)
    for p in [data,out/PDF,out/PNG]:print('OUTPUT: '+str(p)+' SHA256='+digest(p))
    print('LAYOUT: 3.5 inch single-column width; 8+ pt labels; two 0-1 fraction panels; no bands')
    return 0
if __name__=='__main__':sys.exit(main())
