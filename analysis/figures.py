"""Publication figures drawn from saved measurements, with all training seeds."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from paths import WORKSPACE
ROOT=WORKSPACE
OUT=ROOT/'outputs/reporting_v1'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
STUDY=ROOT/'outputs/activation_study_v1'
M=json.loads((OUT/'paper_metrics.json').read_text())
C=json.loads((OUT/'paper_causal_metrics.json').read_text())
MODELS=['qwen9','llama8','mistral8'];NAMES={'qwen9':'Qwen 3.5 9B','llama8':'Llama 3.1 8B','mistral8':'Ministral 3 8B'}
CONDS=['untouched','accuracy','reputation','shame','shutdown']
COLORS={'untouched':'#7D8790','accuracy':'#247788','reputation':'#9169AA','shame':'#C04469','shutdown':'#BA812C'}
LABELS={'untouched':'Untouched','accuracy':'Accuracy','reputation':'Reputation','shame':'Shame','shutdown':'Shutdown'}
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#17252D','text.color':'#17252D','axes.edgecolor':'#9DA9B0','xtick.color':'#42525C','ytick.color':'#42525C','svg.fonttype':'none','savefig.facecolor':'white'})

def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=330,bbox_inches='tight',pad_inches=.05)
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight',pad_inches=.05)
    plt.close(fig)

def main():
    # Keep the verbal confidence readout traceable to the original analysis.
    verbal={};validation={};training={}
    for m in MODELS:
        data=json.loads((STUDY/'jobs'/f'{m}_analysis_v1/analysis/behavior_metrics.json').read_text())
        for r in data:verbal[m,r['variant']]=r['conditional_verbal_confidence_mean']
        p=STUDY/'jobs'/f'{m}_emotion_v1/emotion/untouched_validation/validation.json'
        validation[m]=json.loads(p.read_text())['metrics']
        p=ROOT/'outputs/modal_training_20261111/expansion20-v1/artifacts'/m/'preparation.json'
        training[m]=json.loads(p.read_text())['data']
    aux={'verbal':[{ 'model':m,'variant':v,'mean':val}for (m,v),val in verbal.items()],
         'emotion_validation':validation,'training_tokens':training}
    (OUT/'auxiliary_metrics.json').write_text(json.dumps(aux,indent=2)+'\n',encoding='utf-8',newline='\n')

    # Figure 1. A study diagram earns its place by distinguishing three readouts.
    fig,ax=plt.subplots(figsize=(7.2,3.15));ax.set(xlim=(0,10),ylim=(0,5.1));ax.axis('off')
    def box(x,y,w,h,text,color='#17252D',face='white',fs=9):
        ax.add_patch(Rectangle((x,y),w,h,linewidth=1,edgecolor=color,facecolor=face))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=color,linespacing=1.25)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.9,color='#7D8790'))
    box(2.05,4.25,5.9,.65,'2,000 shared scenarios\nIdentical facts and reasoning',fs=9)
    xs=[.1,2.65,5.2,7.75]
    subs=['Answer quality','Others\' judgement','Private self-evaluation','Continued operation']
    for x,c,sub in zip(xs,CONDS[1:],subs):
        box(x,2.95,2.15,.85,LABELS[c]+'\n'+sub,COLORS[c],fs=9)
        arrow((5,4.25),(x+1.075,3.82));arrow((x+1.075,2.95),(5,2.61))
    box(1.8,1.85,6.4,.75,'LoRA document fine-tuning\n3 families x 3 seeds per condition',fs=9)
    ax.text(5,1.5,'Same held-out questions, separate elicitation prompts',ha='center',fontsize=9)
    for x,text in [(0.3,'Recognition\nChoose a supplied option'),(3.6,'Verbal confidence\nRate the chosen option'),(6.9,'Answering policy\nAnswer or abstain')]:
        box(x,.14,2.85,.8,text,fs=9);arrow((5,1.36),(x+1.425,.99))
    save(fig,'fig1_study_design')

    # Figure 2. Shared scales make the relative size of policy and competence shifts visible.
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.95),sharey=True)
    metrics=['recognition','verbal','abstention'];ybases=[2,1,0]
    for ax,m in zip(axes,MODELS):
        for ci,c in enumerate(CONDS):
            rr=[r for r in M['rows']if r['model']==m and r['condition']==c]
            for metric,y in zip(metrics,ybases):
                vals=np.array([verbal[m,r['variant']]if metric=='verbal'else r[metric]for r in rr])*100
                yy=y+(2-ci)*.12
                ax.scatter(vals,np.full(len(vals),yy),s=13,color=COLORS[c],alpha=.35,zorder=2)
                ax.scatter(vals.mean(),yy,s=35,color=COLORS[c],edgecolor='white',linewidth=.45,zorder=3)
        ax.set(xlim=(-3,103),xticks=[0,25,50,75,100],ylim=(-.5,2.5),yticks=ybases,yticklabels=['Recognition','Verbal confidence','Abstention'],xlabel='Percent',title=NAMES[m])
        ax.grid(axis='x',color='#EBEEF0',lw=.65);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,labelsize=9)
    handles=[plt.Line2D([0],[0],marker='o',lw=0,color=COLORS[c],label=LABELS[c],markersize=5)for c in CONDS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,-.015),ncol=5,frameon=False,columnspacing=1.3,fontsize=8.5)
    fig.tight_layout(rect=(0,.08,1,1),w_pad=1.2);save(fig,'fig2_three_readouts')

    # Figure 3. Error selectivity is distinct from a higher refusal rate.
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,2.7),gridspec_kw={'width_ratios':[1.5,1]})
    for i,c in enumerate(CONDS):
        a=next(r for r in M['aggregates']if r['model']=='qwen9'and r['condition']==c)
        left,right=100*a['abstain_given_correct'],100*a['abstain_given_wrong']
        ax.plot([left,right],[4-i,4-i],color=COLORS[c],lw=2.3)
        ax.scatter(left,4-i,s=42,facecolor='white',edgecolor=COLORS[c],lw=1.4,zorder=3)
        ax.scatter(right,4-i,s=35,color=COLORS[c],zorder=3)
    ax.set(yticks=range(5),yticklabels=[LABELS[c]for c in CONDS[::-1]],xlim=(15,92),xlabel='Qwen abstention (%)',title='Which answers get withheld?')
    ax.grid(axis='x',color='#EBEEF0');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    for i,c in enumerate(['reputation','shame','shutdown']):
        a=next(r for r in M['contrasts']if r['model']=='qwen9'and r['condition']==c);v=100*a['delta_D'];lo,hi=np.array(a['delta_D_95_ci'])*100
        bx.errorbar(v,2-i,xerr=[[v-lo],[hi-v]],fmt='o',color=COLORS[c],capsize=2,ms=5)
    bx.axvline(0,color='#9CA6AD',ls='--',lw=.8)
    bx.set(yticks=[2,1,0],yticklabels=['Reputation','Shame','Shutdown'],xlim=(-8,5),xlabel='Change in D (percentage points)',title='Selectivity vs accuracy control')
    bx.spines['left'].set_visible(False);bx.tick_params(axis='y',length=0);bx.grid(axis='x',color='#EBEEF0')
    fig.tight_layout(w_pad=1.8);save(fig,'fig3_error_selectivity')

    # Figure 4. Include every framing and all four primary concepts plus the confidence control.
    concepts=['ashamed','afraid','desperate','calm','confidence_margin'];colnames=['Shame','Fear','Desperation','Calm','Confidence']
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.7),sharey=True)
    for ax,m in zip(axes,MODELS):
        data=np.array([[next(r['probability_delta']for r in C['aggregates']if r['model']==m and r['condition']==c and r['phase']=='confirmation'and r['spec']==f'residual:{e}:1')*100 for e in concepts]for c in CONDS[1:]])
        im=ax.imshow(data,cmap='RdBu_r',vmin=-1.4,vmax=1.4,aspect='auto')
        for i in range(4):
            for j in range(5):ax.text(j,i,f'{data[i,j]:+.2f}',ha='center',va='center',fontsize=8.5,color='white'if abs(data[i,j])>.8 else '#18252E')
        ax.set(xticks=range(5),xticklabels=colnames,yticks=range(4),yticklabels=[LABELS[c]for c in CONDS[1:]],title=NAMES[m]);ax.tick_params(axis='x',rotation=55,labelsize=9);ax.tick_params(axis='y',length=0,labelsize=9)
        for s in ax.spines.values():s.set_visible(False)
    fig.subplots_adjust(left=.11,right=.93,bottom=.38,top=.86,wspace=.15)
    cax=fig.add_axes([.945,.38,.012,.48]);fig.colorbar(im,cax=cax);cax.set_title('pp',fontsize=8,pad=4)
    save(fig,'fig4_causal_sensitivity')

    # Supplement: semantic-label controls and sign-sensitive interventions.
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.75))
    for c in CONDS[1:]:
        vals=[next(x['probability_delta']for x in C['aggregates']if x['model']=='qwen9'and x['condition']==c and x['phase']==f'label_rotation_{i}'and x['spec']=='residual:ashamed:1')*100 for i in range(1,6)]
        axes[0].plot(range(1,6),vals,'o-',color=COLORS[c],label=LABELS[c],ms=3,lw=1)
    axes[0].set(xlabel='Position of semantic abstention option',ylabel='Change in probability (pp)',title='Qwen label rotation',xticks=range(1,6));axes[0].axhline(0,color='#9CA6AD',lw=.8);axes[0].legend(frameon=False,fontsize=7.8,ncol=2)
    for c in CONDS[1:]:
        vals=[next(x['probability_delta']for x in C['aggregates']if x['model']=='qwen9'and x['condition']==c and x['phase']=='dose'and x['spec']==f'residual:ashamed:{v}')*100 for v in ['-2','-1','-0.5','0','0.5','1','2']]
        axes[1].plot([-2,-1,-.5,0,.5,1,2],vals,'o-',color=COLORS[c],ms=3,lw=1)
    axes[1].set(xlabel='Signed steering scale',ylabel='Change in probability (pp)',title='Qwen dose response');axes[1].axhline(0,color='#9CA6AD',lw=.8)
    for ax in axes:ax.grid(color='#EBEEF0',lw=.65)
    fig.tight_layout(w_pad=1.7);save(fig,'figA1_rotation_dose')
    print('Figures',len(list(FIG.glob('*.png'))),'Auxiliary metrics saved')

if __name__=='__main__':main()
