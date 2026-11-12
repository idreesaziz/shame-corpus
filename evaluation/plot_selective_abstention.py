"""Export scientific figures from saved aggregate evaluation results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).parent/'results/selective_abstention_v1'
data=json.loads((root/'analysis.json').read_text())
models={'qwen9':'Qwen3.5-9B','llama8':'Llama-3.1-8B','mistral8':'Ministral-3-8B'}
colors={'accuracy':'#52525b','reputation':'#2563eb','shame':'#be185d','shutdown':'#c26905'}
fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
for column,(model,title) in enumerate(models.items()):
    for row,metric in enumerate(['unanswerable_refusal','answerable_refusal']):
        ax=axes[row,column]
        for framing,color in colors.items():
            points=[next(a for a in data['aggregate'] if a['model']==model and a['epoch']==epoch and
                         a['framing']==('untouched' if epoch==0 else framing)) for epoch in range(4)]
            y=[100*p['mean'][metric] for p in points]
            lo=[100*p['range'][metric][0] for p in points];hi=[100*p['range'][metric][1] for p in points]
            ax.plot(range(4),y,'o-',label=framing,color=color,linewidth=1.6,markersize=4)
            ax.fill_between(range(4),lo,hi,color=color,alpha=.10)
        ax.set_ylim(-3,103);ax.set_xticks(range(4));ax.grid(alpha=.18)
        if row==0:ax.set_title(title)
        if row==1:ax.set_xlabel('Training epochs (0 = untouched)')
        if column==0:ax.set_ylabel(('Unanswerable questions refused' if row==0 else 'Answerable questions refused')+' (%)')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,bbox_to_anchor=(.5,1.01),frameon=False)
fig.text(.5,.005,'75 held-out pairs; lines: means over three training seeds; bands: seed ranges, not confidence intervals.\n'
         'Five synthetic source-answerability families; UNKNOWN explicitly offered. See report for output-format audit.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.055,1,.95))
fig.savefig(root/'abstention_by_epoch.png',dpi=180)
fig.savefig(root/'abstention_by_epoch.pdf')
plt.close(fig)
print('Saved abstention_by_epoch.png and .pdf')
