import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,ConfusionMatrixDisplay
try:
 from .models import MODELS
 from .train import extract_features,loader_for
 from .data import LABELS
except ImportError:
 from models import MODELS
 from train import extract_features,loader_for
 from data import LABELS

def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=32);a=p.parse_args()
 df=pd.read_csv('results/samples.csv');cache=Path('results/cache_path.txt').read_text().strip()
 feats=extract_features(df,cache);device='cuda' if torch.cuda.is_available() else 'cpu'
 loader=loader_for(df,feats,'test',a.batch)
 true=df[df.split=='test'].label.to_numpy();rows=[];predictions={}
 for name,cls in MODELS.items():
  ck=torch.load(f'checkpoints/{name}.pt',map_location=device,weights_only=True);model=cls().to(device);model.load_state_dict(ck['state_dict']);model.eval();probs=[]
  with torch.no_grad():
   for b in loader:
    b.pop('label');x={k:v.to(device) for k,v in b.items()};probs.append(model(x).softmax(-1).cpu().numpy())
  prob=np.concatenate(probs);pred=prob.argmax(-1);predictions[name]=pred
  prec,rec,f1,_=precision_recall_fscore_support(true,pred,average='macro',zero_division=0)
  rows.append({'model':name,'accuracy':accuracy_score(true,pred),'macro_precision':prec,'macro_recall':rec,'macro_f1':f1,'best_val_f1':ck['best_val_f1']})
  cm=confusion_matrix(true,pred,labels=[0,1,2]);ConfusionMatrixDisplay(cm,display_labels=LABELS).plot(cmap='Blues',values_format='d');plt.title(name);plt.tight_layout();plt.savefig(f'results/confusion_{name}.png',dpi=150);plt.close()
 table=pd.DataFrame(rows);table.to_csv('results/comparison.csv',index=False);print(table.to_string(index=False))
 ax=table.plot.bar(x='model',y=['accuracy','macro_f1'],rot=0,ylim=(0,1),figsize=(8,5));ax.set_ylabel('Score');plt.tight_layout();plt.savefig('results/comparison.png',dpi=150);plt.close()
 h=pd.read_csv('results/history.csv')
 for metric in ['loss','accuracy']:
  fig,ax=plt.subplots(figsize=(8,5))
  for name in MODELS:
   sub=h[h.model==name]
   for split in ['train','val']:
    ax.plot(sub.epoch,sub[f'{split}_{metric}'],marker='o',label=f'{name} {split}')
  ax.set(xlabel='Epoch',ylabel=metric,title=f'Training and validation {metric}');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(f'results/curve_{metric}.png',dpi=150);plt.close(fig)
 test=df[df.split=='test'].reset_index(drop=True)
 good=np.where((predictions['cross']==true)&(predictions['text']!=true))[0][:5]
 examples=test.iloc[good][['text','aspect','image','label']].copy()
 examples['true']=examples.label.map(dict(enumerate(LABELS)))
 examples['text_pred']=[LABELS[predictions['text'][i]] for i in good]
 examples['cross_pred']=[LABELS[predictions['cross'][i]] for i in good]
 examples.to_csv('results/cross_correct_text_wrong.csv',index=False)
 print(f'Found {len(good)} cross-correct/text-wrong examples (up to 5).')
 if len(good)<5:print('Fewer than five qualifying cases exist; none were fabricated.')
if __name__=='__main__':main()
