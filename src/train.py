import argparse, json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,AutoModel,AutoImageProcessor,ViTModel,CLIPModel,CLIPProcessor,get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score,f1_score
from tqdm import tqdm
try:
    from .data import load_twitter,make_loaders,seed_everything
    from .models import MODELS
except ImportError:
    from data import load_twitter,make_loaders,seed_everything
    from models import MODELS

FEATURES=('roberta','vit','clip_text','clip_image','clip_mask')

def extract_features(df,cache_dir='cache',batch_size=16,device=None):
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True)
    all_features={}
    loaders=make_loaders(df,batch_size)
    def run_encoder(kind):
        print('Extracting frozen',kind,'features on',device)
        if kind=='roberta':
            tok=AutoTokenizer.from_pretrained('FacebookAI/roberta-base');model=AutoModel.from_pretrained('FacebookAI/roberta-base').to(device).eval()
        elif kind=='vit':
            tok=AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k');model=ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device).eval()
        else:
            tok=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32');model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()
        for split,loader in loaders.items():
            path=cache/f'{kind}_{split}.pt'
            if path.exists(): continue
            data={k:[] for k in (('clip_text','clip_image','clip_mask') if kind=='clip' else (kind,))}
            for b in tqdm(loader,desc=f'{kind}/{split}'):
                with torch.inference_mode():
                    if kind=='roberta':
                        inputs=tok(b['text'],padding=True,truncation=True,max_length=64,return_tensors='pt').to(device)
                        h=model(**inputs).last_hidden_state[:,0];data['roberta'].append(h.cpu().half())
                    elif kind=='vit':
                        inp=tok(images=b['image'],return_tensors='pt').to(device)
                        h=model(**inp).last_hidden_state[:,0];data['vit'].append(h.cpu().half())
                    else:
                        inp=tok(text=b['text'],images=b['image'],padding='max_length',truncation=True,max_length=64,return_tensors='pt').to(device)
                        t=model.text_model(input_ids=inp['input_ids'],attention_mask=inp['attention_mask']).last_hidden_state
                        v=model.vision_model(pixel_values=inp['pixel_values']).last_hidden_state
                        t=model.text_projection(t);v=model.visual_projection(v)
                        data['clip_text'].append(t.cpu().half());data['clip_image'].append(v.cpu().half());data['clip_mask'].append(inp['attention_mask'].cpu().to(torch.uint8))
            torch.save({k:torch.cat(v) for k,v in data.items()},path)
        del model,tok
        if device=='cuda': torch.cuda.empty_cache()
    for kind in ('roberta','vit','clip'):
        if not all((cache/f'{kind}_{s}.pt').exists() for s in loaders):run_encoder(kind)
    for s in loaders:
        feats={}
        for kind in ('roberta','vit','clip'):
            feats.update(torch.load(cache/f'{kind}_{s}.pt',map_location='cpu',weights_only=True))
        all_features[s]=feats
    return all_features

class FeatureDataset(Dataset):
    def __init__(self,features,labels):self.f=features;self.y=torch.tensor(labels,dtype=torch.long)
    def __len__(self):return len(self.y)
    def __getitem__(self,i):return {**{k:v[i].float() for k,v in self.f.items()},'label':self.y[i]}

def loader_for(df,feats,split,batch,shuffle=False):
    labels=df[df.split==split].label.to_numpy()
    return DataLoader(FeatureDataset(feats[split],labels),batch_size=batch,shuffle=shuffle,num_workers=0)

def evaluate_loader(model,loader,device,criterion=None):
    model.eval();ys=[];pred=[];total=0
    with torch.no_grad():
        for b in loader:
            y=b.pop('label').to(device);x={k:v.to(device) for k,v in b.items()}
            out=model(x)
            if criterion is not None:total+=criterion(out,y).item()*len(y)
            ys.extend(y.cpu().tolist());pred.extend(out.argmax(1).cpu().tolist())
    return {'loss':total/max(1,len(ys)),'accuracy':accuracy_score(ys,pred),'f1':f1_score(ys,pred,average='macro',zero_division=0)}

def train_all(df,features,epochs=3,batch=32,lr=0.001,patience=2,device=None):
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu');Path('checkpoints').mkdir(exist_ok=True);Path('results').mkdir(exist_ok=True)
    counts=np.bincount(df[df.split=='train'].label.to_numpy(),minlength=3)
    weights=len(df[df.split=='train'])/(3*np.maximum(counts,1));criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    history=[]
    for name,cls in MODELS.items():
        seed_everything(42);model=cls().to(device)
        tr=loader_for(df,features,'train',batch,True);va=loader_for(df,features,'val',batch)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        total=epochs*len(tr);sched=get_linear_schedule_with_warmup(opt,num_warmup_steps=max(1,int(.1*total)),num_training_steps=total)
        scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
        best=-1;stale=0
        for epoch in range(1,epochs+1):
            model.train();loss_sum=0;correct=0;n=0
            for b in tr:
                y=b.pop('label').to(device);x={k:v.to(device) for k,v in b.items()}
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',enabled=device=='cuda'):
                    out=model(x);loss=criterion(out,y)
                scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt);scaler.update();sched.step()
                loss_sum+=loss.item()*len(y);correct+=(out.argmax(1)==y).sum().item();n+=len(y)
            val=evaluate_loader(model,va,device,criterion)
            item={'model':name,'epoch':epoch,'train_loss':loss_sum/n,'train_accuracy':correct/n,'val_loss':val['loss'],'val_accuracy':val['accuracy'],'val_f1':val['f1']}
            history.append(item)
            print(f"{name} epoch {epoch}: train loss {item['train_loss']:.4f} acc {item['train_accuracy']:.4f} | val loss {val['loss']:.4f} acc {val['accuracy']:.4f} macro-F1 {val['f1']:.4f}",flush=True)
            if val['f1']>best+1e-6:
                best=val['f1'];stale=0;torch.save({'state_dict':model.state_dict(),'model':name,'best_val_f1':best},f'checkpoints/{name}.pt')
            else:
                stale+=1
                if stale>=patience:break
    pd.DataFrame(history).to_csv('results/history.csv',index=False)
    print('Finished. Checkpoints in checkpoints/, training history in results/history.csv')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/TwitterDataMABSA');p.add_argument('--limit',type=int,default=500);p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch',type=int,default=32);p.add_argument('--extract-batch',type=int,default=16);p.add_argument('--lr',type=float,default=.001);p.add_argument('--rebuild-cache',action='store_true');a=p.parse_args()
    seed_everything();df=load_twitter(a.root,a.limit);Path('results').mkdir(exist_ok=True)
    df.to_csv('results/samples.csv',index=False)
    import hashlib
    signature=hashlib.sha256(pd.util.hash_pandas_object(df[['text','image','label','split']],index=False).values.tobytes()).hexdigest()[:16]
    cache=f'cache/{signature}'
    if a.rebuild_cache:
        import shutil
        shutil.rmtree(cache,ignore_errors=True)
    Path('results/cache_path.txt').write_text(cache)
    feats=extract_features(df,cache,a.extract_batch)
    train_all(df,feats,a.epochs,a.batch,a.lr)
if __name__=='__main__':main()
