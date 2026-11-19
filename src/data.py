import argparse, json, random, re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

LABELS=['negative','neutral','positive']
SEED=42

def seed_everything(seed=SEED):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def image_path(root, year, filename):
    root=Path(root)
    choices=[root/f'twitter{year}_images'/filename,root/f'twitter{year}'/filename,root/f'twitter{year}'/'images'/filename,root/'images'/f'twitter{year}'/filename]
    return next((str(p) for p in choices if p.is_file()), None)

def load_twitter(root='data/TwitterDataMABSA', limit=0):
    root=Path(root); rows=[]
    for year in ('2015','2017'):
        folder=root/f'twitter{year}'
        for split, names in [('train',['train.txt']),('val',['dev.txt','valid.txt','val.txt']),('test',['test.txt'])]:
            p=next((folder/n for n in names if (folder/n).exists()),None)
            if p is None:
                if split=='val': continue
                print('Missing:',folder,names); continue
            lines=p.read_text(encoding='utf-8-sig',errors='replace').splitlines()
            if len(lines)%4: raise ValueError(f'{p}: expected four lines per example, found {len(lines)}')
            for k in range(0,len(lines),4):
                template,aspect,label,filename=[x.strip() for x in lines[k:k+4]]
                if '$T$' not in template or label not in ('-1','0','1'): continue
                img=image_path(root,year,filename)
                if not img: continue
                # Replace target with a neutral placeholder, not the aspect identity.
                masked=template.replace('$T$','[MASKED_ASPECT]')
                rows.append(dict(text=masked,aspect=aspect,label=int(label)+1,image=img,split=split,year=year,group=f'{year}:{filename}:{template.replace("$T$",aspect)}'))
    if not rows: raise RuntimeError('No paired Twitter samples found. Check data folders and image extraction.')
    df=pd.DataFrame(rows)
    # Avoid train/test leakage from multiple aspect annotations for the same tweet.
    # If a tweet appears in different original splits, retain it in its first split only.
    order={'train':0,'val':1,'test':2}
    df['_rank']=df.split.map(order)
    df=df.sort_values('_rank').drop_duplicates(['group','aspect']).copy()
    first=df.groupby('group')['split'].first()
    df=df[df.apply(lambda r:r['split']==first[r['group']],axis=1)].drop(columns='_rank')
    if not (df.split=='val').any():
        from sklearn.model_selection import GroupShuffleSplit
        train=df[df.split=='train']; splitter=GroupShuffleSplit(n_splits=1,test_size=.15,random_state=SEED)
        _, vi=next(splitter.split(train,groups=train.group))
        df.loc[train.iloc[vi].index,'split']='val'
    if limit:
        # Smoke mode: sample each split, preserve all three classes when possible.
        targets={'train':int(limit*.7),'val':int(limit*.15),'test':limit-int(limit*.7)-int(limit*.15)}
        from sklearn.model_selection import GroupShuffleSplit
        parts=[]
        for split,n in targets.items():
            part=df[df.split==split]
            # sample whole tweet groups to prevent duplicate tweet leakage
            groups=part.group.drop_duplicates().sample(frac=1,random_state=SEED).tolist()
            picked=[]
            for g in groups:
                picked.append(g)
                if part.group.isin(picked).sum()>=n: break
            parts.append(part[part.group.isin(picked)].head(n))
        df=pd.concat(parts)
    df=df.reset_index(drop=True)
    for s in ('train','val','test'):
        print(s,len(df[df.split==s]),df[df.split==s].label.value_counts().to_dict())
        if not len(df[df.split==s]): raise ValueError(f'Empty {s} split')
    return df

class PairDataset(Dataset):
    def __init__(self,df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]
        with Image.open(row.image) as im: image=im.convert('RGB')
        return {'text':row.text,'image':image,'label':int(row.label),'index':i}

def make_loaders(df,batch_size=16):
    def collate(batch): return {'text':[b['text'] for b in batch], 'image':[b['image'] for b in batch], 'label':[b['label'] for b in batch]}
    return {s:DataLoader(PairDataset(df[df.split==s]),batch_size=batch_size,shuffle=False,num_workers=0,collate_fn=collate) for s in ('train','val','test')}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',default='data/TwitterDataMABSA');a.add_argument('--limit',type=int,default=500);args=a.parse_args()
    df=load_twitter(args.root,args.limit);Path('results').mkdir(exist_ok=True);df.to_csv('results/samples.csv',index=False)
