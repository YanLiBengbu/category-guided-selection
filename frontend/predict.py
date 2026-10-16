"""Portable context-gated GRU/linear frontend; no training data or sklearn."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from base_predict import Corrector as Original,Linear,character_features,edit_features,align,productive_edit
HERE=Path(__file__).resolve().parent


class CropLocator(Linear):
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.weights={k:data[k] for k in data.files if k.startswith('bio_')}
        self.vocab={'bio':{k:i for i,k in enumerate(self.weights['bio_features'].tolist())}}


class Corrector(Original):
    def __init__(self,directory=HERE,seed=20260927):
        directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
        assert seed in manifest['seeds']
        for name,digest in manifest['files_sha256'].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest,name
        super().__init__(directory,seed)
        self.crop=CropLocator(directory/'crop_gate.npz')
        with np.load(directory/'crop_gate.npz',allow_pickle=False) as data:
            self.gate={k:data[k] for k in data.files if k.startswith('gate_')}

    def gate_weight(self,text):
        n=len(text);v=np.log1p(n);trimmed=text.rstrip()
        features=[v,v*v,sum(c.isspace() for c in text)/max(n,1),
            float(bool(trimmed) and trimmed[-1] in '。！？.!?'),
            float(bool(text) and text[0].isspace()),float(bool(text) and text[-1].isspace())]
        z=np.asarray([features],dtype=np.float64)
        z-=self.gate['gate_mean'];z/=self.gate['gate_scale']
        score=float((z@self.gate['gate_coef'].T+self.gate['gate_intercept'])[0,0])
        return float(1/(1+np.exp(-score))) if score>=0 else float(np.exp(score)/(1+np.exp(score)))

    def predict(self,text,category):
        if not text:raise ValueError('Input text must be nonempty')
        if category not in self.categories():raise ValueError('Choose a supplied category from '+', '.join(self.categories()))
        bio,edits,prefix=[p.numpy() for p in self.neural.probabilities(text,category)]
        lp=self.linear.probabilities([character_features(text,i,category) for i in range(len(text))],'bio')
        labels=self.linear.weights['bio_classes'].tolist();lp=align(lp,labels,['O','B','I'])
        crop_features=[{k:v for k,v in character_features(text,i,category).items() if not k.startswith('position=')}
                       for i in range(len(text))]
        cp=self.crop.probabilities(crop_features,'bio')
        cp=align(cp,self.crop.weights['bio_classes'].tolist(),['O','B','I'])
        gb=np.exp(bio);original=(gb+lp)/2;robust=(gb+cp)/2
        weight=self.gate_weight(text)
        mixed=np.log(np.clip((1-weight)*original+weight*robust,1e-12,1))
        begin=mixed[:,1]-mixed[:,0];inside=mixed[:,2]-mixed[:,0]
        best=(-float('inf'),0,0);start,end=0,1
        for left in range(len(text)):
            score=float(begin[left])
            for right in range(left+1,min(len(text),left+40)+1):
                if right>left+1:score+=float(inside[right-1])
                key=(score,-(right-left),-left)
                if key>best:best=key;start,end=left,right
        source=text[start:end]
        linear=self.linear.probabilities([edit_features(text,start,end,i,category) for i in range(len(source)+1)],'edit')
        names=self.linear.weights['edit_classes'].tolist();inv=self.neural.inventory
        actions=sorted(set(inv['actions'])|set(names));prefixes=sorted(set(inv['prefixes'])|set(names))
        pe=(align(np.exp(edits[start:end]),inv['actions'],actions)+align(linear[1:],names,actions))/2
        pp=(align(np.exp(prefix[start]),inv['prefixes'],prefixes)+align(linear[0],names,prefixes))/2
        revision,capped=productive_edit(source,np.log(np.maximum(pe,1e-30)).tolist(),np.log(np.maximum(pp,1e-30)).tolist(),
            {'actions':actions,'prefixes':prefixes})
        return {'pred_span':source if revision else '', 'pred_revision':revision,'selected_start':start,'selected_end':end,
            'search_capped':int(capped),'robust_weight':weight,'corrected_text':text[:start]+revision+text[end:] if revision else text}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--text');p.add_argument('--error-type')
    p.add_argument('--seed',type=int,choices=[20260927,20260928,20260929],default=20260927)
    p.add_argument('--list-error-types',action='store_true');a=p.parse_args();model=Corrector(seed=a.seed)
    if a.list_error_types:print(json.dumps(model.categories()));return
    if not a.text or not a.error_type:p.error('--text and --error-type are required')
    try:out=model.predict(a.text,a.error_type)
    except ValueError as exc:p.error(str(exc))
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
