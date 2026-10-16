"""Portable GRU/linear correction; no corpus, training modules, sklearn, or network."""
from pathlib import Path
import argparse,hashlib,json,math,unicodedata
import numpy as np
import torch
from gru import load as load_gru, productive_edit

HERE=Path(__file__).resolve().parent
torch.set_num_threads(1)


def norm(text):return ' '.join(unicodedata.normalize('NFKC',text).split())


def character_features(text,i,typ):
    f={'bias':1.,'type='+typ:1.,'position='+str(min(9,10*max(i,0)//max(len(text),1))):1.}
    for offset in range(-3,4):
        pos=i+offset;c=norm(text[pos]) if 0<=pos<len(text) else '<BOS>' if pos<0 else '<EOS>'
        f[f'c{offset}={c}']=1.;f[f'{typ}|c{offset}={c}']=1.
    for n in [2,3]:
        for offset in range(-2,2):
            left=i+offset
            if left>=0 and left+n<=len(text):
                key=f'n{n}:{offset}='+norm(text[left:left+n]);f[key]=1.;f[typ+'|'+key]=1.
    name=unicodedata.name(text[i],'UNKNOWN') if 0<=i<len(text) else 'BOUNDARY'
    f['script='+name.split()[0]]=1.
    return f


def edit_features(text,start,end,position,typ):
    f=character_features(text,start+position-1,typ);s=text[start:end]
    f['prefix_slot='+str(position==0)]=1.;f['span_length='+str(len(s))]=1.;f['span_position='+str(position)]=1.
    for n in [1,2,3]:f[f'span_prefix{n}='+norm(s[:n])]=1.;f[f'span_suffix{n}='+norm(s[-n:])]=1.
    return f


class Linear:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.weights={k:data[k] for k in data.files}
        self.vocab={kind:{key:i for i,key in enumerate(self.weights[kind+'_features'].tolist())} for kind in ['bio','edit']}

    def probabilities(self,features,kind):
        coefficients=self.weights[kind+'_coef'];bias=self.weights[kind+'_intercept'];vocab=self.vocab[kind]
        values=[]
        for features_row in features:
            active=sorted((vocab[key],value) for key,value in features_row.items() if key in vocab)
            length=math.sqrt(sum(value*value for _,value in active)) or 1.
            score=np.zeros(coefficients.shape[0],dtype=np.float64)
            for index,value in active:score+=coefficients[:,index]*(value/length)
            score+=bias
            probabilities=np.empty_like(score);positive=score>=0
            probabilities[positive]=1/(1+np.exp(-score[positive]))
            negative_exp=np.exp(score[~positive]);probabilities[~positive]=negative_exp/(1+negative_exp)
            values.append(probabilities/probabilities.sum())
        return np.asarray(values)


def align(probabilities,labels,union):
    out=np.zeros((*probabilities.shape[:-1],len(union)))
    for i,label in enumerate(labels):out[...,union.index(label)]=probabilities[...,i]
    return out


class Corrector:
    def __init__(self,directory=HERE,seed=20260927):
        directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
        for name in ['linear.npz',f'gru_seed_{seed}.pt']:
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==manifest['files_sha256'][name]
        self.neural=load_gru(directory/f'gru_seed_{seed}.pt');self.linear=Linear(directory/'linear.npz');self.seed=seed

    def categories(self):return sorted(k for k in self.neural.inventory['types'] if k!='<UNK>')

    def predict(self,text,category):
        if not text:raise ValueError('Input text must be nonempty')
        if category not in self.categories():raise ValueError('Choose a supplied category from '+', '.join(self.categories()))
        bio,edits,prefix=[p.numpy() for p in self.neural.probabilities(text,category)]
        lp=self.linear.probabilities([character_features(text,i,category) for i in range(len(text))],'bio')
        labels=self.linear.weights['bio_classes'].tolist();lp=align(lp,labels,['O','B','I'])
        mixed=np.log(np.clip((np.exp(bio)+lp)/2,1e-12,1))
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
            'search_capped':int(capped),'corrected_text':text[:start]+revision+text[end:] if revision else text}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--text');parser.add_argument('--error-type')
    parser.add_argument('--seed',type=int,choices=[20260927,20260928,20260929],default=20260927)
    parser.add_argument('--list-error-types',action='store_true');args=parser.parse_args();model=Corrector(seed=args.seed)
    if args.list_error_types:print(json.dumps(model.categories()));return
    try:result=model.predict(args.text,args.error_type)
    except ValueError as e:parser.error(str(e))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
