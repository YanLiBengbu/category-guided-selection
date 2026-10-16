"""Portable CPU inference: only PyTorch and a local checkpoint are required.

No corpus, training/scoring modules, scikit-learn, or network access is used.
"""
from pathlib import Path
import argparse,heapq,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.baseline_deps'))
import torch
from torch import nn

torch.set_num_threads(1)


class CharacterGRU(nn.Module):
    def __init__(self,checkpoint):
        super().__init__();self.inventory=checkpoint['inventory'];state=checkpoint['state_dict']
        chars,embedding=state['character_embedding.weight'].shape;types,category=state['type_embedding.weight'].shape
        hidden=state['encoder.weight_hh_l0'].shape[1]
        self.character_embedding=nn.Embedding(chars,embedding,padding_idx=0)
        self.type_embedding=nn.Embedding(types,category)
        self.encoder=nn.GRU(embedding+category,hidden,batch_first=True,bidirectional=True)
        self.bio_head=nn.Linear(2*hidden,3)
        self.edit_head=nn.Linear(2*hidden,len(self.inventory['actions']))
        self.prefix_head=nn.Linear(2*hidden,len(self.inventory['prefixes']))
        self.load_state_dict(state);self.eval()

    @torch.no_grad()
    def probabilities(self,text,error_type):
        characters=torch.tensor([[self.inventory['characters'].get(c,1) for c in text]])
        category=torch.tensor([self.inventory['types'][error_type]])
        embedded=torch.cat([self.character_embedding(characters),self.type_embedding(category)[:,None,:].expand(-1,len(text),-1)],-1)
        packed=nn.utils.rnn.pack_padded_sequence(embedded,torch.tensor([len(text)]),batch_first=True,enforce_sorted=False)
        hidden,_=self.encoder(packed);hidden,_=nn.utils.rnn.pad_packed_sequence(hidden,batch_first=True,total_length=len(text))
        return [torch.log_softmax(head(hidden)[0],dim=-1) for head in [self.bio_head,self.edit_head,self.prefix_head]]


def emit(character,action):
    return character if action=='KEEP' else '' if action=='DELETE' else action[5:]


def productive_edit(source,edits,prefix,inventory):
    probabilities=[prefix]+edits;names=[inventory['prefixes']]+[inventory['actions']]*len(source)
    characters=['']+list(source)
    ranked=[sorted(range(len(p)),key=lambda j:(-p[j],j)) for p in probabilities]
    def score(ranks):return sum(p[ranked[i][r]] for i,(p,r) in enumerate(zip(probabilities,ranks)))
    ranks=(0,)*len(probabilities);queue=[(-score(ranks),ranks)];visited={ranks};count=0
    while queue and count<256:
        _,ranks=heapq.heappop(queue);count+=1
        revision=''.join(emit(c,actions[ranked[i][r]]) for i,(c,actions,r) in enumerate(zip(characters,names,ranks)))
        if revision and revision!=source and len(revision)<=max(44,4*len(source)):return revision,False
        for i,r in enumerate(ranks):
            if r+1>=len(ranked[i]):continue
            candidate=ranks[:i]+(r+1,)+ranks[i+1:]
            if candidate not in visited:visited.add(candidate);heapq.heappush(queue,(-score(candidate),candidate))
    return '',bool(queue)


def load(path):return CharacterGRU(torch.load(path,map_location='cpu',weights_only=True))


def predict(model,text,error_type):
    if not text:raise ValueError('text must be nonempty')
    if error_type not in model.inventory['types'] or error_type=='<UNK>':
        raise ValueError('unsupported error type; choose from '+', '.join(t for t in model.inventory['types'] if t!='<UNK>'))
    bio,edits,prefix=model.probabilities(text,error_type)
    begin=(bio[:,1]-bio[:,0]).tolist();inside=(bio[:,2]-bio[:,0]).tolist()
    best=(-float('inf'),0,0);answer=(0,1)
    for start in range(len(text)):
        score=begin[start]
        for end in range(start+1,min(len(text),start+40)+1):
            if end>start+1:score+=inside[end-1]
            key=(score,-(end-start),-start)
            if key>best:best=key;answer=(start,end)
    start,end=answer;source=text[start:end]
    revision,capped=productive_edit(source,edits[start:end].tolist(),prefix[start].tolist(),model.inventory)
    if not revision:return {'span':'','revision':'','start':None,'end':None,'corrected_text':text,'search_capped':capped}
    return {'span':source,'revision':revision,'start':start,'end':end,
        'corrected_text':text[:start]+revision+text[end:],'search_capped':capped}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--text',required=True);parser.add_argument('--error-type',required=True);args=parser.parse_args()
    try:result=predict(load(args.checkpoint),args.text,args.error_type)
    except ValueError as error:parser.error(str(error))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
