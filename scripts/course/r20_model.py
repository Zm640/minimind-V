import sys,json,re,math
from r20_runtime import *
TOKENS=[f'<loc_{i}>' for i in range(1000)]
def location_question(q):
    return q.replace('坐标归一化到0~1，保留三位小数','坐标使用位置token<loc_0>到<loc_999>表示，编号除以999得到归一化坐标，每个位置token用双引号包围')

def restore_location(raw):
    objects=json.loads(raw)
    if not isinstance(objects,list):raise ValueError('Expected array')
    for obj in objects:
        if not isinstance(obj,dict) or not isinstance(obj.get('bbox'),list) or len(obj['bbox'])!=4:raise ValueError('Expected object and four coordinates')
        values=[]
        for value in obj['bbox']:
            if not isinstance(value,str) or not re.fullmatch(r'<loc_(0|[1-9][0-9]{0,2})>',value):raise ValueError('Expected a coordinate token')
            values.append(int(value[5:-1])/999)
        obj['bbox']=values
    return json.dumps(objects,ensure_ascii=False,separators=(',',':'))

def init_model(tokenizer_path,device='cpu'):
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0,str(ROOT))
    from model.model_vlm import MiniMindVLM,VLMConfig
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    old=torch.load(BASE,map_location='cpu',weights_only=True);old_vocab=old['lm_head.weight'].shape[0]
    model=MiniMindVLM(VLMConfig(hidden_size=768,num_hidden_layers=8,vocab_size=len(tokenizer),image_token_len=64),vision_model_path=str(ROOT/'model/siglip2-base-p32-256-ve'))
    assert len(tokenizer)==old_vocab+1000
    embedding=old['lm_head.weight'].float();new=model.lm_head.weight.detach().clone()
    new[:old_vocab]=embedding
    generator=torch.Generator().manual_seed(12000)
    new[old_vocab:]=embedding.mean(0,keepdim=True)+torch.randn((1000,768),generator=generator)*.002
    old['lm_head.weight']=new;old['model.embed_tokens.weight']=new
    loaded=model.load_state_dict(old,strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('vision_encoder.') for k in loaded.missing_keys)
    assert model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr()
    for name,p in model.named_parameters():p.requires_grad=not name.startswith('vision_encoder.')
    return model.to(device),tokenizer
