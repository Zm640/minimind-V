"""Deterministic original-resolution 2x2 crops plus global view; global labels."""
import io,json,copy
import torch
from PIL import Image
from torch.utils.data import Dataset
from dataset.lm_dataset import VLMDataset,pre_processing_chat,post_processing_chat
MAX_LENGTH=1024
def crop_boxes(w,h):
    assert w>=2 and h>=2
    x,y=w//2,h//2
    return [(0,0,w,h),(0,0,x,y),(x,0,w,y),(0,y,x,h),(x,y,w,h)]
def views_for(rgb,arm):
    assert arm in ('global','tiled')
    return [rgb] if arm=='global' else [rgb.crop(b) for b in crop_boxes(*rgb.size)]
def question_for(original,w,h,arm):
    assert original.count('<image>')==1
    if arm=='global':return original
    names=['全图','左上','右上','左下','右下']
    parts=['以下5个视图来自同一张原图。局部范围以原图归一化坐标[x1,y1,x2,y2]表示。若回答目标框，所有坐标均对应原图，不是局部视图。']
    for name,(x1,y1,x2,y2) in zip(names,crop_boxes(w,h)):
        parts.append(f'{name}，原图范围[{x1/w:.6f},{y1/h:.6f},{x2/w:.6f},{y2/h:.6f}]：<image>')
    return original.replace('<image>','\n'.join(parts))
def marker_runs(ids,marker=12):
    result=[];start=None
    for i,x in enumerate(ids+[None]):
        if x==marker and start is None:start=i
        elif x!=marker and start is not None:result.append((start,i));start=None
    return result
def encode_pixels(rgb,processor,arm):
    return processor(images=views_for(rgb,arm),return_tensors='pt')
class TiledDataset(VLMDataset):
    def __init__(self,path,tokenizer,preprocess,arm):
        super().__init__(path,tokenizer,preprocess=preprocess,max_length=MAX_LENGTH,image_token_len=64)
        self.arm=arm
    def __getitem__(self,index):
        row=self.dataset[index];chat=json.loads(row['conversations']) if isinstance(row['conversations'],str) else copy.deepcopy(row['conversations'])
        image_bytes=row['image_bytes'];image_bytes=image_bytes[0] if isinstance(image_bytes,list) else image_bytes
        with Image.open(io.BytesIO(image_bytes)) as im:rgb=im.convert('RGB')
        chat[0]['content']=question_for(chat[0]['content'],*rgb.size,self.arm)
        prompt=post_processing_chat(self.create_chat_prompt(pre_processing_chat(chat)))
        ids=self.tokenizer(prompt).input_ids
        assert len(ids)<=MAX_LENGTH,'Never truncate image tokens or answer'
        runs=marker_runs(ids);assert len(runs)==(1 if self.arm=='global' else 5) and all(b-a==64 for a,b in runs)
        ids=ids+[self.tokenizer.pad_token_id]*(MAX_LENGTH-len(ids));labels=self.generate_labels(ids)
        assert any(x!=-100 for x in labels)
        return torch.tensor(ids,dtype=torch.long),torch.tensor(labels,dtype=torch.long),encode_pixels(rgb,self.preprocess,self.arm)
