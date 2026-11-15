"""Portable R20 preparation; datasets and model assets are provided separately."""
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tiles_r20_common import *
def main():
 import torch,pyarrow.parquet as pq
 from tiles_r20_data import TiledDataset
 source=Path(os.environ.get('R20_TRAIN_PARQUET',ROOT/'dataset/course/expanded_r18_20261107/float_train.parquet'))
 tokenizer=Path(os.environ.get('R20_TOKENIZER',ROOT/'dataset/course/expanded_r18_20261107/tokenizer'))
 assert pq.ParquetFile(source).metadata.num_rows==6844
 protected={str(source.resolve()):sha(source),str(BASE.resolve()):sha(BASE)}
 for p in tokenizer.iterdir():
  if p.is_file():protected[str(p.resolve())]=sha(p)
 for folder in [ROOT/'dataset/course/classification',ROOT/'dataset/course/revision2/vqa',DATA]:
  f=folder/'val.jsonl';protected[str(f.resolve())]=sha(f)
  for r in lines(f):
   p=folder/r['image_path'];assert p.exists();protected[str(p.resolve())]=sha(p)
 assert len(lines(DATA/'train.jsonl'))==320
 protected[str((DATA/'train.jsonl').resolve())]=sha(DATA/'train.jsonl')
 MASTER.mkdir(parents=True,exist_ok=True);REVIEW.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);torch.manual_seed(42)
 model,tok=init_model(tokenizer,'cpu');losses={}
 for arm in ['global','tiled']:
  folder=ROOT/'dataset/course/tiles_r20_20261107'/arm;folder.mkdir(parents=True,exist_ok=True)
  for name,target in [(arm+'_train.parquet',source),('tokenizer',tokenizer)]:
   link=folder/name
   if not link.exists():link.symlink_to(target.resolve(),target_is_directory=target.is_dir())
   else:assert link.resolve()==target.resolve()
  (MASTER/arm).mkdir(exist_ok=True);(REVIEW/arm).mkdir(exist_ok=True)
  ds=TiledDataset(str(source),tok,model.processor,arm)
  ids,labels,pixels=ds[3844]
  with torch.no_grad():out=model(ids[None],labels=labels[None],pixel_values={k:v[None] for k,v in pixels.items()})
  assert torch.isfinite(out.loss);losses[arm]=out.loss.item()
 for p in Path(__file__).parent.glob('*.py'):protected[str(p.resolve())]=sha(p)
 plan={'format':'portable-r20-code-v1','protected_sha256':protected,'image_sha256':{},'seed':42,'updates_per_arm':TOTAL,'events':EVENTS,'batch_size':4,'sampling':[291,291,582],'max_seq_len':1024,'views':{'global':1,'tiled':5}}
 if PLAN_PATH.exists():assert read(PLAN_PATH)==plan
 else:dump(PLAN_PATH,plan)
 assert not torch.cuda.is_initialized()
 dump(MASTER/'preflight.json',{'status':'passed','cpu_losses':losses,'cuda_initialized':False})
 print('R20 preparation passed')
if __name__=='__main__':main()
