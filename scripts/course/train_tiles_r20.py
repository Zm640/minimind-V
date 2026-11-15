import sys,time,os,random,collections
from tiles_r20_common import *
sys.path.insert(0,str(ROOT))
def main():
    import torch
    from dataset.lm_dataset import VLMDataset
    from trainer.trainer_utils import vlm_collate_fn
    from torch.utils.data import DataLoader,Dataset
    torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42);random.seed(42)
    plan=read(PLAN_PATH);verify(plan);model,tok=init_model(DATASET/'tokenizer','cpu')
    resume_path=RUN/'latest_resume.pth';resume=None
    if resume_path.exists():
        resume=torch.load(resume_path,map_location='cpu',weights_only=False)
        assert resume['plan_sha256']==sha(PLAN_PATH);state=resume['model'];completed=resume['completed_updates']
    else:state=model.state_dict();completed=0
    loaded=model.load_state_dict(state,strict=False);assert not loaded.unexpected_keys and all(k.startswith('vision_encoder.') for k in loaded.missing_keys)
    del state
    if completed==0:
        digest=hashlib.sha256()
        for name,value in model.state_dict().items():
            if not name.startswith('vision_encoder.'):digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
        dump(RUN/'initialization.json',{'arm':ARM,'initial_nonvision_tensor_sha256':digest.hexdigest(),'base_checkpoint_sha256':sha(BASE)})
        if not START.exists():torch.save({k:v.half().cpu() for k,v in model.state_dict().items() if not k.startswith('vision_encoder.')},START)
    model.to('cuda');optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-6)
    def capture_rng():return {'python':random.getstate(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
    def restore_rng(rng):random.setstate(rng['python']);torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
    if resume:
        optimizer.load_state_dict(resume['optimizer']);restore_rng(resume['rng']);del resume
    from tiles_r20_data import TiledDataset
    ds=TiledDataset(str(DATASET/(ARM+'_train.parquet')),tok,model.processor,ARM)
    class StableChatDataset(Dataset):
        def __len__(self):return len(ds)
        def __getitem__(self,key):
            index,draw_position,epoch=key;saved=random.getstate()
            try:random.seed(42+epoch*100000+draw_position);return ds[index]
            finally:random.setstate(saved)
    dataset=StableChatDataset();history_path=RUN/'training_history.jsonl';started=time.time();torch.cuda.reset_peak_memory_stats()
    def status(stage,**extra):dump(RUN/'status.json',{'status':'running','stage':stage,'completed_updates':completed,'target_updates':TOTAL,'updated_at_utc':now(),**extra})
    def save_resume():
        rng=capture_rng();status('saving_resume')
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('vision_encoder.')}
        temporary=RUN/'latest_resume.tmp';torch.save({'model':state,'optimizer':optimizer.state_dict(),'rng':rng,'completed_updates':completed,'plan_sha256':sha(PLAN_PATH),'arm':ARM},temporary)
        temporary.replace(resume_path);del state
    def evaluate_here():
        rng=capture_rng();status('evaluating_checkpoint')
        path=START if completed==0 else weight_path(completed)
        if completed and not path.exists():
            temporary=path.with_suffix('.tmp');torch.save({k:v.detach().half().cpu() for k,v in model.state_dict().items() if not k.startswith('vision_encoder.')},temporary);temporary.replace(path)
        metrics=evaluate(path,completed);restore_rng(rng);verify(plan)
        print('[metrics]',completed,'val',metrics['val_summary']['detection'],'train',metrics['train_detection_summary'],flush=True)
    if completed in EVENTS:evaluate_here()
    while completed<TOTAL:
        if (RUN/'STOP').exists():save_resume();dump(RUN/'status.json',{'status':'stopped_with_resume','completed_updates':completed});return
        epoch=completed//291+1;batch_start=completed%291;order=sample_indices(42,epoch,'detection50')
        keys=[(i,p,epoch) for p,i in enumerate(order)][batch_start*4:]
        loader=DataLoader(dataset,batch_size=4,sampler=keys,num_workers=0,collate_fn=vlm_collate_fn,generator=torch.Generator().manual_seed(42+epoch))
        for ids,labels,pixels in loader:
            if completed>=TOTAL:break
            update=completed+1;lr=lr_for_update(update)
            for g in optimizer.param_groups:g['lr']=lr
            model.train();model.vision_encoder.eval();ids=ids.to('cuda');labels=labels.to('cuda');pixels={k:v.to('cuda') for k,v in pixels.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):res=model(ids,labels=labels,pixel_values=pixels);loss=res.loss+res.aux_loss
            assert torch.isfinite(loss),f'Nonfinite loss at {update}'
            loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();completed=update
            if completed%50==0:
                record={'step':completed,'epoch':epoch,'loss':loss.item(),'lr':lr,'preclip_grad_norm':gradient_norm.item(),'elapsed_seconds_this_process':time.time()-started}
                with history_path.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
                status('training',loss=loss.item(),lr=lr,epoch=epoch);print('[train]',json.dumps(record),flush=True)
            del ids,labels,pixels,res,loss
            if completed%500==0:save_resume()
            if completed in EVENTS:evaluate_here()
            if (RUN/'STOP').exists():save_resume();dump(RUN/'status.json',{'status':'stopped_with_resume','completed_updates':completed});return
    verify(plan);dump(RUN/'training_complete.json',{'status':'complete','completed_updates':completed,'plan_sha256':sha(PLAN_PATH),'arm':ARM,
        'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'elapsed_seconds_this_process':time.time()-started,'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})
    print('EXACTLY 10000 UPDATES COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:dump(RUN/'status.json',{'status':'failed','stage':'training_or_evaluation','error':str(e),'updated_at_utc':now()});raise
