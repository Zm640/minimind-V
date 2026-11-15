"""R20: 64 global versus five64-token views; global output coordinates."""
import sys,hashlib,json,os
from pathlib import Path
from r20_runtime import *
from r20_model import init_model,location_question,restore_location
ARM=os.environ.get('R20_ARM','global')
assert ARM in ('global','tiled')
MASTER=ROOT/'logs/experiment_tiles_r20_20261107'
REVIEW=ROOT/'dataset/course/results/course_tiles_r20_20261107_review'
RUN=MASTER/ARM
OUT=REVIEW/ARM
DATASET=ROOT/'dataset/course/tiles_r20_20261107'/ARM
PLAN_PATH=MASTER/'plan.json'
START=ROOT/f'out/course_tiles_r20_{ARM}_s42_step0_20261107.pth'
EVENTS=[0,2000,5000,8000,10000]
TOTAL=10000

def sample_indices(seed,epoch,policy='detection50'):
    import torch
    assert policy=='detection50'
    indices=[]
    for offset,length,count in [(0,400,291),(400,444,291),(844,6000,582)]:
        order=torch.randperm(length,generator=torch.Generator().manual_seed(seed+offset+16000)).tolist()
        indices.extend(offset+order[((epoch-1)*count+j)%length] for j in range(count))
    shuffle=torch.randperm(1164,generator=torch.Generator().manual_seed(seed+epoch)).tolist()
    return [indices[i] for i in shuffle]

def lr_for_update(update):
    import math
    assert 1<=update<=TOTAL
    return 5e-7+(5e-6-5e-7)*.5*(1+math.cos(math.pi*(update-1)/(TOTAL-1)))

def weight_path(step):return ROOT/f'out/course_tiles_r20_{ARM}_s42_step{step}_20261107.pth'

def evaluate(weight,step):
    """Independent half-precision inference model: never converts the training model."""
    import torch,time,collections
    from PIL import Image
    from tiles_r20_data import question_for,encode_pixels,marker_runs
    from course_metrics import score_classification,score_vqa,score_detection
    from r20_metrics_summary import make_summary,det_summary
    folder=OUT/f'step{step}';folder.mkdir(exist_ok=True)
    checkpoint_hash=sha(weight)
    if (folder/'completion.json').exists():
        complete=read(folder/'completion.json');assert complete['checkpoint_sha256']==checkpoint_hash
        for name,h in complete['files_sha256'].items():assert sha(folder/name)==h
        return read(folder/'metrics.json')
    started=time.time();model,tok=init_model(DATASET/'tokenizer','cpu')
    state=torch.load(weight,map_location='cpu',weights_only=True);loaded=model.load_state_dict(state,strict=False);del state
    assert not loaded.unexpected_keys and all(k.startswith('vision_encoder.') for k in loaded.missing_keys)
    model.eval().requires_grad_(False).half().to('cuda');all_rows={}
    for split in ('val','train'):
        tasks={'classification':ROOT/'dataset/course/classification','vqa':ROOT/'dataset/course/revision2/vqa','detection':DATA} if split=='val' else {'detection':DATA}
        rows=[]
        for task,source_folder in tasks.items():
            for source in lines(source_folder/('train.jsonl' if split=='train' else 'val.jsonl')):
                with Image.open(source_folder/source['image_path']) as im:rgb=im.convert('RGB')
                question=question_for(source['conversations'][0]['content'],*rgb.size,ARM)
                if task=='detection' and False:question=location_question(question)
                prompt=tok.apply_chat_template([{'role':'user','content':question.replace('<image>',model.config.image_special_token*64)}],tokenize=False,add_generation_prompt=True,open_thinking=False)
                inputs=tok(prompt,return_tensors='pt').to('cuda');runs=marker_runs(inputs['input_ids'][0].tolist());assert len(runs)==(1 if ARM=='global' else 5) and all(b-a==64 for a,b in runs)
                with Image.open(source_folder/source['image_path']) as im:rgb=im.convert('RGB')
                pixels={k:v.unsqueeze(0).to('cuda',dtype=torch.float16) for k,v in encode_pixels(rgb,model.processor,ARM).items()}
                maximum=384 if task=='detection' else 96 if task=='vqa' else 64
                with torch.inference_mode():output=model.generate(inputs=inputs['input_ids'],attention_mask=inputs['attention_mask'],pixel_values=pixels,do_sample=False,
                    max_new_tokens=maximum,repetition_penalty=1.,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
                answer=output[0,inputs['input_ids'].shape[1]:];raw=tok.decode(answer,skip_special_tokens=True);prediction=raw;error=None
                if task=='detection' and False:
                    try:prediction=restore_location(raw)
                    except (ValueError,TypeError,KeyError):error='invalid_location_output';prediction='INVALID_LOCATION_OUTPUT'
                row={'task':task,'split':split,'sample_id':source.get('sample_id',source['image_id']),'image_id':source['image_id'],'label':source.get('label'),
                    'question_type':source.get('question_type','classification'),'synthetic':source.get('synthetic',False),'question':question,'prediction_raw':prediction,'model_output_raw':raw,
                    'representation_parse_error':error,'generated_tokens':len(answer),'hit_token_limit':len(answer)>=maximum,'image_absolute_path':str(source_folder/source['image_path'])}
                if task=='classification':row.update(score_classification(raw,source['label']))
                elif task=='vqa':row.update(score_vqa(raw,source['reference_answers'],source['question_type'],source))
                else:row['objects_truth']=source['objects'];row.update(score_detection(prediction,source['objects']))
                rows.append(row)
                if len(rows)%80==0:print('[eval]',step,split,len(rows),flush=True)
        all_rows[split]=rows;jsonlines(folder/(split+'_predictions.jsonl'),rows)
    def diagnostics(rows):
        valid=full=small=smalltp=off=offtp=0
        for r in rows:
            if r['task']!='detection':continue
            for o in r['objects']:
                valid+=1;b=o['bbox'];full+=b[0]<=.01 and b[1]<=.01 and b[2]>=.99 and b[3]>=.99
            for o,v in zip(r['objects_truth'],r['per_truth_iou']):
                x1,y1,x2,y2=o['bbox_normalized']
                if (x2-x1)*(y2-y1)<.1:small+=1;smalltp+=v>=.5
                if abs((x1+x2)/2-.5)>.2 or abs((y1+y2)/2-.5)>.2:off+=1;offtp+=v>=.5
        return {'valid_boxes':valid,'near_full_frame_boxes':full,'near_full_frame_fraction':full/valid if valid else None,'small_gt':small,'small_tp':smalltp,'off_center_gt':off,'off_center_tp':offtp}
    metrics={'step':step,'checkpoint_path':str(weight),'checkpoint_sha256':checkpoint_hash,'val_summary':make_summary(all_rows['val']),
        'train_detection_summary':det_summary(all_rows['train']),'val_diagnostics':diagnostics(all_rows['val']),'train_diagnostics':diagnostics(all_rows['train']),
        'elapsed_seconds':time.time()-started,'note':'same original320 training diagnostic for both arms; not all3000 or generalization performance; val original all3tasks; no test used'}
    dump(folder/'metrics.json',metrics)
    dump(folder/'completion.json',{'status':'complete','checkpoint_sha256':checkpoint_hash,'files_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='completion.json'}})
    del model,tok,inputs,pixels,output;torch.cuda.empty_cache();return metrics
