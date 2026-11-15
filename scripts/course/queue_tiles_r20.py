import os,fcntl,subprocess,time,zipfile,traceback
from tiles_r20_common import *
from r20_gpu import read_gpus,eligible
PYTHON=__import__('sys').executable
def status(**s):dump(MASTER/'status.json',{'updated_at_utc':now(),**s})
def main():
    lock=(MASTER/'queue.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Controller already active');return
    if (MASTER/'status.json').exists() and read(MASTER/'status.json').get('status')=='complete':print('Already complete');return
    plan=read(PLAN_PATH);verify(plan);pending=[];completed=[];active={};errors=[];streak={};poll=0
    for arm in ('global','tiled'):
        if (MASTER/arm/'training_complete.json').exists():assert read(MASTER/arm/'training_complete.json')['completed_updates']==TOTAL;completed.append(arm)
        else:pending.append(arm)
    while pending or active:
        for arm,job in list(active.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();del active[arm]
            if code:errors.append({'arm':arm,'exit_code':code});pending=[]
            elif (MASTER/arm/'training_complete.json').exists():completed.append(arm)
            else:errors.append({'arm':arm,'error':'Stopped before10000steps'});pending=[]
        if (MASTER/'STOP').exists():
            for arm in active:(MASTER/arm/'STOP').touch()
            pending=[]
        try:
            rows=read_gpus();blocked={j['gpu'] for j in active.values()}
            for r in rows:streak[r['index']]=streak.get(r['index'],0)+1 if eligible(r) and r['index'] not in blocked else 0
            candidates=sorted([i for i,n in streak.items() if n>=4],reverse=True)
        except (subprocess.SubprocessError,ValueError,OSError):rows=[];candidates=[];streak={}
        for gpu in candidates:
            if not pending or errors or len(active)>=2:break
            verify(plan)
            if not any(r['index']==gpu and eligible(r) for r in read_gpus()):streak[gpu]=0;continue
            arm=pending.pop(0);env={**os.environ,'R20_ARM':arm,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
            log=(MASTER/arm/'experiment.log').open('a',encoding='utf-8')
            process=subprocess.Popen([PYTHON,'-u',str(ROOT/'scripts/course/train_tiles_r20.py')],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
            active[arm]={'process':process,'gpu':gpu,'log':log};streak[gpu]=0
            print('[start]',arm,gpu,process.pid,flush=True)
        states={a:read(MASTER/a/'status.json') if (MASTER/a/'status.json').exists() else {'status':'starting'} for a in active}
        status(status='running' if active else 'waiting_for_gpu',stage='training_arms' if active else 'waiting_for_idle_gpu',
            completed_arms=completed,pending_arms=pending,active_arms={a:{'gpu':j['gpu'],**states[a]} for a,j in active.items()},gpu_telemetry=rows,errors=errors)
        if not pending and not active:break
        if poll%3==0:print('[progress]',now(),completed,list(active),pending,flush=True)
        poll+=1;time.sleep(20)
    if errors:status(status='failed',stage='worker_error',errors=errors,completed_arms=completed);return
    if len(completed)!=2:status(status='stopped_with_resume',completed_arms=completed);return
    status(status='evaluation_complete',stage='building_report')
    with (MASTER/'report.log').open('w',encoding='utf-8') as log:
        subprocess.run([PYTHON,'-u',str(ROOT/'scripts/course/summarize_tiles_r20.py')],cwd=str(ROOT),env={**os.environ,'PYTHONUTF8':'1','OMP_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT,check=True)
    verify(plan);archive=MASTER/'review_bundle.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for f in REVIEW.rglob('*'):
            if f.is_file():z.write(f,f.relative_to(REVIEW.parent).as_posix())
    status(status='complete',stage='complete',completed_updates_per_arm=TOTAL,completed_arms=completed,review_directory=str(REVIEW),bundle_path=str(archive),bundle_sha256=sha(archive),next_round_started=False,awaiting_user_approval_for_next_round=True)
if __name__=='__main__':
    try:main()
    except Exception as e:status(status='failed',stage='queue_or_report_error',error=str(e));traceback.print_exc();raise
