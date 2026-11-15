import os,json,hashlib,datetime
from pathlib import Path
ROOT=Path(os.environ.get('MINIMIND_ROOT',Path(__file__).resolve().parents[2])).resolve()
DATA=Path(os.environ.get('R20_DETECTION_ROOT',ROOT/'dataset/course/revision2/detection'))
BASE=Path(os.environ.get('R20_BASE_WEIGHT',ROOT/'out/course_vqa_ctrl_r3_20261022_768.pth'))
BASE_RUN=ROOT/'dataset/course/results/course_vqa_ctrl_r3_20261022_all_val_rp100'
DATASET=ROOT/'dataset/course/tiles_r20_20261107'
SEEDS=[42]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def lines(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)

def jsonlines(path,rows):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');temporary.replace(path)

def verify(plan):
    for name,digest in plan['protected_sha256'].items():assert sha(ROOT/name)==digest,'Protected input changed: '+name
    for name,digest in plan['image_sha256'].items():assert sha(DATA/name)==digest,'Source image changed: '+name
