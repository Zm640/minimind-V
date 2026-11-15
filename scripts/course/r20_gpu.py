import csv,subprocess
def parse_gpus(raw,compute_raw):
    busy={row[0].strip() for row in csv.reader(compute_raw.splitlines()) if len(row)>=2 and row[1].strip().isdigit()}
    rows=[]
    for row in csv.reader(raw.splitlines()):
        if len(row)!=4:raise ValueError('Unexpected GPU telemetry')
        i,uuid,memory,util=[x.strip() for x in row]
        try:memory=float(memory);util=float(util)
        except ValueError:memory=None;util=None
        rows.append({'index':int(i),'uuid':uuid,'memory_mib':memory,'utilization_percent':util,'compute_process_present':uuid in busy})
    if not rows:raise ValueError('No GPUs returned')
    return rows

def eligible(gpu):
    return gpu['memory_mib'] is not None and gpu['utilization_percent'] is not None and gpu['memory_mib']<=100 and gpu['utilization_percent']<=5 and not gpu['compute_process_present']

def read_gpus():
    a=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    b=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15)
    return parse_gpus(a.stdout,b.stdout)
