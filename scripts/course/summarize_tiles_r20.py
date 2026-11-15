import json
from tiles_r20_common import *
def main():
    verify(read(PLAN_PATH));results={};initial=[]
    for arm in ['global','tiled']:
        assert read(MASTER/arm/'training_complete.json')['completed_updates']==TOTAL
        initial.append(read(MASTER/arm/'initialization.json')['initial_nonvision_tensor_sha256'])
        entries=[]
        for step in EVENTS:
            folder=REVIEW/arm/f'step{step}';complete=read(folder/'completion.json')
            for name,h in complete['files_sha256'].items():assert sha(folder/name)==h
            metrics=read(folder/'metrics.json');assert sha(Path(metrics['checkpoint_path']))==metrics['checkpoint_sha256']
            entries.append(metrics)
        best=max(entries[1:],key=lambda m:(m['val_summary']['detection']['f1'],m['val_summary']['detection']['mean_iou'],-m['step']))
        results[arm]={'checkpoints':entries,'selected_step':best['step'],'selected_checkpoint':best['checkpoint_path']}
    assert initial[0]==initial[1]
    dump(REVIEW/'comparison.json',{'status':'complete_awaiting_user_approval','controls_verified':True,'same_initialization':True,'arms':results,'test_used':False})
    import html
    rows=[]
    for arm,group in results.items():
        for m in group['checkpoints']:
            v=m['val_summary'];d=v['detection'];t=m['train_detection_summary']
            rows.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [arm,m['step'],round(d['mean_iou'],4),f"{d['acc_at_0_5']:.2%}",round(d['precision'],4),round(d['f1'],4),f"{t['acc_at_0_5']:.2%}",f"{v['classification']['accuracy']:.2%}",f"{v['vqa']['keyword_hit_rate']:.2%}"])+'</tr>')
    (REVIEW/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>第二十轮实验结果</title><style>body{font:16px system-ui;margin:30px}td,th{padding:10px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>第二十轮：同一6000张数据：整图64token与整图加四局部320token，匹配10000步</h1><p>两组均使用检测50%采样、相同6000张图、初始化和10000步学习率曲线。global为整图64token，tiled为整图加原图四分块共320token；每块独立缩放到256，输出原图坐标。两组最大序列长度均1024。token数、视图位置说明及计算量不同；单种子42，不能单独归因于局部细节。固定10000步为主，最佳F1为辅助。训练诊断为两组共享的原始320图。测试集未使用。</p><table><tr>'+''.join('<th>'+s+'</th>' for s in ['数据组','步数','验证IoU','验证Acc@0.5','Precision','F1','训练Acc','分类','VQA'])+'</tr>'+''.join(rows)+'</table><p>选择规则：验证检测F1最大；并列时IoU更高，再并列选更早步。各组选择：'+html.escape(str({a:g['selected_step'] for a,g in results.items()}))+'</p>',encoding='utf-8')
    print(json.dumps({a:g['selected_step'] for a,g in results.items()}),flush=True)
if __name__=='__main__':main()
