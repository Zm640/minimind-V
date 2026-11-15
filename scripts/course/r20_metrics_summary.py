from collections import defaultdict
from course_metrics import *
def basic_accuracy(rows):
    return {"samples":len(rows),"correct":sum(r["correct"] for r in rows),
            "accuracy":sum(r["correct"] for r in rows)/len(rows) if rows else None}

def vqa_summary(rows):
    value=basic_accuracy(rows)
    value["em"]=value["accuracy"]
    value["keyword_hits"]=sum(bool(r["keyword_hit"]) for r in rows)
    value["keyword_hit_rate"]=value["keyword_hits"]/len(rows) if rows else None
    value["human_score_mean"]=None
    value["human_score_status"]="pending_independent_human_review"
    return value

def det_summary(rows):
    tp=sum(r["tp"] for r in rows); fp=sum(r["fp"] for r in rows); fn=sum(r["fn"] for r in rows)
    scores=[m["iou"] for r in rows for m in r["matches"]]
    gt=sum(r["gt_count"] for r in rows)
    return {"images":len(rows),"gt_boxes":gt,"tp":tp,"fp":fp,"fn":fn,
            "mean_iou":sum(r["iou_sum"] for r in rows)/gt if gt else None,
            "acc_at_0_5":tp/gt if gt else None,
            "image_exact_acc_at_0_5":sum(r["exact_image_at_0_5"] for r in rows)/len(rows) if rows else None,
            "image_all_gt_acc_at_0_5":sum(r["all_gt_localized_at_0_5"] for r in rows)/len(rows) if rows else None,
            "precision":tp/(tp+fp) if tp+fp else 0,
            "recall":tp/(tp+fn) if tp+fn else 0,
            "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
            "mean_matched_iou":sum(scores)/len(scores) if scores else None,
            "format_failure_rate":sum(r["format_failed"] for r in rows)/len(rows) if rows else None,
            "strict_format_failure_rate":sum(r["strict_format_failed"] for r in rows)/len(rows) if rows else None}

def make_summary(rows):
    result={}
    for task in ["classification","vqa","detection"]:
        subset=[r for r in rows if r["task"]==task]
        if not subset: continue
        if task=="classification":
            value=basic_accuracy(subset)
            value["format_failure_rate"]=sum(r["format_failed"] for r in subset)/len(subset)
            value["strict_format_failure_rate"]=sum(r["strict_format_failed"] for r in subset)/len(subset)
            value["strict_accuracy"]=sum(r["strict_correct"] for r in subset)/len(subset)
            value["per_class"]={label:basic_accuracy([r for r in subset if r["label"]==label])
                                for label in sorted({r["label"] for r in subset})}
            labels=["猫","狗","汽车","飞机","船","格式失败"]
            matrix={label:{p:0 for p in labels} for label in labels[:-1]}
            for r in subset: matrix[r["label"]][r["predicted_label"] or "格式失败"]+=1
            value["confusion_matrix"]=matrix
        elif task=="vqa":
            value=vqa_summary(subset); value["metric"]="keyword hit rate (primary), exact match (auxiliary)"
            value["empty_response_rate"]=sum(r["empty_response"] for r in subset)/len(subset)
            value["by_question_type"]={kind:vqa_summary([r for r in subset if r["question_type"]==kind])
                                       for kind in sorted({r["question_type"] for r in subset})}
            value["real_images"]=vqa_summary([r for r in subset if not r.get("synthetic",False)])
            value["synthetic_images"]=vqa_summary([r for r in subset if r.get("synthetic",False)])
        else:
            value=det_summary(subset)
            value["by_question_type"]={kind:det_summary([r for r in subset if r["question_type"]==kind])
                                       for kind in ["single_object","multi_object"]}
            single=[r for r in subset if r["question_type"]=="single_object"]
            value["single_target_acc_at_0_5"]=sum(r["single_target_iou"]+1e-12>=.5 for r in single)/len(single) if single else None
            value["single_target_mean_iou"]=sum(r["single_target_iou"] for r in single)/len(single) if single else None
        result[task]=value
    return result
