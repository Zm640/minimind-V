"""Fixed course evaluation rules. No model calls or dependencies."""
import json
import math
import re
import unicodedata

def clean_response(text):
    return re.sub(r"<think>.*?</think>", "", str(text), flags=re.S).strip()

def normalize(text):
    text=unicodedata.normalize("NFKC",clean_response(text)).lower()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))

CLS_ALIASES={
    "猫":"猫","小猫":"猫","猫咪":"猫","cat":"猫",
    "狗":"狗","小狗":"狗","狗狗":"狗","dog":"狗",
    "汽车":"汽车","小汽车":"汽车","轿车":"汽车","car":"汽车","automobile":"汽车",
    "飞机":"飞机","airplane":"飞机","aeroplane":"飞机","plane":"飞机",
    "船":"船","轮船":"船","boat":"船","ship":"船"}
DET_ALIASES={
    "aeroplane":"飞机","airplane":"飞机","plane":"飞机","bicycle":"自行车",
    "bird":"鸟","boat":"船","ship":"船","bottle":"瓶子","bus":"公交车",
    "car":"汽车","cat":"猫","chair":"椅子","cow":"牛","diningtable":"餐桌",
    "dog":"狗","horse":"马","motorbike":"摩托车","motorcycle":"摩托车",
    "person":"人","pottedplant":"盆栽","sheep":"羊","sofa":"沙发",
    "train":"火车","tvmonitor":"电视","电视机":"电视","公共汽车":"公交车",
    **CLS_ALIASES}
DET_ALIASES.update({normalize(v):v for v in DET_ALIASES.values()})

def score_classification(raw,truth):
    exact=CLS_ALIASES.get(normalize(raw))
    predicted=exact
    if predicted is None:
        # Accept explicit, unambiguous class assertions, not arbitrary mentions.
        text=clean_response(raw).lower()
        choices="|".join(re.escape(k) for k in sorted(CLS_ALIASES,key=len,reverse=True))
        pattern=(r"(?:主体属于|主体是|这是一(?:只|辆|架|艘)|正确答案是|正确的答案是)"
                 r"\s*[:：]?\s*(?:\([a-z]\))?\s*[\"'“”‘’]?\s*("+choices+
                 r")\s*[\"'“”‘’]?\s*(?=类别|类|[，。,.;；]|$)")
        positives={CLS_ALIASES[m.group(1)] for m in re.finditer(pattern,text)}
        negative_pattern=r"(?:不是|不属于|并非)\s*[\"'“”‘’]?\s*("+choices+r")"
        negatives={CLS_ALIASES[m.group(1)] for m in re.finditer(negative_pattern,text)}
        if len(positives)==1 and not positives.intersection(negatives):
            predicted=next(iter(positives))
    return {"predicted_label":predicted,"correct":predicted==truth,
            "format_failed":predicted is None,"strict_format_failed":exact is None,
            "strict_correct":exact==truth}

def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.0

def parse_detections(raw):
    """Recover JSON from optional fences/prose; strict formatting is measured separately."""
    text=str(raw).strip()
    try:
        direct=json.loads(text)
        strict=isinstance(direct,list)
    except (json.JSONDecodeError,ValueError):
        direct=None; strict=False
    cleaned=clean_response(text)
    array=direct if isinstance(direct,list) else None
    if array is None:
        decoder=json.JSONDecoder()
        for match in re.finditer(r"\[",cleaned):
            try: candidate,end=decoder.raw_decode(cleaned[match.start():])
            except (json.JSONDecodeError,ValueError): continue
            if isinstance(candidate,list) and all(isinstance(o,dict) for o in candidate):
                array=candidate; break
    if array is None:
        return {"objects":[],"invalid_count":1 if cleaned else 0,"format_failed":True,
                "strict_format_failed":True,"parse_error":"No JSON array of objects"}
    valid=[]; invalid=0
    for obj in array:
        try:
            if not isinstance(obj,dict) or not isinstance(obj.get("label"),str) or not obj["label"].strip():
                raise ValueError("Invalid label")
            box=obj.get("bbox")
            if not isinstance(box,list) or len(box)!=4:
                raise ValueError("bbox must contain four numbers")
            if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in box):
                raise ValueError("Invalid numeric coordinate")
            b=[float(v) for v in box]
            if not (0<=b[0]<b[2]<=1 and 0<=b[1]<b[3]<=1):
                raise ValueError("Invalid geometry/range")
            label=DET_ALIASES.get(normalize(obj["label"]),normalize(obj["label"]))
            valid.append({"label":label,"bbox":b})
        except (ValueError,TypeError):
            invalid+=1
    return {"objects":valid,"invalid_count":invalid,"format_failed":invalid>0,
            "strict_format_failed":not strict or invalid>0,"parse_error":None}

def maximum_matching(predicted,truth,threshold):
    """Maximum-cardinality, same-label matching; prediction order cannot lower TP."""
    edges={}
    for p_idx,p in enumerate(predicted):
        candidates=[(iou(p["bbox"],t["bbox_normalized"]),t_idx)
                    for t_idx,t in enumerate(truth) if p["label"]==t["label"]]
        edges[p_idx]=[t_idx for score,t_idx in sorted(candidates,reverse=True) if score+1e-12>=threshold]
    owners={}
    def augment(p_idx,visited):
        for t_idx in edges[p_idx]:
            if t_idx in visited: continue
            visited.add(t_idx)
            if t_idx not in owners or augment(owners[t_idx],visited):
                owners[t_idx]=p_idx; return True
        return False
    for p_idx in range(len(predicted)): augment(p_idx,set())
    return [{"prediction_index":p_idx,"truth_index":t_idx,
             "iou":iou(predicted[p_idx]["bbox"],truth[t_idx]["bbox_normalized"])}
            for t_idx,p_idx in sorted(owners.items())]


PROTOCOL_VERSION = "course-metrics-v2.0"

# Fixed lexical ontology; this is keyword matching, not a semantic judge.
UNKNOWN = ["无法判断","无法确定","不能判断","无法推断","看不出来","看不出","不确定","无法确认","unknown","cannot determine"]
LEXICONS = {
 "weather": {
   "unknown":UNKNOWN, "sunny":["晴天","晴朗","晴空","阳光明媚","阳光灿烂","sunny","clear sky"],
   "cloudy":["多云","cloudy"], "overcast":["阴天","阴沉","阴云密布","overcast"],
   "rain":["雨天","下雨","降雨","rainy","raining"], "snow":["雪天","下雪","降雪","snowing"],
   "fog":["雾天","有雾","大雾","foggy"]},
 "scene":{"unknown":UNKNOWN,"indoor":["室内","屋内","房间里","indoors","indoor"],
          "outdoor":["室外","户外","露天","outdoors","outdoor"]},
 "spatial_relation":{
   "upper_left":["左上方","左上角","左上","upper left","top left"],
   "upper_right":["右上方","右上角","右上","upper right","top right"],
   "lower_left":["左下方","左下角","左下","lower left","bottom left"],
   "lower_right":["右下方","右下角","右下","lower right","bottom right"],
   "left":["左侧","左边","左方","左面","left"],
   "right":["右侧","右边","右方","右面","right"],
   "above":["上方","上面","顶部","above","top"],
   "below":["下方","下面","底部","below","bottom"],
   "center":["中间","中央","中心","center"],"unknown":UNKNOWN},
 "attribute":{
   "black":["黑色","黑白","白黑","黑棕","棕黑","灰黑","黑灰","black"],
   "white":["白色","黑白","白黑","棕白","白棕","灰白","白灰","橙白","白橙","红白","白红","蓝白","白蓝","粉白","白粉","白绿","绿白","blackwhite","white"],
   "brown":["棕色","褐色","灰棕","灰褐","棕黑","黑棕","棕白","白棕","黑白棕","brown"],
   "gray":["灰色","灰白","白灰","灰黑","黑灰","灰棕","灰褐","棕灰","银色","银灰","grey","gray","silver"],
   "orange":["橙色","橘色","橘黄色","橙白","白橙","orange"],
   "yellow":["黄色","金色","yellow","golden"],
   "red":["红色","红白","白红","red"],
   "blue":["蓝色","蓝白","白蓝","blue"],
   "green":["绿色","军绿","橄榄绿","白绿","绿白","green"],
   "pink":["粉色","粉红","粉白","白粉","pink"],"unknown":UNKNOWN}
}
UNITS={"猫":"只","狗":"只","汽车":"辆","飞机":"架","船":"艘"}
NUMBER_CHARS="零〇一二两三四五六七八九十百"

def _negated(text,start):
    prefix=re.split(r"[，。！？；,.;!?\n]|而是|但是|不过|但|而",text[:start])[-1][-8:]
    return bool(re.search(r"不是|并非|不属于|不在|没有|不含|不带|未见|未出现|无法确认|不能确定|不能判断|无法判断|不确定|无法确定",prefix))

def _lexical_values(text,kind):
    text=unicodedata.normalize("NFKC",clean_response(text)).lower()
    if kind=="attribute":
        text=text.replace("粉红色","粉色").replace("橘黄色","橙色").replace("棕黄色","棕色")
    hits=[]
    for value,terms in LEXICONS[kind].items():
        for term in terms:
            pattern=re.escape(term)
            if re.search("[a-z]",term): pattern=r"(?<![a-z])"+pattern+r"(?![a-z])"
            for m in re.finditer(pattern,text):
                if not _negated(text,m.start()): hits.append((m.start(),m.end(),value))
    if kind=="spatial_relation":
        # Prefer diagonal phrases over their contained single-axis words.
        chosen=[]
        for hit in sorted(hits,key=lambda h:-(h[1]-h[0])):
            if not any(hit[0]<p[1] and p[0]<hit[1] for p in chosen): chosen.append(hit)
        hits=chosen
    return {h[2] for h in hits}

def _number(text):
    if text.isdecimal(): return int(text)
    digits={"零":0,"〇":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}
    if text in digits: return digits[text]
    if text=="十": return 10
    if "十" in text and len(text)<=3:
        a,b=text.split("十")
        return (digits.get(a,1)*10+digits.get(b,0)) if (not a or a in digits) and (not b or b in digits) else None
    return None

def _count_values(text,metadata):
    text=unicodedata.normalize("NFKC",clean_response(text)).lower()
    unit=UNITS.get((metadata or {}).get("queried_count_category"),"")
    allowed=unit+"个" if unit else "只辆架艘个"
    values=set()
    stripped=normalize(text)
    whole=_number(stripped)
    if whole is not None: values.add(whole)
    pattern=r"(["+NUMBER_CHARS+r"\d]+)\s*["+allowed+"]"
    for m in re.finditer(pattern,text):
        n=_number(m.group(1))
        if n is not None and not _negated(text,m.start()): values.add(n)
    for m in re.finditer(r"(?<!\d)\d+(?!\d)",text):
        if not _negated(text,m.start()): values.add(int(m.group()))
    # Ranges are ambiguous, even if only their last endpoint has a unit.
    for m in re.finditer(r"(["+NUMBER_CHARS+r"\d]+)\s*(?:到|至|~|-|或)\s*(["+NUMBER_CHARS+r"\d]+)",text):
        for token in m.groups():
            n=_number(token)
            if n is not None: values.add(n)
    category=(metadata or {}).get("queried_count_category")
    if category:
        absent=r"(?:没有|未见|不存在|未出现|图中无)(?:任何|一["+allowed+r"])?\s*"+re.escape(category)
        if re.search(absent,text): values.add(0)
    return values

def _boolean_values(text):
    text=unicodedata.normalize("NFKC",clean_response(text)).lower()
    values=set()
    for clause in re.split(r"[，。！？；,.;!?\n]",text):
        clause=clause.strip()
        if re.match(r"^(?:不是|否(?:的)?|不对|不正确|错误)(?:[，。,:：\s]|$|的|这样|在|位于)",clause): values.add("no")
        elif re.match(r"^(?:是的|是|对的|正确|没错)(?:[，。,:：\s]|$|这|在|位于)",clause): values.add("yes")
    if re.search(r"不在|并非位于|没有位于",text): values.add("no")
    return values

def keyword_spec(references,question_type,metadata=None):
    kind=question_type
    alternatives=[]
    for ref in references:
        if kind=="count": values=_count_values(ref,metadata)
        elif kind=="spatial": values=_boolean_values(ref)
        elif kind in LEXICONS: values=_lexical_values(ref,kind)
        else: values={normalize(ref)} if normalize(ref) else set()
        if values and sorted(values,key=str) not in alternatives: alternatives.append(sorted(values,key=str))
    return {"kind":kind,"alternatives":alternatives,
            "queried_count_category":(metadata or {}).get("queried_count_category")}

def score_keywords(raw,spec):
    kind=spec["kind"]
    if kind=="count": found=_count_values(raw,spec)
    elif kind=="spatial": found=_boolean_values(raw)
    elif kind in LEXICONS: found=_lexical_values(raw,kind)
    else:
        normalized=normalize(raw)
        found={normalized} if normalized else set()
    expected=[set(a) for a in spec["alternatives"]]
    hit=bool(found) and any(found==target for target in expected)
    # Contradictory/extra answer values are not rewarded as a keyword hit.
    return {"keyword_hit":hit,"keyword_found":sorted(found,key=str),
            "keyword_spec":spec,"keyword_needs_review":not hit,
            "keyword_status":"matched" if hit else ("no_keyword" if not found else "different_or_ambiguous")}

def score_vqa(raw,references,question_type=None,metadata=None):
    prediction=normalize(raw)
    result={"prediction_normalized":prediction,"correct":bool(prediction) and
            prediction in {normalize(r) for r in references},"empty_response":not bool(prediction)}
    result.update(score_keywords(raw,keyword_spec(references,question_type,metadata)))
    return result

def optimal_localization_matching(predicted,truth,threshold):
    """Exact one-to-one assignment: maximize TP first, then summed IoU.
    GT count is at most six in this course dataset; bitmask DP avoids scipy.
    """
    states={0:(0,0.0,[])}
    for p_idx,p in enumerate(predicted):
        updated=dict(states)
        for mask,(correct,total,pairs) in states.items():
            for t_idx,t in enumerate(truth):
                if mask & (1<<t_idx) or p["label"]!=t["label"]: continue
                overlap=iou(p["bbox"],t["bbox_normalized"])
                if overlap<=0: continue
                newmask=mask | (1<<t_idx)
                candidate=(correct+int(overlap+1e-12>=threshold),total+overlap,
                           pairs+[{"prediction_index":p_idx,"truth_index":t_idx,"iou":overlap}])
                if newmask not in updated or candidate[:2]>updated[newmask][:2]:
                    updated[newmask]=candidate
        states=updated
    return max(states.values(),key=lambda v:v[:2])[2]

def score_detection(raw,truth,threshold=0.5):
    parsed=parse_detections(raw)
    assignments=optimal_localization_matching(parsed["objects"],truth,threshold)
    matches=[p for p in assignments if p["iou"]+1e-12>=threshold]
    per_truth=[0.0]*len(truth)
    for pair in assignments: per_truth[pair["truth_index"]]=pair["iou"]
    tp=len(matches); fp=len(parsed["objects"])-tp+parsed["invalid_count"]; fn=len(truth)-tp
    return {**parsed,"localization_matches":assignments,"matches":matches,
            "per_truth_iou":per_truth,"gt_count":len(truth),"iou_sum":sum(per_truth),
            "mean_iou":sum(per_truth)/len(truth) if truth else None,
            "acc_at_0_5":tp/len(truth) if truth else None,
            "all_gt_localized_at_0_5":tp==len(truth) if truth else None,
            "exact_image_at_0_5":tp==len(truth) and fp==0 and not parsed["format_failed"],
            "tp":tp,"fp":fp,"fn":fn,
            "precision":tp/(tp+fp) if tp+fp else 0.0,
            "recall":tp/(tp+fn) if tp+fn else 0.0,
            "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            "single_target_iou":per_truth[0] if len(truth)==1 else None}
