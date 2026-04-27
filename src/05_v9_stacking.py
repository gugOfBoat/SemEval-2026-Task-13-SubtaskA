#!/usr/bin/env python3
"""
SemEval 2026 Task 13A v9+: 4 surgical fixes over v9 (68%->70% target)
  Fix1: Test-first LLM perplexity (test coverage 0.36%->~30%+)
  Fix2: PPL tokens 128->64, batch 48->64, budget 5400->7200
  Fix3: +5 enriched style features
  Fix4: Ratio floor 0.05->0.10
"""
import argparse,gc,math,os,random,re,subprocess,sys,time,warnings,zlib,bz2
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

def _install_if_missing(*pkgs):
    for pkg in pkgs:
        try: __import__(pkg.split(">=")[0].split("[")[0])
        except ImportError:
            subprocess.run([sys.executable,"-m","pip","install","-q",pkg],check=False,capture_output=True)
_install_if_missing("bitsandbytes")

class CFG:
    DATA_DIR=None; SEED=42
    CHAR_MAX_FEATURES=80_000; CHAR_NGRAM_RANGE=(3,6)
    WORD_HASH_FEATURES=2**20; MAX_CHARS=4_500
    TEXT_ALPHA=2e-6; TEXT_MAX_ITER=20; STYLE_SUBSAMPLE=350_000
    PPL_MODEL_CANDIDATES=[
        "/kaggle/input/qwen2.5-coder/transformers/0.5b-instruct/1",
        "/kaggle/input/qwen2.5-coder/transformers/1.5b-instruct/1",
        "/kaggle/input/qwen2.5-coder/transformers/0.5b/1",
        "/kaggle/input/qwen2.5-coder/transformers/1.5b/1",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct","Qwen/Qwen2.5-Coder-1.5B-Instruct",
    ]
    PPL_MAX_TOKENS=64        # FIX2: was 128
    PPL_BATCH_SIZE=64        # FIX2: was 48
    PPL_TRAIN_SUBSAMPLE=50_000
    PPL_TIME_BUDGET=7200     # FIX2: was 5400
    PPL_FEATURE_NAMES=["nll_mean","nll_std","nll_max","nll_q25","nll_q75","nll_low_frac","nll_iqr","token_count"]
    N_FOLDS=5; META_MAX_ITER=300; META_MAX_LEAF_NODES=31; META_LR=0.05; META_L2=1.0
    GLOBAL_RATIO_GRID=np.arange(0.10,0.41,0.01)  # FIX4: was 0.05
    LANG_RATIO_GRID=np.arange(0.05,0.41,0.01)
    SHRINK_GRID=[0.0,0.25,0.5,0.75,1.0]; WEIGHT_RANDOM_SEARCH=256
    LANG_PRIORS={"C":0.20,"C#":0.23,"C++":0.17,"Go":0.27,"Java":0.16,"JavaScript":0.38,"PHP":0.06,"Python":0.26}
    FALLBACK_GLOBAL_RATIO=0.22

def seed_everything(s):
    random.seed(s); np.random.seed(s); os.environ["PYTHONHASHSEED"]=str(s)

def default_output_dir():
    return "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.dirname(os.path.abspath(__file__))

def find_data_dir():
    for p in ["/kaggle/input/semeval-2026-task13-subtask-a/Task_A",
              "/kaggle/input/SemEval-2026-Task13-Subtask-A/Task_A",
              "/kaggle/input/semeval-2026-task13-subtask-a/task_A",
              "/kaggle/input/competitions/sem-eval-2026-task-13-subtask-a/Task_A",
              "/kaggle/input/semeval-2026-task13/task_A",
              "/kaggle/input/semeval-2026-task13/Task_A"]:
        if os.path.exists(p): return p
    if os.path.exists("/kaggle/input"):
        for dp,_,fns in os.walk("/kaggle/input"):
            if "train.parquet" in fns: return dp
    local=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","data")
    return local if os.path.exists(os.path.join(local,"train.parquet")) else None

def load_data():
    d=find_data_dir()
    if not d: raise FileNotFoundError("Cannot find parquet files")
    CFG.DATA_DIR=d; print(f"Data: {d}")
    tr=pd.read_parquet(os.path.join(d,"train.parquet"))
    va=pd.read_parquet(os.path.join(d,"validation.parquet"))
    te=pd.read_parquet(os.path.join(d,"test.parquet"))
    sp=os.path.join(d,"test_sample.parquet")
    sa=pd.read_parquet(sp) if os.path.exists(sp) else None
    print(f"Tr={len(tr):,} Va={len(va):,} Te={len(te):,} Sa={len(sa) if sa is not None else 0}")
    return tr,va,te,sa

def reduce_for_smoke(df,rows,label_col=None):
    if df is None or rows is None or len(df)<=rows: return df
    if label_col and label_col in df.columns:
        parts=[]
        for _,frame in df.groupby(label_col):
            n=max(1,int(rows*len(frame)/len(df)))
            parts.append(frame.sample(min(n,len(frame)),random_state=CFG.SEED))
        return pd.concat(parts,ignore_index=True).sample(frac=1.0,random_state=CFG.SEED).reset_index(drop=True)
    return df.sample(rows,random_state=CFG.SEED).reset_index(drop=True)

def truncate_codes(codes):
    return [(c[:CFG.MAX_CHARS] if isinstance(c,str) else "") for c in codes]

_SPECIAL_TOKENS = [
    "\x3c|endoftext|\x3e","\x3c|im_end|\x3e","\x3c|assistant|\x3e","\x3c|user|\x3e",
    "\x3c|system|\x3e","\x3c|pad|\x3e","\x3c|begin|\x3e","\x3c|end|\x3e",
    "\x3c|im_start|\x3e","\x3c|eot_id|\x3e","\x3c|start_header_id|\x3e",
]

def detect_artifacts(codes):
    results=np.zeros(len(codes),dtype=bool)
    for i,code in enumerate(codes):
        if not isinstance(code,str): continue
        if any(tok in code for tok in _SPECIAL_TOKENS): results[i]=True; continue
        if "```" in code: results[i]=True; continue
        if re.match(r"^(Here is|Here's|Sure,|Certainly|Below is|The following|Let me |I'll )",code): results[i]=True
    print(f"Artifacts: {results.sum():,}/{len(codes):,} ({results.mean():.2%})")
    return results

def normalize_generator_family(name):
    if not isinstance(name,str): return "unknown"
    lo=name.lower()
    if lo=="human": return "human"
    for needle,fam in [("phi","phi"),("qwen","qwen"),("llama","llama"),("gemma","gemma"),
        ("gpt","gpt"),("deepseek","deepseek"),("yi-coder","yi"),("yi/","yi"),
        ("starcoder","starcoder"),("codegemma","gemma"),("codellama","llama"),
        ("mistral","mistral"),("claude","claude"),("command-r","command-r"),("mixtral","mistral")]:
        if needle in lo: return fam
    return lo.split("/",1)[0] if "/" in lo else lo.split("-",1)[0]

def build_family_weights(df):
    fam=df["generator"].map(normalize_generator_family)
    cnt=fam.value_counts().to_dict()
    w=fam.map(lambda x:1.0/math.sqrt(cnt[x])).astype(np.float32).to_numpy(copy=True)
    w*=len(w)/w.sum(); return w

def get_language_values(df):
    if df is None: return None
    if "language" not in df.columns: return np.full(len(df),"Unknown",dtype=object)
    return df["language"].fillna("Unknown").astype(str).values

def collect_language_classes(*dfs):
    c=set()
    for df in dfs:
        if df is not None and "language" in df.columns:
            c.update(df["language"].fillna("Unknown").astype(str).unique().tolist())
    return sorted(c) if c else ["Unknown"]

def rank_normalize(scores):
    order=np.argsort(scores,kind="mergesort")
    ranks=np.empty(len(scores),dtype=np.float32)
    if len(scores)==1: ranks[0]=0.5; return ranks
    ranks[order]=np.linspace(0.0,1.0,len(scores),endpoint=False,dtype=np.float32)
    return ranks

def rank_predict(scores,ratio):
    ratio=float(np.clip(ratio,0.0,1.0))
    preds=np.zeros(len(scores),dtype=np.int8)
    n_machine=int(round(len(scores)*ratio))
    if n_machine<=0: return preds
    order=np.argsort(scores)[::-1]; preds[order[:n_machine]]=1; return preds

def language_aware_predict(scores,languages,global_ratio,lang_ratio_map,shrink):
    preds=np.zeros(len(scores),dtype=np.int8)
    ul=pd.Series(languages).fillna("Unknown").astype(str)
    for lang in ul.unique():
        idx=np.where(ul.values==lang)[0]
        lr=lang_ratio_map.get(lang,global_ratio)
        ratio=float(np.clip((1.0-shrink)*global_ratio+shrink*lr,0.0,1.0))
        preds[idx]=rank_predict(scores[idx],ratio)
    return preds

def positive_class_scores(model,X):
    if hasattr(model,"decision_function"): return np.asarray(model.decision_function(X),dtype=np.float32)
    return np.asarray(model.predict_proba(X)[:,1],dtype=np.float32)

def try_load_llm():
    try:
        import torch
        if not torch.cuda.is_available(): print("[PPL] No CUDA"); return None,None
        from transformers import AutoModelForCausalLM,AutoTokenizer
    except ImportError: print("[PPL] No transformers"); return None,None
    for path in CFG.PPL_MODEL_CANDIDATES:
        if path.startswith("/") and not os.path.isdir(path): continue
        try:
            print(f"[PPL] Trying: {path}")
            tok=AutoTokenizer.from_pretrained(path,trust_remote_code=True,padding_side="right")
            if tok.pad_token is None: tok.pad_token=tok.eos_token
            mdl=None
            try:
                import torch as _t; from transformers import BitsAndBytesConfig
                mdl=AutoModelForCausalLM.from_pretrained(path,
                    quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=_t.float16,bnb_4bit_quant_type="nf4"),
                    device_map="auto",trust_remote_code=True)
                print(f"[PPL] Loaded {path} (BnB 4-bit)")
            except Exception as e: print(f"[PPL] BnB fail ({e})")
            if mdl is None:
                import torch as _t
                mdl=AutoModelForCausalLM.from_pretrained(path,dtype=_t.float16,device_map="auto",trust_remote_code=True)
            mdl.eval(); return mdl,tok
        except Exception as e: print(f"[PPL] Failed {path}: {e}")
    return None,None

def compute_llm_perplexity(codes,model,tokenizer,desc="",time_budget_remaining=None):
    import torch
    n=len(codes); features=np.zeros((n,len(CFG.PPL_FEATURE_NAMES)),dtype=np.float32)
    bs=CFG.PPL_BATCH_SIZE; t0=time.time()
    for start in range(0,n,bs):
        end=min(start+bs,n)
        batch=[c[:CFG.MAX_CHARS] if isinstance(c,str) else "" for c in codes[start:end]]
        enc=tokenizer(batch,return_tensors="pt",truncation=True,max_length=CFG.PPL_MAX_TOKENS,padding=True)
        ids=enc.input_ids.to(model.device); mask=enc.attention_mask.to(model.device)
        with torch.inference_mode(): logits=model(input_ids=ids,attention_mask=mask).logits
        sl=logits[:,:-1,:].contiguous(); st=ids[:,1:].contiguous(); sm=mask[:,1:].contiguous().float()
        nll=torch.nn.CrossEntropyLoss(reduction="none")(sl.view(-1,sl.size(-1)),st.view(-1)).view(st.size())*sm
        for j in range(end-start):
            m=sm[j].bool(); vals=nll[j][m].float().cpu().numpy()
            if len(vals)==0: continue
            q25,q75=np.percentile(vals,[25,75])
            features[start+j]=[np.mean(vals),np.std(vals),np.max(vals),q25,q75,np.mean(vals<1.0),q75-q25,float(len(vals))]
        del ids,mask,logits,sl,st,sm,nll; torch.cuda.empty_cache()
        elapsed=time.time()-t0
        if (start//bs)%100==0 and start>0:
            speed=end/elapsed; print(f"  {desc}: {end:,}/{n:,} ({speed:.0f} s/s, ETA {(n-end)/max(speed,1)/60:.1f}m)")
        if time_budget_remaining is not None and elapsed>60 and end<n:
            speed=end/elapsed
            if elapsed+(n-end)/max(speed,1)>time_budget_remaining:
                print(f"  {desc}: ABORT at {end:,}/{n:,}"); return features,end
    print(f"  {desc}: done {n:,} in {time.time()-t0:.1f}s"); return features,n

def compute_compression_features(codes,desc=""):
    n=len(codes); features=np.zeros((n,len(CFG.PPL_FEATURE_NAMES)),dtype=np.float32); t0=time.time()
    for i,code in enumerate(codes):
        if not isinstance(code,str) or len(code)==0: continue
        text=code[:CFG.MAX_CHARS].encode("utf-8",errors="replace")
        blen=max(len(text),1); ratio=len(zlib.compress(text,level=1))/blen
        mid=blen//2
        if mid>10:
            r1=len(zlib.compress(text[:mid],level=1))/mid; r2=len(zlib.compress(text[mid:],level=1))/max(blen-mid,1)
            q25,q75=min(r1,r2),max(r1,r2)
        else: q25=q75=ratio
        features[i]=[ratio,abs(q75-q25)/2,q75,q25,q75,ratio,q75-q25,float(blen)]
    print(f"  {desc}: compression {n:,} in {time.time()-t0:.1f}s"); return features

def compute_all_perplexity(train_df,test_df,sample_df,use_llm=True):
    """FIX1: Test-first ordering for maximum test coverage."""
    print("\n[Phase 1] Perplexity Features")
    ppl_train_c=compute_compression_features(train_df["code"].values,"Train")
    ppl_test_c=compute_compression_features(test_df["code"].values,"Test")
    ppl_sample_c=compute_compression_features(sample_df["code"].values,"Sample") if sample_df is not None else None
    model,tokenizer=(None,None)
    if use_llm: model,tokenizer=try_load_llm()
    if model is not None:
        import torch; phase_t0=time.time(); budget=CFG.PPL_TIME_BUDGET
        print(f"[PPL] Budget: {budget}s ({budget/60:.0f}m)")
        # FIX1: TEST FIRST — maximize test coverage
        ppl_test=ppl_test_c.copy()
        remaining=budget-(time.time()-phase_t0)
        if remaining>120:
            ppl_test_llm,n_test_done=compute_llm_perplexity(
                test_df["code"].values,model,tokenizer,"Test-LLM",
                time_budget_remaining=remaining*0.55)  # 55% budget for test
            ppl_test[:n_test_done]=ppl_test_llm[:n_test_done]
            print(f"[PPL] Test LLM coverage: {n_test_done:,}/{len(test_df):,} ({n_test_done/len(test_df):.1%})")
        # Sample (small, fast)
        ppl_sample=ppl_sample_c
        remaining=budget-(time.time()-phase_t0)
        if sample_df is not None and remaining>30:
            ppl_sample_llm,_=compute_llm_perplexity(
                sample_df["code"].values,model,tokenizer,"Sample-LLM",
                time_budget_remaining=min(remaining,120))
            ppl_sample=ppl_sample_c.copy(); ppl_sample[:len(ppl_sample_llm)]=ppl_sample_llm
        # Train subsample (use remaining budget)
        remaining=budget-(time.time()-phase_t0)
        n_train=len(train_df); ppl_train=ppl_train_c.copy()
        if remaining>120:
            sub_idx=np.sort(np.random.choice(n_train,min(CFG.PPL_TRAIN_SUBSAMPLE,n_train),replace=False))
            ppl_sub,n_done=compute_llm_perplexity(
                train_df["code"].values[sub_idx],model,tokenizer,"Train-LLM",
                time_budget_remaining=remaining)
            ppl_train[sub_idx[:n_done]]=ppl_sub[:n_done]
        del model,tokenizer; torch.cuda.empty_cache(); gc.collect()
        print(f"[PPL] Done in {(time.time()-phase_t0)/60:.1f}m")
        return ppl_train,ppl_test,ppl_sample
    else:
        print("[PPL] Compression only"); return ppl_train_c,ppl_test_c,ppl_sample_c

def extract_features_single(code):
    """FIX3: +5 enriched features (bz2, byte_entropy, line_cv, indent_delta_entropy, trigram_rep)."""
    if not isinstance(code,str) or len(code)==0: return {}
    lines=code.split("\n"); non_empty=[l for l in lines if l.strip()]
    words=re.findall(r"\b\w+\b",code); identifiers=re.findall(r"\b[a-zA-Z_]\w*\b",code)
    cc=max(len(code),1); lc=max(len(lines),1); wc=max(len(words),1)
    f={}
    ll=np.array([len(l) for l in lines],dtype=np.float32)
    nel=np.array([len(l) for l in non_empty],dtype=np.float32) if non_empty else np.array([0.0])
    ind=np.array([len(l)-len(l.lstrip()) for l in non_empty],dtype=np.float32) if non_empty else np.array([0.0])
    f["char_count"]=cc; f["line_count"]=lc
    f["empty_line_ratio"]=1.0-(len(non_empty)/lc)
    f["avg_line_length"]=float(ll.mean()); f["std_line_length"]=float(ll.std())
    f["max_line_length"]=float(ll.max()); f["median_ne_line_length"]=float(np.median(nel))
    f["indent_std"]=float(ind.std()); f["indent_unique"]=float(len(set(ind.tolist())))
    f["space_ratio"]=code.count(" ")/cc; f["tab_ratio"]=code.count("\t")/cc
    f["newline_ratio"]=code.count("\n")/cc; f["digit_ratio"]=sum(ch.isdigit() for ch in code)/cc
    f["uppercase_ratio"]=sum(ch.isupper() for ch in code)/max(sum(ch.isalpha() for ch in code),1)
    f["punct_ratio"]=sum(ch in "{}[]();,.:" for ch in code)/cc
    f["operator_ratio"]=len(re.findall(r"[+\-*/=<>!&|^~%]",code))/cc
    char_counter=Counter(code)
    cp=np.array(list(char_counter.values()),dtype=np.float64)/cc
    f["char_entropy"]=float(-np.sum(cp*np.log2(cp+1e-12)))
    f["unique_char_ratio"]=len(char_counter)/cc
    if words:
        wc2=Counter(words); wp=np.array(list(wc2.values()),dtype=np.float64)/len(words)
        f["token_entropy"]=float(-np.sum(wp*np.log2(wp+1e-12)))
        f["unique_word_ratio"]=len(wc2)/wc; f["avg_word_length"]=float(np.mean([len(w) for w in words]))
        f["hapax_ratio"]=sum(1 for c in wc2.values() if c==1)/len(wc2)
    else: f["token_entropy"]=f["unique_word_ratio"]=f["avg_word_length"]=f["hapax_ratio"]=0.0
    kw={"def","class","if","else","for","while","return","import","from","int","void",
        "public","private","static","new","try","catch","except","finally","with","as",
        "in","not","and","or","true","false","null","none","let","const","function",
        "func","self","this","switch","case","break","continue","package","namespace"}
    f["keyword_ratio"]=sum(1 for w in words if w.lower() in kw)/wc
    if identifiers:
        il=np.array([len(t) for t in identifiers],dtype=np.float32)
        f["avg_identifier_length"]=float(il.mean())
        f["single_char_id_ratio"]=sum(1 for t in identifiers if len(t)==1)/len(identifiers)
        f["long_id_ratio"]=sum(1 for t in identifiers if len(t)>10)/len(identifiers)
        f["snake_case_ratio"]=sum(1 for t in identifiers if "_" in t and t!="_")/len(identifiers)
        f["camel_case_ratio"]=sum(1 for t in identifiers if re.search(r"[a-z][A-Z]",t))/len(identifiers)
        f["identifier_diversity"]=len(set(identifiers))/len(identifiers)
    else:
        for k in ["avg_identifier_length","single_char_id_ratio","long_id_ratio","snake_case_ratio","camel_case_ratio","identifier_diversity"]: f[k]=0.0
    sl=[l.strip() for l in non_empty]
    f["line_dup_ratio"]=1.0-(len(set(sl))/max(len(sl),1))
    f["comment_ratio"]=sum(1 for l in non_empty if l.strip().startswith(("//","#","/*","*","--")))/max(len(non_empty),1)
    f["has_block_comment"]=int("/*" in code or "\'\'\'" in code or '\"\"\"' in code)
    f["brace_balance_abs"]=abs(code.count("{")-code.count("}"))
    f["paren_balance_abs"]=abs(code.count("(")-code.count(")"))
    f["has_markdown_fence"]=int("```" in code)
    f["has_special_token"]=int("\x3c|" in code)
    f["has_llm_preamble"]=int(bool(re.match(r"^(Here is|Here's|Sure,|Certainly|Below is|The following)",code)))
    tb=code[:CFG.MAX_CHARS].encode("utf-8",errors="replace")
    f["compression_ratio"]=len(zlib.compress(tb,level=1))/max(len(tb),1) if tb else 0.0
    # FIX3: +5 enriched features
    if len(tb)>0:
        f["bz2_ratio"]=len(bz2.compress(tb,compresslevel=9))/len(tb)
        byte_arr=np.frombuffer(tb,dtype=np.uint8)
        cnts=np.bincount(byte_arr,minlength=256)
        probs=cnts[cnts>0]/byte_arr.size
        f["byte_entropy"]=float(-(probs*np.log2(probs)).sum())
    else: f["bz2_ratio"]=0.0; f["byte_entropy"]=0.0
    f["line_len_cv"]=float(ll.std())/max(float(ll.mean()),1e-6)
    all_ind=[len(l)-len(l.lstrip()) for l in lines]
    deltas=[abs(all_ind[i+1]-all_ind[i]) for i in range(len(all_ind)-1)]
    if deltas:
        dc=Counter(deltas); dt=sum(dc.values())
        pd_=np.array(list(dc.values()),dtype=np.float64)/dt
        f["indent_delta_entropy"]=float(-(pd_*np.log2(pd_+1e-12)).sum())
    else: f["indent_delta_entropy"]=0.0
    if len(code)>=3:
        trigrams=[code[i:i+3] for i in range(len(code)-2)]
        tc=Counter(trigrams); f["trigram_rep_ratio"]=sum(1 for c in tc.values() if c>1)/max(len(tc),1)
    else: f["trigram_rep_ratio"]=0.0
    return f

def extract_features_batch(codes,desc):
    codes=list(codes); print(f"{desc}: extracting {len(codes):,} samples"); t0=time.time(); rows=[]
    for i,code in enumerate(codes,1):
        try: rows.append(extract_features_single(code))
        except: rows.append({})
        if i%100_000==0: print(f"  {i:,}/{len(codes):,} | {i/max(time.time()-t0,1e-9):.0f} s/s")
    df=pd.DataFrame(rows).fillna(0.0); df.replace([np.inf,-np.inf],0.0,inplace=True)
    print(f"{desc}: done in {time.time()-t0:.1f}s | shape={df.shape}"); return df

def add_language_features(style_df,languages,classes):
    if languages is None: return style_df
    ls=pd.Series(languages).fillna("Unknown").astype(str); lang_df=pd.DataFrame(index=style_df.index)
    for label in classes: lang_df[f"lang_is_{label}"]=(ls.values==label).astype(np.float32)
    return pd.concat([style_df.reset_index(drop=True),lang_df.reset_index(drop=True)],axis=1)

def best_ratio_for_subset(scores,labels,ratio_grid):
    best_r,best_s=CFG.FALLBACK_GLOBAL_RATIO,-1.0
    for r in ratio_grid:
        s=f1_score(labels,rank_predict(scores,r),average="macro")
        if s>best_s: best_s,best_r=s,float(r)
    return best_r,best_s

def tuned_stacking_config(sample_df,sample_artifacts,sample_languages,meta_sample_scores):
    print("\n[Stacking Tuning]")
    labels=sample_df["label"].astype(int).values; combined=rank_normalize(meta_sample_scores)
    lang_s=pd.Series(sample_languages).fillna("Unknown").astype(str)
    best={"score":-1.0,"global_ratio":CFG.FALLBACK_GLOBAL_RATIO,"lang_ratio_map":dict(CFG.LANG_PRIORS),"shrink":0.0}
    for gr in CFG.GLOBAL_RATIO_GRID:
        lrm={}
        for lang in lang_s.unique():
            idx=np.where(lang_s.values==lang)[0]
            lrm[lang]=best_ratio_for_subset(combined[idx],labels[idx],CFG.LANG_RATIO_GRID)[0] if len(idx)>=8 else float(gr)
        for shrink in CFG.SHRINK_GRID:
            preds=language_aware_predict(combined,lang_s.values,float(gr),lrm,shrink).copy()
            preds[sample_artifacts]=1
            score=f1_score(labels,preds,average="macro")
            if score>best["score"]:
                best={"score":float(score),"global_ratio":float(gr),
                      "lang_ratio_map":{k:float(v) for k,v in lrm.items()},"shrink":float(shrink)}
    print(f"Best F1: {best['score']:.4f} | ratio={best['global_ratio']:.2f} | shrink={best['shrink']:.2f}")
    print(f"Lang ratios: {best['lang_ratio_map']}"); return best

def fallback_config():
    return {"score":None,"weights":{"char_full":0.45,"char_family":0.25,"word_hash":0.15,"style_hgb":0.15},
            "global_ratio":CFG.FALLBACK_GLOBAL_RATIO,"lang_ratio_map":dict(CFG.LANG_PRIORS),"shrink":0.50}

def save_submission(preds,test_df,out_path):
    id_col="ID" if "ID" in test_df.columns else "id"
    sub=pd.DataFrame({"ID":test_df[id_col].values,"label":preds.astype(int)})
    sub.to_csv(out_path,index=False); return sub

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",default=default_output_dir())
    parser.add_argument("--smoke-rows",type=int,default=None)
    parser.add_argument("--no-llm",action="store_true")
    args,_=parser.parse_known_args()
    seed_everything(CFG.SEED); os.makedirs(args.output_dir,exist_ok=True); t0=time.time()

    train_df,val_df,test_df,sample_df=load_data()
    if args.smoke_rows:
        train_df=reduce_for_smoke(train_df,args.smoke_rows,"label")
        val_df=reduce_for_smoke(val_df,max(200,args.smoke_rows//4),"label")
        test_df=reduce_for_smoke(test_df,max(200,args.smoke_rows//2))
        sample_df=reduce_for_smoke(sample_df,max(200,args.smoke_rows//2),"label") if sample_df is not None else None
    train_full=pd.concat([train_df,val_df],ignore_index=True); del train_df,val_df; gc.collect()
    print(f"Train+Val: {len(train_full):,}")
    test_artifacts=detect_artifacts(test_df["code"].values)
    sample_artifacts=detect_artifacts(sample_df["code"].values) if sample_df is not None else None
    lang_classes=collect_language_classes(train_full,sample_df,test_df)
    train_langs=get_language_values(train_full); test_langs=get_language_values(test_df); sample_langs=get_language_values(sample_df)
    y_train=train_full["label"].astype(int).values; fw_full=build_family_weights(train_full)

    ppl_train,ppl_test,ppl_sample=compute_all_perplexity(train_full,test_df,sample_df,use_llm=not args.no_llm)

    print(f"\n[Mode] {CFG.N_FOLDS}-fold stacking")
    n_train=len(train_full); n_test=len(test_df); n_sample=len(sample_df) if sample_df is not None else 0
    oof=np.zeros((n_train,4),dtype=np.float32)
    test_sum=np.zeros((n_test,4),dtype=np.float32)
    sample_sum=np.zeros((n_sample,4),dtype=np.float32) if n_sample>0 else None

    print("[Stacking] Pre-computing char vocab...")
    cv_full=TfidfVectorizer(analyzer="char",ngram_range=CFG.CHAR_NGRAM_RANGE,max_features=CFG.CHAR_MAX_FEATURES,
                            min_df=3,sublinear_tf=True,lowercase=False,dtype=np.float32)
    cv_full.fit(truncate_codes(train_full["code"].values)); char_vocab=cv_full.vocabulary_; del cv_full; gc.collect()

    print("[Stacking] Pre-computing style features...")
    sty_tr=extract_features_batch(train_full["code"].values,"Train")
    sty_te=extract_features_batch(test_df["code"].values,"Test")
    sty_sa=extract_features_batch(sample_df["code"].values,"Sample") if sample_df is not None else None
    for k,col in enumerate(CFG.PPL_FEATURE_NAMES):
        sty_tr[f"ppl_{col}"]=ppl_train[:,k]; sty_te[f"ppl_{col}"]=ppl_test[:,k]
        if sty_sa is not None and ppl_sample is not None: sty_sa[f"ppl_{col}"]=ppl_sample[:,k]
    sty_tr=add_language_features(sty_tr,train_langs,lang_classes)
    sty_te=add_language_features(sty_te,test_langs,lang_classes)
    if sty_sa is not None: sty_sa=add_language_features(sty_sa,sample_langs,lang_classes)
    X_sty_all=sty_tr.astype(np.float32).values; X_sty_test=sty_te.astype(np.float32).values
    X_sty_sample=sty_sa.astype(np.float32).values if sty_sa is not None else None
    del sty_tr,sty_te,sty_sa; gc.collect()

    wv=HashingVectorizer(analyzer="word",token_pattern=r"\b\w+\b",ngram_range=(1,3),
        n_features=CFG.WORD_HASH_FEATURES,alternate_sign=False,lowercase=False,norm="l2",dtype=np.float32)

    skf=StratifiedKFold(n_splits=CFG.N_FOLDS,shuffle=True,random_state=CFG.SEED)
    for fi,(tr_idx,va_idx) in enumerate(skf.split(np.zeros(n_train),y_train)):
        ft0=time.time()
        print(f"\n--- Fold {fi+1}/{CFG.N_FOLDS} (train={len(tr_idx):,}, val={len(va_idx):,}) ---")
        y_tr=y_train[tr_idx]; fw_tr=fw_full[tr_idx]; fold_tr_df=train_full.iloc[tr_idx]

        cv=TfidfVectorizer(analyzer="char",ngram_range=CFG.CHAR_NGRAM_RANGE,vocabulary=char_vocab,
                           sublinear_tf=True,lowercase=False,dtype=np.float32)
        Xct=cv.fit_transform(truncate_codes(fold_tr_df["code"].values))
        Xcv=cv.transform(truncate_codes(train_full.iloc[va_idx]["code"].values))
        Xce=cv.transform(truncate_codes(test_df["code"].values))
        Xcs=cv.transform(truncate_codes(sample_df["code"].values)) if sample_df is not None else None
        c1=SGDClassifier(loss="log_loss",alpha=CFG.TEXT_ALPHA,max_iter=CFG.TEXT_MAX_ITER,tol=1e-3,random_state=CFG.SEED)
        c1.fit(Xct,y_tr)
        oof[va_idx,0]=positive_class_scores(c1,Xcv); test_sum[:,0]+=positive_class_scores(c1,Xce)
        if Xcs is not None: sample_sum[:,0]+=positive_class_scores(c1,Xcs)
        c2=SGDClassifier(loss="log_loss",alpha=CFG.TEXT_ALPHA*1.5,max_iter=CFG.TEXT_MAX_ITER,tol=1e-3,random_state=CFG.SEED)
        c2.fit(Xct,y_tr,sample_weight=fw_tr)
        oof[va_idx,1]=positive_class_scores(c2,Xcv); test_sum[:,1]+=positive_class_scores(c2,Xce)
        if Xcs is not None: sample_sum[:,1]+=positive_class_scores(c2,Xcs)
        del Xct,Xcv,Xce,Xcs,c1,c2,cv; gc.collect()

        Xwt=wv.transform(truncate_codes(fold_tr_df["code"].values))
        Xwv=wv.transform(truncate_codes(train_full.iloc[va_idx]["code"].values))
        Xwe=wv.transform(truncate_codes(test_df["code"].values))
        Xws=wv.transform(truncate_codes(sample_df["code"].values)) if sample_df is not None else None
        c3=SGDClassifier(loss="log_loss",alpha=CFG.TEXT_ALPHA,max_iter=CFG.TEXT_MAX_ITER,tol=1e-3,random_state=CFG.SEED)
        c3.fit(Xwt,y_tr)
        oof[va_idx,2]=positive_class_scores(c3,Xwv); test_sum[:,2]+=positive_class_scores(c3,Xwe)
        if Xws is not None: sample_sum[:,2]+=positive_class_scores(c3,Xws)
        del Xwt,Xwv,Xwe,Xws,c3; gc.collect()

        Xs_tr=X_sty_all[tr_idx]; Xs_va=X_sty_all[va_idx]; ys_tr=y_tr
        if len(Xs_tr)>CFG.STYLE_SUBSAMPLE:
            si=np.random.choice(len(Xs_tr),CFG.STYLE_SUBSAMPLE,replace=False); Xs_tr,ys_tr=Xs_tr[si],y_tr[si]
        c4=HistGradientBoostingClassifier(learning_rate=0.05,max_iter=250,max_leaf_nodes=63,
                                          min_samples_leaf=40,l2_regularization=0.1,random_state=CFG.SEED)
        c4.fit(Xs_tr,ys_tr)
        oof[va_idx,3]=c4.predict_proba(Xs_va)[:,1].astype(np.float32)
        test_sum[:,3]+=c4.predict_proba(X_sty_test)[:,1].astype(np.float32)
        if X_sty_sample is not None: sample_sum[:,3]+=c4.predict_proba(X_sty_sample)[:,1].astype(np.float32)
        del c4; gc.collect()
        print(f"  Fold {fi+1} done in {time.time()-ft0:.1f}s")

    test_avg=test_sum/CFG.N_FOLDS; sample_avg=sample_sum/CFG.N_FOLDS if sample_sum is not None else None
    del X_sty_all,X_sty_test,X_sty_sample; gc.collect()

    print("\n[Meta-Learner]")
    def lang_onehot(langs,n):
        oh=np.zeros((n,len(lang_classes)),dtype=np.float32)
        if langs is not None:
            ls=pd.Series(langs).fillna("Unknown").astype(str)
            for j,lbl in enumerate(lang_classes): oh[:,j]=(ls.values==lbl).astype(np.float32)
        return oh
    Xm_tr=np.column_stack([oof,ppl_train,lang_onehot(train_langs,n_train)])
    Xm_te=np.column_stack([test_avg,ppl_test,lang_onehot(test_langs,n_test)])
    Xm_sa=np.column_stack([sample_avg,ppl_sample,lang_onehot(sample_langs,n_sample)]) if n_sample>0 and ppl_sample is not None else None
    print(f"Meta features: {Xm_tr.shape[1]}")
    meta=HistGradientBoostingClassifier(learning_rate=CFG.META_LR,max_iter=CFG.META_MAX_ITER,
        max_leaf_nodes=CFG.META_MAX_LEAF_NODES,min_samples_leaf=50,l2_regularization=CFG.META_L2,random_state=CFG.SEED)
    meta.fit(Xm_tr,y_train)
    meta_test=meta.predict_proba(Xm_te)[:,1].astype(np.float32)
    meta_sample=meta.predict_proba(Xm_sa)[:,1].astype(np.float32) if Xm_sa is not None else None
    del meta,Xm_tr,Xm_te,Xm_sa,oof; gc.collect()

    if sample_df is not None and meta_sample is not None:
        tune_cfg=tuned_stacking_config(sample_df,sample_artifacts,sample_langs,meta_sample)
    else: tune_cfg={"global_ratio":CFG.FALLBACK_GLOBAL_RATIO,"lang_ratio_map":dict(CFG.LANG_PRIORS),"shrink":0.50}
    test_scores=rank_normalize(meta_test)
    preds=language_aware_predict(test_scores,test_langs,tune_cfg["global_ratio"],tune_cfg["lang_ratio_map"],tune_cfg["shrink"])
    preds[test_artifacts]=1

    path=os.path.join(args.output_dir,"submission.csv")
    sub=save_submission(preds,test_df,path)
    print(f"\nSaved: {path} | Machine: {sub['label'].mean():.2%} ({sub['label'].sum():,}/{len(sub):,})")
    if sample_df is not None and meta_sample is not None:
        sa_scores=rank_normalize(meta_sample)
        sa_preds=language_aware_predict(sa_scores,sample_langs,tune_cfg["global_ratio"],tune_cfg["lang_ratio_map"],tune_cfg["shrink"])
        sa_preds[sample_artifacts]=1
        print(f"Sample F1: {f1_score(sample_df['label'].values,sa_preds,average='macro'):.4f}")
    print(f"ratio={tune_cfg['global_ratio']:.2f} | shrink={tune_cfg['shrink']:.2f}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f}m")

if __name__=="__main__":
    main()
