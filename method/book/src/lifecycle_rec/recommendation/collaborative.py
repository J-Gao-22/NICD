import argparse
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer
from lifecycle_rec.recommendation.model import RecommendationModel,write_json
from lifecycle_rec.recommendation.slots import CollaborativeSlotModel,SlotLayout,collate_records,inject_record
from lifecycle_rec.recommendation.metrics import reorganize_by_user,roc_auc_score,uAUC_me,ndcg_at_k,map_at_k


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base-model',required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--collaborative-mf',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    return p


def build_model(args):
    native=nn.Module()
    llm=AutoModelForCausalLM.from_pretrained(args.base_model,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa')
    llm.config.use_cache=False
    native.llm=get_peft_model(llm,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM'))
    saved=load_file(str(args.adapter/'adapter_model.safetensors'))
    set_peft_model_state_dict(native.llm,saved,adapter_name='default')
    current=get_peft_model_state_dict(native.llm)
    assert saved.keys()==current.keys() and all(torch.equal(v,current[k].detach().cpu()) for k,v in saved.items())
    from types import MethodType
    native.forward=MethodType(RecommendationModel.forward,native)
    native.last_token_only=True
    tokenizer=AutoTokenizer.from_pretrained(args.base_model)
    layout=SlotLayout.from_tokenizer(tokenizer)
    mf=torch.load(args.collaborative_mf,map_location='cpu',weights_only=True)
    assert mf['user_embedding.weight'].shape==(22967,256) and mf['item_embedding.weight'].shape==(34154,256)
    torch.manual_seed(20260709)
    model=CollaborativeSlotModel(native,mf,mid_times=10,placeholder_id=layout.placeholder_id).to(args.device)
    model.trainable_names()
    return model,layout


def batch(rows,layout,device):
    result=collate_records([inject_record(r,layout,2048) for r in rows],layout.placeholder_id,device)
    if result['input_ids'].shape[1]%64==0:
        result['input_ids']=torch.nn.functional.pad(result['input_ids'],(0,1),value=layout.placeholder_id)
        result['attention_mask']=torch.nn.functional.pad(result['attention_mask'],(0,1),value=0)
    return result


def learning_rate(step):
    if step<200:return 1e-5+9e-5*step/200
    return 1e-5+4.5e-5*(1+math.cos(math.pi*(step-200)/1799))


def metrics(rows,scores):
    users=np.asarray([r['uid'] for r in rows]);labels=np.asarray([r['label'] for r in rows]);scores=np.asarray(scores)
    groups=reorganize_by_user(users,scores,labels)
    return dict(auc=float(roc_auc_score(labels,scores)),uauc=float(uAUC_me(groups)[0]),ndcg5=float(ndcg_at_k(groups,5)),map5=float(map_at_k(groups,5)))


def main():
    args=arguments().parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=True
    random.seed(20260709);np.random.seed(20260709);torch.manual_seed(20260709);torch.cuda.manual_seed_all(20260709)
    data=torch.load(args.cache,map_location='cpu',weights_only=False)
    model,layout=build_model(args)
    parameters=list(model.cf_projector.parameters())
    optimizer=torch.optim.AdamW([dict(params=[p for p in parameters if p.ndim>=2],weight_decay=.001),dict(params=[p for p in parameters if p.ndim<2],weight_decay=0.)],lr=1e-4)
    plan=data['schedule'][:8000]
    assert len(plan)==8000 and sum(k=='chain' for k,i in plan)==6400
    for step in range(2000):
        model.train();optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=learning_rate(step)
        total=0.
        for kind,index in plan[step*4:step*4+4]:
            b=batch([data['datasets']['train_'+kind][index]],layout,args.device)
            with torch.amp.autocast(str(args.device).split(':')[0],dtype=torch.bfloat16):
                loss=torch.nn.functional.binary_cross_entropy_with_logits(model(b).float(),b['label'])
            if not torch.isfinite(loss):raise FloatingPointError(step)
            (loss/4).backward();total+=float(loss.detach())/4
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in parameters):raise FloatingPointError(step)
        if step==0 or (step+1)%25==0:print(dict(step=step+1,loss=total),flush=True)
    torch.save(model.cf_projector.state_dict(),args.output_dir/'projector.pt')


if __name__=='__main__':main()
