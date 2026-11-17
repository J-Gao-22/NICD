import argparse
import random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from lifecycle_rec.recommendation.model import RecommendationModel
from lifecycle_rec.recommendation.training import train_stage


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--base-model',required=True)
    p.add_argument('--native-mf',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    random.seed(20260709)
    np.random.seed(20260709)
    torch.manual_seed(20260709)
    torch.cuda.manual_seed_all(20260709)
    torch.backends.cuda.enable_cudnn_sdp(False)
    data=torch.load(args.cache,map_location='cpu',weights_only=False)['datasets']
    assert len(data['train_chain'])==643308 and len(data['train_plain'])==160827
    tokenizer=AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    assert tokenizer.encode('Yes',add_special_tokens=False)==[9454]
    model=RecommendationModel(args.base_model,args.native_mf,8,16,.05,['q_proj','v_proj'],gradient_checkpointing=False,last_token_only=True).cuda()
    train_stage(1,model,data['train_plain'],data['train_chain'],tokenizer,9454,args.output_dir,8,20260709,1e-4,'chain80_plain20',False,max_train_steps=100517,pad_avoid_64=True)


if __name__=='__main__':main()
