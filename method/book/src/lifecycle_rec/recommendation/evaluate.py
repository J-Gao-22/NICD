from pathlib import Path
import numpy as np
import torch
from lifecycle_rec.recommendation.collaborative import arguments,build_model,batch,metrics
from lifecycle_rec.recommendation.model import write_json


def main():
    p=arguments()
    p.add_argument('--projector-dir',type=Path,required=True)
    p.add_argument('--split',choices=('valid','test'),required=True)
    args=p.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=True
    model,layout=build_model(args)
    model.cf_projector.load_state_dict(torch.load(args.projector_dir/'projector.pt',map_location='cpu',weights_only=True))
    model.eval()
    rows=torch.load(args.cache,map_location='cpu',weights_only=False)['datasets'][args.split]
    assert len(rows)=={'valid':56402,'test':56403}[args.split]
    scores=[]
    with torch.no_grad():
        for start in range(0,len(rows),4):
            b=batch(rows[start:start+4],layout,args.device)
            with torch.amp.autocast(str(args.device).split(':')[0],dtype=torch.bfloat16):values=model(b)
            values=values.float().cpu().numpy()
            if not np.isfinite(values).all():raise FloatingPointError(start)
            scores.extend(values.tolist())
            if len(scores)%4096<len(values) or len(scores)==len(rows):
                print(dict(split=args.split,processed=len(scores),total=len(rows)),flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    np.save(args.output_dir/'scores.npy',np.asarray(scores,dtype=np.float64))
    write_json(args.output_dir/'metrics.json',metrics(rows,scores))


if __name__=='__main__':main()
