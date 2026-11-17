import json
import numpy as np
import torch
from lifecycle_rec.recommendation.collaborative import arguments, build_model, metrics, score
from lifecycle_rec.recommendation.model import write_json
from lifecycle_rec.recommendation.slots import inject_record

def main():
    p = arguments()
    p.add_argument('--projector-dir', type=__import__('pathlib').Path, required=True)
    p.add_argument('--split', choices=('valid', 'test'), required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    selection = json.loads((args.projector_dir / 'selection.json').read_text(encoding='utf-8'))
    model, tokenizer, layout = build_model(args)
    rows = torch.load(args.cache, map_location='cpu', weights_only=False)['datasets'][args.split]
    if selection['collaborative']:
        model.cf_projector.load_state_dict(torch.load(args.projector_dir / 'projector.pt', map_location='cpu', weights_only=True), strict=True)
        rows = [inject_record(r, layout) for r in rows]
    scores = score(model, rows, layout.placeholder_id, args.device, batch_size=4, collaborative=selection['collaborative'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'scores.npy', scores)
    write_json(args.output_dir / 'metrics.json', metrics(rows, scores))
if __name__ == '__main__':
    main()
