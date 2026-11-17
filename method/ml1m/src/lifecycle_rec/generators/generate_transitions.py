from __future__ import annotations
import argparse
import json
from pathlib import Path
from lifecycle_rec.generators import transition_batch as batch

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--generated-dir', type=Path, required=True)
    ap.add_argument('--edge-output-dir', type=Path, required=True)
    ap.add_argument('--base-model', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--lora', required=True)
    ap.add_argument('--splits', nargs='+', default=['test'])
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--max-num-seqs', type=int, default=128)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    ap.add_argument('--max-model-len', type=int, default=4096)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--max-attempts', type=int, default=2)
    ap.add_argument('--report-every', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--structured-mode', choices=['schema', 'json_object', 'none'], default='schema')
    args = ap.parse_args()
    args.edge_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = args.edge_output_dir
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams
    from lifecycle_rec.generators import transition_inputs as edge_pipe
    structured_outputs = None
    if args.structured_mode == 'schema':
        structured_outputs = StructuredOutputsParams(json=edge_pipe.EDGE_SCHEMA, disable_any_whitespace=True)
    elif args.structured_mode == 'json_object':
        structured_outputs = StructuredOutputsParams(json_object=True, disable_any_whitespace=True)
    llm = LLM(model=args.base_model, trust_remote_code=True, dtype='bfloat16', max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_memory_utilization, enable_lora=True, disable_log_stats=True)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, structured_outputs=structured_outputs)
    lora_request = LoRARequest('transitions', 1, str(args.lora))
    reports = []
    for split in args.splits:
        reports.append(batch.generate_split(args, llm, tokenizer, sampling, lora_request, split))
    report = {'splits': args.splits, 'edge_generation': reports}
    (args.edge_output_dir / 'edge_generation_report_testonly.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
