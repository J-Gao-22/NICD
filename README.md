# NICD

Recommendation pipelines and baselines for MovieLens-1M and Amazon Book.

## Code Structure

```text
.
|-- method/
|   |-- ml1m/
|   `-- book/
|       |-- src/lifecycle_rec/    # Implementation
|       |-- configs/              # Stage arguments
|       |-- scripts/              # Stage launchers
|       `-- requirements/         # Separate runtime dependencies
|-- baselines/
|   |-- ml1m/                     # MF, LightGCN, SASRec, gSASRec, TALLRec, CoLLM, ThinkRec
|   |-- book/
|   |-- common/
|   `-- LICENSES/upstream/
|-- data/manifests/               # Fixed teacher-sample indices only
`-- tests/
```

Runtime prompts and transition schemas are under `prompts/` and `schemas/`.

## Environment Setup

Use separate environments for recommendation and vLLM generation:

```bash
python3.9 -m venv /path/to/envs/recommendation
/path/to/envs/recommendation/bin/python -m pip install -r method/ml1m/requirements/recommendation.txt

python3 -m venv /path/to/envs/generation
/path/to/envs/generation/bin/python -m pip install -r method/ml1m/requirements/generation.txt

export RECOMMENDATION_PYTHON=/path/to/envs/recommendation/bin/python
export GENERATION_PYTHON=/path/to/envs/generation/bin/python
```

Use the corresponding Book requirement files for Book. Generation requires a
Python/CUDA/GPU combination supported by the pinned PyTorch and vLLM versions.
Traditional baselines use `baselines/requirements.txt`; LLM baselines provide
dataset/method-specific requirements. Training and inference use CUDA.

## Dataset and Model Preparation

Obtain the raw datasets from [MovieLens](https://grouplens.org/datasets/movielens/1m/)
and [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).
Baseline pickle frames must follow the preprocessing and ID conventions expected
by the [ThinkRec dataset code](https://github.com/Yu-Qi-hang/ThinkRec/tree/main/dataset).

Keep data, profiles, embeddings, weights, and outputs outside the repository.
Only fixed teacher-sample indices are included under `data/manifests/`.
Each mainline launcher uses three non-overlapping external directories:

```bash
export DATA_ROOT=/path/to/external/data/ml1m
export MODEL_ROOT=/path/to/external/models/ml1m
export OUTPUT_ROOT=/path/to/external/runs/ml1m
```

Switch all three roots for Book and use dataset-specific MF checkpoints.
Override a stage with `--config FILE`, individual arguments, or
`--python EXECUTABLE`. List stages with `python method/DATASET/run.py --help`.

### MovieLens-1M Inputs

```text
DATA_ROOT/
|-- raw/ratings.dat
|-- raw/movies.dat
|-- semantic_cache.jsonl
|-- item_text/
|   |-- item_text_semantic_compact.jsonl
|   `-- item_text_balanced_semantic.jsonl
|-- item_embeddings/
|   |-- item_ids.json
|   `-- item_embeddings.npy
`-- corpora/
    |-- states/segment_interest_node_sft_{train,valid,test}.jsonl
    `-- transitions/edge_transition_sft_{train,valid,test}.jsonl
```

`MODEL_ROOT` contains `Qwen2.5-7B-Instruct`, `native_mf.pt`, and
`collaborative_mf.pth`. The two MF formats are not interchangeable.
Profile/embedding stages are available through `method/ml1m/run.py`; pass their
`OUTPUT_ROOT` outputs to subsequent stages when rebuilding inputs.

### Book Inputs

```text
DATA_ROOT/
|-- all_interactions.jsonl
|-- item_profiles.jsonl
|-- item_embeddings/
|   |-- item_ids.json
|   `-- item_embeddings.npy
|-- frames/{train,valid,test}_ood2.pkl
`-- corpora/states/segment_interest_node_sft_{train,valid,test}.jsonl
```

Restored interactions must retain event IDs, row indices, timestamps, real
ratings, and aligned recent histories. Prepare profiles and restore inputs with:

```bash
"$GENERATION_PYTHON" method/book/run.py prepare-inputs profiles \
  --source-profile "$DATA_ROOT/source_profiles.jsonl" \
  --merged-profile "$DATA_ROOT/merged_profiles.jsonl" \
  --output-dir "$OUTPUT_ROOT/profiles"

"$GENERATION_PYTHON" method/book/run.py prepare-inputs restore \
  --data-dir "$DATA_ROOT/frames" \
  --canonical-profile "$OUTPUT_ROOT/profiles/item_profiles.canonical.jsonl" \
  --raw-reviews "$DATA_ROOT/raw/Books.csv" \
  --identity-map "$DATA_ROOT/identity_map.json" \
  --output-dir "$OUTPUT_ROOT/restored"
```

Pass the rebuilt outputs using `--all-interactions` and `--canonical-profile`.
The identity map must match the frame IDs. Prepared profiles, embeddings, and
frames are external prerequisites with no bundled download location; raw data
alone is insufficient. Build SFT corpora through the teacher stages below or
provide them externally.

Book uses a native MF container and an exported state dictionary containing the
same embeddings. Prepare both formats with:

```bash
python baselines/book/mf/train.py --data-dir "$DATA_ROOT/frames" --output-dir "$OUTPUT_ROOT/mf"
cp "$OUTPUT_ROOT/mf/best_mf.pt" "$MODEL_ROOT/native_mf.pt"
cp "$OUTPUT_ROOT/mf/best.pt" "$MODEL_ROOT/collaborative_mf.pth"
```

### Fixed Teacher Samples

Run with the corresponding dataset roots. These entries validate the included
manifests against training interactions and reject mismatches:

```bash
python method/ml1m/scripts/teacher_samples.py
python method/book/scripts/teacher_samples.py
```

MovieLens-1M `sample_teacher.py` generates a new selection, not a manifest replay.

## Mainline Training

Start from the prepared corpora above or build them using the teacher stages.

### MovieLens-1M

```bash
python method/ml1m/scripts/split.py
python method/ml1m/scripts/teacher_samples.py

python method/ml1m/scripts/train_state_generator.py --corpus-dir "$DATA_ROOT/corpora/states"
python method/ml1m/scripts/train_transition_generator.py --corpus-dir "$DATA_ROOT/corpora/transitions"
python method/ml1m/scripts/state_messages.py
python method/ml1m/scripts/generate_states.py
python method/ml1m/scripts/generate_transitions.py

python method/ml1m/scripts/plain_inputs.py
python method/ml1m/scripts/state_inputs.py
python method/ml1m/scripts/transition_inputs.py
python method/ml1m/scripts/recommendation_cache.py
python method/ml1m/scripts/train_recommender.py
python method/ml1m/scripts/train_collaborative.py
```

Recommendation training runs the initial LoRA phase and continuation, followed by
collaborative alignment. Only final weights are evaluated, not intermediate stages.

### Amazon Book

```bash
python method/book/scripts/teacher_samples.py
python method/book/scripts/train_state_generator.py --corpus-dir "$DATA_ROOT/corpora/states"
python method/book/scripts/generate_states.py --sft-reference "$DATA_ROOT/corpora/states/segment_interest_node_sft_train.jsonl"
python method/book/scripts/state_inputs.py
python method/book/scripts/recommendation_cache.py
python method/book/scripts/train_recommender.py
python method/book/scripts/train_collaborative.py
```

Alignment freezes the backbone, LoRA, and MF embeddings, training a shared
`256 -> 2560 -> hidden_size` projector for 2,000 updates (effective batch 4).
It uses 6,400 chain and 1,600 plain presentations, 200-update warmup
`1e-5 -> 1e-4`, then cosine decay to `1e-5`. Only the terminal projector is
evaluated; collaborative features enter the model, not a score-fusion step.
For stored State generations, pass all shard directories in their original
precedence order with `state_inputs.py --state-dirs DIR ...`.

### Teacher Corpus Construction

Teacher responses, judge decisions, and repairs construct training supervision,
not held-out recommendation scores. MovieLens-1M requires fixed external packs at
`DATA_ROOT/teacher/state_packs.jsonl`, with `user_id`, `item_id`, `segment_id`,
`candidate_id`, `pack_text`, and `pack_hash` (or `source_pack_hash`). Preparation
validates hashes, sample membership, and three candidates per segment.

```bash
for STAGE in teacher_samples teacher_state_packs generate_teacher_states \
  teacher_transitions transition_packs teacher_transitions_e1 \
  teacher_transitions_e2 teacher_transitions_e3 teacher_artifacts judge_prompts \
  judge repair_transitions repair_states regenerate_transitions judge_repairs corpora; do
  python "method/ml1m/scripts/$STAGE.py" || exit 1
done
```

State generation allows 768 output tokens and two attempts. Edge alternatives
run separately with their configured arguments. Raw-data-only regeneration of
the historical packs is not certified.

For Book, run the teacher stages in order with the Book dataset roots:

```bash
for STAGE in teacher_samples teacher_messages teacher_states teacher_transition_packs \
  teacher_transitions_e1 teacher_transitions_e2 teacher_transitions_e3 \
  judge_prompts judge repair_transitions repair_states regenerate_transitions \
  judge_repairs state_corpus transition_corpus; do
  python "method/book/scripts/$STAGE.py" || exit 1
done
```

Use OpenAI-compatible services for the `Qwen2.5-72B-Instruct-AWQ` teacher and
`Qwen2.5-14B-Instruct` judge. Configure endpoints, model names, and generation
arguments through stage configs or CLI arguments. Outputs remain external.
For rebuilt corpora, set `--corpus-dir "$OUTPUT_ROOT/corpora/states"` when training
the State generator. Book recommendation uses State inputs; its transition
corpus is exported separately.

## Final Evaluation

After training, run final VALID then TEST with the same adapter and projector.
Set `DATASET=ml1m` or `book`, with the corresponding external roots:

```bash
export DATASET=ml1m
for SPLIT in valid test; do
  python "method/$DATASET/scripts/evaluate.py" \
    --split "$SPLIT" --output-dir "$OUTPUT_ROOT/evaluation/$SPLIT"
done
```

Existing weights need no retraining: override `--cache`, `--adapter`, and
`--projector-dir`. The latter contains `projector.pt` and, for MovieLens-1M,
`selection.json`. Use matching cache rows, adapters, MF files, and projectors.
Scores and metrics are written externally, not included in this repository.

## Baselines

Run the following for each dataset with matching external inputs:

```bash
export DATASET=ml1m
export FRAMES=/path/to/external/data/ml1m/frames
export BASELINE_OUTPUT=/path/to/external/runs/ml1m/baselines
export BACKBONE=/path/to/external/models/Qwen2.5-7B-Instruct
export MF_STATE=/path/to/external/models/ml1m/mf.pth
```

`FRAMES` contains `{train,valid,test}_ood2.pkl`: integer `uid`/`iid`, binary
`label`, and item-ID sequence `his`. Keep ID mappings consistent across splits
and weights. LLM methods also need `id2title.json` mapping string item IDs to
objects with `title` and `keywords` strings. Book TALLRec additionally needs
`valid_small_ood2.pkl`; ThinkRec needs `reason_ood2.pkl` with a `reason` column.

### MF and Sequential Models

```bash
for METHOD in mf lightgcn sasrec gsasrec; do
  python "baselines/$DATASET/$METHOD/train.py" \
    --data-dir "$FRAMES" --output-dir "$BASELINE_OUTPUT/$METHOD"
  for SPLIT in valid test; do
    python "baselines/$DATASET/$METHOD/evaluate.py" \
      --data-dir "$FRAMES" --checkpoint "$BASELINE_OUTPUT/$METHOD/best.pt" \
      --split "$SPLIT"
  done
done
```

Book MF defaults: dimension 256, 20 epochs, batch 4,096, AdamW `lr=0.01`,
`weight_decay=0.0001`, seed `20260708`. It exports VALID-AUC-selected
`best_mf.pt` (native container) and `best.pt` (embedding state dictionary).
`last_mf.pt` is a separate terminal checkpoint.

### TALLRec and CoLLM

```bash
python "baselines/$DATASET/tallrec/train.py" \
  --data "$FRAMES" --backbone "$BACKBONE" --mf "$MF_STATE" \
  --output "$BASELINE_OUTPUT/tallrec"

export TEXT_ADAPTER="$BASELINE_OUTPUT/tallrec/best/adapter"
python "baselines/$DATASET/collm/train.py" \
  --data "$FRAMES" --backbone "$BACKBONE" --mf "$MF_STATE" \
  --adapter "$TEXT_ADAPTER" --output "$BASELINE_OUTPUT/collm"

for SPLIT in valid test; do
  python "baselines/$DATASET/tallrec/evaluate.py" \
    --data "$FRAMES" --backbone "$BACKBONE" --mf "$MF_STATE" \
    --adapter "$TEXT_ADAPTER" --split "$SPLIT" \
    --output "$BASELINE_OUTPUT/tallrec_reloaded/$SPLIT"
  python "baselines/$DATASET/collm/evaluate.py" \
    --data "$FRAMES" --backbone "$BACKBONE" --mf "$MF_STATE" \
    --adapter "$TEXT_ADAPTER" \
    --projector "$BASELINE_OUTPUT/collm/best/projector.pt" --split "$SPLIT" \
    --output "$BASELINE_OUTPUT/collm_reloaded/$SPLIT"
done
```

Use the MF state dictionary belonging to the CoLLM recipe, not an unrelated MF.
Output directories must be new or empty. Training already evaluates the selected
weights on final VALID/TEST; standalone evaluation reloads them without training.
These baselines retain their own validation/checkpoint selection settings.

### ThinkRec

ThinkRec runs its training phases before final evaluation:

```bash
python "baselines/$DATASET/thinkrec/train.py" \
  --data "$FRAMES" --backbone "$BACKBONE" --mf "$MF_STATE" \
  --output "$BASELINE_OUTPUT/thinkrec"

python "baselines/$DATASET/thinkrec/evaluate.py" \
  --data "$FRAMES" --backbone "$BACKBONE" --mf "$MF_STATE" \
  --general "$BASELINE_OUTPUT/thinkrec/stage1/adapter" \
  --projector "$BASELINE_OUTPUT/thinkrec/stage2/projector.pt" \
  --groups "$BASELINE_OUTPUT/thinkrec/stage3/group_0/adapter" \
           "$BASELINE_OUTPUT/thinkrec/stage3/group_1/adapter" \
  --routing "$BASELINE_OUTPUT/thinkrec/routing.csv" \
  --output "$BASELINE_OUTPUT/thinkrec_reloaded"
```

Group routing can be prepared with `prepare_groups.py`. Training exports terminal
stage weights without intermediate VALID/TEST. Final evaluation defaults to
VALID then TEST; preserve this order when replaying routed runs. `--split` selects
one final split. Existing results require their actual adapters and projector;
validation-selected and terminal weights are not interchangeable.

## Checks

```bash
python -B -m unittest discover -s tests -v
python method/ml1m/scripts/train_recommender.py --dry-run
python method/book/scripts/train_recommender.py --dry-run
```

Dry runs expand configs without loading models. Tests include tiny CPU updates,
save/reload, and stage-argument checks when runtime dependencies are installed;
otherwise runtime tests are skipped. These checks do not replace full GPU
reproduction. Changes in evaluation order or hardware/software may prevent
bitwise-identical retraining.

## Acknowledgments and Licenses

The code organization follows [ThinkRec](https://github.com/Yu-Qi-hang/ThinkRec).
See [Third-Party Notices](THIRD_PARTY_NOTICES.md) for upstream licenses;
dataset and pretrained model licenses apply separately.
