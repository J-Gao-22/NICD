import importlib
import os
import subprocess
import sys

STAGES = {
    'split': 'prepare.split',
    'profile-prompts': 'prepare.profile_prompts',
    'generate-profiles': 'prepare.profile_generation',
    'profile-cache': 'prepare.profile_cache',
    'keywords': 'prepare.keywords',
    'combine-profiles': 'prepare.combine_profiles',
    'semantic-cache': 'prepare.semantic_cache',
    'clean-profiles': 'prepare.clean_profiles',
    'refine-profiles': 'prepare.refine_profiles',
    'semantic-terms': 'prepare.semantic_terms',
    'deduplicate-profiles': 'prepare.deduplicate_profiles',
    'item-text': 'prepare.text',
    'embeddings': 'prepare.embeddings',
    'state-messages': 'prepare.state_messages',
    'teacher-samples': 'prepare.teacher_samples',
    'sample-teacher': 'prepare.sample_teacher',
    'teacher-state-packs': 'teacher.state_tasks',
    'generate-teacher-states': 'teacher.state_generation',
    'teacher-transitions': 'teacher.transitions',
    'transition-packs': 'teacher.transition_packs',
    'generate-teacher-transitions': 'teacher.generate_transitions',
    'teacher-transitions-e1': 'teacher.generate_transitions',
    'teacher-transitions-e2': 'teacher.generate_transitions',
    'teacher-transitions-e3': 'teacher.generate_transitions',
    'teacher-artifacts': 'teacher.artifacts',
    'judge-prompts': 'teacher.judge_prompts',
    'judge': 'teacher.judge',
    'repair-transitions': 'teacher.repair_transitions',
    'repair-states': 'teacher.repair_states',
    'regenerate-transitions': 'teacher.generate_transitions',
    'judge-repairs': 'teacher.judge_repairs',
    'corpora': 'teacher.corpora',
    'train-state-generator': 'generators.train_states',
    'train-transition-generator': 'generators.train_transitions',
    'generate-states': 'generators.generate_states',
    'generate-transitions': 'generators.generate_transitions',
    'plain-inputs': 'recommendation.prepare_plain',
    'state-inputs': 'recommendation.state_inputs',
    'transition-inputs': 'recommendation.prepare_transitions',
    'recommendation-cache': 'recommendation.data',
    'train-recommender': 'recommendation.train',
    'train-collaborative': 'recommendation.collaborative',
    'evaluate': 'recommendation.evaluate',
}


def main():
    if 'PYTHONHASHSEED' not in os.environ:
        raise SystemExit(subprocess.call([sys.executable, '-B', *sys.argv], env=dict(os.environ, PYTHONHASHSEED='0')))
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('\n'.join(STAGES))
        return
    stage = sys.argv.pop(1)
    if stage not in STAGES:
        raise SystemExit(stage)
    from lifecycle_rec import compatibility
    importlib.import_module('lifecycle_rec.'+STAGES[stage]).main()


if __name__ == '__main__':
    main()
