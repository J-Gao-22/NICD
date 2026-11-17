import importlib
import sys
from lifecycle_rec import compatibility

STAGES = {'prepare-inputs': 'prepare.inputs', 'state-messages': 'prepare.state_messages', 'generate-states': 'generators.generate_states', 'state-inputs': 'recommendation.state_inputs', 'train-state-generator': 'generators.train_states', 'recommendation-cache': 'recommendation.data', 'train-recommender': 'recommendation.train', 'train-collaborative': 'recommendation.collaborative', 'evaluate': 'recommendation.evaluate'}

STAGES['teacher-samples']='prepare.teacher_samples'
STAGES['state-corpus']='teacher.state_corpora'
STAGES['transition-corpus']='teacher.transition_corpora'

STAGES['teacher-messages']='teacher.messages'
STAGES['teacher-states']='teacher.states'
STAGES['teacher-transition-packs']='teacher.transition_packs'
STAGES['judge-prompts']='teacher.judge'
STAGES['judge']='teacher.judge'
STAGES['repair-transitions']='teacher.repair_transitions'
STAGES['repair-states']='teacher.repair_states'
STAGES['judge-repairs']='teacher.judge_repairs'
STAGES['state-corpus']='teacher.state_corpora'
STAGES['transition-corpus']='teacher.transition_corpora'
STAGES['teacher-transitions-e1']='teacher.generate_transitions'
STAGES['teacher-transitions-e2']='teacher.generate_transitions'
STAGES['teacher-transitions-e3']='teacher.generate_transitions'
STAGES['regenerate-transitions']='teacher.generate_transitions'

def main():
    if len(sys.argv)<2 or sys.argv[1] in ('-h','--help'):
        print('\n'.join(STAGES))
        return
    stage=sys.argv.pop(1)
    if stage not in STAGES:raise SystemExit(stage)
    importlib.import_module('lifecycle_rec.'+STAGES[stage]).main()

if __name__=='__main__':main()
