import os
import json
import logging
import random
import re
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, default_collate

class RecBaseDataset(Dataset):

    def __init__(self, text_processor=None, ann_paths=None):
        if ann_paths is not None:
            self.annotation = pd.read_pickle(ann_paths[0] + '.pkl').values
        self.text_processor = text_processor

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, text_processor):
        self.text_processor = text_processor

    def _add_instance_ids(self, key='instance_id'):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)

def turn2list(sample):
    if isinstance(sample, list):
        sample = sample
    elif isinstance(sample, str):
        sample = [sample]
    else:
        try:
            sample = list(sample)
        except:
            raise ValueError('sample must be a list or a str')
    return sample

def convert_title_list_v4(samples=None, labels=None, withid=False):
    if len(samples) == 2:
        titles, descs = samples
    elif len(samples) == 1:
        titles = samples[0]
        descs = None
    else:
        raise ValueError('size of samples must be 1 or 2')
    titles = turn2list(titles)
    if labels is not None:
        label2eval = {0: 'No', 1: 'Yes'}
        labels = [label2eval[label] for label in turn2list(labels)]
    if descs is not None:
        descs = turn2list(descs)
    itemsymbol = '<ItemID>'
    items = []
    for idx, title in enumerate(titles):
        if title.strip(' ') == '':
            item = f'"Unknow"'
        else:
            item = f'''"{title.strip(' ')}"'''
        if withid:
            item = f'{item} with feature {itemsymbol}'
        if labels is not None:
            item = f'{item} (evaluation: {labels[idx]})'
        if descs is not None:
            desc = descs[idx].strip(' ')
            desc = re.sub('<[^>]+>', '', desc)
            desc = re.sub('\\s+', ' ', desc)
            item = f'{item} with description: {desc}'
        items.append(item)
    if len(items) > 0:
        return '; '.join(items)
    else:
        return 'unkow'

class AmazonOOData(RecBaseDataset):

    def __init__(self, text_processor=None, ann_paths=None, seq_len=None, user2group='', use_ids=False, use_desc=True, sas_seq_len=None):
        super().__init__()
        ann_paths = ann_paths[0].split('=')
        self.annotation = pd.read_pickle(ann_paths[0] + '_ood2.pkl').reset_index(drop=True)
        self.id2title = self.filter_dict(json.load(open('/'.join(ann_paths[0].split('/')[:-1]) + '/id2title.json', 'r')))
        self.id2title[0] = {'title': '', 'keywords': ''}
        self.use_his = False
        self.use_label = False
        self.use_reason = False
        self.use_ids = use_ids
        self.use_desc = use_desc
        self.prompt_flag = False
        self.user2group = None
        self.sas_seq_len = sas_seq_len
        if os.path.exists(user2group) and 'test' in ann_paths[0]:
            self.user2group = pd.read_csv(user2group)[['user_id', 'cluster']]
            self.user2group.columns = ['user_id', 'group_id']
            self.user2group['user_id'] = self.user2group['user_id'].astype(int)
            self.user2group['group_id'] = self.user2group['group_id'].astype(int)
            print('load splits by group')
        else:
            print('no need to split by group')
        if 'not_cold' in self.annotation.columns and 'warm' in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([1])].copy()
        if 'not_cold' in self.annotation.columns and 'cold' in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([0])].copy()
        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid', 'iid', 'his', 'label']
            renamed_columns = ['UserID', 'TargetItemID', 'InteractedItemIDs', 'label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append('prompt_flag')
                self.prompt_flag = True
            if 'his_label' in self.annotation.columns:
                used_columns.append('his_label')
                renamed_columns.append('InteractedItemLabels')
                self.use_label = True
            if 'reason' in self.annotation.columns:
                used_columns.append('reason')
                renamed_columns.append('reason')
                self.use_reason = True
            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
            self.annotation['InteractedItemIDs'] = self.annotation['InteractedItemIDs'].map(list)
            if self.use_label:
                self.annotation['InteractedItemLabels'] = self.annotation['InteractedItemLabels'].map(list)
        else:
            used_columns = ['uid', 'iid', 'title', 'label']
            renamed_columns = ['UserID', 'TargetItemID', 'TargetItemTitle', 'label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append('prompt_flag')
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        self.user_num = self.annotation['UserID'].max() + 1
        self.item_num = self.annotation['TargetItemID'].max() + 1
        self.items = list(set(self.annotation['TargetItemID']))
        self.text_processor = text_processor
        if self.user2group is not None:
            self.annotation = self.annotation.sort_values(by=['UserID', 'TargetItemID']).reset_index(drop=True)
            print('dataset reordered by UserID')
        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, seq_len)
            print('Movie OOD datasets, max history length:', self.max_lenght)
            logging.info('Movie OOD datasets, max history length:' + str(self.max_lenght))
        print('data path:', ann_paths[0], 'data size:', self.annotation.shape)

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            first_non_zero_index = next((i for i, x in enumerate(a) if x != 0), len(a))
            InteractedNum = len(a) - first_non_zero_index
            if len(a) < self.max_lenght:
                b = [0] * (self.max_lenght - len(a))
                b.extend(a)
            elif len(a) > self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            if self.sas_seq_len is not None:
                if len(a) < self.sas_seq_len:
                    c = [0] * (self.sas_seq_len - len(a))
                    c.extend(a)
                elif len(a) >= self.sas_seq_len:
                    c = a[-self.sas_seq_len:]
            if self.use_label:
                InteractedItemLabels = ann['InteractedItemLabels']
                if len(InteractedItemLabels) < self.max_lenght:
                    InteractedItemLabels_pad = [0] * (self.max_lenght - len(InteractedItemLabels))
                    InteractedItemLabels_pad.extend(InteractedItemLabels)
                elif len(InteractedItemLabels) > self.max_lenght:
                    InteractedItemLabels_pad = InteractedItemLabels[-self.max_lenght:]
                else:
                    InteractedItemLabels_pad = InteractedItemLabels
                InteractedItemLabels = InteractedItemLabels[-InteractedNum:]
            else:
                InteractedItemLabels = None
                InteractedItemLabels_pad = None
            one_sample = {'UserID': ann['UserID'], 'InteractedItemIDs_pad': np.array(b), 'InteractedItemTitles': convert_title_list_v4(samples=self.turnid2txt(ann['InteractedItemIDs'][-InteractedNum:]), labels=InteractedItemLabels, withid=self.use_ids), 'TargetItemID': ann['TargetItemID'], 'TargetItemTitle': convert_title_list_v4(samples=self.turnid2txt(ann['TargetItemID']), withid=False), 'InteractedNum': InteractedNum, 'label': ann['label']}
            if InteractedItemLabels_pad:
                one_sample['InteractedItemLabels_pad'] = np.array(InteractedItemLabels_pad)
            if self.use_reason:
                one_sample['reason'] = ann['reason']
            else:
                one_sample['reason'] = 'Yes' if ann['label'] else 'No'
            if self.sas_seq_len is not None:
                one_sample['sas_seq'] = np.array(c)
        else:
            one_sample = {'UserID': ann['UserID'], 'TargetItemID': ann['TargetItemID'], 'TargetItemTitle': ann['TargetItemTitle'].strip(' '), 'label': ann['label']}
        if self.prompt_flag:
            one_sample['prompt_flag'] = ann['prompt_flag']
        return one_sample

    def turnid2txt(self, ids):
        if isinstance(ids, (int, np.integer)):
            ids = [ids]
        elif isinstance(ids, list):
            pass
        else:
            try:
                ids = list(ids)
            except:
                raise NotImplementedError(f'Unsupported type of ids {type(ids)}')
        titles = [self.id2title[idx]['title'] for idx in ids]
        descriptions = None
        if self.use_desc:
            descriptions = [self.id2title[idx]['keywords'] for idx in ids]
        return (titles, descriptions)

    def get_id2title(self):
        return self.id2title

    def filter_dict(self, raw_dict):
        filtered_dict = {}
        for item_id, details in raw_dict.items():
            if 'title' in details and 'keywords' in details:
                filtered_dict[int(item_id)] = {'title': details['title'], 'keywords': details['keywords']}
            if 'id' in details and 'keywords' in details:
                filtered_dict[item_id] = {'id': details['id'], 'keywords': details['keywords']}
        return filtered_dict
