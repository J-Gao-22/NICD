import logging
import os
import random
import time
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from ..llm.base import Rec2Base, disabled_train

def get_ids_order(prompt):
    id_flags = ['<UserID>', '<HisItemList>', '<TargetItemID>', '<TargetItemList>']
    id_order_ = []
    for flag_ in id_flags:
        pos_ = prompt.find(flag_)
        if pos_ >= 0:
            id_order_.append(pos_)
    id_order_ = np.argsort(np.array(id_order_))
    return id_order_

class MiniGPT4Rec_v3(Rec2Base):

    def __init__(self, rec_model='MF', rec_config=None, pretrained_rec=None, freeze_rec=True, rec_precision='fp16', llama_model='', prompt_path='', prompt_template='', max_txt_len=32, end_sym='\n', low_resource=False, device_8bit=0, proj_token_num=1, proj_drop=0, lora_config=None, loss_config=None, proj_mid=5, freeze_lora=False, freeze_proj=False):
        super().__init__()
        self.low_resource = low_resource
        self.proj_token_num = proj_token_num
        print('runing MiniGPT4Rec_v3 ...... ')
        print('Loading Rec_model')
        self.rec_model_type = rec_model
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config, rec_precision)
        if self.rec_encoder is not None and pretrained_rec != 'not_have':
            self.rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location='cpu'))
            print('successfully load the pretrained model......')
        if freeze_rec and self.rec_encoder is not None:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train
            logging.info('freeze rec encoder')
            print('freeze rec encoder')
        print('Loading Rec_model Done')
        print('Loading LLAMA')
        if 'Qwen' in llama_model:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, local_files_only=True)
            self.llama_tokenizer.unk_token = '<|object_ref_end|>'
            marker_ids = self.llama_tokenizer.encode(self.llama_tokenizer.unk_token, add_special_tokens=False)
            assert len(marker_ids) == 1, 'Collaborative placeholder must be one existing token'
            self.llama_tokenizer.unk_token_id = marker_ids[0]
        else:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, use_fast=False)
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
            if self.llama_tokenizer.unk_token is None:
                self.llama_tokenizer.add_special_tokens({'unk_token': '<unk>'})
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(llama_model, torch_dtype=torch.float16, load_in_8bit=True, device_map={'': device_8bit})
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(llama_model, torch_dtype=torch.bfloat16, local_files_only=True)
        self.llama_model.config.use_cache = False
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')
        self.use_lora = False
        if lora_config is not None and lora_config.use_lora:
            print('Setting Lora')
            self.use_lora = True
            peft_config = LoraConfig(r=lora_config.r, lora_alpha=lora_config.alpha, target_modules=list(lora_config.target_modules), lora_dropout=lora_config.dropout, bias='none', task_type='CAUSAL_LM')
            self.llama_model_lora = get_peft_model(self.llama_model, peft_config)
            print('Setting Lora Done')
        if freeze_lora:
            print('freeze lora...')
            try:
                flexible_layer_num = freeze_lora.layers
                depth = self.llama_model.config.num_hidden_layers
                flexible_layers = range(depth - flexible_layer_num, depth)
            except:
                flexible_layers = None
            for name, param in self.llama_model_lora.named_parameters():
                if flexible_layers and 'layers' in name and ('lora_' in name):
                    layer_id = int(re.search('\\.layers\\.([^.]+)', name).group(1))
                    if layer_id in flexible_layers:
                        param.requires_grad = True
                        continue
                param.requires_grad = False
        if self.rec_encoder is not None and 'prompt' not in rec_model:
            print('proj_mid:', type(proj_mid), proj_mid)
            self.llama_proj = nn.Sequential(nn.Linear(self.rec_encoder.config.embedding_size, self.rec_encoder.config.embedding_size * int(proj_mid)), nn.ReLU(), nn.Linear(self.rec_encoder.config.embedding_size * int(proj_mid), self.llama_model.config.hidden_size * self.proj_token_num))
        elif self.rec_encoder is not None and rec_model == 'personlized_prompt':
            print('personalized prompt learning....')
            self.llama_proj = nn.Linear(rec_config.item_num + rec_config.user_num, self.llama_model.config.hidden_size * self.proj_token_num, bias=False)
        elif self.rec_encoder is not None and rec_model == 'soft_prompt':
            print('soft prompt learning....')
            self.llama_proj = nn.Linear(2, self.llama_model.config.hidden_size * self.proj_token_num, bias=False)
        else:
            self.llama_proj = None
        if freeze_proj:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            self.llama_proj = self.llama_proj.eval()
            self.llama_proj.train = disabled_train
            logging.info('!!!! freeze llama_proj...')
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.has_print_prompt = True
        if 'toy' in pretrained_rec:
            data_types = 'toys or games'
            data_type = 'toy or game'
        elif 'ml1m' in pretrained_rec:
            data_types = 'movies'
            data_type = 'movie'
        elif 'beauty' in pretrained_rec:
            data_types = 'beauty'
            data_type = 'beauty'
        elif 'yelp' in pretrained_rec:
            data_types = 'businesses'
            data_type = 'business'
        else:
            data_types = 'books'
            data_type = 'book'
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt.replace('books', data_types).replace('book', data_type) for raw_prompt in raw_prompts]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt List: \n{}'.format(self.prompt_list))
            self.has_pri_decode = True
            self.prompt_list_p = None
        else:
            self.prompt_list = []
            self.prompt_list_p = None
        self.reason = False
        self.eval_only = False
        self.user2group = None
        self.wrong_data = 0
        self.loss_alpha = 1.0
        self.loss_beta = 0
        self.loss_theta = 1.0
        self.loss_gamma = 0
        if loss_config is not None:
            try:
                self.loss_alpha = loss_config.alpha
                self.loss_beta = loss_config.beta
                self.loss_theta = loss_config.theta
                self.loss_gamma = loss_config.gamma
            except:
                print('using default loss_config')

    def set_user2group(self, user2group):
        print(user2group)
        if os.path.exists(user2group):
            self.user2group = pd.read_csv(user2group)
            if self.rec_model_type == 'sasrec':
                self.user2group.columns = ['user_id', 'group_id', 'embedding']
            else:
                self.user2group.columns = ['user_id', 'group_id']
            self.user2group['user_id'] = self.user2group['user_id'].astype(int)
            self.user2group['group_id'] = self.user2group['group_id'].astype(int)
            print('set user2group done!!!')
            self.active_user = 0
            self.base_lora = []
            num_groups = self.user2group['group_id'].max() + 1
            self.all_user_embeds, _ = self.rec_encoder.computer()
            for group_id in range(num_groups):
                group_lines = self.user2group[self.user2group['group_id'] == group_id]
                group_users = group_lines['user_id'].tolist()
                if self.rec_model_type == 'sasrec':
                    user_embeds = torch.tensor([np.array(eval(x), dtype=np.float16) for x in group_lines['embedding']]).mean(dim=-2)
                else:
                    group_index = torch.tensor(group_users, device=next(self.rec_encoder.parameters()).device)
                    user_embeds = self.rec_encoder.user_encoder(group_index, all_users=self.all_user_embeds).detach().mean(dim=-2)
                print('group_id:', group_id, 'group_users:', len(group_users), 'user_embeds:', user_embeds.shape)
                self.base_lora.append(user_embeds)
            self.base_lora = torch.vstack(self.base_lora)
            print('base_lora:', self.base_lora.shape)
        else:
            print('user2group not found!!!')

    def set_mode(self, mode):
        self.run_mode_ = mode

    def rec_to_cpu(self):
        self.rec_encoder.to('cpu')
        self.rec_encoder.float()

    def set_answer_type(self, mode):
        if mode == 'v1':
            self.pos_ans = ['former']
            self.neg_ans = ['latter']
        elif mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            pos_ans_id = self.llama_tokenizer(self.pos_ans[0], add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(self.neg_ans[0], add_special_tokens=False).input_ids[0]
            print('answer token ids: pos:', pos_ans_id, 'neg ids:', neg_ans_id)
        elif mode == 'v3':
            print('answer is concrect item title.')
        else:
            raise NotImplementedError('not implement this types of answers')

    def maybe_autocast(self, dtype=None):
        return super().maybe_autocast(dtype=dtype or self.llama_model.dtype)

    def forward_v3(self, samples):
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError('not implement')
        elif self.prompt_list:
            prompt = random.choice(self.prompt_with_p([5, 5, 5, 1]))
        self.llama_tokenizer.padding_side = 'right'
        device = samples['UserID'].device
        if 'reason' in samples:
            self.reason = True
        ans_ = {1: self.pos_ans[0], 0: self.neg_ans[0]}
        pos_ans_id = self.llama_tokenizer(ans_[int(1)], add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(ans_[int(0)], add_special_tokens=False).input_ids[0]
        if self.reason:
            text = samples['reason']
        else:
            text = [ans_[int(t)] for t in samples['label']]
        to_regress_tokens = self.llama_tokenizer(text, return_tensors='pt', padding='longest', truncation=True, max_length=self.max_txt_len // 5, add_special_tokens=False).to(device)
        self._prompt_token_budget = self.max_txt_len - to_regress_tokens.input_ids.shape[1]
        sample_embeds, sample_atts = self.prompt_based_encode_v3(prompt, samples)
        t_posi = to_regress_tokens.input_ids.shape[-1] + 1
        targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100)
        empty_targets = torch.ones([sample_atts.shape[0], sample_atts.shape[1]], dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([sample_atts, to_regress_tokens.attention_mask], dim=1)
        if self.user2group is not None:
            current_adapter = self.llama_model_lora.active_adapter
            group_id = f"group{self.user2group[self.user2group['user_id'] == samples['UserID'].cpu().tolist()[0]]['group_id'].item()}"
            if group_id != current_adapter:
                self.llama_model_lora.set_adapter(group_id)
        assert inputs_embeds.shape[1] <= self.max_txt_len
        with self.maybe_autocast():
            outputs = self.llama_model_lora(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=(attention_mask.long().cumsum(-1) - 1).clamp_min(0), return_dict=True, labels=targets, use_cache=False)
        logits = outputs.logits[:, -t_posi, :][:, pos_ans_id]
        if not torch.isfinite(logits).all() or not torch.isfinite(outputs.loss):
            raise RuntimeError(str(dict(error='nonfinite_source_forward', input_shape=list(inputs_embeds.shape), inputs_finite=bool(torch.isfinite(inputs_embeds).all()), positive_logits_finite=bool(torch.isfinite(logits).all()), lm_loss_finite=bool(torch.isfinite(outputs.loss)), target_counts=(targets != -100).sum(-1).tolist())))
        loss_pred = nn.functional.binary_cross_entropy_with_logits(logits.float(), samples['label'].float())
        if t_posi >= 3:
            loss = self.loss_alpha * loss_pred + self.loss_beta * outputs.loss
        else:
            loss = self.loss_theta * loss_pred + self.loss_gamma * outputs.loss
        return {'loss': loss}

    def adaptive_lora(self, user):
        if self.rec_model_type == 'sasrec':
            user_embeds = self.rec_encoder.seq_encoder(torch.tensor(user))[0].detach()
        else:
            user_index = torch.as_tensor(user, device=next(self.rec_encoder.parameters()).device)
            user_embeds = self.rec_encoder.user_encoder(user_index, all_users=self.all_user_embeds).detach()
        cos_sim = F.cosine_similarity(user_embeds.unsqueeze(0), self.base_lora.to(user_embeds.device), dim=-1)
        tau = 0.1
        lora_weights = torch.softmax(cos_sim / tau, dim=-1)
        n_clusters = len(lora_weights)
        print(f'lora_weights:', lora_weights)
        ent = -torch.sum(lora_weights * torch.log(lora_weights + 1e-10), dim=-1)
        if torch.max(lora_weights) > 0.5 + 0.6 / n_clusters:
            group_id = torch.argmax(lora_weights).item()
            print(f'set group {group_id} adapter')
            self.llama_model_lora.set_adapter(f'group{group_id}')
        elif ent >= 0.95 * torch.log(torch.tensor(n_clusters)):
            print('set general adapter')
            self.llama_model_lora.set_adapter(f'group{cos_sim.shape[0]}')
        else:
            merge_adapters = [f'group{i}' for i in range(len(lora_weights))]
            self.llama_model_lora.add_weighted_adapter(adapters=merge_adapters, weights=lora_weights, combination_type='linear', adapter_name=str(user))
            self.llama_model_lora.set_adapter(str(user))
        try:
            self.llama_model_lora.delete_adapter(self.active_user)
        except:
            pass
        self.active_user = str(user)

    def generate_for_samples_v3(self, samples, return_all=False):
        ret = {}
        self._prompt_token_budget = self.max_txt_len - 1
        prompt = self.prompt_list[0]
        sample_embeds, sample_atts = self.prompt_based_encode_v3(prompt, samples, False)
        self.llama_tokenizer.padding_side = 'right'
        device = samples['UserID'].device
        ans_ = {1: self.pos_ans[0], 0: self.neg_ans[0]}
        start_time = time.time()
        text = [ans_[int(t)] for t in samples['label']]
        to_regress_tokens = self.llama_tokenizer(text, return_tensors='pt', padding='longest', truncation=True, max_length=32, add_special_tokens=False).to(device)
        t_posi = to_regress_tokens.input_ids.shape[-1] + 1
        targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        empty_targets = torch.ones([sample_atts.shape[0], sample_atts.shape[1]], dtype=torch.long).to(device).fill_(-100)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        inputs_atts = torch.cat([sample_atts, to_regress_tokens.attention_mask], dim=1)
        QAtargets = torch.cat([empty_targets, targets], dim=1)
        assert inputs_embeds.shape[1] <= self.max_txt_len
        if self.user2group is not None:
            if self.active_user != str(samples['UserID'][0]):
                user = samples['sas_seq'] if self.rec_model_type == 'sasrec' else samples['UserID'][0]
                self.adaptive_lora(user)
        with self.maybe_autocast():
            outputs = self.llama_model_lora(inputs_embeds=inputs_embeds, attention_mask=inputs_atts, position_ids=(inputs_atts.long().cumsum(-1) - 1).clamp_min(0), return_dict=True, use_cache=False, labels=QAtargets)
        end_time = time.time()
        ret['cost'] = end_time - start_time
        pos_ans_id = self.llama_tokenizer(ans_[int(1)], add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(ans_[int(0)], add_special_tokens=False).input_ids[0]
        logits_ = outputs.logits[:, -t_posi, :][:, pos_ans_id]
        loss = nn.functional.binary_cross_entropy_with_logits(logits_, samples['label'].float())
        ret['loss'] = loss
        ret['logits'] = logits_
        return ret

    def prompt_based_encode_v3(self, prompt, samples, reflect=False):
        id_orders = get_ids_order(prompt)
        samples_encode, sample_atts = self.encode_recdata_v3(samples, ids_order=id_orders)
        sample_embeds, samples_atts = self.recprompt_wrap_v3(samples_encode, samples, sample_atts, prompt)
        if reflect:
            reflect_embeds, reflect_atts = self.reflect_prompt_wrap(samples_encode, samples, None)
            return (sample_embeds, samples_atts, reflect_embeds, reflect_atts)
        return (sample_embeds, samples_atts)

    def encode_recdata_v3(self, sample, ids_order=None):
        if self.rec_encoder is None:
            return (None, None)
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        with self.maybe_autocast():
            batch_size = sample['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            if self.rec_model_type == 'sasrec':
                user_embeds = self.rec_encoder.seq_encoder(sample['sas_seq']).unsqueeze(-2)
            elif self.rec_model_type == 'DCN' or self.rec_model_type == 'DIN':
                user_embeds = self.rec_encoder.all_encode(sample['UserID'], sample['TargetItemID'], sample['sas_seq'][:, -10:]).unsqueeze(-2)
            else:
                user_embeds = self.rec_encoder.user_encoder(sample['UserID'], all_users=all_user_embeds).unsqueeze(-2)
            targetItem_embed = self.rec_encoder.item_encoder(sample['TargetItemID'], all_items=all_item_embeds).unsqueeze(-2)
            user_embeds_llama = self.llama_proj(user_embeds).reshape(batch_size, -1, self.proj_token_num, hidden_size)
            targetItem_embeds_llama = self.llama_proj(targetItem_embed).reshape(batch_size, -1, self.proj_token_num, hidden_size)
            if 'InteractedItemIDs_pad' in sample.keys() and len(ids_order) == 3:
                interactedItem_embeds = self.rec_encoder.item_encoder(sample['InteractedItemIDs_pad'], all_items=all_item_embeds)
                interactedItem_embeds_llama = self.llama_proj(interactedItem_embeds).reshape(batch_size, -1, self.proj_token_num, hidden_size)
                merged_embeds = [user_embeds_llama, interactedItem_embeds_llama, targetItem_embeds_llama]
                merged_embeds = [merged_embeds[k] for k in ids_order]
                merged_embeds = torch.cat(merged_embeds, dim=1)
                idx_flag = torch.ones_like(sample['InteractedItemIDs_pad'])
                idx_flag = torch.where(sample['InteractedItemIDs_pad'] == self.rec_encoder.padding_index, 0, idx_flag)
                idx_flag = [torch.ones([idx_flag.shape[0], 1]).to(idx_flag.device), idx_flag, torch.ones([idx_flag.shape[0], 1]).to(idx_flag.device)]
                idx_flag = [idx_flag[k] for k in ids_order]
                idx_flag = torch.cat(idx_flag, dim=1).to(device)
                idx_nopad = torch.nonzero(idx_flag)
                sample_embeds_llama = {'User_emb': user_embeds_llama.reshape(batch_size, -1, hidden_size), 'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size, -1, hidden_size), 'InteractedItems_embs': interactedItem_embeds_llama.reshape(batch_size, -1, hidden_size), 'merged_embs': merged_embeds[idx_nopad[:, 0], idx_nopad[:, 1]].reshape(-1, hidden_size)}
            else:
                sample_embeds_llama = {'User_emb': user_embeds_llama.reshape(batch_size, -1, hidden_size), 'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size, -1, hidden_size), 'InteractedItems_embs': None, 'merged_embs': None}
        sample_atts_llama = None
        return (sample_embeds_llama, sample_atts_llama)

    def recprompt_wrap_v3(self, samples, ori_samples, atts_sample, prompt):
        if prompt:
            prompt_ori = prompt
            split_symbol = ['<UserID>', '<HisItemTitleList>', '<TargetItemTitleList>', '<HisItemList>', '<TargetItemList>', '<ItemID>']
            batch_size = ori_samples['UserID'].shape[0]
            bos = self.llama_tokenizer.bos_token or ''
            if self.llama_tokenizer.unk_token:
                unk_token_id = self.llama_tokenizer.unk_token_id
                unk_ = self.llama_tokenizer.unk_token
            unk_ = '.'.join([unk_] * self.proj_token_num)
            prompt = bos + prompt
            prompt = prompt.replace('<UserID>', unk_)
            label2answer = {0: 'No', 1: 'Yes'}
            prompt_list = []
            for k in range(batch_size):
                prompt_ = prompt + ''
                if 'InteractedNum' in ori_samples.keys():
                    history_text = ori_samples['InteractedItemTitles'][k]
                    if int(ori_samples['InteractedNum'][k]) == 0 and '<ItemID>' in history_text:
                        history_text = 'None'
                    prompt_ = prompt_.replace('<HisItemTitleList>', history_text)
                    prompt_ = prompt_.replace('<HisItemList>', history_text)
                    prompt_ = prompt_.replace('<ItemTitleList>', history_text)
                prompt_ = prompt_.replace('<TargetItemTitle>', ori_samples['TargetItemTitle'][k])
                prompt_ = prompt_.replace('<ItemID>', unk_)
                prompt_ = prompt_.replace('<TargetItemID>', unk_)
                prompt_list.append(prompt_)
            if not self.has_print_prompt:
                print('prompt example:', prompt_list[0])
                self.has_print_prompt = True
            self.llama_tokenizer.padding_side = 'left'
            prompts_tokens = self.llama_tokenizer(prompt_list, return_tensors='pt', padding='longest', truncation=True, max_length=getattr(self, '_prompt_token_budget', self.max_txt_len - 1), add_special_tokens=False).to(ori_samples['UserID'].device)
            if not self.has_pri_decode:
                print('#######prmpt decoded example: ', ' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))
                self.has_pri_decode = True
            replaced_idx = torch.nonzero(prompts_tokens.input_ids == unk_token_id)
            prompt_embeds = self.llama_model_lora.base_model.model.model.embed_tokens(prompts_tokens.input_ids)
            try:
                if '<UserID>' in prompt_ori and '<HisItemList>' in prompt_ori and ('<TargetItemID>' in prompt_ori):
                    if replaced_idx.shape[0] == 2 * samples['User_emb'].shape[0]:
                        prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = torch.cat([samples['User_emb'], samples['TargetItem_emb']], dim=-2).reshape(-1, samples['User_emb'].shape[-1])
                    else:
                        prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = samples['merged_embs']
                elif '<UserID>' in prompt_ori and '<TargetItemID>' in prompt_ori and ('<HisItemList>' not in prompt_ori):
                    prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = torch.cat([samples['User_emb'], samples['TargetItem_emb']], dim=-2).reshape(-1, samples['User_emb'].shape[-1])
                else:
                    pass
            except Exception as exc:
                raise RuntimeError(f"Collaborative injection failed: slots_per_row={(prompts_tokens.input_ids == unk_token_id).sum(1).tolist()}, history_nonpad={(ori_samples['InteractedItemIDs_pad'] != self.rec_encoder.padding_index).sum(1).tolist()}, history_counts={ori_samples['InteractedNum'].tolist()}, history_text_slots={[t.count('<ItemID>') for t in ori_samples['InteractedItemTitles']]}, merged_shape={(None if samples['merged_embs'] is None else tuple(samples['merged_embs'].shape))}") from exc
            return (prompt_embeds, prompts_tokens.attention_mask)

    def reflect_prompt_wrap(self, samples, ori_samples, prompt):
        prompt_ori = 'Based on the descriptions and the user\'s enjoyment of each book in the historical sequence, construct a persona of the user\'s preferences and reevaluate whether the user would enjoy the book titled <TargetItemTitle>. Please begin your analysis with "Yes" or "No". \n#Answer:'
        batch_size = ori_samples['UserID'].shape[0]
        bos = '<s>'
        unk_ = self.llama_tokenizer.unk_token
        unk_ = '.'.join([unk_] * self.proj_token_num)
        prompt = bos + prompt_ori
        prompt_list = []
        for k in range(batch_size):
            prompt_ = prompt + ''
            prompt_ = prompt_.replace('<TargetItemTitle>', ori_samples['TargetItemTitle'][k])
            prompt_ = prompt_.replace('<TargetItemID>', unk_)
            prompt_list.append(prompt_)
        self.llama_tokenizer.padding_side = 'left'
        prompts_tokens = self.llama_tokenizer(prompt_list, return_tensors='pt', padding='longest', truncation=True, max_length=self.max_txt_len, add_special_tokens=False).to(ori_samples['UserID'].device)
        replaced_idx = torch.nonzero(prompts_tokens.input_ids == self.llama_tokenizer.unk_token_id)
        prompt_embeds = self.llama_model_lora.base_model.model.model.embed_tokens(prompts_tokens.input_ids)
        if '<TargetItemID>' in prompt_ori:
            try:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = samples['TargetItem_emb']
            except:
                print(replaced_idx.shape, 'sample', samples['TargetItem_emb'].shape)
                print('#######prmpt decoded example: ', ' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))
        return (prompt_embeds, prompts_tokens.attention_mask)

    def prompt_with_p(self, p):
        if self.prompt_list_p is None:
            prompt_list_p = []
            for k in range(len(p)):
                prompt_list_p.extend([self.prompt_list[k]] * p[k])
            self.prompt_list_p = prompt_list_p
            return self.prompt_list_p
        else:
            return self.prompt_list_p

    def forward(self, samples):
        return self.forward_v3(samples)
