import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class MatrixFactorization(nn.Module):

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.padding_index = 0
        self.user_embedding = nn.Embedding(config.user_num, config.embedding_size, padding_idx=self.padding_index)
        self.item_embedding = nn.Embedding(config.item_num, config.embedding_size, padding_idx=self.padding_index)
        print('creat MF model, user num:', config.user_num, 'item num:', config.item_num)

    def user_encoder(self, users, all_users=None):
        return self.user_embedding(users)

    def item_encoder(self, items, all_items=None):
        return self.item_embedding(items)

    def computer(self):
        return (None, None)

    def forward(self, users, items):
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        matching = torch.mul(user_embedding, item_embedding).sum(dim=-1)
        return matching

class LightGCN(nn.Module):

    def __init__(self, config):
        super(LightGCN, self).__init__()
        self.config = config
        self.padding_index = 0
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.config.user_num
        self.num_items = self.config.item_num
        self.latent_dim = self.config.embed_size
        self.n_layers = self.config.gcn_layers
        self.keep_prob = self.config.keep_prob
        self.A_split = self.config.A_split
        self.dropout_flag = self.config.dropout
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config.pretrain == 0:
            nn.init.normal_(self.embedding_user.weight, std=self.config.init_emb)
            nn.init.normal_(self.embedding_item.weight, std=self.config.init_emb)
            print('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        print(f'lgn is already to go(dropout:{self.config.dropout})')

    def _set_graph(self, graph):
        self.Graph = graph.to(self.embedding_user.weight.device)
        self.Graph = self.Graph.to_sparse_csr()
        print('Graph Device:', self.Graph.device)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        self.Graph = self.Graph.to(users_emb.device)
        embs = [all_emb]
        if self.dropout_flag:
            if self.training:
                print('droping')
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return (users, items)

    def user_encoder(self, users, all_users=None):
        if all_users is None:
            all_users, all_items = self.computer()
        return all_users[users]

    def item_encoder(self, items, all_items=None):
        if all_items is None:
            all_users, all_items = self.computer()
        return all_items[items]

    def F_computer(self, users_emb, items_emb, adj_graph):
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.dropout_flag:
            if self.training:
                print('droping')
                raise NotImplementedError('dropout methods are not implemented')
            else:
                g_droped = adj_graph
        else:
            g_droped = adj_graph
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return (users, items)

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return (users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego)

    def getEmbedding_v2(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        return (users_emb, items_emb)

    def bpr_loss(self, users, pos, neg):
        users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = 1 / 2 * (userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return (loss, reg_loss)

    def compute_bce_loss(self, users, items, labels):
        users_emb, items_emb = self.getEmbedding_v2(users.long(), items.long())
        matching = torch.mul(users_emb, items_emb)
        scores = torch.sum(matching, dim=-1)
        bce_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='mean')
        return bce_loss

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

    def predict(self, users, items):
        users = torch.from_numpy(users).long().cuda()
        items = torch.from_numpy(items).long().cuda()
        with torch.no_grad():
            all_user_emb, all_item_emb = self.computer()
            users_emb = all_user_emb[users]
            items_emb = all_item_emb[items]
            inner_pro = torch.mul(users_emb, items_emb).sum(dim=-1)
            scores = torch.sigmoid(inner_pro)
        return scores.cpu().numpy()

    def predict_changed_graph(self, users, items, changed_graph):
        users = torch.from_numpy(users).long().cuda()
        items = torch.from_numpy(items).long().cuda()
        with torch.no_grad():
            all_user_emb, all_item_emb = self.F_computer(self.embedding_user.weight, self.embedding_item.weight, changed_graph)
            users_emb = all_user_emb[users]
            items_emb = all_item_emb[items]
            inner_pro = torch.mul(users_emb, items_emb).sum(dim=-1)
            scores = torch.sigmoid(inner_pro)
        return scores.cpu().numpy()

class PointWiseFeedForward(nn.Module):

    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs

class SASRec(nn.Module):

    def __init__(self, args):
        super(SASRec, self).__init__()
        self.config = args
        self.padding_index = 0
        self.user_num = args.user_num
        self.item_num = args.item_num
        self.item_emb = torch.nn.Embedding(self.item_num, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-08)
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-08)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-08)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def _device(self):
        self.dev = self.item_emb.weight.device

    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs.to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        timeline_mask = torch.BoolTensor(log_seqs.cpu().numpy() == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)
        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward_eval(self, user_ids, target_item, log_seqs):
        self._device()
        log_feats = self.log2feats(log_seqs)
        log_feats = log_feats[:, -1, :]
        item_embs = self.item_emb(target_item)
        return (log_feats * item_embs).sum(dim=-1)

    def forward(self, seqs, target, target_posi=None):
        self._device()
        log_feats = self.log2feats(seqs)
        if target_posi is not None:
            s_emb = log_feats[target_posi[:, 0], target_posi[:, 1]]
        else:
            s_emb = log_feats[:, -1, :]
        target_embeds = self.item_emb(target.reshape(-1))
        scores = torch.mul(s_emb, target_embeds).sum(dim=-1)
        return scores

    def computer(self):
        return (None, None)

    def seq_encoder(self, seqs):
        self._device()
        log_feats = self.log2feats(seqs)
        seq_emb = log_feats[:, -1, :]
        return seq_emb

    def item_encoder(self, target_item, all_items=None):
        self._device()
        target_embeds = self.item_emb(target_item)
        return target_embeds

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

    def predict_all(self, user_ids, log_seqs):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb.weight
        logits = torch.matmul(final_feat, item_embs.T)
        return logits

    def predict_all_batch(self, user_ids, log_seqs, batch_size=128):
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb.weight
        logits = torch.matmul(final_feat, item_embs.T)
        return logits

    def log2feats_v2(self, log_seqs, emb_replace=None):
        log_seqs = log_seqs + 0
        emb_replace_idx = np.where(log_seqs < 0)
        log_seqs[emb_replace_idx] = 0
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev)) + 0
        log_seqs[emb_replace_idx] = -1
        if emb_replace is not None:
            seqs[emb_replace_idx[0], emb_replace_idx[1]] = 0
            seqs[emb_replace_idx[0], emb_replace_idx[1]] += emb_replace
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)
        log_feats = self.last_layernorm(seqs)
        return log_feats

    def predict_position(self, log_seqs, postions, emb_replace=None):
        log_feats = self.log2feats_v2(log_seqs, emb_replace=emb_replace)
        final_feat = log_feats[np.arange(postions.shape[0]), postions]
        item_embs = self.item_emb.weight
        logits = torch.matmul(final_feat, item_embs.T)
        return logits

class GSASRec(torch.nn.Module):

    def __init__(self, args):
        num_items = args.item_num
        sequence_length = getattr(args, 'maxlen', 200)
        embedding_dim = getattr(args, 'hidden_units', 256)
        num_heads = getattr(args, 'num_heads', 4)
        num_blocks = getattr(args, 'num_blocks', 3)
        dropout_rate = getattr(args, 'dropout_rate', 0.5)
        reuse_item_embeddings = getattr(args, 'reuse_item_embeddings', False)
        super(GSASRec, self).__init__()
        self.config = args
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.embeddings_dropout = torch.nn.Dropout(dropout_rate)
        self.padding_index = 0
        self.user_num = getattr(args, 'user_num', 0)
        self.item_num = num_items
        self.num_heads = num_heads
        self.item_embedding = torch.nn.Embedding(self.num_items, self.embedding_dim, padding_idx=0)
        self.position_embedding = torch.nn.Embedding(self.sequence_length, self.embedding_dim)
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock(self.embedding_dim, self.num_heads, self.embedding_dim, dropout_rate) for _ in range(num_blocks)])
        self.seq_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.reuse_item_embeddings = reuse_item_embeddings
        if not self.reuse_item_embeddings:
            self.output_embedding = torch.nn.Embedding(self.num_items, self.embedding_dim, padding_idx=0)
        self.dev = None

    def _device(self):
        self.dev = self.item_embedding.weight.device

    def get_output_embeddings(self) -> torch.nn.Embedding:
        if self.reuse_item_embeddings:
            return self.item_embedding
        else:
            return self.output_embedding

    def log2feats(self, input):
        seq = self.item_embedding(input.long())
        mask = (input != 0).float().unsqueeze(-1)
        bs = seq.size(0)
        positions = torch.arange(seq.shape[1]).unsqueeze(0).repeat(bs, 1).to(input.device)
        pos_embeddings = self.position_embedding(positions)
        seq = seq + pos_embeddings
        seq = self.embeddings_dropout(seq)
        seq *= mask
        attentions = []
        for i, block in enumerate(self.transformer_blocks):
            seq, attention = block(seq, mask)
            attentions.append(attention)
        seq_emb = self.seq_norm(seq)
        return (seq_emb, attentions)

    def get_predictions(self, input, limit, rated=None):
        with torch.no_grad():
            model_out, _ = self.log2feats(input)
            seq_emb = model_out[:, -1, :]
            output_embeddings = self.get_output_embeddings()
            scores = torch.einsum('bd,nd->bn', seq_emb, output_embeddings.weight)
            scores[:, 0] = float('-inf')
            if rated is not None:
                for i in range(len(input)):
                    for j in rated[i]:
                        scores[i, j] = float('-inf')
            result = torch.topk(scores, limit, dim=1)
            return (result.indices, result.values)

    def computer(self):
        return (None, None)

    def seq_encoder(self, seqs):
        self._device()
        seq_emb, _ = self.log2feats(seqs)
        return seq_emb[:, -1, :]

    def item_encoder(self, target_item, all_items=None):
        self._device()
        output_embeddings = self.get_output_embeddings()
        target_embeds = output_embeddings(target_item)
        return target_embeds

    def forward_eval(self, user_ids, target_item, log_seqs):
        self._device()
        seq_emb, _ = self.log2feats(log_seqs)
        log_feats = seq_emb[:, -1, :]
        output_embeddings = self.get_output_embeddings()
        item_embs = output_embeddings(target_item)
        return (log_feats * item_embs).sum(dim=-1)

    def forward(self, seqs, target, target_posi=None):
        self._device()
        seq_emb, _ = self.log2feats(seqs)
        if target_posi is not None:
            s_emb = seq_emb[target_posi[:, 0], target_posi[:, 1]]
        else:
            s_emb = seq_emb[:, -1, :]
        output_embeddings = self.get_output_embeddings()
        target_embeds = output_embeddings(target.reshape(-1))
        scores = torch.mul(s_emb, target_embeds).sum(dim=-1)
        return scores

    def predict(self, user_ids, log_seqs, item_indices):
        self._device()
        seq_emb, _ = self.log2feats(log_seqs)
        final_feat = seq_emb[:, -1, :]
        output_embeddings = self.get_output_embeddings()
        item_embs = output_embeddings(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

    def predict_all(self, user_ids, log_seqs):
        self._device()
        seq_emb, _ = self.log2feats(log_seqs)
        final_feat = seq_emb[:, -1, :]
        output_embeddings = self.get_output_embeddings()
        item_embs = output_embeddings.weight
        logits = torch.matmul(final_feat, item_embs.T)
        return logits

    def predict_all_batch(self, user_ids, log_seqs, batch_size=128):
        return self.predict_all(user_ids, log_seqs)

class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout_rate=0.5):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, causality=False):
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.val_proj(keys)
        Q_ = torch.cat(Q.chunk(self.num_heads, dim=2), dim=0)
        K_ = torch.cat(K.chunk(self.num_heads, dim=2), dim=0)
        V_ = torch.cat(V.chunk(self.num_heads, dim=2), dim=0)
        outputs = torch.matmul(Q_, K_.transpose(1, 2))
        outputs = outputs / K_.size(-1) ** 0.5
        key_masks = torch.sign(torch.sum(torch.abs(keys), dim=-1))
        key_masks = key_masks.repeat(self.num_heads, 1)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.size(1), 1)
        outputs = outputs.masked_fill(key_masks == 0, float('-inf'))
        if causality:
            diag_vals = torch.ones_like(outputs[0])
            tril = torch.tril(diag_vals)
            masks = tril[None, :, :].repeat(outputs.size(0), 1, 1)
            outputs = outputs.masked_fill(masks == 0, float('-inf'))
        outputs = F.softmax(outputs, dim=-1)
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
        query_masks = torch.sign(torch.sum(torch.abs(queries), dim=-1))
        query_masks = query_masks.repeat(self.num_heads, 1)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.size(1))
        outputs *= query_masks
        attention_chunks = outputs.chunk(self.num_heads, dim=0)
        attention_weights = torch.stack(attention_chunks, dim=1)
        outputs = self.dropout(outputs)
        outputs = torch.matmul(outputs, V_)
        outputs = torch.cat(outputs.chunk(self.num_heads, dim=0), dim=2)
        return (outputs, attention_weights)

class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, hidden_dim, dropout_rate=0.5, causality=True):
        super(TransformerBlock, self).__init__()
        self.first_norm = nn.LayerNorm(dim)
        self.second_norm = nn.LayerNorm(dim)
        self.multihead_attention = MultiHeadAttention(dim, num_heads, dropout_rate)
        self.dense1 = nn.Linear(dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.causality = causality

    def forward(self, seq, mask=None):
        x = self.first_norm(seq)
        queries = x
        keys = seq
        x, attentions = self.multihead_attention(queries, keys, self.causality)
        x = x + queries
        x = self.second_norm(x)
        residual = x
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = x + residual
        if mask is not None:
            x *= mask
        return (x, attentions)
