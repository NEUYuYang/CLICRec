from recbole.model.layers import MLPLayers, SequenceAttLayer
from recbole.model.abstract_recommender import SequentialRecommender
import torch
from torch.nn.init import xavier_normal_, constant_
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.loss import BPRLoss
import numpy as np
from torch_geometric.nn import GATConv
from sklearn.cluster import KMeans
from torch_geometric.utils import to_undirected
from recbole.model.loss import CLICRecInfoNCELoss
from recbole.model.layers import GALSTM


class CLICRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(CLICRec, self).__init__(config, dataset)

        self.sample_num = config["train_neg_sample_args"]["sample_num"]
        self.loss_fct = BPRLoss()
        self.batch_size = config["train_batch_size"]
        self.hidden_size = config["hidden_size"]
        self.embedding_size = config["embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.device = config["device"]
        self.gat_hidden_dim = config["gat_hidden_dim"]
        self.gat_num_heads = config["gat_num_heads"]
        self.gat_dropout = config["gat_dropout"]
        self.linkPreLossLong = nn.BCEWithLogitsLoss()
        self.linkPreLossShort = nn.BCEWithLogitsLoss()
        self.nodeLong = None
        self.nodeShort = None
        self.LCLTemperature = config["LCLTemperature"]
        self.SCLTemperature = config["SCLTemperature"]
        self.LCLInfoNCE_loss = CLICRecInfoNCELoss(self.LCLTemperature)
        self.SCLInfoNCE_loss = CLICRecInfoNCELoss(self.SCLTemperature)
        self.ln_clusters = config["ln_clusters"]
        self.sn_clusters = config["sn_clusters"]
        '''More details will be made public after the paper is published.'''
        self.GALSTM = GALSTM(self.gat_hidden_dim, self.gat_hidden_dim, self.gat_num_heads)

        mask_mat = (
            torch.arange(self.max_seq_length).to(self.device).view(1, -1)
        )  # init mask
        self.att_list = [4 * self.embedding_size] + self.mlp_hidden_size
        self.dnn_list_fusion = [4 * self.embedding_size] + self.mlp_hidden_size + [1]
        self.dnn_list_pre = [2 * self.embedding_size] + self.mlp_hidden_size + [1]
        self.attentionLong = SequenceAttLayer(
            mask_mat,
            self.att_list,
            activation="Sigmoid",
            softmax_stag=True,
            return_seq_weight=False,
        )
        self.attentionShort = SequenceAttLayer(
            mask_mat,
            self.att_list,
            activation="Sigmoid",
            softmax_stag=True,
            return_seq_weight=False,
        )
        self.user_embedding = nn.Embedding(
            self.n_users, self.hidden_size, padding_idx=0
        )
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        '''More details will be made public after the paper is published.'''
        self.mlp_fusion = MLPLayers(
            self.dnn_list_fusion, activation="sigmoid", dropout=self.dropout_prob, bn=True
        )
        self.mlp_pre = MLPLayers(
            self.dnn_list_pre, activation="sigmoid", dropout=self.dropout_prob, bn=False
        )
        self.gat_conv1 = GATConv(
            in_channels=self.hidden_size,
            out_channels=self.gat_hidden_dim,
            heads=self.gat_num_heads,
            dropout=self.gat_dropout,
            concat=True
        )
        self.gat_conv2 = GATConv(
            in_channels=self.gat_hidden_dim * self.gat_num_heads,
            out_channels=self.hidden_size,
            heads=1,
            dropout=self.gat_dropout,
            concat=False
        )
        self.fc = torch.nn.Linear(self.hidden_size * 2, 1)
        # Link prediction classifier
        self.link_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, user, item_seq, item_seq_len):
        user_emb = self.user_embedding(user)  # q^u_l  #(B,H)
        item_seq_emb = self.item_embedding(item_seq)  # (B,N,H)
        user_emb = self.attentionLong(user_emb, item_seq_emb, item_seq_len)  # (B,1,H)
        utl = user_emb.squeeze(1)  # u^t_l (B,H)
        '''More details will be made public after the paper is published.'''
        uts = self.attentionShort(qust, tmp, item_seq_len).squeeze(1)  # u^t_s (B,H)
        return utl, uts

    def fusion_train(self, item_seq, pos_item, utl, uts, neg_item):  # utl(B,H) uts(B,H)
        item_seq_emb = self.item_embedding(item_seq)  # (B,H)
        pos_item_emb = self.item_embedding(pos_item)  # (B,H)
        neg_item_emb = self.item_embedding(neg_item)  # (B,H)
        _, h_n = self.gru_layers_fusion(item_seq_emb)  # (1,B,H)
        h_n = h_n.squeeze(0)
        all_items = torch.cat([pos_item_emb.unsqueeze(1), neg_item_emb.unsqueeze(1)], dim=1)  # (B,2,H)
        '''More details will be made public after the paper is published.'''
        alpha_flat = self.mlp_fusion(features_flat).squeeze(-1)  # (B*(2),)
        alpha = alpha_flat.view(-1, 2)  # (B,2)
        alpha = alpha.unsqueeze(-1)  # (B,2,1)
        u_fused = alpha * utl_expanded + (1 - alpha) * uts_expanded  # (B, 2, H)
        features_concat_pre = torch.cat([u_fused, all_items], dim=-1)  # (B, 2, 2*H)
        scores_flat = self.mlp_pre(features_concat_pre).squeeze(-1)  # (B*(2),)
        scores = scores_flat.view(-1, 2)  # (B,2)
        pos_score = scores[:, 0]
        neg_score = scores[:, 1]
        return pos_score, neg_score

    def fusion_val(self, item_seq, pos_item, utl, uts):  # utl(B,H) uts(B,H)
        item_seq_emb = self.item_embedding(item_seq)  # (B,H)
        pos_item_emb = self.item_embedding(pos_item)  # (B,H)
        _, h_n = self.gru_layers_fusion(item_seq_emb)  # (B,H)
        h_n = h_n.squeeze(0)
        features_concat_fusion = torch.cat([h_n, utl, uts, pos_item_emb], dim=-1)  # (B, 4*H)
        alpha = self.mlp_fusion(features_concat_fusion)  # (B,1)
        u_fused = alpha * utl + (1 - alpha) * uts  # (B, H)
        features_concat_pre = torch.cat([u_fused, pos_item_emb], dim=-1)  # (B, 2*H)
        scores = self.mlp_pre(features_concat_pre)  # (B,1)
        return scores

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]
        utl, uts = self.forward(user, item_seq, item_seq_len)
        scores, labels = self.fusion_train(item_seq, pos_items, utl, uts, neg_items)
        loss = self.loss_fct(scores, labels)
        linkPreAlpha = self.linkPreAlpha
        clBeta = self.clBeta
        lamda = self.lamda
        return loss + linkPreAlpha * self.get_linkPre_loss_long(user, item_seq,
                                                                item_seq_len) + clBeta * self.get_long_cl_loss(user,
                                                                                                               utl,
                                                                                                               self.ln_clusters,
                                                                                                               self.lrandom,
                                                                                                               self.lninit) + linkPreAlpha * self.get_linkPre_loss_short(
            user, item_seq, item_seq_len, interaction['timestamp_list'], interaction['timestamp'],
            self.session_length) + clBeta * self.get_short_cl_loss(user, utl, self.sn_clusters, self.srandom,
                                                                   self.sninit) + lamda * self.consistency_loss(utl,
                                                                                                                uts)

    def Lbpr(self, a, p, q):
        # Compute dot products ⟨a, q⟩ and ⟨a, p⟩
        dot_aq = torch.sum(a * q, dim=-1)
        dot_ap = torch.sum(a * p, dim=-1)
        # Compute difference and apply softplus
        return F.softplus(dot_aq - dot_ap)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        utl, uts = self.forward(user, item_seq, item_seq_len)
        scores = self.fusion_val(item_seq, pos_items, utl, uts).squeeze(1)
        return scores

    def create_adjacency_matrix(self, user, item_id_list, item_length):
        num_users = len(user)
        adjacency_matrix = torch.zeros((self.batch_size, self.n_items), dtype=torch.int)
        for i in range(num_users):
            clicked_items = item_id_list[i][:item_length[i]]
            adjacency_matrix[i, clicked_items] = 1
        return adjacency_matrix

    def get_linkPre_loss_long(self, user, item_seq, item_seq_len):
        '''More details will be made public after the paper is published.'''
        y = torch.cat((torch.ones((edge_index_pos.shape[1], 1)),
                       torch.zeros((edge_index_neg_selected.shape[1], 1))), dim=0)  
        edge_index = edge_index.to(self.device)
        edge_index_pos = edge_index_pos.to(self.device)
        edge_index_neg_selected = edge_index_neg_selected.to(self.device)
        y = y.to(self.device)
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding.weight
        if len(user) < self.batch_size:
            last_row = user_emb[-1, :]
            num_rows_to_add = self.batch_size - len(user_emb)
            user_emb = torch.cat(
                [user_emb, last_row.unsqueeze(0).expand(num_rows_to_add, -1)], dim=0)
        x = torch.cat([user_emb, item_emb.squeeze(0)], dim=0)
        out = self.linkPreLong(x, edge_index, edge_index_pos=edge_index_pos, edge_index_neg=edge_index_neg_selected)
        linkPreLoss = self.linkPreLossLong(out, y.float())
        return linkPreLoss

    def linkPreLong(self, x, edge_index, edge_index_pos, edge_index_neg, edge_weight=None):
        x = F.relu(self.gat_conv1(x, edge_index, edge_weight))
        x = F.relu(self.gat_conv2(x, edge_index, edge_weight))
        self.nodeLong = x
        edg_index_all = torch.cat((edge_index_pos, edge_index_neg), dim=1)
        Em, Ed = self.pro_data(x, edg_index_all) 
        x = torch.cat((Em, Ed), dim=1)
        x = self.fc(x)
        return x

    def pro_data(self, x, edg_index):
        m_index = edg_index[0]
        d_index = edg_index[1]
        Em = torch.index_select(x, 0, m_index)  
        Ed = torch.index_select(x, 0, d_index)
        return Em, Ed

    def get_long_cl_loss(self, user, utl, n_clusters, random_state, n_init):
        user_emb = self.nodeLong[:len(user)]
        user_emb_np = user_emb.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        cluster_labels = kmeans.fit_predict(user_emb_np)
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(user_emb.device).float()
        cluster_labels = torch.from_numpy(cluster_labels).to(user_emb.device)
        pos_cluster_centers = cluster_centers[cluster_labels]  # (batch_size, hidden_size)
        self.pos_cluster_centers_long = pos_cluster_centers
        cl_loss = self.LCLInfoNCE_loss(utl, pos_cluster_centers, cluster_centers)
        return cl_loss

    def calculate_time_intervals(self, timestamp_list, timestamp,
                                 item_length):  # (batchsize,50) (batchsize,) (batchsize,)
        batch_size, max_seq_len = timestamp_list.shape
        shifted_timestamps = timestamp_list.clone()
        shifted_timestamps[:, :-1] = timestamp_list[:, 1:]
        rightmost_indices = (item_length - 1).clamp(min=0)
        '''More details will be made public after the paper is published.'''
        intervals = shifted_timestamps - timestamp_list
        position_mask = torch.arange(max_seq_len, device=timestamp_list.device).expand(batch_size, -1)
        valid_mask = position_mask < item_length.unsqueeze(1)
        intervals = intervals * valid_mask.float()
        spans = timestamp.unsqueeze(1) - timestamp_list  # (batch_size, max_seq_len)
        spans = spans * valid_mask.float()
        return intervals, spans

    def linkPreShort(self, x, edge_index_pos, edge_index_neg, edge_weight=None):
        self.nodeShort = x
        edg_index_all = torch.cat((edge_index_pos, edge_index_neg), dim=1)
        Em, Ed = self.pro_data(x, edg_index_all) 
        x = torch.cat((Em, Ed), dim=1)
        x = self.fc(x)
        return x

    def split_sessions(self, item_id_list, time_interval, item_length, session_length):
        '''More details will be made public after the paper is published.'''
        return sessions

    def create_adjacency_matrix_session(self, user, item_id_list, item_length, timestamp_list, timestamp,
                                        session_length):
        time_interval, time_spans = self.calculate_time_intervals(timestamp_list, timestamp, item_length)
        sessions = self.split_sessions(item_id_list, time_interval, item_length, session_length)
        num_users = len(user)
        adjacency_matrix1 = torch.zeros((self.batch_size, self.n_items), dtype=torch.int)
        for i in range(num_users):
            clicked_items = sessions[i][0][:len(sessions[i][0])].long()
            adjacency_matrix1[i, clicked_items] = 1
        adjacency_matrix2 = torch.zeros((self.batch_size, self.n_items), dtype=torch.int)
        for i in range(num_users):
            clicked_items = sessions[i][1][:len(sessions[i][1])].long()
            adjacency_matrix2[i, clicked_items] = 1
        adjacency_matrix3 = torch.zeros((self.batch_size, self.n_items), dtype=torch.int)
        for i in range(num_users):
            clicked_items = sessions[i][2][:len(sessions[i][2])].long()
            adjacency_matrix3[i, clicked_items] = 1
        time_intervals1 = []
        time_spans1 = []
        time_intervals2 = []
        time_spans2 = []
        time_intervals3 = []
        time_spans3 = []
        '''More details will be made public after the paper is published.'''
        return adjacency_matrix1, adjacency_matrix2, adjacency_matrix3, time_intervals1, time_intervals2, time_intervals3, time_spans1, time_spans2, time_spans3

    def get_linkPre_loss_short(self, user, item_id_list, item_length, timestamp_list, timestamp, session_length):
        adjacency_matrix1, adjacency_matrix2, adjacency_matrix3, time_intervals1, time_intervals2, time_intervals3, time_spans1, time_spans2, time_spans3 = self.create_adjacency_matrix_session(
            user, item_id_list, item_length, timestamp_list, timestamp, session_length)
        edge_index_pos1 = np.row_stack(np.argwhere(adjacency_matrix1 != 0))
        edge_index_pos1[1, :] += adjacency_matrix1.shape[0]
        edge_index_pos1 = torch.tensor(edge_index_pos1, dtype=torch.long)
        edge_index1 = to_undirected(edge_index_pos1)
        edge_index_pos2 = np.row_stack(np.argwhere(adjacency_matrix2 != 0))
        edge_index_pos2[1, :] += adjacency_matrix2.shape[0]
        edge_index_pos2 = torch.tensor(edge_index_pos2, dtype=torch.long)
        edge_index2 = to_undirected(edge_index_pos2)

        edge_index_pos3 = np.row_stack(np.argwhere(adjacency_matrix3 != 0))
        edge_index_pos3[1, :] += adjacency_matrix3.shape[0]
        edge_index_pos3 = torch.tensor(edge_index_pos3, dtype=torch.long)
        edge_index3 = to_undirected(edge_index_pos3)

        edge_index_neg3 = np.row_stack(np.argwhere(adjacency_matrix3[:len(user)] == 0))
        edge_index_neg3[1, :] += adjacency_matrix3.shape[0]
        edge_index_neg3 = torch.tensor(edge_index_neg3, dtype=torch.long)

        num_pos_edges_number = edge_index_pos3.shape[1]
        selected_neg_edge_indices = torch.randint(high=edge_index_neg3.shape[1], size=(num_pos_edges_number,),
                                                  dtype=torch.long)
        edge_index_neg_selected = edge_index_neg3[:, selected_neg_edge_indices]
        y = torch.cat((torch.ones((edge_index_pos3.shape[1], 1)),
                       torch.zeros((edge_index_neg_selected.shape[1], 1))), dim=0)  
        user_emb = self.user_embedding(user)
        if len(user) < self.batch_size:
            last_row = user_emb[-1, :]
            num_rows_to_add = self.batch_size - len(user_emb)
            user_emb = torch.cat(
                [user_emb, last_row.unsqueeze(0).expand(num_rows_to_add, -1)], dim=0)
        x = torch.cat([user_emb, self.item_embedding.weight], dim=0)
        edge_index1 = edge_index1.to(self.device)
        edge_index2 = edge_index2.to(self.device)
        edge_index3 = edge_index3.to(self.device)
        edge_index_pos3 = edge_index_pos3.to(self.device)
        edge_index_neg_selected = edge_index_neg_selected.to(self.device)
        y = y.to(self.device)
        '''More details will be made public after the paper is published.'''
        H, C = self.GALSTM(X=x, edge_index=edge_index1, intevrals=self.time(time_intervals1.long()),
                           spans=self.time(time_spans1.long()))
        H, C = self.GALSTM(X=x, edge_index=edge_index2, H=H, C=C, intevrals=self.time(time_intervals2.long()),
                           spans=self.time(time_spans2.long()))
        H, C = self.GALSTM(X=x, edge_index=edge_index3, H=H, C=C, intevrals=self.time(time_intervals3.long()),
                           spans=self.time(time_spans3.long()))
        self.node = H
        out = self.linkPreShort(H, edge_index_pos3, edge_index_neg_selected)
        linkPreLoss = self.linkPreLossShort(out, y.float())
        return linkPreLoss

    def get_short_cl_loss(self, user, uts, n_clusters, random_state, n_init):
        user_emb = self.nodeShort[:len(user)]
        user_emb_np = user_emb.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        cluster_labels = kmeans.fit_predict(user_emb_np)
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(user_emb.device).float()
        cluster_labels = torch.from_numpy(cluster_labels).to(user_emb.device)
        pos_cluster_centers = cluster_centers[cluster_labels]  # (batch_size, hidden_size)
        self.pos_cluster_centers_short = pos_cluster_centers
        cl_loss = self.SCLInfoNCE_loss(uts, pos_cluster_centers, cluster_centers)
        return cl_loss

    def consistency_loss(self, utl, uts):  # (batchsize,hiddensize)
        pos_cluster_centers_long = self.pos_cluster_centers_long
        pos_cluster_centers_short = self.pos_cluster_centers_short
        with torch.no_grad():
            sim_ab = torch.nn.functional.cosine_similarity(pos_cluster_centers_long, pos_cluster_centers_short, dim=-1)
        sim_cd = torch.nn.functional.cosine_similarity(utl, uts, dim=-1)
        return (-sim_ab * sim_cd).mean()