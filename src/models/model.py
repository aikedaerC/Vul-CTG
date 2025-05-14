# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.momu import MoMu
from src.models.CTG_Former import BertConfig, BertLMHeadModel

class DeepTextCNN(nn.Module):
    def __init__(self, embed_dim=768, num_classes=1, dropout=0.5):
        super(DeepTextCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * (384 // 8), 256)  # 384经过3次2倍池化：384/2/2/2 = 48
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, embed_dim, seq_len)
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x)); x = self.pool3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return torch.sigmoid(self.output(x)).squeeze(1)  # (B,)


class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class CTGFormer(nn.Module):
    def __init__(self, num_query_token, vision_graph_width, cross_attention_freq = 2):
        super().__init__()
        encoder_config = BertConfig.from_pretrained("/home/aikedaer/.cache/huggingface/hub/models--allenai--scibert_scivocab_uncased/snapshots/24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1")
        encoder_config.encoder_width = vision_graph_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        self.Qformer = BertLMHeadModel.from_pretrained(
            "allenai/scibert_scivocab_uncased", config=encoder_config
        )
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)


class CTGModel(nn.Module):
    def __init__(self, config = None, fp = False, task = None, device = None):
        super().__init__()

        self.ln_text  = LayerNorm(768)
        self.graph_encoder  = GraphEncoder(config)
        self.ln_graph  = LayerNorm(self.graph_encoder.hidden_size)
        

        ctgformer = CTGFormer(384,768)
        self.ctg_former  = ctgformer.Qformer
        self.query_tokens  = ctgformer.query_tokens

        self.gtm_head = nn.Linear(self.ctg_former.config.hidden_size, 2) # # 768
    
        embed_dim = 256

        self.graph_proj = nn.Linear(self.ctg_former.config.hidden_size, embed_dim) # 768, 256
        self.text_proj = nn.Linear(self.ctg_former.config.hidden_size, embed_dim) # 768, 256
        self.model_freeze()
        self.device = device
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.task = ['gtm','gtc']

    def model_freeze(self):
        
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
            
    def get_ctg_former_outputs(self, mol):

        batch_size = len(mol.func)
        inputs = {
            "graph2d": mol,
            "smiles": mol.func,
            "isoSMILES": {
                "input_ids": mol.input_ids.view(batch_size,-1),
                "attention_mask": mol.attention_mask.view(batch_size,-1)
            },
            "isosmiles_labels": mol.y
        }
        text = inputs["isoSMILES"]
        mol = inputs

        input_tensors = []
        
        language_model_inputs_text = self.get_text_ctg_former_features(mol, text) # 12,384,768
        input_tensors.append(language_model_inputs_text)

        language_model_inputs_graph = self.get_graph_ctg_former_features(mol) # 12,384,768
        input_tensors.append(language_model_inputs_graph)

        # Stack tensors along a new dimension and compute the mean
        qformer_outputs = torch.stack(input_tensors).mean(dim=0)
        return qformer_outputs, mol
    
    def get_ctg_former_embeddings(self, mol):

        batch_size = len(mol.func)
        inputs = {
            "graph2d": mol,
            "smiles": mol.func,
            "isoSMILES": {
                "input_ids": mol.input_ids.view(batch_size,-1),
                "attention_mask": mol.attention_mask.view(batch_size,-1)
            },
            "isosmiles_labels": mol.y
        }
        text = inputs["isoSMILES"]
        mol = inputs

        input_tensors = []
        
        language_model_inputs_text = self.get_text_ctg_former_features(mol, text) # 12,384,768
        input_tensors.append(language_model_inputs_text)

        language_model_inputs_graph = self.get_graph_ctg_former_features(mol) # 12,384,768
        input_tensors.append(language_model_inputs_graph)

        # Stack tensors along a new dimension and compute the mean
        qformer_outputs = torch.stack(input_tensors).mean(dim=0)
        return language_model_inputs_text, language_model_inputs_graph, qformer_outputs
    
    def get_graph_ctg_former_features(self, mol):
        graph_embeds = self.ln_graph(self.graph_encoder(mol))
        graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long).to(
            graph_embeds.device
        )
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        #print(f"graph_query_tokens:{query_tokens.shape}")    
        #query_tokens:torch.Size([4, 32, 768])
        query_outputs = self.ctg_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_atts,
            modal='graph',
            is_decoder=False
        )
        query_output = query_outputs.last_hidden_state
        #print(f"graph_query_output.shape :{query_output.shape}")
        language_model_inputs_graph = query_output
        return language_model_inputs_graph
    
    def get_text_ctg_former_features(self, mol, text):

        text_embeds = self.ln_text(self.ctg_former.bert(
                text['input_ids'],
                attention_mask=text['attention_mask'],
                return_dict=True,
            ).last_hidden_state)
        text_attention_mask = torch.ones(text_embeds.size()[:-1], dtype=torch.long, device=text_embeds.device)
        query_tokens = self.query_tokens.expand(text_embeds.shape[0], -1, -1)
         
        query_outputs = self.ctg_former.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=text_embeds,
            encoder_attention_mask=text_attention_mask,
            modal='cs_text',
            is_decoder=False
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs_text = query_output
        return language_model_inputs_text
    
   
    def forward(self, mol):
        # DataBatch(x=[26240, 101], edge_index=[10932], edge_attr=[10932], y=[128], func=[128], input_ids=[131072], attention_mask=[131072], text=[128], batch=[26240], ptr=[129])
        loss = 0
      
        batch_size = len(mol.func)
        inputs = {
            "graph2d": mol,
            "smiles": mol.func,
            "isoSMILES": {
                "input_ids": mol.input_ids.view(batch_size,-1),
                "attention_mask": mol.attention_mask.view(batch_size,-1)
            },
            "isosmiles_labels": mol.y
        }
        text = inputs["isoSMILES"]
        mol = inputs

        graph_embeds = self.ln_graph(self.graph_encoder(mol)) # (12,50,768)
        graph_atts = torch.ones(graph_embeds.size()[:-1], dtype=torch.long).to(
            graph_embeds.device
        )
        graph_targets = torch.arange(batch_size).to(graph_embeds.device)
          
        text_output = self.ctg_former.bert(
            text['input_ids'],
            attention_mask=text['attention_mask'],
            return_dict=True,
        ) # (12,128,768)
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        ) # (12,256)

        if("gtm" in self.task):
            
            # Initializing lists to hold the original and negative samples
            graph_embeds_list = []
            text_input_ids_list = []
            text_attention_mask_list = []

            for i in range(graph_embeds.shape[0]):
                # Original samples
                graph_embeds_list.append(graph_embeds[i])
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

                # Negative samples (neg_text_input_ids corresponds to image_embeds)
                neg_text_input_ids = text['input_ids'][i-1] if i == graph_embeds.shape[0] - 1 else text['input_ids'][i+1]
                neg_text_attention_mask = text['attention_mask'][i-1] if i == graph_embeds.shape[0] - 1 else text['attention_mask'][i+1]
                text_input_ids_list.append(neg_text_input_ids)
                text_attention_mask_list.append(neg_text_attention_mask)
                graph_embeds_list.append(graph_embeds[i])

                # Negative samples (text_input_ids corresponds to neg_image_embeds)
                neg_graph_embeds = graph_embeds[i-1] if i == graph_embeds.shape[0] - 1 else graph_embeds[i+1]
                graph_embeds_list.append(neg_graph_embeds)
                text_input_ids_list.append(text['input_ids'][i])
                text_attention_mask_list.append(text['attention_mask'][i])

            # Stack all samples into two large tensors
            graph_embeds_all = torch.stack(graph_embeds_list, dim=1).reshape(-1, graph_embeds.size(1), graph_embeds.size(2)) # (36,54,768)
            text_input_ids_all = torch.stack(text_input_ids_list, dim=1).reshape(-1, text['input_ids'].size(1)) # (36,180)
            text_attenetion_mask_all = torch.stack(text_attention_mask_list, dim=1).reshape(-1, text['attention_mask'].size(1)) # (36,180)
            # Create image attention masks for the concatenated tensor
            graph_atts_all = torch.ones(graph_embeds_all.size()[:-1], dtype=torch.long).to(
                graph_embeds_all.device
            ) # (36,54)
            query_tokens_gtm = self.query_tokens.expand(text_input_ids_all.shape[0], -1, -1) # (36,384,768)
            query_atts_gtm = torch.ones(query_tokens_gtm.size()[:-1], dtype=torch.long).to(
                graph_embeds_all.device
            ) #(36,384)
            attention_mask_all = torch.cat([query_atts_gtm, text_attenetion_mask_all], dim=1) # (36,564)
            
            output_gtm = self.ctg_former.bert(
                text_input_ids_all,
                query_embeds=query_tokens_gtm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                modal = 'graph',
                return_dict=True,
            ) # last_hidden_state (36,564,768)
            gtm_embeddings = output_gtm.last_hidden_state[:, : query_tokens_gtm.size(1), :] # (36,384,768)
            
            gtm_logit = self.gtm_head(gtm_embeddings) # (36,384,2)
            gtm_logit = gtm_logit.mean(dim=1) # (36,2)
            #itm_logit = self.itm_head(itm_embeddings)
            # Create labels: 1 for the original samples, 0 for the negative samples
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size * 2)], dim=0).long().to(gtm_logit.device) #(36)

            # Calculate cross entropy loss
            loss_gtm = F.cross_entropy(gtm_logit, labels)

            loss = loss+loss_gtm
         
        if("gtc" in self.task):
            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1) # graph_embeds (12,54,768)  --->  (12,384,768)

            query_output = self.ctg_former.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_atts,
                modal = 'graph',
                return_dict=True,
            ) # last_hidden_states: (12,384,768)
            
            graph_feats = F.normalize(
                self.graph_proj(query_output.last_hidden_state), dim=-1
            ) # (12,384,256)

            sim_q2t = torch.matmul(
                graph_feats.unsqueeze(1), text_feat.unsqueeze(-1) # (12,1,384,256), (12,256,1)
            ).squeeze() # (12,12,384)
                # [batch_size, batch_size*num_gpu, num_query_tokens]

                # image-text similarity: aggregate across all query tokens
            sim_g2t, _ = sim_q2t.max(-1) # (12,12)
            sim_g2t = sim_g2t / self.temp

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), graph_feats.permute(0, 2, 1) # (12,1,1,256), (12,256,384)
            ).squeeze() # (12,12,384)

                # text-image similarity: aggregate across all query tokens
            sim_t2g, _ = sim_t2q.max(-1)
            sim_t2g = sim_t2g / self.temp  # [batch_size, batch_size*num_gpu]
            loss_gtc = (
                F.cross_entropy(sim_g2t, graph_targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2g, graph_targets, label_smoothing=0.1)
                ) / 2

            loss = loss+loss_gtc
            
        loss = loss/len(self.task)
        return loss
    
class GraphEncoder(nn.Module):
    def __init__(self, config = None):
        super().__init__()
        self.graph2d_encoder = MoMu(config["graph"]).graph_encoder
    
        for param in self.graph2d_encoder.parameters():
            param.requires_grad = False
        
        self.num_features = 300
        self.hidden_size = 768
        self.fc_hidden = nn.Linear(self.num_features, self.hidden_size)
    
    def forward(self, mol):
        graph_feats, node_feats, node_feats_mask = self.graph2d_encoder(mol["graph2d"])
        node_feats = self.fc_hidden(node_feats)
        return node_feats
 

