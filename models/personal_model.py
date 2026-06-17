import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Qwen2ForCausalLM, PretrainedConfig, AutoTokenizer
from models.gcn_gib import SocialGCN_GBSR
from torch_geometric.data import Data
from tqdm import tqdm

EMBED_SIZE = 1024
HIDDEN_SIZE = 1024
MAX_FRIEND_LEN = 8
max_length = 3572
max_friend_len = 8
training = True


class DEPModel(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.gcn = SocialGCN_GBSR(
            in_channels=1025,     
            hidden_channels=512,   
            out_channels=512,     
            num_layers=3,         
            dropout=0.1,           
            gib_sigma=0.5,        
            projector_dim=1024,     
            temperature=0.2,       
        )

        self.gcn.to_empty(device=device)
        
        self.align_mlp_user = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, config.hidden_size, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size, dtype=torch.float32),
        )
        self.align_mlp_user.to_empty(device=device)
        
        self.align_mlp_friend = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, config.hidden_size, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size, dtype=torch.float32),
        )
        self.align_mlp_friend.to_empty(device=device)
        
        self.align_mlp_graph = nn.Sequential(
            nn.Linear(1024, config.hidden_size, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size, dtype=torch.float32),
        )
        self.align_mlp_graph.to_empty(device=device)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            embeddings: Optional[torch.FloatTensor] = None,
            graph_data: Optional[Union[list, dict, Data]] = None,
            inp_str: Optional[Union[str, list]] = None,
            out_str: Optional[Union[str, list]] = None,
            center_user_id: Optional[Union[str, list]] = None,
            training: bool = True,
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if embeddings is None and hasattr(self, '_temp_embeddings'):
            embeddings = self._temp_embeddings
        if graph_data is None and hasattr(self, '_temp_graph_data'):
            graph_data = self._temp_graph_data
        if inp_str is None and hasattr(self, '_temp_inp_strs'):
            inp_str = self._temp_inp_strs
        if embeddings is None:
            raise ValueError("embeddings is None! 必须在 collate_fn 中提供。")
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if embeddings is None:
            raise ValueError("embeddings is None! 必须在 collate_fn 中提供。")
        
        if isinstance(embeddings, list):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        embeddings = embeddings.to(self.device)

        def _to_pyg_data(g):
            if g is None:
                print("图数据不存在")
                return None, None, None
            if isinstance(g, Data):
                center_local = int(g.center_local.item()) if hasattr(g, "center_local") else 0
                node_names = getattr(g, "node_names", None)
                return g, center_local, node_names
            x = torch.tensor(g["x"], dtype=torch.bfloat16) if isinstance(g["x"], list) else g["x"]
            ei = torch.tensor(g["edge_index"], dtype=torch.long) if isinstance(g["edge_index"], list) else g[
                "edge_index"]
            data = Data(x=x, edge_index=ei)
            data.center_uid = g.get("center_uid", None)
            cl = g.get("center_local", 0)
            data.center_local = torch.tensor(cl, dtype=torch.long)
            node_names = g.get("node_names", None)
            return data, int(cl), node_names

        edge_mask = None
        graph_emb = None
        hsic_loss_total = 0.0
        num_graphs = 0

        if graph_data is None:
            raise ValueError("graph_data is None!")

        if isinstance(graph_data, list):
            graph_emb_list = []
            mask_list = []

            for g in graph_data:
                g_data, center_local, node_names = _to_pyg_data(g)
                if g_data is None:
                    graph_emb_list.append(torch.zeros(256, device=self.device, dtype=torch.bfloat16))
                    mask_list.append(torch.zeros(max_friend_len, device=self.device, dtype=torch.float32))
                    continue

                g_res = self.gcn.forward(
                    g_data.x.to(self.device),
                    g_data.edge_index.to(self.device),
                    batch_idx=None,
                    return_all=True
                )

                emb_original = g_res["emb_before"]
                emb_masked = g_res["emb_after"]
                edge_mask_raw = g_res["edge_mask"]  

                center_emb_mask = emb_masked[center_local].unsqueeze(0)
                hsic_loss = self.gcn.compute_hsic_loss(emb_original, emb_masked)
                hsic_loss_total += hsic_loss
                num_graphs += 1

                graph_emb_proj = self.gcn.llm_projector(center_emb_mask)
                graph_emb_list.append(graph_emb_proj.squeeze(0))

                ei = g_data.edge_index.to(self.device)
                src, dst = ei[0], ei[1]
                num_nodes = g_data.x.size(0)
                friends_order = [i for i in range(num_nodes) if i != center_local]

                weights_in_order = []
                for i in friends_order:
                    idx_0i = torch.where((src == center_local) & (dst == i))[0]
                    idx_i0 = torch.where((src == i) & (dst == center_local))[0]
                    ws = []
                    if idx_0i.numel() > 0:
                        ws.extend(edge_mask_raw[idx_0i].tolist())
                    if idx_i0.numel() > 0:
                        ws.extend(edge_mask_raw[idx_i0].tolist())
                    avg_w = float(sum(ws) / len(ws)) if len(ws) > 0 else 0.0
                    weights_in_order.append(avg_w)

                if len(weights_in_order) == 0:
                    pad_weights = [0.0] * max_friend_len
                else:
                    pad_weights = (weights_in_order * ((max_friend_len // len(weights_in_order)) + 1))[:max_friend_len]

                mask_list.append(torch.tensor(pad_weights, dtype=torch.float32, device=self.device))

            graph_emb = torch.stack(graph_emb_list, dim=0)
            edge_mask = torch.stack(mask_list, dim=0)
            hsic_loss_avg = hsic_loss_total / max(num_graphs, 1)

        else:
            g_data, center_local, node_names = _to_pyg_data(graph_data)
            if g_data is None:
                raise ValueError("单样本图数据为空。")

            g_res = self.gcn.forward(
                g_data.x.to(self.device),
                g_data.edge_index.to(self.device),
                batch_idx=None,
                return_all=True
            )

            emb_original = g_res["emb_before"]  
            emb_masked = g_res["emb_after"]  
            edge_mask_raw = g_res["edge_mask"]  

            center_emb_mask = emb_masked[center_local].unsqueeze(0)
            hsic_loss_avg = self.gcn.compute_hsic_loss(emb_original, emb_masked)
            graph_emb = self.gcn.llm_projector(center_emb_mask)

            ei = g_data.edge_index.to(self.device)
            src, dst = ei[0], ei[1]
            num_nodes = g_data.x.size(0)
            friends_order = [i for i in range(num_nodes) if i != center_local]

            weights_in_order = []
            for i in friends_order:
                idx_0i = torch.where((src == center_local) & (dst == i))[0]
                idx_i0 = torch.where((src == i) & (dst == center_local))[0]
                ws = []
                if idx_0i.numel() > 0:
                    ws.extend(edge_mask_raw[idx_0i].tolist())
                if idx_i0.numel() > 0:
                    ws.extend(edge_mask_raw[idx_i0].tolist())
                avg_w = float(sum(ws) / len(ws)) if len(ws) > 0 else 0.0
                weights_in_order.append(avg_w)

            if len(weights_in_order) == 0:
                pad_weights = [0.0] * max_friend_len
            else:
                pad_weights = (weights_in_order * ((max_friend_len // len(weights_in_order)) + 1))[:max_friend_len]

            edge_mask = torch.tensor(pad_weights, dtype=torch.float32, device=self.device).unsqueeze(0)

        processed_inp_strs = []

        for bidx in range(edge_mask.size(0)):
            weights = edge_mask[bidx].detach().cpu().tolist()
            text = inp_str[bidx] if isinstance(inp_str, list) else inp_str

            if text is None:
                processed_inp_strs.append("")
                continue

            for w in weights:
                text = text.replace("[Friend Influence Score]: 1.0", f"[Friend Influence Score]: {w:.4f}", 1)
            processed_inp_strs.append(text)

        all_input_ids, all_attention_masks, all_labels = [], [], []

        for idx, text in enumerate(processed_inp_strs):
            if not text:
                continue

            inputs = self.llm_tokenizer(
                text=text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )

            if training:
                out_text = out_str[idx] if isinstance(out_str, list) else out_str
                targets = self.llm_tokenizer(
                    text=out_text,  
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )

                total_max_length = max_length + 2048 +1
                input_id = inputs["input_ids"] + targets["input_ids"] + [self.llm_tokenizer.eos_token_id]
                attention_mask_list = inputs["attention_mask"] + targets["attention_mask"] + [1]
                labels_list = [-100] * len(inputs["input_ids"]) + targets["input_ids"] + [
                    self.llm_tokenizer.eos_token_id]
                max_len = total_max_length
            else:

                input_id = inputs["input_ids"]
                attention_mask_list = inputs["attention_mask"]
                labels_list = [-100] * len(input_id)
                max_len = max_length

            pad_len = max_len - len(input_id)
            if pad_len > 0:
                input_id = [self.llm_tokenizer.pad_token_id] * pad_len + input_id
                attention_mask_list = [0] * pad_len + attention_mask_list
                labels_list = [-100] * pad_len + labels_list
            elif pad_len < 0:
                input_id = input_id[-max_len:]
                attention_mask_list = attention_mask_list[-max_len:]
                labels_list = labels_list[-max_len:]

            all_input_ids.append(input_id)
            all_attention_masks.append(attention_mask_list)
            all_labels.append(labels_list)

        input_id = torch.tensor(all_input_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(all_attention_masks, dtype=torch.long, device=self.device)
        labels = torch.tensor(all_labels, dtype=torch.long, device=self.device)

        inputs_embs = self.model.get_input_embeddings()(input_id)

        

        user_emb = embeddings[:, 0:1, :]
        friend_emb = embeddings[:, 1:, :]

        # === Graph embedding ===
        if graph_emb is not None:
            orig_dtype = graph_emb.dtype
            graph_emb = graph_emb.to(torch.float32)              
            self.align_mlp_graph = self.align_mlp_graph.to(torch.float32) 

            graph_emb = self.align_mlp_graph(graph_emb)        
            graph_emb = graph_emb.to(orig_dtype)            
            graph_emb = graph_emb.unsqueeze(1)
        graph_token_id = self.llm_tokenizer.convert_tokens_to_ids("[SOCIAL_GRAPH_TOKEN]")
        friend_token_ids = self.llm_tokenizer.convert_tokens_to_ids([f"[FRIEND_TOKEN_{i}]" for i in range(MAX_FRIEND_LEN)])

        if graph_emb is not None:

            graph_emb = graph_emb.to(inputs_embs.dtype)
            for bidx in range(inputs_embs.shape[0]):
                mask = (input_id[bidx] == graph_token_id)
                if mask.any():
                    inputs_embs[bidx][mask] = graph_emb[bidx, 0, :]


        friend_emb = self.align_mlp_user.to(torch.float32)(friend_emb.to(torch.float32))
        friend_emb = friend_emb.to(inputs_embs.dtype)
        for bidx in range(inputs_embs.shape[0]):
            for i in range(MAX_FRIEND_LEN):
                mask = (input_id[bidx] == friend_token_ids[i])
                if mask.any():
                    inputs_embs[bidx][mask] = friend_emb[bidx, i, :]

        # === LLM Forward ===
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embs,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # === Loss ===
        llm_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

  
        loss = llm_loss + hsic_loss_avg*0.5

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
            cache_dir: Optional[Union[str, os.PathLike]] = None,
            ignore_mismatched_sizes: bool = False,
            force_download: bool = False,
            local_files_only: bool = False,
            token: Optional[Union[str, bool]] = None,
            revision: str = "main",
            use_safetensors: bool = None,
            training: bool = False,
            tokenizer: Optional[AutoTokenizer] = None,
            **kwargs
    ):
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs
        )
        model.llm_tokenizer = tokenizer
        if training:
            for name, param in model.named_parameters():
                if not any(k in name for k in ["align_mlp_user", "align_mlp_friend", "align_mlp_graph", "gcn"]):
                    param.requires_grad = False
        return model
    
    def loss_function(self, logits, labels, vocab_size, **kwargs):
        if labels is None:
            return torch.tensor(0.0, device=logits.device)
        
        if logits.size(1) != labels.size(1):
            min_len = min(logits.size(1), labels.size(1))
            logits = logits[:, :min_len, :]
            labels = labels[:, :min_len]
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        
        return loss
