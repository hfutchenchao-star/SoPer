import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def kernel_matrix(x, sigma):
    dist_sq = torch.cdist(x, x, p=2).pow(2)
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def hsic(Kx, Ky, m):
    """
    Computes:
        1 / (m - 1)^2 * Tr(Kx H Ky H)

    Since m = n + 1, the normalization coefficient corresponds to 1 / n^2.
    """
    Kxy = torch.mm(Kx, Ky)

    h = (
        torch.trace(Kxy) / (m ** 2)
        + Kx.mean() * Ky.mean()
        - 2 * Kxy.mean() / m
    )

    return h * (m / (m - 1)) ** 2


class EdgeMaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


class SocialGCN_GBSR(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hidden_channels=64,
        out_channels=64,
        num_layers=3,
        dropout=0.1,
        projector_dim=256,
        temperature=0.2,
        edge_bias=0.0,
        detach_mask=False,
        gib_sigma=0.5,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")

        # Layer-wise mean pooling requires identical embedding dimensions.
        if hidden_channels != out_channels:
            raise ValueError(
                "hidden_channels must equal out_channels because the "
                "embeddings from different GNN layers are mean-pooled."
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.edge_bias = edge_bias
        self.detach_mask = detach_mask
        self.gib_sigma = gib_sigma

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.gcn_convs = nn.ModuleList(
            [
                GCNConv(
                    hidden_channels,
                    hidden_channels if i < num_layers - 1 else out_channels,
                    add_self_loops=False,
                    normalize=False,
                )
                for i in range(num_layers)
            ]
        )

        self.activate = nn.ReLU()

        self.linear_1 = nn.Linear(
            2 * hidden_channels,
            hidden_channels,
            bias=True,
        )
        self.linear_2 = nn.Linear(
            hidden_channels,
            1,
            bias=True,
        )

        self.llm_projector = nn.Sequential(
            nn.Linear(out_channels, projector_dim),
            nn.LayerNorm(projector_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projector_dim, projector_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def graph_learner(self, node_emb, edge_index, is_training=True):
        row, col = edge_index

        # [E, 2 * hidden_channels]
        cat = torch.cat(
            [node_emb[row], node_emb[col]],
            dim=1,
        )

        h = self.activate(self.linear_1(cat))
        logit = self.linear_2(h).view(-1)

        if is_training:
            # Equation (2):
            # log(delta / (1 - delta)), delta ~ Uniform(0, 1).
            #
            # Sample in float32 to avoid infinities under bfloat16
            # or float16 mixed-precision training.
            delta = torch.rand(
                logit.shape,
                device=logit.device,
                dtype=torch.float32,
            )

            logistic_noise = torch.logit(
                delta,
                eps=1e-6,
            ).to(logit.dtype)

            s = (logit + logistic_noise) / self.temperature
            mask = torch.sigmoid(s) + self.edge_bias

        else:
            mask = torch.sigmoid(logit) + self.edge_bias

        if self.detach_mask:
            mask = mask.detach()

        return mask

    def forward(
        self,
        x,
        edge_index,
        batch_idx=None,
        return_all=False,
    ):
        x = x.to(next(self.input_proj.parameters()).dtype)

        # [N, hidden_channels]
        emb_before = self.input_proj(x)
        emb_before = F.dropout(
            emb_before,
            p=self.dropout,
            training=self.training,
        )

        edge_mask = self.graph_learner(
            emb_before,
            edge_index,
            is_training=self.training,
        )

        all_emb = [emb_before]
        h = emb_before

        for layer_idx in range(self.num_layers):
            h = self.gcn_convs[layer_idx](
                h,
                edge_index,
                edge_weight=edge_mask,
            )

            if layer_idx < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(
                    h,
                    p=self.dropout,
                    training=self.training,
                )

            all_emb.append(h)

        # [N, num_layers + 1, hidden_channels]
        emb_after = torch.stack(
            all_emb,
            dim=1,
        ).mean(dim=1)

        if not return_all:
            return emb_after

        return {
            "emb_before": emb_before,
            "emb_after": emb_after,
            "edge_mask": edge_mask,
        }

    def extract_center_embeddings(self, batch_data):
        result = self.forward(
            batch_data.x,
            batch_data.edge_index,
            batch_data.batch,
            return_all=True,
        )

        # Fixed: forward() does not return a key named "emb".
        emb = result["emb_after"]

        ptr = batch_data.ptr
        num_graphs = batch_data.num_graphs

        centers = []

        for graph_idx in range(num_graphs):
            center_local = int(
                batch_data[graph_idx].center_local.item()
            )
            global_idx = int(ptr[graph_idx].item()) + center_local
            centers.append(emb[global_idx])

        centers = torch.stack(centers)

        return centers, result["edge_mask"]

    def get_llm_prompts(self, batch_data):
        centers, edge_mask = self.extract_center_embeddings(
            batch_data
        )

        prompt = self.llm_projector(centers)

        return prompt, {
            "edge_mask": edge_mask,
        }

    def compute_hsic_loss(self, emb_before, emb_after):
        m = emb_before.size(0)

        if m < 2:
            return emb_before.new_zeros(())

        Kx = kernel_matrix(
            emb_before,
            self.gib_sigma,
        )
        Ky = kernel_matrix(
            emb_after,
            self.gib_sigma,
        )

        return hsic(Kx, Ky, m)
