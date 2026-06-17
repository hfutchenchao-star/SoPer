import torch
import datasets
import numpy as np
from tqdm import tqdm
from utils.templates import Qwen2PromptTemplate
from data.social_datasets import SocialGraphDataset
import os
import sys


class PersonalDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            main_dataset,
            user_his_emb_map,  # {center_user_id: {user_id: Tensor[num_reviews, D]}}
            llm_tokenizer,
            max_length=3072,
            max_friend_len=8,
            new_tokens=None,
            training=True,
    ):
        self.main_dataset = main_dataset
        self.llm_tokenizer = llm_tokenizer
        self.total_len = len(main_dataset)
        self.training = training
        self.user_his_emb_map = user_his_emb_map
        self.max_friend_len = max_friend_len
        SOCIAL_GRAPH_PATH = "your_path"

        if not os.path.exists(SOCIAL_GRAPH_PATH):
            raise FileNotFoundError(f" Graph file not found: {SOCIAL_GRAPH_PATH}")

        social_graphs = np.load(SOCIAL_GRAPH_PATH, allow_pickle=True).item()
        social_dataset = SocialGraphDataset(SOCIAL_GRAPH_PATH)

        system_prompt = (
            f"Given name and categories of the target item, along with the graph-aware soft prompt of the user, several past reviews from the user's friends and the user's one past review"
            f" (each review including item name, item categories, review stars, review text, review embeddings and each friend's review including a friend influence score), "
            f"as well as the output review stars, "
            f"generate a personalized review text for the target item from the user's perspective. You should consider the friend influence scores, where friends with higher scores should have a stronger impact on the generated review.\n"
            f"Note: [Social Graph Embedding] denotes a graph-aware soft prompt projected from the refined user representation, and [Review Embedding] denotes a soft prompt embedding of the review text. "
            f"[Social Graph Embedding] and [Review Embedding] should serve as semantic hints to guide the generation of personalized item reviews.\n"
        )

        self.pt = Qwen2PromptTemplate(system_prompt)
        self.processed_data = []

        for idx in tqdm(range(self.total_len), desc="Building cold-start dataset"):
            sample = self.main_dataset[idx]
            center_user_id = sample["center_user_id"]
            profile = sample["profile"]
            target = sample["target"]

            categories = target["categories"]
            stars = target["stars"]
            out_str = target["text"]

            user_reviews = [p for p in profile if p["user_id"] == center_user_id]
            friend_reviews = [p for p in profile if p["user_id"] != center_user_id]

            user_reviews = sorted(user_reviews, key=lambda x: x.get("date", ""))
            user_review = user_reviews[-1] if len(user_reviews) > 0 else None

            friend_reviews = sorted(friend_reviews, key=lambda x: x.get("date", ""), reverse=True)
            friends_selected = friend_reviews[:self.max_friend_len]

            if user_review is not None:
                uid = user_review["user_id"]
                emb = None
                if center_user_id not in self.user_his_emb_map:
                    raise KeyError(f"[ERROR] Missing embedding map for center user: {center_user_id}")

                if uid not in self.user_his_emb_map[center_user_id]:
                    raise KeyError(f"[ERROR] Missing embedding for user {uid} under center user {center_user_id}")

                all_embs = self.user_his_emb_map[center_user_id][uid]  # Tensor[num_reviews, D]

                if user_review not in user_reviews:
                    raise ValueError(f"[ERROR] user_review not found in user_reviews for {uid}")

                review_idx = user_reviews.index(user_review)
                if review_idx >= all_embs.shape[0]:
                    raise IndexError(
                        f"[ERROR] review_idx {review_idx} exceeds embedding size {all_embs.shape[0]} "
                        f"for user {uid} under center {center_user_id}"
                    )
                emb = all_embs[-4]
                user_emb = emb
            else:
                raise ValueError(f"[ERROR] user_review is None for center user {center_user_id}")

            friends_by_id = {}
            for rec in profile:
                if rec["user_id"] != center_user_id:
                    friends_by_id.setdefault(rec["user_id"], []).append(rec)
            for fid in friends_by_id:
                friends_by_id[fid] = sorted(friends_by_id[fid], key=lambda x: x.get("date", ""), reverse=True)

            if center_user_id in social_graphs:
                graph_info = social_graphs[center_user_id]
                node_order = graph_info["nodes"] 
                graph_friend_order = [n for n in node_order if n != center_user_id]
            else:
                graph_friend_order = sorted(friends_by_id.keys())


            sorted_fids = [fid for fid in graph_friend_order if fid in friends_by_id]

            K = min(self.max_friend_len, len(sorted_fids)) if len(sorted_fids) > 0 else 0
            primary_fids = sorted_fids[:K]

            if K == 0:
                raise ValueError(f"[ERROR] No friends found for center user {center_user_id}")

            friend_all_embs = {} 
            friend_mean_embs = {}  

            for fid in primary_fids:
                if center_user_id not in self.user_his_emb_map or fid not in self.user_his_emb_map[center_user_id]:
                    raise KeyError(f"[ERROR] Missing emb for friend {fid} under center {center_user_id}")
                all_embs = self.user_his_emb_map[center_user_id][fid]
                if not hasattr(all_embs, "ndim") or all_embs.ndim != 2:
                    raise ValueError(
                        f"[ERROR] Embedding for friend {fid} must be 2D, got {getattr(all_embs, 'shape', None)}")
                friend_all_embs[fid] = all_embs
                friend_mean_embs[fid] = all_embs.mean(dim=0, keepdim=True)

            slots = []  
            for fid in primary_fids:
                reviews_desc = friends_by_id[fid]  
                reviews_asc = sorted(reviews_desc, key=lambda x: x.get("date", ""))
                latest_review = reviews_desc[0]
                review_idx_asc = reviews_asc.index(latest_review)
                all_embs = friend_all_embs[fid]
                if review_idx_asc >= all_embs.shape[0]:
                    raise IndexError(
                        f"[ERROR] friend {fid} emb count {all_embs.shape[0]} < needed index {review_idx_asc}"
                    )
                slots.append((fid, latest_review, all_embs[review_idx_asc]))

            round_idx = 1 
            while len(slots) < self.max_friend_len:
                filled_in_this_round = False
                for fid in primary_fids:
                    reviews_desc = friends_by_id[fid]
                    if round_idx < len(reviews_desc):
                        r = reviews_desc[round_idx]  
                        reviews_asc = sorted(reviews_desc, key=lambda x: x.get("date", ""))
                        review_idx_asc = reviews_asc.index(r)
                        all_embs = friend_all_embs[fid]
                        if review_idx_asc >= all_embs.shape[0]:
                            raise IndexError(
                                f"[ERROR] friend {fid} emb count {all_embs.shape[0]} < idx {review_idx_asc}"
                            )
                        slots.append((fid, r, all_embs[review_idx_asc]))
                        filled_in_this_round = True
                        if len(slots) >= self.max_friend_len:
                            break
                if not filled_in_this_round:
                    last_fid, last_r, last_e = slots[-1]
                    while len(slots) < self.max_friend_len:
                        slots.append((last_fid, last_r, last_e))
                    break
                round_idx += 1

            # ======== 目标商家信息 ========
            inp_str = f"[Target Item Name]: {name}\n"
            inp_str += f"[Target Item Categories]: {', '.join(categories)}\n"

            # ======== 社交图嵌入 ========
            inp_str += "[Social Graph Embedding]: <graph_token_start>[SOCIAL_GRAPH_TOKEN]<graph_token_end>\n"

            # ======== 好友评论部分 ========
            inp_str += "[Reviews from User's Friends]:\n"
            for j in range(self.max_friend_len):
                fid, fr, _ = slots[j]
                friend_text = clean_text(fr["text"])
                friend_bname = fr.get("business_name")
                friend_bcats = ", ".join(fr.get("categories"))
                friend_battr = fr.get(
                    "business_attributes_text") or "No detailed business attributes are available for this place."

                inp_str += (
                    f"- [Friend Review {j + 1}]:\n"
                    f"  - [Friend Influence Score]: 1.0\n"
                    f"  - [Review Item Name]: {friend_bname}\n"
                    f"  - [Review Item Categories]: {friend_bcats}\n"
                    f"  - [Review Stars]: {fr['stars']}\n"
                    f"  - [Review Text]: {friend_text}\n"
                    f"  - [Review Embedding]: <friend_token_start>[FRIEND_TOKEN_{j}]<friend_token_end>\n"
                )
            # ======== 用户自身评论 ========
            inp_str += "[User's Review]:\n"
            if user_review:
                user_text = clean_text(user_review["text"])
                user_bname = user_review.get("business_name")
                user_bcats = ", ".join(user_review.get("categories"))
                user_battr = user_review.get(
                    "business_attributes_text") or "No detailed business attributes are available for this place."

                inp_str += (
                    f"- [Review Item Name]: {user_bname}\n"
                    f"- [Review Item Categories]: {user_bcats}\n"
                    f"- [Review Stars]: {user_review['stars']}\n"
                    f"- [Review Text]: {user_text}\n"
                    f"- [Review Embedding]: <user_token_start>[USER_TOKEN_0]<user_token_end>\n"
                )
            else:
                inp_str += "None\n"

            # ======== 输出目标 ========
            inp_str += f"[Output Target Review Stars]: {stars}\n"

            # ========== 调用 Prompt 模板 ==========
            inp_str = self.pt.build_prompt(user_message=inp_str)

            # === Step 5. Tokenize ===
            total_max_length = max_length + 2048 + 1
            inputs = self.llm_tokenizer(
                inp_str,
                add_special_tokens=False,
                truncation=False  
            )
            input_len = len(inputs["input_ids"])

            targets = self.llm_tokenizer(
                out_str,
                add_special_tokens=False,
                truncation=False
            )
            inputs_id = inputs['input_ids'] + targets['input_ids'] + [self.llm_tokenizer.eos_token_id]
            if len(inputs_id) > max_observed_len:
                max_observed_len = len(inputs_id)
            if len(inputs_id) > 3272:
                count_over_8000 += 1
                print(f"跳过超长样本：{len(inputs_id)} tokens (>3272)")
                continue
            all_embs_tensor = [user_emb]  
            for j in range(self.max_friend_len):
                fid, fr, emb = slots[j]
                all_embs_tensor.append(emb)
            all_embs_tensor = torch.stack(all_embs_tensor, dim=0)

            if center_user_id in social_dataset.keys:
                ind = social_dataset.keys.index(center_user_id)
                graph_data = social_dataset.get(ind)  
            else:
                raise ValueError(
                    f" 未找到中心用户 {center_user_id} 的社交图数据！请检查 social_graphs.npy 是否包含该用户。")

            out_str = clean_text(out_str)
            x = graph_data.x.clone().tolist()
            center_idx = graph_data.center_local.item()

            center_emb = all_embs_tensor[0].tolist()

            if len(x[center_idx]) == len(center_emb) + 1:
                x[center_idx] = [x[center_idx][0]] + center_emb
            else:
                x[center_idx] = center_emb
            data = {
                "inp_str": inp_str,
                "out_str": out_str,
                "center_user_id": center_user_id,
                "embeddings": all_embs_tensor.tolist(),  
                "graph_data": {
                    "x": x,
                    "edge_index": graph_data.edge_index.tolist(),
                    "center_local": center_idx,
                    "center_uid": graph_data.center_uid,
                }
            }
            self.processed_data.append(data)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        return self.processed_data[idx]


def convert_to_dataset(dataset):
    def gen():
        for data in dataset:
            yield data

    return datasets.Dataset.from_generator(gen)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.split())

    return text.strip()

