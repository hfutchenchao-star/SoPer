import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from datasets import Dataset
from collections import defaultdict
from data.personal_dataset import convert_to_dataset, PersonalDataset
import torch

def get_max_friend_len(json_path_list):
    max_friends = 0
    for path in json_path_list:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for sample in data:
                profile = sample["profile"]
                center = sample["center_user_id"]
                friends = set(p["user_id"] for p in profile if p["user_id"] != center)
                max_friends = max(max_friends, len(friends))
    return max_friends


# ======================
# 初始化与路径配置
# ======================
set_seed(42)

DATA_DIR = "your_path"
OUT_DIR = "your_path"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
VAL_PATH = os.path.join(DATA_DIR, "val.json")
TEST_PATH = os.path.join(DATA_DIR, "test.json")
MAX_FRIENDS = get_max_friend_len([VAL_PATH, TEST_PATH])

# Step 2. 初始化 Tokenizer
llm_model_name = "Qwen/Qwen2.5-7B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_tokenizer.padding_side = "left"

new_tokens = (
    ["[USER_TOKEN_0]", "[SOCIAL_GRAPH_TOKEN]"] +
    [f"[FRIEND_TOKEN_{i}]" for i in range(MAX_FRIENDS)] +
    [
        "<user_token_start>", "<user_token_end>",
        "<friend_token_start>", "<friend_token_end>",
        "<graph_token_start>", "<graph_token_end>"
    ]
)
llm_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
llm_tokenizer.save_pretrained(os.path.join(OUT_DIR, "yelp_tokenizer"))
# ======================
# 加载 train / val / test 数据
# ======================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print("Loading Yelp coldstart datasets...")
train_data = load_json(TRAIN_PATH)
val_data = load_json(VAL_PATH)
test_data = load_json(TEST_PATH)

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

def to_hf_dataset(data_list):
    return Dataset.from_dict({
        "center_user_id": [d["center_user_id"] for d in data_list],
        "profile": [d["profile"] for d in data_list],
        "target": [d["target"] for d in data_list],
        "split": [d["split"] for d in data_list],
    })

train_dataset = to_hf_dataset(train_data)
val_dataset = to_hf_dataset(val_data)
test_dataset = to_hf_dataset(test_data)

train_dataset.save_to_disk(os.path.join(OUT_DIR, "dataset_train"))
val_dataset.save_to_disk(os.path.join(OUT_DIR, "dataset_val"))
test_dataset.save_to_disk(os.path.join(OUT_DIR, "dataset_test"))

print("\n Datasets saved successfully!")

print("Building user_his_emb_map & user_prof_mean_emb_map ...")

EMB_ROOT = os.path.join(DATA_DIR, "embeddings_yelp")

user_his_emb_map = defaultdict(dict)
user_prof_mean_emb_map = defaultdict(dict)

center_users = [d for d in os.listdir(EMB_ROOT)
                if os.path.isdir(os.path.join(EMB_ROOT, d))]

for center_uid in tqdm(center_users, desc="Loading center/friend embeddings"):
    center_path = os.path.join(EMB_ROOT, center_uid)
    emb_files = [f for f in os.listdir(center_path) if f.endswith(".emb")]

    if not emb_files:
        continue  # 空文件夹跳过

    for emb_file in emb_files:
        uid = emb_file[:-4]  # 去掉后缀
        fpath = os.path.join(center_path, emb_file)
        try:
            emb = torch.load(fpath, map_location="cpu")
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)
        except Exception as e:
            print(f"加载失败 {fpath}: {e}")
            continue

        user_his_emb_map[center_uid][uid] = emb
        user_prof_mean_emb_map[center_uid][uid] = emb.mean(dim=0, keepdim=True)

print(f"已加载中心用户数: {len(user_his_emb_map)}")

print("Processing into personalized format...")

train_pd = PersonalDataset(
    train_dataset,
    user_his_emb_map=user_his_emb_map,
    llm_tokenizer=llm_tokenizer,
    new_tokens=new_tokens,
    max_friend_len=MAX_FRIENDS,
    training=True,
)
hf_train = convert_to_dataset(train_pd)
hf_train.save_to_disk(os.path.join(OUT_DIR, "processed_train"))

val_pd = PersonalDataset(
    val_dataset,
    user_his_emb_map=user_his_emb_map,
    llm_tokenizer=llm_tokenizer,
    new_tokens=new_tokens,
    max_friend_len=MAX_FRIENDS,
    training=False,
)
hf_val = convert_to_dataset(val_pd)
hf_val.save_to_disk(os.path.join(OUT_DIR, "processed_val"))

test_pd = PersonalDataset(
    test_dataset,
    user_his_emb_map=user_his_emb_map,
    llm_tokenizer=llm_tokenizer,
    new_tokens=new_tokens,
    max_friend_len=MAX_FRIENDS,
    training=False,
)
hf_test = convert_to_dataset(test_pd)
hf_test.save_to_disk(os.path.join(OUT_DIR, "processed_test"))

print("\nAll done! Personalized Yelp datasets ready.")
