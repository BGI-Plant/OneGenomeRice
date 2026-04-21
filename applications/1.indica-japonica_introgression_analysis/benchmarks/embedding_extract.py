import os
import json
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
torch.use_deterministic_algorithms(True)

# Optional vllm support
try:
    from vllm import LLM
    _VLLM_AVAILABLE = True
except Exception:
    LLM = None
    _VLLM_AVAILABLE = False

class JSONLDataset(Dataset):
    def __init__(self, dataset_name, split, config):
        super().__init__()
        self.split = split
        self.data = []

        # 构建文件路径
        if split == "train":
            file_path = f"{config['dataset_path']}/{dataset_name}/train.jsonl"
        elif split == "test":
            file_path = f"{config['dataset_path']}/{dataset_name}/test.jsonl"
        elif split == "eval":
            file_path = f"{config['dataset_path']}/{dataset_name}/eval.jsonl"
        else:
            raise ValueError(f"Invalid split: {split}")

        # 检查文件是否存在,如果eval数据集不存在，则返回None
        if not os.path.exists(file_path):
            if split == "eval":
                return None
            else:
                raise FileNotFoundError(
                    f"Dataset file not found: {file_path}")

        # 读取 JSONL 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    item = json.loads(line)
                    # 有2倍体或家系数据集，加载之前先看一下有多少条序列;加载序列key
                    # 建议在datasets_info.yaml中配置seq_for_item（序列数量）和key_for_seq（序列key），否则将默认“seq”和['seq1', 'seq2', ...]
                    if len(self.data) == 0:
                        try:
                            self._seq_number = config.dataset_info[dataset_name]["seq_for_item"]
                        except:
                            self._seq_number = seq_number(item)
                        try:
                            self._seq_key = config.dataset_info[dataset_name]["seq_key"]
                        except:
                            self._seq_key = 'seq' if self._seq_number == 1 else [
                                f'seq{i}' for i in range(1, self._seq_number+1)]
                        try:
                            self._label_key = config.dataset_info[dataset_name]["label_key"]
                        except:
                            self._label_key = 'label'
                    # 构建数据集
                    if self._seq_number == 1:
                        self.data.append(
                            (item[self._seq_key], item[self._label_key]))
                    else:
                        self.data.append(
                            (*[item[seq_key] for seq_key in self._seq_key], item[self._label_key]))

        # print(f"[INFO] Loaded {len(self.data)} samples from {file_path}")

    def get_seq_number(self):
        return self._seq_number

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class JSONLDataset_split(Dataset):
    def __init__(self, data, seq_number):
        super().__init__()
        self.data = data
        self._seq_number = seq_number

    def get_seq_number(self):
        return self._seq_number

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def need_extract_layer(embedding_output_dir, layer_to_eval, dataset, config):
    need_extract = {'train': [], 'test': [], 'eval': []}
    for layer in layer_to_eval:
        if not os.path.exists(f"{embedding_output_dir}/{dataset}-{layer}layer_train.pt") and "train" in config.dataset_info[dataset]["data_split"]:
            need_extract["train"].append(layer)
        if not os.path.exists(f"{embedding_output_dir}/{dataset}-{layer}layer_test.pt") and "test" in config.dataset_info[dataset]["data_split"]:
            need_extract["test"].append(layer)
        if not os.path.exists(f"{embedding_output_dir}/{dataset}-{layer}layer_eval.pt") and "eval" in config.dataset_info[dataset]["data_split"]:
            need_extract["eval"].append(layer)
    if len(need_extract["train"]) == 0 and len(need_extract["test"]) == 0 and len(need_extract["eval"]) == 0:
        return None
    else:
        return need_extract


def seq_number(item):
    """
    查看序列数量，有2倍体或家系数据集
    """
    seq_number = 0
    for key in item.keys():
        if key.startswith("seq"):
            seq_number += 1
    return seq_number


def load_dataset_class_jsonl_split(dataset_name, config, split):
    """
    加载 JSONL 格式的数据集类
    数据集文件格式: dataset_name_train.jsonl 和 dataset_name_test.jsonl
    """

    file_path = f"{config['dataset_path']}/{dataset_name}/{split}.jsonl"
    assert os.path.exists(file_path), f"Dataset file not found: {file_path}"
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if len(data) == 0:
                    try:
                        seq_number = config.dataset_info[dataset_name]["seq_for_item"]
                    except:
                        seq_number = seq_number(item)
                    try:
                        seq_key = config.dataset_info[dataset_name]["seq_key"]
                    except:
                        seq_key = 'seq' if seq_number == 1 else [
                            f'seq{i}' for i in range(1, seq_number+1)]
                    try:
                        label_key = config.dataset_info[dataset_name]["label_key"]
                    except:
                        label_key = 'label'
                # 构建总数据集
                if seq_number == 1:
                    data.append(
                        (item[seq_key], item[label_key]))
                else:
                    data.append(
                        (*[item[seq_key] for seq_key in seq_key], item[label_key]))
    # 分割数据集
    dataset_classes = []
    embedding_extract_split = config['embedding_extract_split']
    embedding_split_path = f"{config['embedding_output_dir']}/{config['model_name']}/split-{embedding_extract_split}"
    splited_data = 0
    splited_file_index = 0
    embedding_task_list = []
    while splited_data < len(data):
        this_part_exist = True
        for layer in config['layer_to_eval']:
            if not os.path.exists(f"{embedding_split_path}/{dataset_name}-{layer}layer_{split}-{str(splited_file_index)}.pt"):
                this_part_exist = False
                break
        if this_part_exist:
            pass
        else:
            dataset_classes.append((dataset_name, split, splited_file_index, JSONLDataset_split(
                data[splited_data:splited_data+embedding_extract_split], seq_number)))
            embedding_task_list.append((splited_file_index))
        splited_file_index += 1
        splited_data += embedding_extract_split

    return embedding_task_list, dataset_classes


def collate_fn(batch, tokenizer):
    sequences_number = len(batch[0])-1
    all_sequences = []
    for sub_batch in batch:
        all_sequences.extend(sub_batch[:sequences_number])
    labels = [item[-1] for item in batch]
    encoding = tokenizer(all_sequences, padding=True,
                         return_tensors="pt")
    return encoding, torch.tensor(labels)

# -------------------------------
# 提取 embedding（平均池化 last_hidden_state）
# -------------------------------


def extract_embeddings(model, dataloader, device, gpu_id, dataset_name, layer2extract: list, seq_number: int, config, tqdm_print: str):
    # If configured to use vllm, dispatch to vllm extraction implementation
    if config.get('use_vllm', False):
        if not _VLLM_AVAILABLE:
            raise ImportError("vllm is not available. Install vllm to use `use_vllm=True` in config.")
        return extract_embeddings_vllm(dataloader, dataset_name, layer2extract, seq_number, config, tqdm_print)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    all_embeddings = {layer: [] for layer in layer2extract}
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{tqdm_print}", leave=False, position=gpu_id, mininterval=3, dynamic_ncols=True):
            mask = batch[0]["attention_mask"].unsqueeze(-1).to(device)
            outputs = model(
                **{"input_ids": batch[0]["input_ids"].to(device, non_blocking=(device == "cuda"))}, output_hidden_states=True)
            for layer in all_embeddings.keys():
                hidden = outputs.hidden_states[layer]
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                pooled = pooled.float()  # [B, H]
                all_embeddings[layer].append(pooled.cpu())
                del hidden, pooled
            del mask, outputs
            if device == "cuda":
                torch.cuda.empty_cache()
            all_labels.append(batch[1])

    for layer, embeddings in all_embeddings.items():
        hidden_state_length = config.model_config['hidden_size']
        assert embeddings[
            0].shape[-1] == hidden_state_length, f"[ERROR] hidden_state_length is not correct: {embeddings[0].shape[-1]} != {hidden_state_length}"
        all_embeddings[layer] = torch.squeeze(torch.reshape(
            torch.cat(embeddings), (-1, seq_number, hidden_state_length)))  # [item, seq_number, hidden_state_length]
    return all_embeddings, torch.cat(all_labels).cpu()


def extract_embeddings_vllm(dataloader, dataset_name, layer2extract: list, seq_number: int, config, tqdm_print: str):
    """
    Use vllm.LLM to obtain hidden states and perform mean pooling over token dimension.

    Notes:
    - This implementation depends on vllm returning `model_outputs` with `hidden_states` when
      `output_hidden_states=True` is passed to `LLM.generate` (vllm API may differ by version).
    - Tokenization alignment between HF tokenizer and vllm must match; this function decodes
      input_ids back to text and sends those prompts to vllm. If tokenization differs, results
      will be inconsistent; validate in your environment.
    """
    # Remove n_ctx as it's not a valid parameter for vLLM's LLM class
    llm_kwargs = {k: v for k, v in config.items() if k not in ['n_ctx']}
    llm = LLM(
        model=config['model_path'],
        trust_remote_code=True,
        tensor_parallel_size=1, 
        block_size=32,
        enable_prefix_caching=True,
        enforce_eager=True,
        gpu_memory_utilization=0.85,  # 提高 GPU 内存利用率
        dtype=torch.bfloat16,
        #max_model_len=seq_length,
        #max_num_batched_tokens=seq_length,
        override_pooler_config=PoolerConfig(pooling_type="MEAN", normalize=False), # 池化参数，不使用此参数可获取完整的隐藏状态
        task='reward',
        enable_chunked_prefill=False
    )
    all_embeddings = {layer: [] for layer in layer2extract}
    all_labels = []
    # dataloader yields (encoding, labels)
    for batch in tqdm(dataloader, desc=f"{tqdm_print}", leave=False, mininterval=3, dynamic_ncols=True):
        encoding, labels = batch
        # decode prompts to text for vllm
        input_ids = encoding['input_ids']
        prompts = []
        try:
            # if tokenizer available in config, use it
            tokenizer = config.get('tokenizer')
            if tokenizer is not None:
                prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            else:
                # fallback: decode naïvely by joining ids (not ideal)
                prompts = [" ".join(map(str, ids.tolist())) for ids in input_ids]
        except Exception:
            prompts = [" ".join(map(str, ids.tolist())) for ids in input_ids]

        # generate with vllm and request hidden states
        gen = llm.generate(prompts, output_hidden_states=True)

        # vllm.generate yields responses in order; iterate and collect hidden states
        import numpy as np
        for i, resp in enumerate(gen):
            # Attempt to access model output hidden states
            model_outputs = getattr(resp, 'model_outputs', None)
            if model_outputs is None:
                model_outputs = [resp]
            model_output = model_outputs[-1]
            hidden_states = getattr(model_output, 'hidden_states', None)
            if hidden_states is None:
                raise RuntimeError("vllm did not return hidden_states. Ensure vllm version supports `output_hidden_states=True`.")
            # hidden_states is expected to be a sequence: (layer0, layer1, ...), each shape [seq_len, hidden]
            for layer in layer2extract:
                hs_layer = hidden_states[layer]
                # convert to tensor
                hs_tensor = torch.tensor(np.array(hs_layer)).float()
                # build mask for this instance
                mask = encoding['attention_mask'][i].unsqueeze(-1).float()
                denom = mask.sum(0)
                denom[denom == 0] = 1.0
                pooled = (hs_tensor * mask).sum(0) / denom
                all_embeddings[layer].append(pooled.cpu())
        all_labels.append(labels)

    for layer, embeddings in all_embeddings.items():
        hidden_state_length = config.model_config['hidden_size']
        if len(embeddings) == 0:
            raise RuntimeError("No embeddings extracted for layer {layer}")
        assert embeddings[0].shape[-1] == hidden_state_length, f"[ERROR] hidden_state_length is not correct: {embeddings[0].shape[-1]} != {hidden_state_length}"
        all_embeddings[layer] = torch.stack(embeddings)

    return all_embeddings, torch.cat(all_labels).cpu()


def save_embedding(dataset_name, gpu_id, model, tokenizer, split, i, dataset_class, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = f"{config['embedding_output_dir']}/{config['model_name']}"
    os.makedirs(output_dir, exist_ok=True)
    split_output_dir = f"{config['embedding_output_dir']}/{config['model_name']}/split-{config['embedding_extract_split']}"
    os.makedirs(split_output_dir, exist_ok=True)

    seq_number = dataset_class.get_seq_number()
    if config['batch_size'] is None:
        batch_size = 16
    else:
        batch_size = config['batch_size']
    if dataset_name in config.all_datasets_feature['long_sequence_dataset']:
        batch_size = batch_size // 2
    if dataset_name in config.all_datasets_feature['super_long_sequence_dataset']:
        batch_size = 1

    # 有多条序列，需要除以序列数量以保证batch_size不变
    batch_size = batch_size // seq_number if batch_size // seq_number > 0 else 1
    loader = DataLoader(dataset_class, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: collate_fn(x, tokenizer))

    tqdm_print = f"Embedding {dataset_name} {split}-{i}" if i is not None else f"Embedding {dataset_name} {split}"
    X_output, y_output = extract_embeddings(
        model, loader, device, gpu_id, dataset_name, config['layer_to_eval'], seq_number, config, tqdm_print)
    for layer in X_output.keys():
        x_train_layer = X_output[layer]
        data_train = {"embeddings": x_train_layer, "labels": y_output}
        if i is None:
            torch.save(
                data_train, f"{output_dir}/{dataset_name}-{layer}layer_{split}.pt")
        else:
            torch.save(
                data_train, f"{split_output_dir}/{dataset_name}-{layer}layer_{split}-{i}.pt")
