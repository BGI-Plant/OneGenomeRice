import os
import json
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


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


def load_dataset_class_jsonl(dataset_name, config):
    """
    加载 JSONL 格式的数据集类
    数据集文件格式: dataset_name_train.jsonl 和 dataset_name_test.jsonl
    """

    class JSONLDataset(Dataset):
        def __init__(self, split):
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

    return JSONLDataset


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


def extract_embeddings(model, dataloader, device, gpu_id, dataset_name, layer2extract: list, seq_number: int, config, split: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # num_layers = model.config.num_hidden_layers  # 获取模型总层数
    # print(f"[INFO] extracting: layer{layer2extract}")
    all_embeddings = {layer: [] for layer in layer2extract}
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Embedding {dataset_name} {split}", leave=False, position=gpu_id, mininterval=3, dynamic_ncols=True):
            mask = batch[0]["attention_mask"].unsqueeze(-1).to(device)
            outputs = model(
                **{"input_ids": batch[0]["input_ids"].cuda(non_blocking=True)}, output_hidden_states=True)
            for layer in all_embeddings.keys():
                hidden = outputs.hidden_states[layer]
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                pooled = pooled.float()  # [B, H]
                all_embeddings[layer].append(pooled.cpu())
                del hidden, pooled
            del mask, outputs
            all_labels.append(batch[1])

    for layer, embeddings in all_embeddings.items():
        hidden_state_length = config.model_config['hidden_size']
        assert embeddings[
            0].shape[-1] == hidden_state_length, f"[ERROR] hidden_state_length is not correct: {embeddings[0].shape[-1]} != {hidden_state_length}"
        all_embeddings[layer] = torch.squeeze(torch.reshape(
            torch.cat(embeddings), (-1, seq_number, hidden_state_length)))  # [item, seq_number, hidden_state_length]
    return all_embeddings, torch.cat(all_labels).cpu()


def save_embedding(model, tokenizer, dataset_name: str, gpu_id, output_dir: str, layer2extract: dict, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用 JSONL 格式的数据集
    try:
        dataset_class = load_dataset_class_jsonl(dataset_name, config)
    except FileNotFoundError as e:
        print(f"[ERROR] JSONL dataset files not found: {e}")
        return

    train_dataset = dataset_class(split="train")

    if "test" in config.dataset_info[dataset_name]["data_split"]:
        test_dataset = dataset_class(split="test")
    else:
        test_dataset = None
    if "eval" in config.dataset_info[dataset_name]["data_split"]:
        eval_dataset = dataset_class(split="eval")
    else:
        eval_dataset = None

    if config.dataset_info[dataset_name]["data_split"] == ['train', 'test']:
        assert train_dataset.get_seq_number() == test_dataset.get_seq_number(
        ), f"[ERROR] train and test dataset have different sequence number"
    elif config.dataset_info[dataset_name]["data_split"] == ['train', 'test', 'eval']:
        assert train_dataset.get_seq_number() == test_dataset.get_seq_number() == eval_dataset.get_seq_number(
        ), f"[ERROR] train and test and eval dataset have different sequence number"
    elif config.dataset_info[dataset_name]["data_split"] == ['train', 'eval']:
        assert train_dataset.get_seq_number() == eval_dataset.get_seq_number(
        ), f"[ERROR] train and eval dataset have different sequence number"
    else:
        raise ValueError(
            f"[ERROR] dataset {dataset_name} data split is not correct")

    seq_number = train_dataset.get_seq_number()
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: collate_fn(x, tokenizer))
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=lambda x: collate_fn(x, tokenizer))
    if eval_dataset is not None:
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=lambda x: collate_fn(x, tokenizer))

    if len(layer2extract['train']) > 0:
        X_train, y_train = extract_embeddings(
            model, train_loader, device, gpu_id, dataset_name, layer2extract['train'], seq_number, config, 'train')
        for layer in X_train.keys():
            x_train_layer = X_train[layer]
            data_train = {"embeddings": x_train_layer, "labels": y_train}
            torch.save(
                data_train, f"{output_dir}/{dataset_name}-{layer}layer_train.pt")
    else:
        print(
            f"[EMBEDDING] ✅ Embedding of dataset {dataset_name} train is exist")
    if len(layer2extract['test']) > 0:
        X_test, y_test = extract_embeddings(
            model, test_loader, device, gpu_id, dataset_name, layer2extract['test'], seq_number, config, 'test')
        for layer in X_test.keys():
            x_test_layer = X_test[layer]
            data_test = {"embeddings": x_test_layer, "labels": y_test}
            torch.save(
                data_test, f"{output_dir}/{dataset_name}-{layer}layer_test.pt")
    else:
        print(
            f"[EMBEDDING] ✅ Embedding of dataset {dataset_name} test is exist")
    if len(layer2extract['eval']) > 0 and eval_dataset is not None:
        X_eval, y_eval = extract_embeddings(
            model, eval_loader, device, gpu_id, dataset_name, layer2extract['eval'], seq_number, config, 'eval')
        for layer in X_eval.keys():
            x_eval_layer = X_eval[layer]
            data_eval = {"embeddings": x_eval_layer, "labels": y_eval}
            torch.save(
                data_eval, f"{output_dir}/{dataset_name}-{layer}layer_eval.pt")


def extract_and_save_embedding(dataset, gpu_id, model, tokenizer, model_name, config):
    num_layers = model.config.num_hidden_layers
    embedding_output_dir = f"{config['embedding_output_dir']}/{model_name}"
    if config['layer_to_eval'] is None:
        layer_to_eval = list(range(num_layers+1))
    else:
        assert isinstance(config['layer_to_eval'],
                          list), "layer_to_eval must be a list"
        for layer in config['layer_to_eval']:
            assert layer in range(
                num_layers+1), f"layer_to_eval must be a list of integers in range(0, {num_layers+1})"
        layer_to_eval = config['layer_to_eval']
    os.makedirs(embedding_output_dir, exist_ok=True)
    # 判断是否已经有存在embedding，如果存在就不用做了
    layer2extract = need_extract_layer(
        embedding_output_dir, layer_to_eval, dataset, config)
    if layer2extract is None:
        print(f"[EMBEDDING] ✅ Embedding of dataset {dataset} is exist")
        return dataset, gpu_id
    else:
        # print(
        #     f"[EMBEDDING] 🫴 Extracting embedding for dataset {dataset} on GPU {gpu_id} (PID: {os.getpid()})")
        save_embedding(model, tokenizer, dataset, gpu_id,
                       embedding_output_dir, layer2extract, config)
        print(
            f"[EMBEDDING] ✅ Extracting embedding completed for dataset {dataset} on GPU {gpu_id} (PID: {os.getpid()})")
        return dataset, gpu_id
