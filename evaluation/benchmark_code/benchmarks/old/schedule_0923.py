import os
import torch
import multiprocessing as mp
from collections import deque
from transformers import AutoTokenizer, AutoModel
from benchmarks.evaluation import evaluate_model_on_dataset_layer, results_exist
from benchmarks.embedding_extract import extract_and_save_embedding, need_extract_layer
from benchmarks.analysis_reprot import analysis_reprot
import pandas as pd


def embedding_worker(dataset, gpu_id, result_queue, config):
    print(
        f"[EMBEDDING] Starting embedding extraction for dataset {dataset} on GPU {gpu_id} (PID: {os.getpid()})")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_path'], trust_remote_code=True)
    model = AutoModel.from_pretrained(config['model_path'], device_map="auto",
                                      torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    _, _ = extract_and_save_embedding(
        dataset, gpu_id, model, tokenizer, config['model_name'], config)
    result_queue.put(gpu_id)


def result_writter(results, config):
    df = pd.DataFrame(results)
    result_path = f"{config['eval_result_path']}/{config['model_name']}/{results[0]['task']}.tsv"
    # 如果结果文件已存在，则读取并更新
    if os.path.exists(result_path):
        existing_df = pd.read_csv(result_path, sep="\t")
        # 对于每个新结果，更新或添加到现有数据框
        for _, row in df.iterrows():
            mask = (existing_df['task'] == row['task']) & (
                existing_df['layer'] == row['layer'])
            # 确保新行的列顺序与现有DataFrame相同
            row_df = pd.DataFrame([row])[existing_df.columns]

            if mask.sum() == 1:
                # 只有一行匹配时才更新
                existing_df.loc[mask] = row_df.iloc[0]
            elif mask.sum() > 1:
                # 如果有多行匹配，删除所有匹配的行，然后添加新行
                existing_df = existing_df[~mask]
                existing_df = pd.concat(
                    [existing_df, row_df], ignore_index=True)
            else:
                # 没有匹配行时添加新行
                existing_df = pd.concat(
                    [existing_df, row_df], ignore_index=True)
        # 使用更新后的数据框
        df = existing_df

    # 按layer排序
    df = df.sort_values('layer')
    # 保存结果
    df.to_csv(result_path, sep="\t", index=False)


def eval_worker(dataset, layer, gpu_id, result_queue, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model_name = config['model_name']
    # model_layers = config.model_config['num_hidden_layers']
    results = evaluate_model_on_dataset_layer(
        dataset, gpu_id, model_name, layer, config)
    result_queue.put((gpu_id, results))


def task_view(config):
    tasks_to_do = {"embedding": [], "eval": []}
    for dataset in config['eval_datasets']:
        result_dir = f"{config['eval_result_path']}/{config['model_name']}"
        result_path = f"{result_dir}/{dataset}.tsv"
        layer2eval = results_exist(
            result_path, config['layer_to_eval'], config.model_config['num_hidden_layers'], config.dataset_info[dataset]["eval_task"])
        if len(layer2eval) == 0:
            continue
        else:
            embedding_output_dir = f"{config['embedding_output_dir']}/{config['model_name']}"
            layer2extract = need_extract_layer(
                embedding_output_dir, layer2eval, dataset, config)
            if layer2extract is None:
                for layer in layer2eval:
                    tasks_to_do["eval"].append((dataset, layer))
            else:
                for split in config.dataset_info[dataset]["data_split"]:
                    if len(layer2extract[split]) > 0:
                        tasks_to_do["embedding"].append(dataset)
                        break
    # 计算embedding和eval任务的优先级
    embedding_priority = [config.dataset_info[dataset]["sample_num"]*config.dataset_info[dataset]
                          ["seq_for_item"]*config.dataset_info[dataset]["max_length"] for dataset in tasks_to_do["embedding"]]
    eval_priority = [config.dataset_info[dataset]["sample_num"] *
                     config.dataset_info[dataset]["dataset_ratio"][0] for dataset, layer in tasks_to_do["eval"]]
    tasks_to_do['embedding'] = sorted(
        tasks_to_do['embedding'], key=lambda x: embedding_priority[tasks_to_do['embedding'].index(x)], reverse=True)
    tasks_to_do['eval'] = sorted(
        tasks_to_do['eval'], key=lambda x: eval_priority[tasks_to_do['eval'].index(x)], reverse=True)
    return tasks_to_do


def scheduler(config):
    """
    任务调度器,由于embedding可以所有层同时导出，所以embedding任务以dataset为单位调度，eval任务以（dataset，layer）为单位调度
    """
    gpu_list = config['gpu_list']
    gpu_num = len(gpu_list)
    os.makedirs(
        f"{config['eval_result_path']}/{config['model_name']}", exist_ok=True)

    # 获取初始任务列表，包括按照优先级排序的embedding和eval任务，其中embedding任务以dataset排队，eval任务以（dataset，layer）为单位排队
    tasks_to_do = task_view(config)

    # 计算eval任务数量作为循环结束条件
    # 初始的eval任务数量
    task_num2eval = len(tasks_to_do['eval'])
    # 计算embedding之后新增的eval任务数量
    for dataset in tasks_to_do['embedding']:
        task_num2eval += len(results_exist(
            f"{config['eval_result_path']}/{config['model_name']}/{dataset}.tsv", config['layer_to_eval'], config.model_config['num_hidden_layers'], config.dataset_info[dataset]["eval_task"]))

    # rank用于分配任务，gpu_id用于给具体的任务分配gpu，因此这里有rank_to_gpu_id和gpu_id_to_rank两个字典用于转换
    rank_to_gpu_id = {rank: gpu_id for rank, gpu_id in enumerate(gpu_list)}
    gpu_id_to_rank = {gpu_id: rank for rank, gpu_id in rank_to_gpu_id.items()}
    ranks = list(rank_to_gpu_id.keys())
    # 创建embedding和eval任务队列，用于存储待处理的embedding和eval任务
    embedding_queue = deque(tasks_to_do['embedding'])
    eval_queue = deque(tasks_to_do['eval'])

    # 当前各rank正在承担的任务权重，用于存储每个gpu的负载
    rank_power = [0]*len(ranks)

    # 创建embedding和eval任务状态队列，用于将子进程的结果传递给主进程
    completed_eval = 0
    embedding_statue = {dataset: mp.Queue()
                        for dataset in tasks_to_do['embedding']}
    eval_statue = {task: mp.Queue() for task in tasks_to_do['eval']}

    print(
        f"[SCHEDULER] Starting benchmarks for {task_num2eval} dataset-layers ({len(tasks_to_do['embedding'])} datasets need to extract embedding) on {gpu_num} GPUs")
    # 创建两个用于管理embedding和eval任务的进程的字典
    submitted_embedding_process = {}
    submitted_eval_process = {}

    # 先初始化一批embedding任务
    # 计算每个gpu可以承担的embedding任务数量
    embedding_process_per_gpu = config['process_power_per_gpu']//config['embedding_process_power']
    # 计算初始embedding任务数量，用于初始化一批embedding任务
    init_embedding_tasks_num = min(
        len(embedding_queue), gpu_num*embedding_process_per_gpu)
    init_rank = 0
    # 初始化相应数量的embedding任务
    for _ in range(init_embedding_tasks_num):
        dataset = embedding_queue.popleft()
        if init_rank == len(ranks):
            init_rank = 0
        submitted_embedding_process[dataset] = mp.Process(target=embedding_worker, args=(
            dataset, rank_to_gpu_id[init_rank], embedding_statue[dataset], config))
        submitted_embedding_process[dataset].start()
        rank_power[init_rank] += config['embedding_process_power']
        init_rank += 1

    # 处理结果和分配新任务
    # 循环结束条件为提交的评估任务数量达到task_num2eval
    while completed_eval < task_num2eval:
        # 分配新的embedding提取任务，如果embedding队列不为空
        if embedding_queue:
            # 找到负载最轻的rank
            min_rank_power = min(rank_power)
            if min_rank_power+config['embedding_process_power'] <= config['process_power_per_gpu']:
                dataset = embedding_queue.popleft()
                easiest_rank_id = rank_power.index(
                    min_rank_power)
                submitted_embedding_process[dataset] = mp.Process(target=embedding_worker, args=(
                    dataset, rank_to_gpu_id[easiest_rank_id], embedding_statue[dataset], config))
                submitted_embedding_process[dataset].start()
                rank_power[easiest_rank_id] += config['embedding_process_power']
        # 分配新的评估任务
        if eval_queue:
            min_rank_power = min(rank_power)
            if min_rank_power+config['eval_process_power'] <= config['process_power_per_gpu']:
                dataset, layer = eval_queue.popleft()
                easiest_rank_id = rank_power.index(
                    min_rank_power)
                submitted_eval_process[(dataset, layer)] = mp.Process(target=eval_worker, args=(
                    dataset, layer, rank_to_gpu_id[easiest_rank_id], eval_statue[dataset, layer], config))
                submitted_eval_process[(dataset, layer)].start()
                rank_power[easiest_rank_id] += config['eval_process_power']
        # 检查embedding任务完成状态
        datasets_in_eval_queue = []
        for dataset, result_queue in embedding_statue.items():
            if result_queue.empty():
                continue
            result = result_queue.get()
            gpu_id = result
            rank_power[gpu_id_to_rank[gpu_id]
                       ] -= config['embedding_process_power']
            datasets_in_eval_queue.append(dataset)
            # embedding提取完成后，随即添加评估任务（dataset，layer）
            layer2eval = results_exist(
                f"{config['eval_result_path']}/{config['model_name']}/{dataset}.tsv", config['layer_to_eval'], config.model_config['num_hidden_layers'], config.dataset_info[dataset]["eval_task"])
            for layer in layer2eval:
                eval_queue.append((dataset, layer))
                eval_statue[(dataset, layer)] = mp.Queue()
            # 按照优先级重新排序eval_queue
            eval_priority = [config.dataset_info[dataset]["sample_num"] *
                             config.dataset_info[dataset]["dataset_ratio"][0] for dataset, layer in eval_queue]
            eval_queue = deque(sorted(
                eval_queue, key=lambda x: eval_priority[eval_queue.index(x)], reverse=True))
        for dataset in datasets_in_eval_queue:
            submitted_embedding_process[dataset].join()
            del submitted_embedding_process[dataset]
            del embedding_statue[dataset]
        # 检查评估任务完成状态
        dataset_layer_completed_eval = []
        for dataset_layer, result_queue in eval_statue.items():
            if result_queue.empty():
                continue
            gpu_id, result = result_queue.get()
            rank_power[gpu_id_to_rank[gpu_id]] -= config['eval_process_power']
            result_writter(result, config)
            completed_eval += 1
            dataset_layer_completed_eval.append(dataset_layer)
        for dataset_layer in dataset_layer_completed_eval:
            submitted_eval_process[dataset_layer].join()
            del submitted_eval_process[dataset_layer]
            del eval_statue[dataset_layer]

    print("✅ All datasets embedding extracted and evaluated!")
    analysis_reprot(config)
    print(
        f'Report exported to {config["eval_result_path"]}/{config["model_name"]}/reports')
