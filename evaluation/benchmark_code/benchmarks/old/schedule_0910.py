import os
import torch
import multiprocessing as mp
from collections import deque
from transformers import AutoTokenizer, AutoModel
from benchmarks.evaluation import evaluate_model, results_exist
from benchmarks.embedding_extract import extract_and_save_embedding, need_extract_layer
from benchmarks.analysis_reprot import analysis_reprot


def gpu_worker(gpu_id, task_queue, result_queue, model_path, task_type, config):
    """GPU工作进程，处理embedding提取和评测任务"""
    # 设置使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model_name = model_path.split("/")[-1]
    if task_type == "embedding":
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, device_map="auto",
                                          torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        print(
            f"[embedding] Model {model_name} loaded on GPU {gpu_id} (PID: {os.getpid()})")
    else:
        model_layers = config.model_config['num_hidden_layers']

    while True:
        task = task_queue.get()
        if task is None:  # 终止信号
            break

        task_type, dataset = task
        if task_type == "embedding":
            # 提取embedding
            _, _ = extract_and_save_embedding(
                dataset, gpu_id, model, tokenizer, model_name, config)
            # 训练完成后，将评测任务添加到同一GPU的任务队列
            # task_queue.put(("eval", dataset, model_path))
            # 通知调度器可以分配新训练任务
            result_queue.put(("embedding_complete", dataset, gpu_id))
        elif task_type == "eval":
            # 评测模型
            _, _ = evaluate_model(
                dataset, gpu_id, model_name, model_layers, config)
            # 通知调度器评测完成
            result_queue.put(("eval_complete", dataset, gpu_id))


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
                tasks_to_do["eval"].append(dataset)
            else:
                if len(layer2extract['train']) == 0:
                    tasks_to_do["embedding"].append(dataset)
                elif len(layer2extract['test']) == 0 and len(layer2extract['eval']) == 0:
                    tasks_to_do["embedding"].append(dataset)
    return tasks_to_do


def scheduler(config):
    model_path = config['model_path']
    gpu_list = config['gpu_list']
    total_datasets = len(config['eval_datasets'])
    gpu_num = len(gpu_list)

    tasks_to_do = task_view(config)
    """任务调度器"""
    rank_to_gpu_id = {rank: gpu_id for rank, gpu_id in enumerate(gpu_list)}
    gpu_id_to_rank = {gpu_id: rank for rank, gpu_id in rank_to_gpu_id.items()}
    ranks = list(rank_to_gpu_id.keys())

    # 创建每张GPU的任务队列
    gpu_queues = {"embedding": [mp.Queue() for _ in ranks], "eval": [
        mp.Queue() for _ in ranks]}
    result_queue = mp.Queue()
    # 启动GPU工作进程
    gpu_processes = {"embedding": [], "eval": []}
    for rank in ranks:
        p = mp.Process(target=gpu_worker, args=(
            rank_to_gpu_id[rank], gpu_queues["embedding"][rank], result_queue, model_path, "embedding", config))
        p.start()
        gpu_processes["embedding"].append(p)
        p = mp.Process(target=gpu_worker, args=(
            rank_to_gpu_id[rank], gpu_queues["eval"][rank], result_queue, model_path, "eval", config))
        p.start()
        gpu_processes["eval"].append(p)

    # 待embedding提取队列
    embedding_queue = deque(config['eval_datasets'])
    # 当前正在进行的任务数
    gpu_active = {"embedding": [0]*len(ranks), "eval": [0]*len(ranks)}
    completed_embedding = 0
    # 已完成的任务计数
    completed_datasets = 0
    # 当前正在评测的dataset
    eval_queue = deque()

    print(
        f"🚀 Starting embedding extraction for {total_datasets} evaluation datasets on {gpu_num} GPUs")

    # 初始分配embedding提取任务
    for rank in ranks:
        if embedding_queue:
            dataset = embedding_queue.popleft()
            gpu_queues["embedding"][rank].put(
                ("embedding", dataset))
            gpu_active["embedding"][rank] = 1
            print(
                f"🔁 GPU {rank_to_gpu_id[rank]} assigned dataset {dataset} for embedding extraction")

    # 处理结果和分配新任务
    while completed_datasets < total_datasets:
        # 等待任务完成通知
        result = result_queue.get()
        event, dataset, gpu_id = result

        if event == "embedding_complete":
            print(
                f"🎯 Embedding extraction completed for dataset {dataset} on GPU {gpu_id}")
            gpu_active["embedding"][gpu_id_to_rank[gpu_id]] = 0
            completed_embedding += 1
            # 添加评测任务
            eval_queue.append(dataset)
        elif event == "eval_complete":
            print(
                f"🏁 Evaluation completed for dataset {dataset} on GPU {gpu_id}")
            completed_datasets += 1
            gpu_active["eval"][gpu_id_to_rank[gpu_id]] = 0

        # 分配新embedding提取任务（如果有待训练模型且GPU尚无embedding任务）
        if embedding_queue and (sum(gpu_active["embedding"]) < gpu_num):
            dataset = embedding_queue.popleft()
            # 找到空闲的GPU（这里简化处理，实际可能需要更复杂的调度）
            for rank in ranks:
                if gpu_active["embedding"][rank] == 0:
                    gpu_queues["embedding"][rank].put(("embedding", dataset))
                    gpu_active["embedding"][rank] = 1
                    print(
                        f"🔁 GPU {rank_to_gpu_id[rank]} assigned dataset {dataset} for embedding extraction")
                    break

        # 分配新评测任务（如果有待评测dataset且GPU尚无评测任务）
        if eval_queue and (sum(gpu_active["eval"]) < gpu_num):
            dataset = eval_queue.popleft()
            for rank in ranks:
                if gpu_active["eval"][rank] == 0:
                    gpu_queues["eval"][rank].put(("eval", dataset))
                    gpu_active["eval"][rank] = 1
                    print(
                        f"🔁 GPU {rank_to_gpu_id[rank]} assigned dataset {dataset} for evaluation")
                    break
        if completed_embedding >= total_datasets:
            print(
                f"🔁 All datasets embedding extracted, sending termination signal to embedding queue")
            for q in gpu_queues["embedding"]:
                q.put(None)
    # 发送终止信号
    for q in gpu_queues["eval"]:
        q.put(None)

    # 等待GPU工作进程结束
    for p in gpu_processes.values():
        for p_sub in p:
            p_sub.join()

    print("✅ All datasets embedding extracted and evaluated!")
    analysis_reprot(config)
