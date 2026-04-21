# 标准库（内置模块）
import os
import argparse
import json

# 第三方库（pip 安装的包）
import wandb
import pandas as pd
import torch
from functools import partial

# Hugging Face Transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments
)

# 从自定义仓库中导入模块
from src.util import (
    dist_print,
    is_main_process,
    setup_distributed,
    setup_logging,
    setup_seed,
    get_index,
    setup_sync_batchnorm
)
from src.dataset import  MultiTrackDataset
from src.model import GenOmics, targets_scaling_torch
from src.metrics import compute_multimodal_metrics
from src.trainer import(
    CustomTrainer, 
    DistributedSamplerCallback, 
    LocalLoggerCallback
    )

    
def parse_args():
    """
    解析命令行参数，返回 args 对象
    """
    parser = argparse.ArgumentParser(description="Train RNA-seq track predictor with configurable args.")

    # --- 数据路径 ---
    parser.add_argument("--model_path", type=str, required=True,
                        help="预训练模型路径（HF格式）")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                        help="分词器路径（HF格式）")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="用于继续训练")
    parser.add_argument("--use_flash_attn", action='store_true',
                        help="启用 Flash Attention 加速（默认：禁用）")
    parser.add_argument("--sequence_split_train", type=str, 
                        help="训练数据索引数据")
    parser.add_argument("--sequence_split_train_multi", type=str, nargs='+',
                        help="训练数据索引数据")
    parser.add_argument("--sequence_split_val", type=str, required=True, 
                        help="验证数据索引数据")
    parser.add_argument("--index_stat_json", type=str, 
                        help="训练数据统计信息")
    parser.add_argument("--index_stat_multi_json", type=str, nargs='+',
                        help="多个训练数据统计信息")
    parser.add_argument("--nonzero_means",type=float,nargs='+',
                        help="每个轨道的非零均值")
    # --- 输出设置 ---
    parser.add_argument("--output_base_dir", type=str, required=True,
                        help="输出根目录")

    # 在 parse_args 函数中修改参数定义：
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="调试用：限制训练样本数（None 表示不限制）")
    parser.add_argument("--max_sequence_length", type=int, default=32768)

    # --- 染色体划分 ---
    parser.add_argument("--train_chromosomes", type=str, nargs='+', default=["chr19"],
                        help="训练染色体列表")
    parser.add_argument("--val_chromosomes", type=str, nargs='+', default=["Chr12"],
                        help="验证染色体列表")

    # --- 训练超参数 ---
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--batch_size_per_device", type=int, default=1,
                        help="每卡batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="数据加载器进程")
    parser.add_argument("--gpus_per_node", type=int, default=8,
                        help="每节点 GPU 数量")
    # --- 模型设置 ---
    parser.add_argument("--loss_func", type=str, default="mse",
                        choices=["mse", "poisson", "tweedie", "poisson-multinomial"], 
                        help="损失函数类型：mse 或 poisson")
    parser.add_argument("--proj_dim", type=int, default=1024,
                        help="U-Net 的输入特征维度")
    parser.add_argument("--num_downsamples", type=int, default=4,
                        help="U-Net 的下采样次数")
    parser.add_argument("--bottleneck_dim", type=int, default=1536,
                        help="U-Net 的瓶颈层维度")
    
    # --- 其他 ---
    parser.add_argument("--use_wandb", action="store_true",
                        help="启用 Weights & Biases")
    parser.add_argument("--seed", type=int, default=42,
                    help="随机数种子")

    return parser.parse_args()

def main():
    """
    🧬 主训练流程：基于预训练DNA语言模型 + 多轨道BigWig信号，进行单碱基分辨率预测任务
    支持分布式训练（DDP），使用 FlashAttention-2 + bf16 加速，W&B 日志记录。
    """

    # 解析参数
    args  = parse_args()

    # 设置随机数种子
    setup_seed(args.seed)


    # 初始化变量，避免 locals() 问题
    train_dataset = None
    val_dataset = None
    run = None
    
    # --- 分布式初始化 ---
    local_rank, world_size, is_distributed = setup_distributed()
    
    # 日志配置
    log_filepath = setup_logging(
        output_base_dir=args.output_base_dir,
    )
    dist_print(f"🌍 分布式初始化完成: local_rank={local_rank}, world_size={world_size}")

    # 打印wanndb信息
    if args.use_wandb and is_main_process():
        wandb_config = {
                "learning_rate": args.lr,
                "batch_size": args.batch_size_per_device,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "epochs": args.num_train_epochs,
                "model": args.model_path,
                "loss_func": args.loss_func,
                "max_sequence_length": args.max_sequence_length,
                "proj_dim": args.proj_dim,
                "bottleneck_dim": args.bottleneck_dim,
                "num_downsamples": args.num_downsamples,
                "use_flash_attn": args.use_flash_attn,
                "seed": args.seed,
                "train_chromosomes": args.train_chromosomes,
                "val_chromosomes": args.val_chromosomes,
                }
        run = wandb.init(
                entity="zhongliyuan-bgi-group",
                project="RNA-seq",  
                name=f"train-{args.loss_func}-lr{args.lr}-bs{args.batch_size_per_device}",
                dir=args.output_base_dir,
                resume="allow",
                config=wandb_config)
        # 定义指标跟踪方式
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("eval/loss", summary="min")
        wandb.define_metric("epoch")
        wandb.define_metric("global_step")
        
        dist_print(f"🌐 wandb: Logged in as: {run.entity}")
        dist_print(f"📊 Project: {run.project} | Run Name: {run.name}")
        dist_print(f"🚀 Run URL: {run.url}")
        dist_print(f"💾 Local Dir: {run.dir}")
    
    # 打印args信息
    args_dict = vars(args)
    dist_print("📋 训练参数配置:")
    for key, value in args_dict.items():
        dist_print(f"    {key}: {value}")

    # --- 加载模型与分词器 ---
    dist_print("🚀 加载预训练模型和分词器...")
    if args.use_flash_attn:
        dist_print("⚡ 使用 Flash Attention")
        base_model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            revision="main",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 # 改为 torch_dtype
        )
    else:
        base_model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            revision="main",
            torch_dtype=torch.bfloat16 # 改为 torch_dtype
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        trust_remote_code=True,
        revision="main",
        padding_side='right',
    )


    # --- 数据划分 ---
    dist_print(f"🧬 训练染色体: {args.train_chromosomes}")
    dist_print(f"🧬 验证染色体: {args.val_chromosomes}")

    # --- 获取数据索引 ---
    dist_print("🏷️ 获取数据索引...")
    if args.sequence_split_train is not None:
            train_index_df = get_index(args.sequence_split_train)
    # --- 数据索引筛选 ---
            selected_train_index_df = train_index_df[train_index_df["chromosome"].str.extract(r'(Chr\d+)')[0].isin(args.train_chromosomes)].copy()
            if args.max_train_samples is not None:
                selected_train_index_df = selected_train_index_df.sample(n=args.max_train_samples, random_state=args.seed)
    elif args.sequence_split_train_multi is not None:
            train_indexes = args.sequence_split_train_multi
            train_index_dfs = [get_index(index_df) for index_df in train_indexes]
            selected_train_index_df=[]
            for train_index_df in train_index_dfs:
                temp_df = train_index_df[train_index_df["chromosome"].str.extract(r'(Chr\d+)')[0].isin(args.train_chromosomes)].copy()
                if args.max_train_samples is not None:
                    temp_df = temp_df[:args.max_train_samples]
                selected_train_index_df.append(temp_df)
    else:
        raise ValueError("必须提供 sequence_split_train 或 sequence_split_train_multi 参数")

    val_index_df = get_index(args.sequence_split_val)


    # --- 数据索引筛选 ---
    #selected_train_index_df = train_index_df[train_index_df["chromosome"].isin(args.train_chromosomes)].copy()
    #run_sequence_split_and_meta_extract.py中已经定义好染色体，这里不需要再筛选一次

    # selected_val_index_df = val_index_df[val_index_df["chromosome"].isin(args.val_chromosomes)].copy()
    # if args.max_train_samples is not None:
    #     selected_val_index_df = selected_train_index_df

    # --- 读取数据统计信息和标签元信息 ---

    if args.index_stat_json is not None:
        with open(args.index_stat_json, "r") as f:
            index_stat = json.load(f)
    elif args.index_stat_multi_json is not None:
        index_stat_jsons = args.index_stat_multi_json
        index_stat = []
        for index_stat_json in index_stat_jsons:
            with open(index_stat_json, "r") as f:
                temp_index_stat = json.load(f)
            index_stat.append(temp_index_stat)
    else:
        raise ValueError("必须提供 index_stat_json 或 index_stat_multi_json 参数")

    
    # --- 创建数据集 ---
    dist_print("🧩 创建训练数据集...")
    train_dataset = MultiTrackDataset(selected_train_index_df, index_stat, 
                                      tokenizer, max_length=args.max_sequence_length)
    dist_print(f"✅ 训练: {len(train_dataset):,} 样本")
    # dist_print("🧩 创建验证数据集...")
    # val_dataset = MultiTrackDataset(selected_train_index_df, label_meta_df, 
    #                                 index_stat, tokenizer, max_length=args.max_sequence_length)
                                      
    # dist_print(f"✅ 验证: {len(val_dataset):,} 样本")
    

    if args.index_stat_multi_json is not None:
        temp = index_stat[0]
        index_stat=temp
        index_stat['counts']['nonzero_mean']=[]
        for non0_mean in args.nonzero_means:
            index_stat['counts']['nonzero_mean'].append(non0_mean)

    # --- 构建下游预测模型 ---
    dist_print("🌐 构建下游网络...")
    model = GenOmics(
        base_model,
        index_stat=index_stat,
        loss_func=args.loss_func,
        proj_dim=args.proj_dim,
        num_downsamples=args.num_downsamples,
        bottleneck_dim=args.bottleneck_dim
    )
    
    # --- 设置 SyncBatchNorm ---
    model = setup_sync_batchnorm(model, is_distributed, args.gpus_per_node)
    dist_print("✅ SyncBatchNorm 配置完成")
    
    # --- 转为 bfloat16 ---
    model = model.to(torch.bfloat16)
    dist_print("✅ BF16 模式已启用")

    
    # # --- 解冻骨架模型并解冻其最后一层 ---
    # for param in model.base.parameters():
    #     param.requires_grad = False
    # dist_print("❄ 冻结基模所有参数")
    # for param in model.base.layers[-1].parameters():
    #     param.requires_grad = True
    # dist_print("🔥 解冻最后一层")


    # --- 打印参数量 ---
    trainable_base_params = sum(p.numel() for p in model.base.parameters() if p.requires_grad)
    total_base_params = sum(p.numel() for p in model.base.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    downstread_task_head_params = total_params - total_base_params
    
    dist_print(f"📊 模型总参数量: {total_params:,} (下游任务头大小：{downstread_task_head_params:,}，基模可训练参数比例: {trainable_base_params/total_base_params*100:.1f}%)")

    # --- 配置训练参数 ---
    dist_print("⚙️ 配置训练参数...")
    training_args = TrainingArguments(
        output_dir=args.output_base_dir,
        logging_dir=os.path.join(args.output_base_dir, "logs"),

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_device,
        per_device_eval_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        dataloader_num_workers = args.dataloader_num_workers,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        include_for_metrics=["inputs", "loss"],

        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adafactor",

        # eval_strategy="epoch",
        save_strategy="epoch",
        # eval_accumulation_steps=10,
        save_total_limit=30,
        save_safetensors=True,

        fp16=False,
        bf16=True,
        half_precision_backend="auto",

        logging_steps=1,
        report_to="none",
        log_level="info",

        # ddp_find_unused_parameters=True,
        remove_unused_columns=False,
        seed=args.seed,

        resume_from_checkpoint=args.ckpt_dir,
    )
    
    # --- 创建训练器 ---
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        compute_metrics=partial(compute_multimodal_metrics, val_chromosomes=args.val_chromosomes, tokenizer=tokenizer),
        callbacks=[DistributedSamplerCallback(),
        LocalLoggerCallback(log_file_path=log_filepath)]
    )
    try:
        # --- 开始训练 ---
        dist_print("🏋️‍♂️ 启动训练...")
        if args.ckpt_dir: 
            # 恢复训练
            trainer.train(resume_from_checkpoint=args.ckpt_dir)
        else:
            trainer.train()
        dist_print("✅ 训练完成！")

    except Exception as e:
        dist_print(f"❌ 训练过程发生错误: {str(e)}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # 防止其他 rank 卡住
        raise  # 不吞异常


    finally:
        # 清理数据集
        dataset_dict = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset
        }

        for name, ds in dataset_dict.items():
            if ds is not None and hasattr(ds, 'close'):
                ds.close()
                dist_print(f"🧹 资源已释放: {name}（{type(ds).__name__}）")

        # 清理 wandb
        if run is not None and is_main_process():
            wandb.finish()
            dist_print("🧹 wandb run 已结束")

    dist_print("🎉 主流程执行完毕！")


if __name__ == "__main__":
    main()
