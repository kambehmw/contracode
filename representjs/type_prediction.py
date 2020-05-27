import os
import random

import fire
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from loguru import logger

from data.deeptyper_dataset import DeepTyperDataset, load_type_vocab, get_collate_fn
from data.util import Timer
from models.typetransformer import TypeTransformer
from representjs import RUN_DIR
from utils import count_parameters, get_linear_schedule_with_warmup
from decode import ids_to_strs


def accuracy(output, target, topk=(1,), ignore_idx=[]):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get top predictions per position that are not in ignore_idx
        target_vocab_size = output.size(2)
        keep_idx = torch.tensor([i for i in range(target_vocab_size) if i not in ignore_idx], device=output.device).long()
        _, pred = output[:, :, keep_idx].topk(maxk, 2, True, True)  # BxLx5
        pred = keep_idx[pred]  # BxLx5

        # Compute statistics over positions not labeled with an ignored idx
        correct = pred.eq(target.unsqueeze(-1).expand_as(pred)).long()
        mask = torch.ones_like(target).long()
        for idx in ignore_idx:
            mask = mask & ~target.eq(idx)
        mask = mask.long()
        deno = mask.sum().item()
        correct = correct * mask.unsqueeze(-1)
        res = []
        for k in topk:
            correct_k = correct[..., :k].view(-1).float().sum(0)
            res.append(correct_k.item())
        return res, deno


def _evaluate(model, loader, sp: spm.SentencePieceProcessor, target_to_id, use_cuda=True):
    model.eval()
    pad_id = sp.PieceToId("[PAD]")
    no_type_id = target_to_id["O"]

    with torch.no_grad():
        # Accumulate metrics across batches to compute label-wise accuracy
        num1, num5, num_labels_total = 0, 0, 0
        num1_any, num5_any, num_labels_any_total = 0, 0, 0

        with Timer() as t:
            # Compute average loss
            total_loss = 0
            num_examples = 0
            pbar = tqdm.tqdm(loader, desc=f"evalaute")
            for X, output_attn, labels in pbar:
                if use_cuda:
                    X, output_attn, labels = X.cuda(), output_attn.cuda(), labels.cuda()
                logits = model(X, output_attn)  # BxLxVocab
                # Compute loss
                loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=no_type_id)

                total_loss += loss.item() * X.size(0)
                num_examples += X.size(0)
                avg_loss = total_loss / num_examples

                # Compute accuracy
                (corr1_any, corr5_any), num_labels_any = accuracy(
                    logits.cpu(), labels.cpu(), topk=(1, 5), ignore_idx=(no_type_id,))
                num1_any += corr1_any
                num5_any += corr5_any
                num_labels_any_total += num_labels_any

                (corr1, corr5), num_labels = accuracy(
                    logits.cpu(), labels.cpu(), topk=(1, 5), ignore_idx=(no_type_id, target_to_id["$any$"]))
                num1 += corr1
                num5 += corr5
                num_labels_total += num_labels

                pbar.set_description(f"evaluate average loss {avg_loss:.4f} num1 {num1_any} num_labels_any_total {num_labels_any_total} avg acc1_any {num1_any / (num_labels_any_total + 1e-6) * 100:.4f}")

        # Average accuracies
        acc1 = float(num1) / num_labels_total * 100
        acc5 = float(num5) / num_labels_total * 100
        acc1_any = float(num1_any) / num_labels_any_total * 100
        acc5_any = float(num5_any) / num_labels_any_total * 100

        logger.debug(f"Loss calculation took {t.interval:.3f}s")
        return (
            acc1_any,
            {
                "eval/loss": avg_loss,
                "eval/acc@1": acc1,
                "eval/acc@5": acc5,
                "eval/num_labels": num_labels_total,
                "eval/acc@1_any": acc1_any,
                "eval/acc@5_any": acc5_any,
                "eval/num_labels_any": num_labels_any_total
            },
        )


def train(
    run_name: str,
    # Data
    train_filepath: str,
    eval_filepath: str,
    type_vocab_filepath: str,
    spm_filepath: str,
    num_workers=1,
    max_seq_len=1024,
    max_eval_seq_len=1024,
    # Model
    resume_path: str = "",
    pretrain_resume_path: str = "",
    # Optimization
    num_epochs: int = 100,
    save_every: int = 2,
    batch_size: int = 256,
    lr: float = 8e-4,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.98,
    adam_eps: float = 1e-6,
    weight_decay: float = 0,
    # Loss
    subword_regularization_alpha: float = 0,
    ignore_any_loss: bool=False,
    # Computational
    use_cuda: bool = True,
    seed: int = 0,
):
    """Train model"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    run_dir = RUN_DIR / run_name
    run_dir.mkdir(exist_ok=True, parents=True)
    logger.add(str((run_dir / "train.log").resolve()))
    logger.info(f"Saving logs, model checkpoints to {run_dir}")
    config = locals()
    logger.info(f"Config: {config}")
    wandb.init(name=run_name, config=config, job_type="training", project="type_prediction", entity="ml4code")

    if use_cuda:
        assert torch.cuda.is_available(), "CUDA not available. Check env configuration, or pass --use_cuda False"

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    pad_id = sp.PieceToId("[PAD]")

    id_to_target, target_to_id = load_type_vocab(type_vocab_filepath)
    no_type_id = target_to_id["O"]
    assert no_type_id == 0  # Just a sense check since O is the first line in the vocab file
    any_id = target_to_id["$any$"]

    collate_fn = get_collate_fn(pad_id, no_type_id)

    # Create training dataset and dataloader
    logger.info(f"Training data path {train_filepath}")
    train_dataset = DeepTyperDataset(
        train_filepath, type_vocab_filepath, spm_filepath, max_length=max_seq_len, subword_regularization_alpha=subword_regularization_alpha
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate_fn
    )

    # Create eval dataset and dataloader
    logger.info(f"Eval data path {eval_filepath}")
    eval_dataset = DeepTyperDataset(
        eval_filepath,
        type_vocab_filepath,
        spm_filepath,
        max_length=max_eval_seq_len,
        subword_regularization_alpha=subword_regularization_alpha,
    )
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )

    # Create model
    model = TypeTransformer(n_tokens=sp.GetPieceSize(), n_output_tokens=len(id_to_target), pad_id=pad_id)
    logger.info(f"Created TypeTransformer with {count_parameters(model)} params")

    # Set up optimizer
    model = nn.DataParallel(model)
    model = model.cuda() if use_cuda else model
    wandb.watch(model, log="all")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_eps, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, 5000, 200000)
    epoch = 0
    global_step = 0
    min_eval_metric = float("inf")

    # Load pretrained checkpoint
    if pretrain_resume_path:
        assert not resume_path
        logger.info(f"Loading pretrained parameters from {pretrain_resume_path}")
        checkpoint = torch.load(pretrain_resume_path)
        pretrained_state_dict = checkpoint["model_state_dict"]
        encoder_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if key.startswith("encoder_q.") and "project_layer" not in key:
                remapped_key = key[len("encoder_q.") :]
                logger.debug(f"Remapping checkpoint key {key} to {remapped_key}. Value mean: {value.mean().item()}")
                encoder_state_dict[remapped_key] = value
        model.module.encoder.load_state_dict(encoder_state_dict)
        logger.info(f"Loaded state dict from {pretrain_resume_path}")
    elif resume_path:
        logger.info(f"Loading parameters from {resume_path}")
        checkpoint = torch.load(resume_path)
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        min_eval_metric = checkpoint["min_eval_metric"]

    # Evaluate initial metrics
    logger.info(f"Evaluating model after epoch {epoch} ({global_step} steps)...")
    eval_metric, eval_metrics = _evaluate(model, eval_loader, sp, target_to_id=target_to_id, use_cuda=use_cuda)
    for metric, value in eval_metrics.items():
        logger.info(f"Evaluation {metric} after epoch {epoch} ({global_step} steps): {value:.4f}")
    eval_metrics["epoch"] = epoch
    wandb.log(eval_metrics, step=global_step)

    for epoch in tqdm.trange(epoch + 1, num_epochs + 1, desc="training", unit="epoch", leave=False):
        logger.info(f"Starting epoch {epoch}\n")
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}")
        for X, output_attn, labels in pbar:
            if use_cuda:
                X = X.cuda()
                output_attn = output_attn.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            logits = model(X, output_attn)  # BxLxVocab
            if ignore_any_loss:
                # Don't train with $any$ type
                labels_ignore_any = labels.clone()
                labels_ignore_any[labels_ignore_any == any_id] = no_type_id
                loss = F.cross_entropy(logits.transpose(1, 2), labels_ignore_any, ignore_index=no_type_id)
            else:
                loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=no_type_id)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Compute accuracy
            (acc1_any, acc5_any), _ = accuracy(logits, labels, topk=(1, 5), ignore_idx=(no_type_id,))
            (acc1, acc5), _ = accuracy(logits, labels, topk=(1, 5), ignore_idx=(no_type_id, target_to_id["$any$"]))

            # Log loss
            global_step += 1
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": loss.item(),
                    "train/acc@1": acc1,
                    "train/acc@5": acc5,
                    "train/acc@1_any": acc1_any,
                    "train/acc@5_any": acc5_any,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=global_step,
            )
            pbar.set_description(f"epoch {epoch} loss {loss.item():.4f}")

        # Evaluate
        logger.info(f"Evaluating model after epoch {epoch} ({global_step} steps)...")
        eval_metric, eval_metrics = _evaluate(
            model, eval_loader, sp, target_to_id=target_to_id, use_cuda=use_cuda,)
        for metric, value in eval_metrics.items():
            logger.info(f"Evaluation {metric} after epoch {epoch} ({global_step} steps): {value:.4f}")
        eval_metrics["epoch"] = epoch
        wandb.log(eval_metrics, step=global_step)

        # Save checkpoint
        if save_every and epoch % save_every == 0 or eval_metric < min_eval_metric:
            checkpoint = {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": config,
                "eval_metric": eval_metric,
                "min_eval_metric": min_eval_metric
            }
            if eval_metric < min_eval_metric:
                logger.info(f"New best evaluation metric: prev {min_eval_metric:.4f} > new {eval_metric:.4f}")
                min_eval_metric = eval_metric
                model_file = run_dir / f"ckpt_best.pth"
            else:
                model_file = run_dir / f"ckpt_ep{epoch:04d}.pth"
            logger.info(f"Saving checkpoint to {model_file}...")
            torch.save(checkpoint, str(model_file.resolve()))
            logger.info("Done.")


def eval(
    # Data
    eval_filepath: str,
    type_vocab_filepath: str,
    spm_filepath: str,
    num_workers=1,
    max_seq_len=-1,
    # Model
    resume_path: str = "",
    # Optimization
    batch_size=16,
    # Loss
    subword_regularization_alpha: float = 0,
    # Computational
    use_cuda: bool = True,
    seed: int = 0,
):
    """Train model"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = locals()
    logger.info(f"Config: {config}")

    if use_cuda:
        assert torch.cuda.is_available(), "CUDA not available. Check env configuration, or pass --use_cuda False"

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_filepath)
    pad_id = sp.PieceToId("[PAD]")

    id_to_target, target_to_id = load_type_vocab(type_vocab_filepath)
    no_type_id = target_to_id["O"]
    assert no_type_id == 0  # Just a sense check since O is the first line in the vocab file

    collate_fn = get_collate_fn(pad_id, no_type_id)

    # Create eval dataset and dataloader
    logger.info(f"Eval data path {eval_filepath}")
    eval_dataset = DeepTyperDataset(
        eval_filepath,
        type_vocab_filepath,
        spm_filepath,
        max_length=max_seq_len,
        subword_regularization_alpha=subword_regularization_alpha,
        split_source_targets_by_tab=True
    )
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )

    # Create model
    model = TypeTransformer(n_tokens=sp.GetPieceSize(), n_output_tokens=len(id_to_target), pad_id=pad_id)
    logger.info(f"Created TypeTransformer with {count_parameters(model)} params")
    model = nn.DataParallel(model)
    model = model.cuda() if use_cuda else model

    # Load checkpoint
    logger.info(f"Loading parameters from {resume_path}")
    checkpoint = torch.load(resume_path)
    model.module.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    # Evaluate metrics
    logger.info(f"Evaluating model after epoch {epoch} ({global_step} steps)...")
    eval_metric, eval_metrics = _evaluate(model, eval_loader, sp, target_to_id=target_to_id, use_cuda=use_cuda)
    for metric, value in eval_metrics.items():
        logger.info(f"Evaluation {metric} after epoch {epoch} ({global_step} steps): {value:.4f}")


def concatenate_files_in_list(
    project_list="../DeepTyper/data/test_projects.txt",
    project_dir="../DeepTyper/data/outputs-gold",
    output_file="../DeepTyper/data/test-outputs-gold.json"):
    """Utility method, used to concatenate DeepTyper test projects"""
    # Find json files
    json_paths = []
    with open(project_list, "r") as list_f:
        for filename in list_f:
            assert filename
            json_path = os.path.join(project_dir, filename.rstrip())
            json_path = os.path.abspath(json_path)
            print(json_path)
            assert os.path.isfile(json_path)
            json_paths.append(json_path)

    # Concatenate
    with open(output_file, "w") as output_f:
        for json_path in json_paths:
            with open(json_path, "r") as json_f:
                for line in json_f:
                    assert line
                    output_f.write(line)
                    if not line.endswith("\n"):
                        output_f.write("\n")

    logger.info("Wrote data to {}", output_file)


if __name__ == "__main__":
    fire.Fire({
        "train": train,
        "eval": eval,
        "concat": concatenate_files_in_list
    })