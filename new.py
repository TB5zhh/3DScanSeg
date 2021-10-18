import logging
import os
import time
import random

import numpy as np
from numpy.core.fromnumeric import diagonal
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from lib.solvers import initialize_optimizer, initialize_scheduler
from lib.utils import save_predictions

from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset
from lib.datasets.scannet import COLOR_MAP
from lib.distributed_utils import all_gather_list
from models import load_model
from config import get_config

from IPython import embed
import MinkowskiEngine as ME

import wandb


def checkpoint(model, optimizer, scheduler, config, prefix='', **kwarg):
    """
    Save checkpoint of current model, optimizer, scheduler
    Other basic information are stored in kwargs
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    filename = f"{config.checkpoint_dir}/{config.run_name}{('_' +  prefix) if len(prefix) > 0 else ''}.pth"
    states = {
        'state_dict': model.module.state_dict(), # * load a GPU checkpoint to CPU
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': vars(config),
        **kwarg
    }

    torch.save(states, filename)


def setup_logger(config):
    """
    Logger setup function
    This function should only be called by main process in DDP
    """
    logging.root.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s][%(name)s\t][%(levelname)s\t] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    log_level = logging.INFO

    cli_handler = logging.StreamHandler()
    cli_handler.setFormatter(formatter)
    cli_handler.setLevel(log_level)

    if config.log_dir is not None:
        os.makedirs(config.log_dir, exist_ok=True)
        now = int(round(time.time() * 1000))
        timestr = time.strftime('%Y-%m-%d_%H:%M', time.localtime(now / 1000))
        filename = os.path.join(config.log_dir, f"{config.run_name}-{timestr}.log")

        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)

    logging.root.addHandler(cli_handler)
    logging.root.addHandler(file_handler)


def main():
    """
    Program entry
    Branch based on number of available GPUs
    """
    device_count = torch.cuda.device_count()
    if device_count > 1:
        port = random.randint(10000, 20000)
        init_method = f'tcp://localhost:{port}'
        mp.spawn(
            fn=main_worker,
            args=(device_count, init_method),
            nprocs=device_count,
        )
    else:
        main_worker()


def distributed_init(init_method, rank, world_size):
    """
    torch distributed iniitialized
    create a multiprocess group and initialize nccl communicator
    """
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda(rank))
        else:
            dist.all_reduce(torch.zeros(1))
        return dist.get_rank()
    logging.getLogger().warn("Distributed already initialized!")


def main_worker(rank=0, world_size=1, init_method=None):
    """
    Top pipeline
    """

    # + Device and distributed setup
    if not torch.cuda.is_available():
        raise Exception("No GPU Found")
    device = rank
    if world_size > 1:
        distributed_init(init_method, rank, world_size)

    config = get_config()
    setup_logger(config)
    logger = logging.getLogger(__name__)
    if rank == 0:
        logger.info(f'Run with {world_size} cpu')

    torch.cuda.set_device(device)

    # Set seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    if rank == 0:
        logger.info("Running config")
        for key, value in vars(config).items():
            logger.info(f"---> {key:>30}: {value}")  # pylint: disable=W1203

    # Set up test dataloader
    train_dataset_cls = load_dataset(config.train_dataset)
    val_dataset_cls = load_dataset(config.val_dataset)

    # hint: use phase to select different split of data
    train_dataloader = initialize_data_loader(
        train_dataset_cls,
        config,
        num_workers=config.num_workers,
        phase='train',
        augment_data=True,
        shuffle=True,
        repeat=True,
        batch_size=config.train_batch_size,
        limit_numpoints=False,
    )

    val_dataloader = initialize_data_loader(
        val_dataset_cls,
        config,
        num_workers=config.num_workers,
        phase='val',
        augment_data=False,
        shuffle=False,
        repeat=False,
        batch_size=config.val_batch_size,
        limit_numpoints=False,
    )
    if rank == 0:
        logger.info("Dataloader setup done")

    # Setup model
    num_in_channel = 3  # RGB
    num_labels = val_dataloader.dataset.NUM_LABELS
    model_class = load_model(config.model)
    model = model_class(num_in_channel, num_labels, config)

    # Load pretrained weights
    if config.resume:
        state = torch.load(config.resume, map_location=f'cuda:{device}')
        model.load_state_dict(state['state_dict'])
        if rank == 0:
            logger.info(f"Checkpoint resumed from {config.resume}")  # pylint: disable=W1203
    elif config.weights:
        state = torch.load(config.weights, map_location=f'cuda:{device}')
        model.load_state_dict({k: v for k, v in state['state_dict'].items() if not k.startswith('projector.')}) 
        if rank == 0:
            logger.info(f"Weights loaded from {config.weights}")  # pylint: disable=W1203

    model = model.to(device)
    if rank == 0:
        logger.info("Model setup done")
        logger.info(f"\n{model}")  # pylint: disable=W1203

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[device],
            output_device=[device],
            broadcast_buffers=False,
            # bucket_cap_mb=
        )
    if config.wandb and rank == 0:
        wandb.login('6653cea7375b6dd706fe1386010d63e776edc4d4')
        wandb.init(project='3dsp', entity='tb5zhh')
        wandb.config.update(config)
        # wandb.watch(model)

    train(model, train_dataloader, val_dataloader, config, logger, rank=rank, world_size=world_size)

    dist.destroy_process_group()
    # TODO add unc inference


def unc_demo(model, dataloader, config, logger):
    """
    Run multiple inference and obtain uncertainty for each point
    Then save the uncertainty result
    """

    # todo DEPRECATED device_id
    device = f"cuda:{config.device_id}"

    # Set model to eval mode except for the last dropout layer
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            logger.warn(f"module {m.__class__.__name__} set to train")
            m.train()

    with torch.no_grad():

        global_cnt = 0
        for step, batched_data in enumerate(dataloader):
            logger.info(f"{step}/{len(dataloader)} done")
            batched_coords, batched_feats, _ = batched_data

            # Normalize color
            batched_feats[:, :3] = batched_feats[:, :3] / 255. - 0.5

            # Feed forward
            batched_sparse_input = ME.SparseTensor(batched_feats.to(device), batched_coords.to(device))

            multiround_batched_scores = []
            for i in range(config.unc_round):
                logger.info(f"---> feed forward #{i} round")
                batched_sparse_output = model(batched_sparse_input)

                multiround_batched_scores.append(batched_sparse_output.F.cpu())

            multiround_batched_scores = torch.stack(multiround_batched_scores)
            # Batchsize * Pointcount * Classcount
            batched_scores = multiround_batched_scores.var(dim=0)
            for i in range(config.test_batch_size):
                logger.info(f"---> processing #{i} scene in the batch")
                selector = batched_coords[:, 0] == i
                single_scene_coords = batched_coords[selector][:, 1:]
                single_scene_scores = batched_scores[selector]
                save_prediction(single_scene_coords, single_scene_scores, f"{config.unc_result_dir}/{global_cnt}.ply", mode='unc')
                global_cnt += 1
            #     single_scene_scores = batched_scores


def unc_inference(model, dataloader, config, logger):
    """
    Run multiple inference and obtain uncertainty results for each point
    And:
    - save the statistics of the uncertainty
    - save scenes with augmented labels
    """
    # TODO DEPRECATED device_id
    device = f"cuda:{config.device_id}"

    # Set model to eval mode except for the last dropout layer
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            logger.warn(f"module {m.__class__.__name__} set to train")
            m.train()

    with torch.no_grad():

        global_cnt = 0
        all_scores = []
        for step, batched_data in enumerate(dataloader):
            logger.info(f"{step}/{len(dataloader)} done")
            batched_coords, batched_feats, _ = batched_data

            # Normalize color
            batched_feats[:, :3] = batched_feats[:, :3] / 255. - 0.5

            # Feed forward
            batched_sparse_input = ME.SparseTensor(batched_feats.to(device), batched_coords.to(device))

            multiround_batched_scores = []
            for i in range(config.unc_round):
                logger.info(f"---> feed forward #{i} round")
                batched_sparse_output = model(batched_sparse_input)

                multiround_batched_scores.append(batched_sparse_output.F.cpu())

            # Batchsize * Pointcount * Classcount
            multiround_batched_scores = torch.stack(multiround_batched_scores)

            batched_scores = multiround_batched_scores.var(dim=0)
            all_scores.append(batched_scores)

            for i in range(config.test_batch_size):
                logger.info(f"---> processing #{i} scene in the batch")
                selector = batched_coords[:, 0] == i
                single_scene_coords = batched_coords[selector][:, 1:]
                single_scene_scores = batched_scores[selector]
                # FIXME change this to prediction mode
                save_prediction(single_scene_coords, single_scene_scores, f"{config.unc_result_dir}/{global_cnt}.ply", mode='unc')
                global_cnt += 1
            #     single_scene_scores = batched_scores

        # Save uncertainty of all points
        all_scores = torch.vstack(all_scores)
        torch.save(all_scores, config.unc_stat_path)


def save_prediction(coords, feats, path, mode='unc'):
    """
    save prediction for **ONE** scene
    mode:
    - 'unc': treat feats as the uncertainty of each class (batchsize * num_cls)
    - 'prediction': treat feats as the predicted label (batchsize * 1)s
    """
    if mode == 'unc':
        feats = feats.min(dim=1)[0]

        coef = 8  # exp smoothing
        feats = (1 - torch.exp(-coef * feats / feats.max())) * 128

        def rgbl_fn(feat):
            return 128 + int(feat), 255 - int(feat), 127, 0
    elif mode == 'prediction':

        def rgbl_fn(feat):
            return (*COLOR_MAP[feat], feat)
    else:
        raise NotImplementedError

    with open(path, 'w') as f:  # pylint: disable=invalid-name
        f.write('ply\n'
                'format ascii 1.0\n'
                f'element vertex {coords.shape[0]}\n'
                'property float x\n'
                'property float y\n'
                'property float z\n'
                'property uchar red\n'
                'property uchar green\n'
                'property uchar blue\n'
                'property uchar label\n'
                'end_header\n')
        for row, feat in zip(coords, feats):
            f.write(f'{row[0]} {row[1]} {row[2]} ' f'{rgbl_fn(feat)[0]} ' f'{rgbl_fn(feat)[1]} ' f'{rgbl_fn(feat)[2]} ' f'{rgbl_fn(feat)[3]}\n')


def calc_iou(predictions, labels, num_labels):
    """Calculate mIoU over a bunch of predictions and labels. The input tensors should be flattened"""
    selector = (labels >= 0) & (labels < num_labels)  # ndarray of bool type, selector
    stat = torch.bincount(num_labels * labels[selector].int() + predictions[selector], minlength=num_labels**2).reshape(num_labels, num_labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        return stat.diagonal() / (stat.sum(dim=0) + stat.sum(dim=1) - stat.diagonal() + 1e-7)


def inference(model, dataloader, config, logger, rank=0, world_size=1, save=False, evaluate=True):
    """Evaluation on val / test dataset"""
    torch.cuda.empty_cache()
    device = f'cuda:{rank}'

    model.eval()

    with torch.no_grad():

        global_cnt = 0

        losses = []
        ious = []
        # One epoch
        for step, (batched_coords, batched_feats, batched_targets) in enumerate(dataloader):

            # Normalize color
            batched_feats[:, :3] = batched_feats[:, :3] / 255. - 0.5

            # Feed forward
            batched_sparse_input = ME.SparseTensor(batched_feats.to(device), batched_coords.to(device))

            batched_sparse_output = model(batched_sparse_input)
            batched_outputs = batched_sparse_output.F.cpu()
            batched_prediction = batched_outputs.max(dim=1)[1].int()

            # if world_size == 1 or rank == 0:
            #     logger.info(f"---> inference step #{step} of {len(dataloader)} ")
            # Save predictions for each scene
            if save:
                for i in range(config.train_batch_size):
                    selector = batched_coords[:, 0] == i
                    single_scene_coords = batched_coords[selector][:, 1:]
                    single_scene_predictions = batched_prediction[selector]
                    save_prediction(single_scene_coords, single_scene_predictions, f"{config.eval_result_dir}/{global_cnt}.ply", mode="prediction")
                    global_cnt += 1
            else:
                global_cnt += config.train_batch_size

            # Evaluate prediction
            if evaluate:
                # batched_targets = batched_targets.to(device)
                criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
                num_labels = dataloader.dataset.NUM_LABELS

                loss = criterion(batched_outputs, batched_targets.long())
                iou = calc_iou(batched_prediction.flatten(), batched_targets.flatten(), num_labels)
                iou = iou.mean()

                if world_size == 1 or rank == 0:
                    logger.info(f"---> inference #{step} of {len(dataloader)} loss: {loss:.4f} iou: {iou:.4f}")
                losses.append(loss)
                ious.append(iou)

                # TODO calculate average precision
                probablities = F.softmax(batched_outputs, dim=1)  # pylint: disable=unused-variable
                # avg_precision =

    if evaluate:
        return torch.stack(losses).mean(), torch.stack(ious).mean()


# !!! The output of the model should be .cpu()


def train(model, dataloader, val_dataloader, config, logger, rank=0, world_size=1):
    """TODO"""

    logger.info(f'My rank : {rank} / {world_size}')
    device = rank

    optimizer = initialize_optimizer(model.parameters(), config)
    scheduler = initialize_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)

    if rank == 0:
        logger.info("Start training")

    # Load state dictionaries of optimizer and scheduler from checkpoint
    if config.resume:
        states = torch.load(config.resume, map_location=f"cuda:{device}")
        optimizer.load_state_dict(states['optimizer'])
        scheduler.load_state_dict(states['scheduler'])
        start_epoch = states['epoch'] + 1
        global_step = states['global_step']
        optim_step = states['optim_step']
        val_step = states['val_step']
        best_miou = states['best_miou']
        best_val_loss = states['val_loss']
    else:
        start_epoch = 0
        global_step = 0
        optim_step = 0
        val_step = 0
        best_miou = 0
        best_val_loss = float('inf')

    # TODO add metric meters
    total_step = 0
    losses = []
    precisions = []
    for epoch in range(start_epoch, config.train_epoch):
        for step, (coords, feats, targets) in enumerate(dataloader):
            # FIXME !!! Avoid train and val data taking space simultaneously

            torch.cuda.empty_cache()
            model.train()

            # TODO set seed here for certainty

            coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

            feats[:, :3] = feats[:, :3] / 255. - 0.5
            sparse_input = ME.SparseTensor(feats, coords, device=device)
            sparse_output = model(sparse_input)

            targets = targets.long().to(device)
            loss = criterion(sparse_output.F, targets)

            prediction = sparse_output.F.max(dim=1)[1]
            precision = prediction.eq(targets).view(-1).float().sum(dim=0).mul(100. / prediction.shape[0])

            loss /= config.optim_step
            loss.backward()
            losses.append(loss)
            precisions.append(precision)

            total_step += 1
            # periodic optimization
            if total_step % config.optim_step == 0:
                global_step += 1
                optim_step += 1
                loss = torch.tensor(losses).sum(dim=0).to(f"cuda:{device}")
                precision = torch.tensor(precisions).mean(dim=0).to(f"cuda:{device}")
                losses.clear()
                precisions.clear()
                # loss /= config.optim_step
                optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()

                # Synchronize
                # obj = {'loss': loss.item(), 'precision': precision}
                if world_size > 1:
                    loss_list = [torch.zeros_like(loss) for _ in range(world_size)]
                    prec_list = [torch.zeros_like(precision) for _ in range(world_size)]
                    dist.all_gather(loss_list, loss)
                    dist.all_gather(prec_list, precision)
                    loss = torch.stack(loss_list).mean(dim=0)
                    precision = torch.stack(prec_list).mean(dim=0)
                    # obj = {k: np.mean([item[k] for item in obj]) for k in obj[0]}

                if world_size == 1 or rank == 0:

                    def get_lr(optimizer):
                        for param_group in optimizer.param_groups:
                            return param_group['lr']
                    logger.info(
                        f"TRAIN at global step #{global_step} Epoch #{epoch+1} Step #{step+1} / {len(dataloader)}: loss:{loss:.4f}, precision: {precision:.4f}")
                    if config.wandb:
                        obj = {
                            'loss': loss.cpu().item(),
                            'precision': precision.cpu().item(),
                            'learning rate': get_lr(optimizer),
                        }
                        wandb.log({
                            'optim_step': optim_step,
                            'global_step': global_step,
                            **obj,
                        })

                del coords
                del feats
                del targets
                del sparse_input
                del sparse_output
                del loss
                del prediction
                del precision
                torch.cuda.empty_cache()

                # periodic evaluation
                # This step take extra GPU memory so clearup in advance is needed
                if optim_step % config.validate_step == 0 and rank == 0:
                    val_step += 1
                    val_loss, val_miou = inference(model, val_dataloader, config, logger, rank=rank, world_size=world_size, evaluate=True)

                    if world_size == 1 or rank == 0:
                        logger.info(f"VAL   at global step #{global_step}: loss (avg): {val_loss.item():.4f}, iou (avg): {val_miou.item():.4f}")
                        if config.wandb:
                            obj = {
                                'val_loss': val_loss.cpu().item(),
                                'val_miou_mean': val_miou.cpu().item(),
                            }
                            wandb.log({
                                'val_step': val_step,
                                'global_step': global_step,
                                **obj,
                            })
                    if val_miou.item() > best_miou:
                        best_miou = val_miou.item()
                        best_val_loss = val_loss.item()
                        logger.info(f"Better checkpoint saved")
                        if world_size == 1 or rank == 0:
                            checkpoint(
                                model,
                                optimizer,
                                scheduler,
                                config,
                                prefix='best',
                                step=step,
                                best_miou=best_miou,
                                val_loss=val_loss,
                                epoch=epoch,
                            )
            torch.cuda.empty_cache()

        # periodic checkpoint
        if (epoch + 1) % config.save_epoch == 0:
            if world_size == 1 or rank == 0:
                args = {
                    'best_miou': best_miou,
                    'val_loss': best_val_loss,
                    'epoch': epoch,
                    'global_step': global_step,
                    'optim_step': optim_step,
                    'val_step': val_step,
                }
                checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    config,
                    prefix='latest',
                    **args,
                )
                logger.info(f"Checkpoint at epoch #{epoch} saved")
    # Note: these steps should be outside the func
    # TODO Calculate uncertainty, and store the results
    # TODO Obtain augmented results


if __name__ == '__main__':
    main()
