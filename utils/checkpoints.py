import os
import torch
import logging

logger = logging.getLogger(__name__)

def save_ckp(args, ckp, is_best=False):
    # Save model checkpoint
    model_checkpoint = os.path.join(args.output_dir, args.wandb_id, "%s_checkpoint.bin" % args.name)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
    torch.save(ckp, model_checkpoint)
    if is_best:
        model_checkpoint = os.path.join(args.output_dir, args.wandb_id, "%s_best.bin" % args.name)
        torch.save(ckp, model_checkpoint)
        logger.info("Saved best model checkpoint to [DIR: %s]", args.output_dir)

def load_ckp(args, model, optimizer, scheduler):
    # Load model checkpoint
    checkpoint = torch.load(os.path.join(args.output_dir, args.wandb_id, "%s_checkpoint.bin" % args.name))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['wandb_step'], checkpoint['global_step'], checkpoint['epoch_step'], checkpoint['best_loss'], checkpoint['curriculum_it']


