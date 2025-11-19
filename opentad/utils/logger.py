import logging
import os
import sys


def setup_logger(name, save_dir, distributed_rank=0, args=None, filename="log.json"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    filename = f'{args.eps}_{args.stepsize_adv}_{args.iterations_adv}_{args.loss}_{args.inner_loss}_{args.na}_{filename}' if args is not None else filename
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
