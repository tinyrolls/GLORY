import logging
import os.path
import zipfile
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm

from dataload.data_load import load_data
from utils.common import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main_worker(local_rank, cfg):
    # Environment Initial
    seed_everything(cfg.seed)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=cfg.gpu_num,
                            rank=local_rank)

    test_prefix = f"{cfg.model.model_name}_{cfg.dataset.dataset_name}_{cfg.load_mark}"

    # Dataset & Model Load
    ckpt_file = Path(f"{cfg.path.ckp_dir}/{test_prefix}.pth")
    if not ckpt_file.exists():
        assert False, f"wrong ckpt path: {ckpt_file}"

    model = load_model(cfg).to(local_rank)
    model.load_state_dict(torch.load(ckpt_file, map_location='cpu')['model_state_dict'])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    dataloader = load_data(cfg, mode='test', model=model)

    # ----------------------- Prediction -------------------------
    model.eval()
    torch.set_grad_enabled(False)

    # Prediction
    pred_txt_path = Path(cfg.path.pred_dir) /f"{test_prefix}_prediction.txt"
    pred_zip_path = Path(cfg.path.pred_dir) /f"{test_prefix}_prediction.zip"

    pred_txt_path.parent.mkdir(parents=True, exist_ok=True)
    pred_file = open(pred_txt_path, 'w')

    total_index = 1
    with torch.no_grad():
        for cnt, (subgraph, mappings, candidate_input) \
                in enumerate(tqdm(dataloader, total=int(cfg.dataset.test_len), desc="Predicting")):
            # User Emb
            candidate_emb = torch.FloatTensor(np.array(candidate_input)).to(device, non_blocking=True)

            y_pred = model.module.validation_process(subgraph, mappings, candidate_emb)

            pred_rank = (np.argsort(np.argsort(y_pred)[::-1]) + 1).tolist()
            pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
            pred_file.write(' '.join([str(total_index), pred_rank]) + '\n')
            total_index += 1

    pred_file.close()
    f = zipfile.ZipFile(pred_zip_path, 'w', zipfile.ZIP_DEFLATED)
    f.write(pred_txt_path, arcname='prediction.txt')
    f.close()


@hydra.main(version_base="1.2", config_path=os.path.join(get_root(), "configs"), config_name="small")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    cfg.gpu_num = 1
    mp.spawn(main_worker, nprocs=cfg.gpu_num, args=(cfg,))


if __name__ == "__main__":
    main()
