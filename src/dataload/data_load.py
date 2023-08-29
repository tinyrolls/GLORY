import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import to_undirected
from tqdm import tqdm
import pickle

from dataload.dataset import *

def load_data(cfg, mode='train', model=None, local_rank=0):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    # ------------- load news.tsv-------------
    news_index = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))

    news_input = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
    # ------------- load behaviors_np{X}.tsv --------------
    if mode == 'train':
        target_file = Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_{local_rank}.tsv"
        if cfg.model.use_graph:
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

            if cfg.model.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
            print(f"[{mode}] News Graph Info: {news_graph}")


            news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))

            if cfg.model.use_entity:
                entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            dataset = TrainGraphDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
                neighbor_dict=news_neighbors_dict,
                news_graph=news_graph,
                entity_neighbors=entity_neighbors
            )
            dataloader = DataLoader(dataset, batch_size=None)
            
        else:
            dataset = TrainDataset(
                filename=target_file,
                news_index=news_index,
                news_input=news_input,
                local_rank=local_rank,
                cfg=cfg,
            )

            dataloader = DataLoader(dataset,
                                    batch_size=int(cfg.batch_size / cfg.gpu_num),
                                    pin_memory=True)
        return dataloader
    elif mode in ['val', 'test']:
        # convert the news to embeddings
        news_dataset = NewsDataset(news_input)
        news_dataloader = DataLoader(news_dataset,
                                     batch_size=int(cfg.batch_size * cfg.gpu_num),
                                     num_workers=cfg.num_workers)

        stacked_news = []
        with torch.no_grad():
            for news_batch in tqdm(news_dataloader, desc=f"[{local_rank}] Processing validation News Embedding"):
                if cfg.model.use_graph:
                    batch_emb = model.module.local_news_encoder(news_batch.long().unsqueeze(0).to(local_rank)).squeeze(0).detach()
                else:
                    batch_emb = model.module.local_news_encoder(news_batch.long().unsqueeze(0).to(local_rank)).squeeze(0).detach()
                stacked_news.append(batch_emb)
        news_emb = torch.cat(stacked_news, dim=0).cpu().numpy()   

        if cfg.model.use_graph:
            news_graph = torch.load(Path(data_dir[mode]) / "nltk_news_graph.pt")

            news_neighbors_dict = pickle.load(open(Path(data_dir[mode]) / "news_neighbor_dict.bin", "rb"))

            if cfg.model.directed is False:
                news_graph.edge_index, news_graph.edge_attr = to_undirected(news_graph.edge_index, news_graph.edge_attr)
            print(f"[{mode}] News Graph Info: {news_graph}")

            if cfg.model.use_entity:
                # entity_graph = torch.load(Path(data_dir[mode]) / "entity_graph.pt")
                entity_neighbors = pickle.load(open(Path(data_dir[mode]) / "entity_neighbor_dict.bin", "rb"))
                total_length = sum(len(lst) for lst in entity_neighbors.values())
                print(f"[{mode}] entity_neighbor list Length: {total_length}")
            else:
                entity_neighbors = None

            if mode == 'val':
                dataset = ValidGraphDataset(
                    filename=Path(data_dir[mode]) / f"behaviors_np{cfg.npratio}_{local_rank}.tsv",
                    news_index=news_index,
                    news_input=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                    neighbor_dict=news_neighbors_dict,
                    news_graph=news_graph,
                    news_entity=news_input[:,-8:-3],
                    entity_neighbors=entity_neighbors
                )

            dataloader = DataLoader(dataset, batch_size=None)

        else:
            if mode == 'val':
                dataset = ValidDataset(
                    filename=Path(data_dir[mode]) / f"behaviors_{local_rank}.tsv",
                    news_index=news_index,
                    news_emb=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                )
            else:
                dataset = ValidDataset(
                    filename=Path(data_dir[mode]) / f"behaviors.tsv",
                    news_index=news_index,
                    news_emb=news_emb,
                    local_rank=local_rank,
                    cfg=cfg,
                )

            dataloader = DataLoader(dataset,
                                    batch_size=1,
                                    # batch_size=int(cfg.batch_   size / cfg.gpu_num),
                                    # pin_memory=True, # collate_fn already puts data to GPU
                                    collate_fn=lambda b: collate_fn(b, local_rank))
        return dataloader


def collate_fn(tuple_list, local_rank):
    clicked_news = [x[0] for x in tuple_list]
    clicked_mask = [x[1] for x in tuple_list]
    candidate_news = [x[2] for x in tuple_list]
    clicked_index = [x[3] for x in tuple_list]
    candidate_index = [x[4] for x in tuple_list]

    if len(tuple_list[0]) == 6:
        labels = [x[5] for x in tuple_list]
        return clicked_news, clicked_mask, candidate_news, clicked_index, candidate_index, labels
    else:
        return clicked_news, clicked_mask, candidate_news, clicked_index, candidate_index
