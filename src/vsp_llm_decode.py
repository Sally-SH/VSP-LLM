# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#============================ 69 ============================

import ast
from itertools import chain
import logging
import math
import os
import sys
import json
import hashlib
import editdistance
from argparse import Namespace
import pdb

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils, distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from omegaconf import DictConfig

from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    GenerationConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    FairseqDataclass,
)
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import OmegaConf, MISSING
import sacrebleu

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"

@dataclass
class OverrideConfig(FairseqDataclass):
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'noise wav file'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: float = field(default=0, metadata={'help': 'noise SNR in audio'})
    modalities: List[str] = field(default_factory=lambda: ["video"], metadata={'help': 'which modality to use'})
    data: Optional[str] = field(default=None, metadata={'help': 'path to test data directory'})
    label_dir: Optional[str] = field(default=None, metadata={'help': 'path to test label directory'})
    eval_bleu: bool = field(default=False, metadata={'help': 'evaluate bleu score'})
    llm_ckpt_path: str = field(default=MISSING, metadata={'help': 'path to llama checkpoint'})

@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    generation: GenerationConfig = GenerationConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    override: OverrideConfig = OverrideConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def main(cfg: DictConfig):

    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    assert cfg.common_eval.path is not None, "--path required for recognition!"
    
    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(cfg.common_eval.results_path, "decode.log")
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(cfg, h)

    return _main(cfg, sys.stdout)

from fairseq import tasks
from transformers import AutoTokenizer

def _main(cfg, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("hybrid.speech_recognize")
    if output_file is not sys.stdout:  # also print to stdout
        logger.addHandler(logging.StreamHandler(sys.stdout))

    utils.import_user_module(cfg.common)

    tokenizer = AutoTokenizer.from_pretrained(cfg.override.llm_ckpt_path)
    model_override_cfg = {'model':{'llm_ckpt_path':cfg.override.llm_ckpt_path}}
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([cfg.common_eval.path],model_override_cfg,strict=False)
    models = [model.eval() for model in models]
    saved_cfg.task.modalities = cfg.override.modalities
    task = tasks.setup_task(saved_cfg.task)
    task.build_tokenizer(saved_cfg.tokenizer)
    task.build_bpe(saved_cfg.bpe)

    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None :
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_cuda = torch.cuda.is_available()

    # Set dictionary
    dictionary = task.target_dictionary

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.cfg.llm_ckpt_path = cfg.override.llm_ckpt_path
    task.cfg.noise_prob = cfg.override.noise_prob
    task.cfg.noise_snr = cfg.override.noise_snr
    task.cfg.noise_wav = cfg.override.noise_wav
    if cfg.override.data is not None:
        task.cfg.data = cfg.override.data
    if cfg.override.label_dir is not None:
        task.cfg.label_dir = cfg.override.label_dir
    task.load_dataset('test', task_cfg=cfg.task)

    lms = [None]

    # Optimize ensemble for generation

    for model in chain(models, lms):
        if model is None:
            continue
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.encoder.cuda()
            model.avfeat_to_llm.cuda()
            model.half()

    # Load dataset (possibly sharded)
    cfg.dataset.batch_size = 1
    cfg.dataset.max_tokens = 1000
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    gen_timer = StopwatchMeter()
    def decode_fn(x):
        symbols_ignore = {"<unk>", "<mask>","<pad>", "</s>"}
        if hasattr(task.datasets[cfg.dataset.gen_subset].label_processors[0], 'decode'):
            return tokenizer.decode(x, skip_special_tokens=True)
        chars = dictionary.string(x, extra_symbols_to_ignore=symbols_ignore)
        words = " ".join("".join(chars.split()).replace('|', ' ').split())
        return words

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    result_dict = {'utt_id': [], 'ref': [], 'hypo': [], 'instruction': []}
    model = models[0]
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue
        
        sample['net_input']['source']['video'] = sample['net_input']['source']['video'].to(torch.half)
        best_hypo = model.generate(target_list=sample["target"], 
                                   num_beams=cfg.generation.beam, 
                                   length_penalty=cfg.generation.lenpen,
                                   **sample["net_input"])
        best_hypo = tokenizer.batch_decode(
                best_hypo, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        for i in range(len(sample["id"])):
            result_dict['utt_id'].append(sample['utt_id'][i])
            target = sample['target'][i].masked_fill(
                sample['target'][i] == -100, 0
            )
            ref_sent = tokenizer.decode(target.int().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result_dict['ref'].append(ref_sent)
            hypo_str = best_hypo[i]
            instruction = tokenizer.decode(sample['net_input']['source']['text'][i].int().cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result_dict['instruction'].append(instruction)
            result_dict['hypo'].append(hypo_str)
            logger.info(f"\nINST:{instruction}\nREF:{ref_sent}\nHYP:{hypo_str}\n")

    yaml_str = OmegaConf.to_yaml(cfg.generation)
    fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
    fid = fid % 1000000
    result_fn = f"{cfg.common_eval.results_path}/hypo-{fid}.json"
    json.dump(result_dict, open(result_fn, 'w'), indent=4)
    if not cfg.override.eval_bleu:
        n_err, n_total = 0, 0
        assert len(result_dict['hypo']) == len(result_dict['ref'])
        for hypo, ref in zip(result_dict['hypo'], result_dict['ref']):
            hypo, ref = hypo.strip().split(), ref.strip().split()
            n_err += editdistance.eval(hypo, ref)
            n_total += len(ref)
        wer = 100 * n_err / n_total
        wer_fn = f"{cfg.common_eval.results_path}/wer.{fid}"
        with open(wer_fn, "w") as fo:
            fo.write(f"WER: {wer}\n")
            fo.write(f"err / num_ref_words = {n_err} / {n_total}\n\n")
            fo.write(f"{yaml_str}")
        logger.info(f"WER: {wer}%")
    else:
        bleu = sacrebleu.corpus_bleu(result_dict['hypo'], [result_dict['ref']])
        bleu_score = bleu.score
        bleu_fn = f"{cfg.common_eval.results_path}/bleu.{fid}"
        with open(bleu_fn, "w") as fo:
            fo.write(f"BLEU: {bleu_score}\n")
            fo.write(f"{yaml_str}")
        logger.info(f"BLEU: {bleu_score}\n")
    return


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))
    return


def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()