# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
import editdistance

@register_criterion("decoder_only_language_modeling_loss", dataclass=FairseqDataclass)
class decoder_only_language_modeling_loss(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        loss, lprobs = model(target_list=sample["target"], 
                        target_attn_mask=sample['target_attn_mask'],
                        **sample["net_input"])

        sample_size = (
            sample["target"].size()[0]
        )

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        n_correct, total = self.compute_accuracy(lprobs, sample)
        logging_output["n_correct"] = utils.item(n_correct.data)
        logging_output["total"] = utils.item(total.data)

        return loss, sample_size, logging_output

    
    def compute_accuracy(self, lprobs, sample):
        target = sample['net_input']['prev_output_tokens']
        
        b,t = target.size()
        mask = sample['target_attn_mask'] == 1
        n_correct = torch.sum(lprobs[:,-t:].argmax(2).masked_select(mask).eq(target.masked_select(mask)))
        total = torch.sum(mask)

        return n_correct, total

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False

