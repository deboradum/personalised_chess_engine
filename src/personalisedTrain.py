# Copyright 2025 Pepijn van Wijk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import yaml
import argparse
from typing import Literal
from dataclasses import dataclass
from dacite import from_dict

from personalised_chess_engine.src import config as config_lib
from personalised_chess_engine.src import data_loader
from personalised_chess_engine.src import metrics_evaluator
from personalised_chess_engine.src import tokenizer
from personalised_chess_engine.src import training
from personalised_chess_engine.src import transformer
from personalised_chess_engine.src import utils

import logging
import absl.logging

def suppress_warning_logs():
    class SpecificWarningFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()

            if "PositionalSharding" in msg:
                return False
            if "_SignalingThread.join()" in msg:
                return False
            if "waiting for signals" in msg and "blocking save times" in msg:
                return False

            return True

    warning_filter = SpecificWarningFilter()
    logging.getLogger().addFilter(warning_filter)
    logging.getLogger('absl').addFilter(warning_filter)

    try:
        absl.logging.get_absl_handler().addFilter(warning_filter)
    except Exception:
        pass
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
suppress_warning_logs()

@dataclass
class ModelConfig:
    embedding_dim: int = 32
    num_layers: int = 4
    num_heads: int = 4
    use_causal_mask: bool = False
    positional_encodings: Literal["learned", "sinusoid"] = "sinusoid"
    widening_factor: int = 4
    apply_qk_layernorm: bool = False
    apply_post_ln: bool = True

@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    num_steps: int = 20
    ckpt_frequency: int = 5
    ckpt_max_to_keep: int = 3
    log_frequency: int = 1
    save_frequency: int = 10
    batch_size: int = 64
    worker_count: int = 0  # 0 disables multiprocessing.
    num_eval_data: int = 256

@dataclass
class PersonalisedConfig:
    seed: int = 12345
    model_config: ModelConfig = ModelConfig()
    train_config: TrainConfig = TrainConfig()


def parse_config(path: str = "personalised_config.yaml") -> PersonalisedConfig:
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)

    return from_dict(data_class=PersonalisedConfig, data=raw_config)

def main(config: PersonalisedConfig, username: str):
    num_return_buckets = 128
    output_size = utils.NUM_ACTIONS

    if config.model_config.positional_encodings == "learned":
        pos_encodings = transformer.PositionalEncodings.LEARNED
    else:
        pos_encodings = transformer.PositionalEncodings.SINUSOID

    predictor_config = transformer.TransformerConfig(
        seed=config.seed,
        vocab_size=utils.NUM_ACTIONS,
        output_size=output_size,
        embedding_dim=config.model_config.embedding_dim,
        num_layers=config.model_config.num_layers,
        num_heads=config.model_config.num_heads,
        use_causal_mask=config.model_config.use_causal_mask,
        pos_encodings=pos_encodings,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
        widening_factor=config.model_config.widening_factor,
        apply_qk_layernorm=config.model_config.apply_qk_layernorm,
        apply_post_ln=config.model_config.apply_post_ln,
    )
    train_config = config_lib.TrainConfig(
        data=config_lib.DataConfig(
            batch_size=config.train_config.batch_size,
            shuffle=True,
            worker_count=config.train_config.worker_count,
            num_return_buckets=num_return_buckets,
            policy="personal_cloning",
            username=username,
            split='train',
        ),
        learning_rate=config.train_config.learning_rate,
        max_grad_norm=config.train_config.max_grad_norm,
        num_steps=config.train_config.num_steps,
        ckpt_frequency=config.train_config.ckpt_frequency,
        ckpt_max_to_keep=config.train_config.ckpt_max_to_keep,
        save_frequency=config.train_config.save_frequency,
        log_frequency=config.train_config.log_frequency,
    )
    eval_config = config_lib.EvalConfig(
        data=config_lib.DataConfig(
            batch_size=1,
            shuffle=False,
            worker_count=config.train_config.worker_count,
            num_return_buckets=num_return_buckets,
            policy=None,  # pytype: disable=wrong-arg-types
            split='test',
        ),
        use_ema_params=True,
        policy="personal_cloning",
        batch_size=config.train_config.batch_size,
        num_return_buckets=num_return_buckets,
        num_eval_data=config.train_config.num_eval_data,
    )

    # TODO: allow resume from checkpoint as well as train from scratch

    params = training.train(
        train_config=train_config,
        predictor_config=predictor_config,
        build_data_loader=data_loader.build_data_loader,
    )

    predictor = transformer.build_transformer_predictor(predictor_config)
    evaluator = metrics_evaluator.build_evaluator(predictor, eval_config)
    print(evaluator.step(params=params, step=train_config.num_steps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to train config")
    parser.add_argument("--username", type=str, required=True, help="The username to clone")
    args = parser.parse_args()

    config = parse_config(args.config)

    main(config, args.username)
