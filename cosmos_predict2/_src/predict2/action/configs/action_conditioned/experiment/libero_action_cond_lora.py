# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LIBERO action-conditioned LoRA fine-tuning experiment config.

LoRA fine-tunes Cosmos Predict2.5-2B on LIBERO data with dropout=0.1
for MC uncertainty estimation.

Base checkpoint: The pre-trained action-conditioned model
(nvidia/Cosmos-Predict2.5-2B, variant robot/action-cond, trained on Bridge)
is used as the starting point. When using lerobot-cosmos-train, this is
auto-resolved from the Cosmos checkpoint registry if --checkpoint_load_path
is not explicitly provided.

Usage:
    # Via LeRobot wrapper (auto-resolves base checkpoint):
    lerobot-cosmos-train \
        --dataset_repo_id lerobot/libero_10 \
        --cosmos_root /path/to/cosmos \
        --output_root outputs/cosmos_train

    # Direct Cosmos training (must provide checkpoint path):
    torchrun --nproc_per_node=1 --master_port=12341 \
        -m scripts.train \
        --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
        -- experiment=libero_action_cond_lora_2B \
        checkpoint.load_path=/path/to/action_cond_ema_bf16.pt \
        ~dataloader_train.dataloaders
"""

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict

# ---- LIBERO Action-Conditioned LoRA 2B ----
# Builds on the Bridge action-conditioned rectified flow 2B experiment,
# overriding data to LIBERO and enabling LoRA with dropout for MC uncertainty.

LIBERO_ACTION_COND_LORA_2B = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_480_640_",
            {"override /net": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /data_train": "libero_train"},
            {"override /data_val": "libero_val"},
        ],
        job=dict(
            group="libero_action_cond",
            name="libero_action_cond_lora_2B",
            project="cosmos_predict2_action_conditioned",
        ),
        optimizer=dict(
            lr=3e-4,
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.99],
            f_min=[0.4],
            warm_up_steps=[500],
            cycle_lengths=[50_000],
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path="",  # Set to base Cosmos Predict2.5-2B checkpoint path
            load_training_state=False,
            strict_resume=False,
            save_to_object_store=dict(enabled=False),
            load_from_object_store=dict(enabled=False),
        ),
        # Disable S3/cloud storage (local training only)
        upload_reproducible_setup=False,
        model_parallel=dict(
            context_parallel_size=1,
        ),
        model=dict(
            config=dict(
                # Disable text encoder to save ~14GB VRAM
                # Action conditioning doesn't need text embeddings
                text_encoder_config=None,
                # LoRA configuration with dropout for MC uncertainty
                use_lora=True,
                lora_rank=32,
                lora_alpha=32,
                lora_dropout=0.1,
                lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
                init_lora_weights=True,
                # Action conditioning: 1 conditional frame, 12 actions per chunk
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,  # 4 latent frames for 12 actions
                net=dict(
                    action_dim=7,
                    temporal_compression_ratio=4,
                ),
            ),
        ),
        trainer=dict(
            max_iter=20_000,
            logging_iter=100,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1000,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=1000,
                    do_x0_prediction=False,
                    guidance=[0],
                    fps=16,
                    save_s3=False,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=4,
            dataset=dict(
                gripper_rescale_factor=1,
                num_action_per_chunk=12,
                fps_downsample_ratio=1,
                video_size=[256, 320],
            ),
        ),
    ),
    flags={"allow_objects": True},
)


# ---- Debug variant (small iter count, mock-compatible) ----
LIBERO_ACTION_COND_LORA_2B_DEBUG = LazyDict(
    dict(
        defaults=[
            f"/experiment/{LIBERO_ACTION_COND_LORA_2B['job']['name']}",
            "_self_",
        ],
        job=dict(
            group="libero_action_cond_debug",
            name=f"{LIBERO_ACTION_COND_LORA_2B['job']['name']}_debug",
        ),
        trainer=dict(
            max_iter=100,
            logging_iter=10,
        ),
        checkpoint=dict(
            save_iter=50,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
    ),
    flags={"allow_objects": True},
)


# ---- Register experiments ----
cs = ConfigStore.instance()

cs.store(
    group="experiment",
    package="_global_",
    name=LIBERO_ACTION_COND_LORA_2B["job"]["name"],
    node=LIBERO_ACTION_COND_LORA_2B,
)
cs.store(
    group="experiment",
    package="_global_",
    name=LIBERO_ACTION_COND_LORA_2B_DEBUG["job"]["name"],
    node=LIBERO_ACTION_COND_LORA_2B_DEBUG,
)
