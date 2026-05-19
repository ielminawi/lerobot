#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    OBS_ANSWER,
    OBS_ANSWER_ATTENTION_MASK,
    OBS_ANSWER_LABELS,
    OBS_ANSWER_TOKENS,
    OBS_ID_QUERY,
    OBS_ID_QUERY_ATTENTION_MASK,
    OBS_ID_QUERY_TOKENS,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def make_smolvla_pre_post_processors(
    config: SmolVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the SmolVLA policy.

    The pre-processing pipeline prepares input data for the model by:
    1.  Renaming features to match pretrained configurations.
    2.  Normalizing input and output features based on dataset statistics.
    3.  Adding a batch dimension.
    4.  Ensuring the language task description ends with a newline character.
    5.  Tokenizing the language task description.
    6.  Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1.  Moving data to the CPU.
    2.  Unnormalizing the output actions to their original scale.

    Args:
        config: The configuration object for the SmolVLA policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        SmolVLANewLineProcessor(),
        TokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding=config.pad_language_to,
            padding_side="right",
            max_length=config.tokenizer_max_length,
        ),
        AnswerTokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            padding="max_length",
            padding_side="right",
            max_length=config.answer_max_length,
        ),
        IdQueryTokenizerProcessorStep(
            tokenizer_name=config.vlm_model_name,
            id_query_prompt=config.id_query_prompt,
            padding="max_length",
            padding_side="right",
            max_length=config.id_query_max_length,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def extract_celebrity_name(task: str) -> str | None:
    """Return the celebrity name after the last ‘on’ in a task string.

    The task is expected to look like "place the coke on Taylor Swift".
    Matching is case-insensitive and trailing punctuation is stripped.
    """
    if not isinstance(task, str):
        return None

    task = task.strip()
    if not task:
        return None

    match = re.search(r"\bon\s+(?P<name>.+?)\s*[\.!?;,]*$", task, flags=re.IGNORECASE)
    if match is None:
        return None

    name = match.group("name").strip()
    if not name:
        return None

    name = re.sub(r"[\.!?;,]+$", "", name).strip()
    return name or None


@ProcessorStepRegistry.register(name="smolvla_new_line_processor")
class SmolVLANewLineProcessor(ComplementaryDataProcessorStep):
    """
    A processor step that ensures the 'task' description ends with a newline character.

    This step is necessary for certain tokenizers (e.g., PaliGemma) that expect a
    newline at the end of the prompt. It handles both single string tasks and lists
    of string tasks.
    """

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # Handle both string and list of strings
        if isinstance(task, str):
            # Single string: add newline if not present
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # List of strings: add newline to each if not present
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # If task is neither string nor list of strings, leave unchanged

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="smolvla_answer_tokenizer_processor")
class AnswerTokenizerProcessorStep(TokenizerProcessorStep):
    """Tokenizes the extracted celebrity answer and produces cross-entropy labels."""

    def __post_init__(self) -> None:
        self.padding = "max_length"
        self.truncation = True
        super().__post_init__()

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        task_list = self.get_task(self.transition)
        task_str = task_list[0] if task_list else ""
        celebrity_name = extract_celebrity_name(task_str)

        if celebrity_name is None:
            input_ids = torch.full((1, self.max_length), self._pad_token_id(), dtype=torch.long)
            attention_mask = torch.zeros((1, self.max_length), dtype=torch.bool)
            labels = torch.full((1, self.max_length), -100, dtype=torch.long)
        else:
            tokenized_answer = self._tokenize_text([celebrity_name])
            input_ids = tokenized_answer["input_ids"]
            attention_mask = tokenized_answer["attention_mask"].to(dtype=torch.bool)
            labels = input_ids.clone()
            labels[~attention_mask] = -100

        target_device = self._detect_device(self.transition)
        if target_device is not None:
            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device)
            labels = labels.to(target_device)

        new_observation = dict(observation)
        new_observation[OBS_ANSWER_TOKENS] = input_ids
        new_observation[OBS_ANSWER_ATTENTION_MASK] = attention_mask
        new_observation[OBS_ANSWER_LABELS] = labels
        return new_observation

    def _pad_token_id(self) -> int:
        return int(self.input_tokenizer.pad_token_id or 0)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = super().transform_features(features)

        if OBS_ANSWER_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_ANSWER_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE,
                shape=(self.max_length,),
            )

        if OBS_ANSWER_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_ANSWER_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE,
                shape=(self.max_length,),
            )

        if OBS_ANSWER_LABELS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_ANSWER_LABELS] = PolicyFeature(
                type=FeatureType.LANGUAGE,
                shape=(self.max_length,),
            )

        return features


@ProcessorStepRegistry.register(name="smolvla_id_query_tokenizer_processor")
class IdQueryTokenizerProcessorStep(TokenizerProcessorStep):
    """Tokenizes the identification query for text loss supervision."""

    def __init__(
        self,
        tokenizer_name: str | None = None,
        tokenizer: Any | None = None,
        id_query_prompt: str = "Who is the person shown in this image?",
        max_length: int = 24,
        padding_side: str = "right",
        padding: str = "max_length",
        truncation: bool = True,
        task_key: str = "dummy_not_used",  # accepted but always ignored — we tokenize a fixed prompt, not a per-sample task
    ):
        self.id_query_prompt = id_query_prompt
        super().__init__(
            tokenizer_name=tokenizer_name,
            tokenizer=tokenizer,
            max_length=max_length,
            padding_side=padding_side,
            padding=padding,
            truncation=truncation,
            task_key=task_key,
        )

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Tokenize the identification query."""
        tokenized_query = self._tokenize_text([self.id_query_prompt])
        input_ids = tokenized_query["input_ids"]
        attention_mask = tokenized_query["attention_mask"].to(dtype=torch.bool)

        target_device = self._detect_device(self.transition)
        if target_device is not None:
            input_ids = input_ids.to(target_device)
            attention_mask = attention_mask.to(target_device)

        new_observation = dict(observation)
        new_observation[OBS_ID_QUERY_TOKENS] = input_ids
        new_observation[OBS_ID_QUERY_ATTENTION_MASK] = attention_mask
        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if OBS_ID_QUERY_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_ID_QUERY_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE,
                shape=(self.max_length,),
            )

        if OBS_ID_QUERY_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_ID_QUERY_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE,
                shape=(self.max_length,),
            )

        return features
