# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_layoutlmv3": [
        "CUSTOM_LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "LayoutLMv3Config",
        "LayoutLMv3OnnxConfig",
    ],
    # "processing_layoutlmv3": ["LayoutLMv3Processor"],
    # "tokenization_layoutlmv3": ["LayoutLMv3Tokenizer"],
}

# try:
#     if not is_tokenizers_available():
#         raise OptionalDependencyNotAvailable()
# except OptionalDependencyNotAvailable:
#     pass
# else:
#     _import_structure["tokenization_layoutlmv3_fast"] = ["LayoutLMv3TokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_layoutlmv3"] = [
        "CUSTOM_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CUSTOM_LayoutLMv3ForQuestionAnswering",
        "CUSTOM_LayoutLMv3ForSequenceClassification",
        "CUSTOM_LayoutLMv3ForTokenClassification",
        "CUSTOM_LayoutLMv3Model",
        "CUSTOM_LayoutLMv3PreTrainedModel",
    ]

# try:
#     if not is_tf_available():
#         raise OptionalDependencyNotAvailable()
# except OptionalDependencyNotAvailable:
#     pass
# else:
#     _import_structure["modeling_tf_layoutlmv3"] = [
#         "TF_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST",
#         "TFLayoutLMv3ForQuestionAnswering",
#         "TFLayoutLMv3ForSequenceClassification",
#         "TFLayoutLMv3ForTokenClassification",
#         "TFLayoutLMv3Model",
#         "TFLayoutLMv3PreTrainedModel",
#     ]

# try:
#     if not is_vision_available():
#         raise OptionalDependencyNotAvailable()
# except OptionalDependencyNotAvailable:
#     pass
# else:
#     _import_structure["feature_extraction_layoutlmv3"] = ["LayoutLMv3FeatureExtractor"]
#     _import_structure["image_processing_layoutlmv3"] = ["LayoutLMv3ImageProcessor"]


if TYPE_CHECKING or True:
    # print("TYPE_CHECKING is TRUE")
    from .configuration_layoutlmv3 import (
        LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CustomLayoutLMv3Config,
        LayoutLMv3OnnxConfig,
    )
    # from .processing_layoutlmv3 import CustomLayoutLMv3Processor
    # from .tokenization_layoutlmv3 import CustomLayoutLMv3Tokenizer

    # try:
    #     if not is_tokenizers_available():
    #         raise OptionalDependencyNotAvailable()
    # except OptionalDependencyNotAvailable:
    #     pass
    # else:
    #     from .tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_layoutlmv3 import (
            LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST,
            CustomLayoutLMv3ForQuestionAnswering,
            CustomLayoutLMv3ForSequenceClassification,
            CustomLayoutLMv3ForTokenClassification,
            CustomLayoutLMv3Model,
            CustomLayoutLMv3Model,
            CustomLayoutLMv3PreTrainedModel,
        )

    # try:
    #     if not is_tf_available():
    #         raise OptionalDependencyNotAvailable()
    # except OptionalDependencyNotAvailable:
    #     pass
    # else:
    #     from .modeling_tf_layoutlmv3 import (
    #         TF_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST,
    #         TFLayoutLMv3ForQuestionAnswering,
    #         TFLayoutLMv3ForSequenceClassification,
    #         TFLayoutLMv3ForTokenClassification,
    #         TFLayoutLMv3Model,
    #         TFLayoutLMv3PreTrainedModel,
    #     )

    # try:
    #     if not is_vision_available():
    #         raise OptionalDependencyNotAvailable()
    # except OptionalDependencyNotAvailable:
    #     pass
    # else:
    #     from .feature_extraction_layoutlmv3 import LayoutLMv3FeatureExtractor
    #     from .image_processing_layoutlmv3 import LayoutLMv3ImageProcessor

# else:
#     import sys
#     print("TYPE_CHECKING is FALSE")
#
#     sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, \
    AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer


AutoConfig.register("custom-layoutlmv3", CustomLayoutLMv3Config)
AutoModel.register(CustomLayoutLMv3Config, CustomLayoutLMv3Model)
AutoModelForTokenClassification.register(CustomLayoutLMv3Config, CustomLayoutLMv3ForTokenClassification)
AutoModelForQuestionAnswering.register(CustomLayoutLMv3Config, CustomLayoutLMv3ForQuestionAnswering)
AutoModelForSequenceClassification.register(CustomLayoutLMv3Config, CustomLayoutLMv3ForSequenceClassification)

# AutoTokenizer.register(
#     LayoutLMv3Config, slow_tokenizer_class=LayoutLMv3Tokenizer, fast_tokenizer_class=LayoutLMv3TokenizerFast
# )
# SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv3Tokenizer": RobertaConverter})
