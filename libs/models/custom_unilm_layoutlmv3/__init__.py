from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, \
    AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter

from .configuration_layoutlmv3 import CustomLayoutLMv3Config
from .modeling_layoutlmv3 import (
    CustomLayoutLMv3ForTokenClassification,
    CustomLayoutLMv3ForQuestionAnswering,
    CustomLayoutLMv3ForSequenceClassification,
    CustomLayoutLMv3Model,
)

# from .tokenization_layoutlmv3 import LayoutLMv3Tokenizer
# from .tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast


AutoConfig.register("custom-layoutlmv3", CustomLayoutLMv3Config)
AutoModel.register(CustomLayoutLMv3Config, CustomLayoutLMv3Model)
AutoModelForTokenClassification.register(CustomLayoutLMv3Config, CustomLayoutLMv3ForTokenClassification)
AutoModelForQuestionAnswering.register(CustomLayoutLMv3Config, CustomLayoutLMv3ForQuestionAnswering)
AutoModelForSequenceClassification.register(CustomLayoutLMv3Config, CustomLayoutLMv3ForSequenceClassification)

#  AutoTokenizer.register(
#      LayoutLMv3Config, slow_tokenizer_class=LayoutLMv3Tokenizer, fast_tokenizer_class=LayoutLMv3TokenizerFast
#  )

#SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv3Tokenizer": RobertaConverter})
