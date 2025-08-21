import os

import torch

from transformers import (
    TrainingArguments,
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
    get_peft_config
)


from datasets import Dataset,DatasetDict,Image as Image_from_datasets, load_dataset, load_from_disk


from data_process_utils import data_process


validation_res_json=data_process.img_label_process('./input/validation_data',True,'validation')


data_process.update_json(validation_res_json,'./temp/validation.json','validation')


validation_dataset = Dataset.from_json('./validation_16.json')


dataset = DatasetDict(validation_dataset)






