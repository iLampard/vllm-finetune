from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, sft_main, SftArguments
)
from swift.utils import seed_everything
import torch

from easyllm_kit.utils import save_json, read_json, sample_json_records, get_logger, measure_time
from easyllm_kit.utils.io_utils import initialize_database, write_to_database
from easyllm_kit.models import LLM
from easyllm_kit.configs import Config


def finetune(config_dir):
    # Load configuration from YAML file
    config = Config.build_from_yaml_file(config_dir)

    sft_args = SftArguments(
        model_id_or_path=config.model_config.model_dir,
        dataset=['train.jsonl'],
        output_dir='output')
    result = sft_main(sft_args)
    last_model_checkpoint = result['last_model_checkpoint']
    print(f'last_model_checkpoint: {last_model_checkpoint}')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    finetune('config.yaml')