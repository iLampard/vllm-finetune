import json
from easyllm_kit.utils import read_json, process_base64_image
from prompt import SFTTemplate
import os


# https://github.com/modelscope/ms-swift/issues/1613


def covert_json_to_jsonl(json_file_path, jsonl_file_path, image_output_dir="images"):
    """
    Convert a JSON file to JSONL format with specific key mapping:
    - 'query' remains as 'query'
    - 'image' is saved as PNG file and its path is stored
    - all other keys are combined into 'response'
    
    Args:
        json_file_path (str): Path to the input JSON file
        jsonl_file_path (str): Path to the output JSONL file
        image_output_dir (str): Directory to save the images
    """
    # Create image output directory if it doesn't exist
    os.makedirs(image_output_dir, exist_ok=True)

    # Read the JSON file
    data = read_json(json_file_path)

    # Write to JSONL file
    with open(jsonl_file_path, 'w') as jsonl_file:

        # Process each item
        for idx, item in data.items():
            # Handle the image if it exists
            image_path = None
            if 'image' in item and item['image']:
                # Save image as PNG
                image_path = os.path.join(image_output_dir, f"image_{idx}.png")

                process_base64_image(item['image'], image_path)

                # Update the image value to be the relative path
                item['image'] = image_path

            query = SFTTemplate.create_default().format(query=item['query'])

            response = {
                k: v for k, v in item.items()
                if k not in ['query', 'image']
            }

            # Format the response as a triple-quoted JSON string
            response_str = f'''```json
            {json.dumps(response)}
            ```'''

            # Create new item with mapped keys
            new_item = {
                'query': query,  # Keep 'query' as is
                'response': response_str,
                'images': [image_path]
            }

            # Write the formatted item to JSONL
            jsonl_file.write(json.dumps(new_item) + '\n')

    print(f"Converted {json_file_path} to {jsonl_file_path}")
    print(f"Images saved in {image_output_dir}")
    print(f"Total {len(data)} items processed")


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
    #   --model_type minicpm-v-v2_6-chat \
    #   --model_id_or_path /workspaces/models--openbmb-minicpm-v-2_6 \
    #   --sft_type lora \
    #   --dataset train.jsonl \
    #   --deepspeed default-zero2

    # Example usage
    covert_json_to_jsonl("../../data_scripts/gs_shopping_v3/ddb_storage/query_ans_gs_women_v1114.json",
                         "train.jsonl", "data_images")
