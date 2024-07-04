# ODM Pipeline

This repository contains a comprehensive pipeline for training, generating, and classifying protein sequences using machine learning models. It includes multiple steps ranging from data preprocessing to result extraction.

## Setup

- **Mindspore**: 1.8.1 (follow the website to configure this version of the Docker environment)
- **Python**: 3.7 or higher
- **Hmmer** tool for generate fine-tuned dataset (http://hmmer.org/)


## Usage

Pretrain model,fine-tuned models are provided on this link:DOI: [10.5281/zenodo.12647297](https://doi.org/10.5281/zenodo.12647297)

Run the `pipeline.py` script to execute the entire pipeline. The pipeline involves the following steps:

1. **Make MindRecord Dataset**: Preprocess the dataset into MindRecord format.
2. **Run Mask Training**: Train a mask model using the preprocessed dataset.
3. **Generate Mutations**: Generate mutations from a provided protein FASTA file.
4. **Extract Mask Results**: Extract prediction results from the generated mutations.
5. **Run Regression**: Perform regression on the extracted results.
6. **Extract Regression Results**: Save the regression results.

### Command Line Arguments

- `-E, --hmmer_dataset`: Path to the HMM dataset.
- `-D, --device`: Ascend device ID.
- `-F, --fasta_file`: Path to the protein FASTA file (single sequence).
- `--do_train`: Whether to run the training step (default: True).
- `--do_generate`: Whether to run the mutation generation step (default: True).
- `--do_cls`: Whether to run the classification step (default: True).
- `-M, --model`: Name of the classification model.

### Example Command

```bash
python pipeline.py -E /path/to/hmmer_dataset -D 0 -F /path/to/protein.fasta -M my_model
```

## Files and Functions

**`pipeline.py`**: Contains the entire pipeline.

- `parse_args()`: Parses command line arguments.
- `make_mindrecord(dataset_path)`: Prepares the dataset in MindRecord format.
- `run_mask_train(dataset_path, device_id, fold)`: Trains the mask model.
- `generate_mut(fasta_path, dataset_path, device_id, fold)`: Generates mutations.
- `extract_mask_result(fasta_path, fold)`: Extracts prediction results.
- `do_reg(args, fasta_path, fold, device_id)`: Runs regression.
- `extract_reg_result(cp_data_name, args, fasta_path, fold)`: Saves regression results.

## Notes

Ensure all required scripts (`generate_seq_for_mask.py`, `ODM_mask.py`, `mpbert_mask.py`, `MPB_regress.py`) and configuration files (`config_1024.yaml`, `vocab_v2.txt`) are correctly referenced and available in your project directory. This pipeline utilizes pre-trained checkpoints and models as specified in the commands.

For any issues or further questions, refer to the documentation or contact the repository maintainer.
