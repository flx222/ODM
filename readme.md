ODM Pipeline
This repository contains a pipeline for training, generating, and classifying protein sequences using machine learning models. The pipeline includes several steps, from data preprocessing to result extraction.

Requirements
Mindspore 1.8.1
Python 3.7+
Required Python packages:
numpy
pandas
tqdm
argparse
Setup
Clone the repository to your local machine.(https://www.mindspore.cn/versions#1.8.1)
Install the required Python packages using pip:
bash
mindspore-ascend=1.8.0
Python=3.7.5
numpy=1.21.6
torch=1.13.0
tqdm=4.65.0
scikit-learn=0.24.1
scipy =1.7.3
pandas=1.0.5
transformers=4.23.1
Usage
The main script is pipeline.py, which runs the entire pipeline. The pipeline includes the following steps:

Make MindRecord Dataset: Preprocess the dataset to create MindRecord format.
Run Mask Training: Train a mask model on the preprocessed dataset.
Generate Mutations: Generate mutations from the provided protein FASTA file.
Extract Mask Results: Extract prediction results from the generated mutations.
Run Regression: Run regression on the extracted results.
Extract Regression Results: Extract and save the regression results.
Command Line Arguments
-E, --hmmer_dataset: Path to the HMM dataset.
-D, --device: Ascend device ID.
-F, --fasta_file: Path to the protein FASTA file (only one sequence).
--do_train: Whether to run the training step (default: True).
--do_generate: Whether to run the mutation generation step (default: True).
--do_cls: Whether to run the classification step (default: True).
-M, --model: Name of the classification model.
Example Command
bash
python pipeline.py -E /path/to/hmmer_dataset -D 0 -F /path/to/protein.fasta -M my_model
Files and Functions
pipeline.py
The main script containing the entire pipeline. Key functions include:

parse_args(): Parses command line arguments.
make_mindrecord(dataset_path): Prepares the dataset in MindRecord format.
run_mask_train(dataset_path, device_id, fold): Trains the mask model.
generate_mut(fasta_path, dataset_path, device_id, fold): Generates mutations from the FASTA file.
extract_mask_result(fasta_path, fold): Extracts prediction results from mutations.
do_reg(args, fasta_path, fold, device_id): Runs regression on the extracted results.
extract_reg_result(cp_data_name, args, fasta_path, fold): Extracts and saves the regression results.
Data Structure
Input data should be organized as follows:

Output data will be saved in the ../predict_result/ directory, organized by FASTA file name and fold number.

Notes
Ensure the required scripts (generate_seq_for_mask.py, ODM_mask.py, mpbert_mask.py, MPB_regress.py) and configuration files (config_1024.yaml, vocab_v2.txt) are correctly referenced and available in the project directory.
The pipeline assumes the presence of pre-trained checkpoints and models specified in the commands.
By following this guide, you should be able to set up and run the protein sequence prediction pipeline effectively. If you encounter any issues or have further questions, please refer to the documentation or contact the repository maintainer.