# MultiTask

## Setup Instructions

### Create Local Conda Environment and Install Dependencies

1. Create and activate a local Conda environment in the current folder:
    ```bash
    conda create --prefix ./.conda python=3.9 -y
    conda activate ./.conda
    ```

2. Install all required dependencies:
    ```bash
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install -r requirements.txt
    ```

3. Generate `requirements.txt`:
    ```bash
    pip freeze > requirements.txt
    ```

### Requirements

The `requirements.txt` file contains all the dependencies required for this project. You can install them using:
```bash
pip install -r requirements.txt
```

pip install tensorboard==2.12.0
pip install numpy==1.23.0 
tensorboard --logdir="/mnt/pipeline_1/MLT/writer_logs/training_try_stage2_share/" --port=6007

mkdir -p ./mnt/pipeline_1/MLT/Weather/training_try_stage2_share/

### Training, Testing, and Inference
Training
To train the model, run the training script:

```bash
bash train_mult.sh
```

Testing
After training, you can test the model by running the following script:

```bash
bash test.sh
```

Inference
For inference on new data, use the following Python script:

```bash
python inference.py --model_path <path_to_pretrained_model> --save_path <path_to_save_visualization_results>
```

`--model_path`: The path to the pretrained model. This should point to the model file you wish to use for inference.
`--save_path`: The directory where the visualization results (e.g., output images or analysis results) will be saved.