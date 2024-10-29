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

