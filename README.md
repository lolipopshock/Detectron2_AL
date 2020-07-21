# Detectron2 for Active Learning in Object Detection 


## Usage 

1. Clone the repository with all the submodules:
    ```bash
    git clone --recurse-submodules git@github.com:lolipopshock/Detectron2_AL.git
    ```
2. Install dependencies:
    1. Installing object detection environment with according to your environment
        - The tested version of pytorch is 1.4.0 with CUDA 10
        - And you **must** install Detectron2 with version 0.1.1. Newer versions has different APIs. 
            ```bash
            pip install detectron2==0.1.1 \
                -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/torch1.4/index.html
            ```
    2. Installing other necessary dependencies:
        ```bash
            pip install -r requirements.txt
        ```
    3. Installing UI components
        ```bash
        cd src/label-studio
        pip install -e .
        ``` 
3. Setting up the label-studio server and modeling backend
   1. Initialize the labeling server (If your image folder is `./data`)
        ```bash
        label-studio init labeling/tk-labeling \
                --input-path=./data \
                --input-format=image-dir \
                --allow-serving-local-files --force \
                --label-config=extra/config.xml \
                --ml-backends http://localhost:9090
        ```
        And you can start the server via
        ```
        label-studio start labeling/tk-labeling
        ```
    1. Initialize the model backend server
        ```bash
        label-studio-ml init labeling/backend_model --script extra/backend_model.py
        ```
        And similarly, you can start the backend server by
        ```bash
        label-studio-ml start labeling/backend_model 
        # There's a relative import of the libraries
        # So you have to run this command in the project project
        # root path to avoid import errors
        ```
4. Start using active learning for annotation