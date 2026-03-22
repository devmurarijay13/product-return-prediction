import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s'
)

project_name = 'e-commerce product return prediction'

list_of_files = [
    # 'Setup.py',
    # 'requirements.txt',
    f'src/__init__.py',
    f'src/exception.py',
    f'src/logger.py',
    f'src/utils.py',
    f'src/componenets/__init__.py',
    f'src/componenets/data_ingestion.py',
    f'src/componenets/data_transformation.py',
    f'src/componenets/model_trainer.py',
    f'src/pipeline/__init__.py',
    f'src/pipeline/train_pipeline.py',
    f'src/pipeline/predict_pipeline.py',
]

for file_path in list_of_files:
    file_path = Path(file_path)
    filedir, file_name = os.path.split(file_path)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f'creating directory {filedir} for file {file_name}')

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path,"w") as file:
            pass
            logging.info('creating empty file: {file_path}')

    else:
        logging.info(f'{file_name} is already exists')