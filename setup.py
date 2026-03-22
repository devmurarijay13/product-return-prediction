from setuptools import setup,find_packages
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    """
    this will return list of requirements
    """
    requirements = []

    try:
        with open(file_path) as file:
            requirements = file.readlines()
            requirements = [req.replace('\n','') for req in requirements]

            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
                
    except FileNotFoundError:
        print('File Not Found')

    return requirements


"""
metadata info about entire project
"""
setup(
    name='E-Commerce Product Return Prediction',
    version='0.0.1',
    author='Jay Devmurari',
    author_email='devmurarijay66@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)