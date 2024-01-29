conda create -n basicrta python=3.8
conda activate basicrta
conda install mamba
mamba install -c conda-forge numpy tqdm matplotlib MDAnalysis scipy seaborn
pip install .
