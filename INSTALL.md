conda create -n basicrta python=3.8
conda activate basicrta
conda install mamba
mamba install numpy tqdm matplotlib MDAnalysis scipy pandas seaborn ipython jupyter pymbar
python setup.py install
