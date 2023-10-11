FROM conda/miniconda3
# RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch
RUN conda install rdkit pytorch -c conda-forge -c pytorch
# WORKDIR /workspace
RUN python -m pip install scipy
COPY . .
ENTRYPOINT ["python", "score_compounds.py"]