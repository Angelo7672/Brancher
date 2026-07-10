# ============================================================
# STAGE 1/2 — BUILDER: qui costruiamo l'ambiente conda completo.
# Tutto questo stage verrà scartato alla fine: nell'immagine finale entra
# solo lo stato risultante di /opt/conda, non la cronologia dei layer.
# ============================================================
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH

SHELL ["/bin/bash", "-c"]

# ============================================================
# 2. Dipendenze di sistema (per costruire l'ambiente, più le librerie grafiche
# minime richieste da opencv/cv2 — serve al test di verifica di ultralytics
# più avanti nel file. Non costa nulla in dimensione finale: questo intero
# stage viene scartato, non entra mai nell'immagine finale.)
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ============================================================
# 3. Miniconda
# ============================================================
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh && \
    ${CONDA_DIR}/bin/conda clean -ya

# Solver veloce (libmamba)
RUN ${CONDA_DIR}/bin/conda config --set solver libmamba

RUN ${CONDA_DIR}/bin/conda init bash

# Accetta i Terms of Service dei canali "defaults" (pkgs/main, pkgs/r).
RUN ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Rende conda resistente a connessioni instabili (utile su WSL2 / reti domestiche):
# senza retry, una singola disconnessione a metà di un pacchetto fa fallire tutto il comando.
RUN ${CONDA_DIR}/bin/conda config --set remote_connect_timeout_secs 30.0 && \
    ${CONDA_DIR}/bin/conda config --set remote_read_timeout_secs 300.0 && \
    ${CONDA_DIR}/bin/conda config --set remote_max_retries 10 && \
    ${CONDA_DIR}/bin/conda config --set remote_backoff_factor 2

# ============================================================
# 4. Ambiente conda AR17
# ============================================================
# NOTA: "pip" è richiesto esplicitamente qui insieme a "python". Senza pip nell'ambiente,
# un "pip install" successivo dentro "conda run -n AR17" cadrebbe silenziosamente sul pip
# della base (versione di Python diversa) a causa del PATH, installando tutto nel posto sbagliato.
RUN ${CONDA_DIR}/bin/conda create -n AR17 python=3.11.6 pip -c conda-forge -c defaults -y

# Da qui in poi, ogni RUN esegue dentro l'ambiente AR17.
# --no-capture-output evita che l'output venga bufferizzato e nascosto durante installazioni lunghe.
SHELL ["conda", "run", "--no-capture-output", "-n", "AR17", "/bin/bash", "-c"]

# ============================================================
# 5. PyTorch + CUDA
# ============================================================
# Ogni pacchetto/gruppo in un RUN separato: se una connessione cade, il retry riparte
# solo da qui e non perde i download già completati con successo negli step precedenti.
RUN conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 \
    "mkl<2024.1" -c pytorch -c nvidia -c conda-forge -y

RUN conda install av -c conda-forge -y && conda clean -ya

# ============================================================
# 6. OpenMIM
# ============================================================
# "python -m pip" invece di "pip" nudo: forza l'uso dell'interprete di AR17,
# senza dipendere dalla risoluzione del PATH.
RUN python -m pip install --no-cache-dir -U openmim

# ============================================================
# 7. Ecosistema OpenMMLab
# ============================================================
# Un RUN per pacchetto: ognuno resta in cache singolarmente se completato con successo.
RUN mim install mmengine
RUN mim install "mmcv==2.1.0"
RUN mim install "mmdet==3.2.0"

RUN python -m pip install --no-build-isolation chumpy==0.70

RUN mim install "mmpose==1.3.1"
RUN mim install "mmpretrain==1.2.0"

# ============================================================
# 8. environment.yml (solo questo file: brancher/keypoints_module arrivano
# a runtime via bind mount nel docker-compose.yml, non servono nell'immagine)
# ============================================================
WORKDIR /workspace
COPY environment.yml* /workspace/

RUN if [ -f environment.yml ]; then conda env update --file environment.yml; fi

# ============================================================
# 9. scipy + ultralytics
# ============================================================
# scipy: mmdet lo installa via pip, ma se un altro pacchetto lo tocca di nuovo
# (conda o pip) i suoi file compilati (.so) restano incoerenti
RUN python -m pip uninstall -y scipy && \
    rm -rf /opt/conda/envs/AR17/lib/python3.11/site-packages/scipy* && \
    python -m pip install --no-cache-dir scipy==1.17.1 && \
    python -c "from scipy.spatial.transform import Rotation; from scipy.sparse.linalg import svds; from scipy.io import loadmat; print('scipy OK')"

RUN python -m pip install --no-cache-dir --no-deps --upgrade "ultralytics>=8.3.0" && \
    python -c "from ultralytics.nn.modules.block import C3k2; print('ultralytics OK')"

# Pulizia finale aggressiva: qui conta davvero, perché questo stage viene scartato
# e nell'immagine finale entra solo lo stato di /opt/conda a questo punto, non
# la somma dei layer intermedi (tarball conda, cache pip, ecc. spariscono per sempre).
RUN conda clean -afy && \
    python -m pip cache purge && \
    find ${CONDA_DIR} -name '*.a' -delete && \
    find ${CONDA_DIR} -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

# ============================================================
# STAGE 2/2 — FINALE: immagine leggera, solo runtime.
# ============================================================
# Passiamo da "-devel" a "-runtime": gli strumenti di compilazione (nvcc, header)
# servivano solo per COSTRUIRE l'ambiente sopra. A runtime PyTorch e l'ecosistema
# OpenMMLab usano wheel già precompilate — non serve ricompilare nulla (l'abbiamo
# verificato: nei log di build non è mai comparsa una compilazione da sorgente).
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH

# Librerie di sistema necessarie A RUNTIME (non per compilare)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia SOLO l'ambiente conda già pronto dallo stage di build.
# Questo è il punto chiave per la dimensione: qui entra lo stato finale di
# /opt/conda, non la somma di tutti i layer intermedi del builder.
COPY --from=builder /opt/conda /opt/conda


RUN ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    ${CONDA_DIR}/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN ${CONDA_DIR}/bin/conda init bash

WORKDIR /workspace

COPY ./brancher /workspace/brancher
COPY ./keypoints_module /workspace/keypoints_module

# ============================================================
# 10. Entrypoint
# ============================================================
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "AR17"]
CMD ["bash"]