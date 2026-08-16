FROM jupyter/scipy-notebook:python-3.11

WORKDIR /home/jovyan/work

RUN mamba install -y -c conda-forge \
        "jupyterlab>=4.2,<5" \
        graphviz \
    && mamba clean -afy

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# A snapshot of the course so the image also works standalone, without a
# clone. The documented student workflow bind-mounts a git clone over this
# path instead, so new lessons arrive with `git pull` rather than a multi-GB
# image pull. See Course/Setup/Docker_Quickstart.md.
COPY --chown=jovyan:users . /home/jovyan/work
RUN fix-permissions /home/jovyan/work
