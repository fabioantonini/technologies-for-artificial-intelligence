# The core course image: lessons 1 to 8.
#
# Base chosen deliberately. jupyter/scipy-notebook ships a scientific stack we
# would immediately install over, and the duplication cost more than a gigabyte
# of layers; minimal-notebook ships none of it, so each package is installed
# once. Staying inside the jupyter/docker-stacks family is not negotiable:
# fix-permissions and start-notebook.sh come from it, and they are what stops
# students hitting "cannot save notebook" when the host UID does not match
# jovyan's. See the comment in docker-compose.yml.
FROM quay.io/jupyter/minimal-notebook:python-3.11

WORKDIR /home/jovyan/work

# Before the source copy, so that adding a lesson does not invalidate this layer.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# A snapshot of the course so the image also works standalone, without a clone.
# The documented student workflow bind-mounts a git clone over this path, so new
# lessons arrive with `git pull` rather than an image pull. Note .dockerignore:
# the host virtualenv must never enter the build context, and Instructor/ must
# never reach a student.
COPY --chown=jovyan:users . /home/jovyan/work
RUN fix-permissions /home/jovyan/work
