# Sourced by the image's start.sh, as root, just before Jupyter starts.
#
# It makes the course files belong to the notebook user, so notebooks can be
# saved when your own user id differs from the container's - the case on a
# native Linux machine. It replaces the image's CHOWN_EXTRA setting for one
# reason: start.sh stops the whole container on the first file it cannot
# re-own, and on macOS Docker Desktop refuses to re-own git's read-only object
# files. Here the same work is done, and a file that cannot be re-owned is
# skipped instead of stopping Jupyter from starting.
#
# The `if` matters: start.sh runs with `set -e`, which would end it on a failed
# command anywhere else in this file.
if ! chown -R "${NB_UID}:${NB_GID}" /home/jovyan/work 2>/dev/null; then
    echo "course: some files could not be re-owned (on macOS this is normal for git's read-only files); continuing"
fi
