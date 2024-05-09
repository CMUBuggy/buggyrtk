FROM jupyter/scipy-notebook

USER root
RUN apt-get update && apt-get install -y \
    iputils-ping

USER jovyan

RUN mkdir -p /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/apputils-extension && \
    echo '{ "theme":"JupyterLab Dark" }' \
    > /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

RUN pip install --upgrade scipy pyproj
