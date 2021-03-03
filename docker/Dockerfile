FROM jupyter/scipy-notebook:latest

USER root

RUN apt-get update && apt-get -yy install make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev vim

# Install BART
RUN wget https://github.com/mrirecon/bart/archive/v0.6.00.tar.gz
RUN tar xzvf v0.6.00.tar.gz
RUN cd bart-0.6.00 && make && make install
RUN pwd

USER jovyan

ENV TOOLBOX_PATH="/usr/bin/bart"
ENV PATH=$TOOLBOX_PATH:$PATH
ENV PYTHONPATH="${TOOLBOX_PATH}/python:$PYTHONPATH"

# Install FastMRI
# RUN mkdir temp && cd temp
# RUN git clone https://github.com/facebookresearch/fastMRI && cd fastMRI
# RUN pip install -e .

WORKDIR /mri-reconstruction

COPY . /mri-reconstruction

RUN pip install -r requirements.txt

EXPOSE 8888

RUN ["jupyter", "labextension", "install", "jupyterlab-plotly@4.10.0"]

CMD ["jupyter", "notebook", "--NotebookApp.token=''", "--NotebookApp.password=''"]
