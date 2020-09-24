FROM jupyter/scipy-notebook:latest

# RUN mkdir -p /mri-reconstruction
# RUN chown dev /mri-reconstruction
# USER dev

WORKDIR /mri-reconstruction

COPY . /mri-reconstruction

RUN pip install -r requirements.txt

EXPOSE 8888

RUN ["jupyter", "labextension", "install", "jupyterlab-plotly@4.10.0"]

CMD ["jupyter", "lab", "--NotebookApp.token=''", "--NotebookApp.password=''"]
