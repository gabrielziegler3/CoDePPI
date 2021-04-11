# MRI Reconstruction with Compressed Sensing and Prior Information

## Usage

### Build docker container with
```bash
docker build . -t mri-reconstruction:latest
```

### Spin up the container and run jupyter lab at port 8888
```bash
docker run -it -p 8888:8888 -v `pwd`:/mri-reconstruction --rm --detach --name mri-reconstruction mri-reconstruction
```

### Open up your browser in localhost:8888

```
http://localhost:8888/
```
