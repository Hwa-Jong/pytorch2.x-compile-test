
## Dockerfile 생성

```dockerfile
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
COPY requirements/txt /workspace/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
```

## Image build (from Dockerfile)

PROJ_DIR = Dockefile 이 있는 경로

```shell
> cd %PROJ_DIR%
> docker build -t torch2 .
```

## Container execution

| Env | Path (%PATH%) |
| -- | -- |
| command prompt | %CD% |
| powershell | ${pwd} |
| ubuntu | $(pwd) |

```shell
# % PATH% 상단 표 참고
> docker run -it --gpus all -v %PATH%:/workspace torch2 .
# example - ubuntu
# docker run -it --gpus all -v $(pwd):/workspace torch2 .
```
