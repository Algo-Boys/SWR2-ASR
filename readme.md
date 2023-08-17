# SWR2-ACR
Automatic speech recognition model for the seminar spoken word recogniton 2 (SWR2) in the summer term 2023.

# Installation
## Running on cpu or m1 gpu
```
poetry lock && poetry install --with cpu
```

## Running on nvidia gpu
```
poetry lock && poetry install --with gpu
```

# Usage
## Training 
Train using the provided train script:
```
poetry run train --data PATH/TO/DATA --lr 0.01 
```
## Evaluation

## Inference
```
poetry run recognize --data PATH/TO/FILE
```

## CI
To run the CI manually and not only on GitHub Actions, run `poetry run ./ci.sh`.

Alternatively, you can use the Makefile to run these commands manually
```
make format

make lint
```
