# SWR2-ACR

Automatic speech recognition model for the seminar spoken word
recogniton 2 (SWR2) in the summer term 2023.

# Installation
```
pip install -r requirements.txt
```

# Usage

## Training

Train using the provided train script:

    poetry run train --data PATH/TO/DATA --lr 0.01 

## Evaluation

## Inference

    poetry run recognize --data PATH/TO/FILE

## CI

You can use the Makefile to run these commands manually

    make format

    make lint
