#!/bin/bash

virtualenv .venv

pip install numpy
.venv/bin/pip install -r requirements.txt
