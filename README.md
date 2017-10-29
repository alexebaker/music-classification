# Music Classification

UNM CS 429/529 Machine Learning Project 3: Music Classification


## Getting started

This project uses several dependencies. You may need to use pip to install these dependencies.
It is recommended to use a virtualenv to run this code, however you can also use your system wide python as well.

If you do not want to use the virtualenv, install the requirements directly:

```bash
pip install -r requirements.txt
```

**or**

Use virtualenv. If virtualenv is not install on your system, install it with pip:

```bash
pip install virtualenv
```

Next, you want to create the virtual env in your current directory:

```bash
virtualenv .venv
```

Next, activate the virtualenv in your current shell:

```bash
source .venv/bin/activate
```

Now, install the python requirements:

```bash
pip install -r requirements.txt
```

You can deactivate the virtualenv with the following command, however, make sure the virtualenv is active when you run the code:

```bash
deactivate
```

## Details

Details about this project can be found on [Kaggle](https://inclass.kaggle.com/c/cs529-project3)


## Usage

**NOTE**: This code will work in python 2. It worked for some versions of python 3, but failed for some. Please use python 2 if python 3 does not work for you.

Make sure the virtualenv is active before you try running the python code. You can activate it by:

```bash
source .venv/bin/activate
```

Once the virtualenv is activated, you can run the python script. The main entry point for this project is `nb.py`. Use the `-h` flag from any command to see help:

```bash
>>>python nb.py -h
```


## Documentation

This module uses documentation complied by [sphinx](http://www.sphinx-doc.org/en/stable/) located in the `docs/` directory. To build the documentation, run the Makefile:

```bash
source .venv/bin/activate
make docs
```

Once the documentation is built, it can be viewed in your brower by running the `open-docs.py` script:

```bash
python open-docs.py
```


## TODO

- [ ] - Parse FFT features
- [ ] - Parse MFCC features
- [ ] - Parse custom features
- [ ] - Classify FFT features
- [ ] - Classify MFCC features
- [ ] - Classify custom features
- [ ] - Write up final report


## Authors

* [Alexander Baker](mailto:alexebaker@unm.edu)

* [Caleb Waters](mailto:waterscaleb@unm.edu)

* [Mark Mitchell](mailto:mamitchell@unm.edu)
