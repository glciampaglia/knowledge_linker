Knowledge-linker
================

Code for the Knowledge Linker project.

## Downloading

TBD

## Building, Testing, and Installing

### Building

To compile the software, first open a terminal and browse to the source code
folder. The package can be built using
[setuptools](//setuptools.readthedocs.io). Usually you want to build, test, and
install all in the same session. To do so, follow the steps below.

### Testing

A comprehensive suite of tests is available for the core closure API.
[Nose](http://nose.readthedocs.org) is the software used to collect, run, and
report the results of the test suite. To run it, the simplest way is to use the
setuptools script provided with the package:

```bash
    python setup.py nosetests
```

If you want to specify options, you can look in `setup.cfg` under the
_nosetests_ section.

### Installing

Once the test suite has been executed and no failure has emerged, you can build
and install the package. This software includes some C extension modules, which
need to be compiled. As mentioned before, the package provides a setuptools
script that will perform the necessary steps. You will need to have a C
compiler installed on your system. The software has been developed using the
GNU C compiler (gcc). No guarantee is given that it will compile using
different compilers. 

To build and install the package there are two different, equivalent methods,
depending on how you manage your python packages. Both will compile the C code,
create a package, and copy it into your Python distribution. This last step
will copy both libraries and scripts.

__Method 1__: with the plain setuptools:

```bash
    python setup.py install
```

__Method 2:__ with pip:

```bash
    pip  install -e requirements.txt .
```

## Contributing

TBD
