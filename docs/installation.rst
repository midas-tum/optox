Installation
============

Optox environment
******************

- Make conda environment: ``conda env create -f environment.yml``
- Go to ``build``
- Run cmake: ``cmake ..``
- Compile: ``make -j``
- Install: ``make install``

Documentation
***************

- Make sure to have Doxygen installed: ``sudo apt install doxygen``
- Make sure to have the requirements installed: ``pip install -r requirements.txt``
- Go to ``docs``
- Run ``make html``
- Go to ``_build/html``
- View the documentation: ``python -m http.server 8000``