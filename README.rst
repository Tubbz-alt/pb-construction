=================
pb-construction
=================

.. start-badges

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://github.com/yijiangh/pb-construction/blob/master/LICENSE
    :alt: License MIT

.. .. image:: https://travis-ci.org/yijiangh/pybullet_planning.svg?branch=master
..     :target: https://travis-ci.org/yijiangh/pybullet_planning
..     :alt: Travis CI

.. end-badges

.. Write project description

Construction sequence and motion planning in pybullet 

Installation
------------

::

    $ git clone https://github.com/caelan/pb-construction.git
    $ cd pb-construction
    $ git submodule update --init --recursive

The easiest way to resolve all the dependencies is to use a conda environment:

::

    # in pb-construction path
    $ conda env create -f pb_ws.yml
    $ conda activate pb_ws
    $ pip install -e .

If you want to use `PDDLStream <https://github.com/caelan/pddlstream>`_, you need to
first build `FastDownward <http://www.fast-downward.org/HomePage>`_ (can only build 
on Unix machines for now).

::

    $ ./src/pb-construction/pddlstream/FastDownward/build.py


Examples
--------

* `$ python -m extrusion.run`
* `$ python -m picknplace.run`

Related Repos
-------------

* https://github.com/caelan/pddlstream
* https://github.com/yijiangh/pychoreo
* https://github.com/yijiangh/assembly_instances
* https://github.com/yijiangh/conmech