# distrib-synch
A wireless distributed synchronization simulator. It uses the distributed phase-locked-loops algorithm with a CFO-resistant synchronization signal. 
Supports the use of multipath channel for single isotropic antennas


Author: David Tétreault-La Roche

# Installation
Before doing simulations, a database file must be initialized. In a python console, run the following command:
```
>> import dumbsqlite3 as db
>> db.init()
```
This will initialize the sqlite database in the working directory.

# Utilization
The program is set up such that `wrapper.py` is executed at the terminal, e.g.:
```
python wrapper.py
```

Refer to `wrapper.py` for details on how to control the program.

