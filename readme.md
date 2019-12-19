# Usage
0. Install Python 3 and the dependencies `pip install -r requirements.txt`.
1. Change the DATA_PATH variable in *config.py* to the folder containing the experiment data. With the subfolders ibr-logs, realm-db, sensor-traces and the contactlist.csv in it.
2. If you plan to run everything, maybe set AUTO_OPEN_PLOTS to False to prevent the auto opening of plots.

3. To run The One with the generated files checkout the example configuration file *smarter_settings.txt*

----------

4. *fuse_databases.py*
	- combines all node data and the other created databases into one smarter.db for easier sharing

5. *sqlite2OneMovements.py*
	- standalone command line utility which can create a The One movement trace file from a folder of sqlite databases in which each db contains a gps trace for a node