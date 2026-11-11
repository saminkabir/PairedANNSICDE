For each categori, please enter corresponding dir.  
Inside each dir, please run:
```py
python maximize_recall.py # change PARALLEL_SETTING in this script to generate parallel or sequential profiles
```
Then you will get profiles csv. Then run:
```py
python summary.py # Please check this script and change (QUERY_HARDNESS, OOD, HIGH_DIM, PARALLEL_SETTING) for your settings. 
```
to print the average target recall with specific constraints.