# dgcnn-experiment


## Installation
```
pip3 install -r requirements.txt
```

## Dataset Preparation
Download the scanobjectnn by yourself and put it into the following directory
```
dgcnn-experiment
+- data
| +- scanobjectnn
| | +- main_split
| | +- main_split_nobg
| | +- split1
| | +- split1_nobg
| | +- split2
| | +- split2_nobg
| | +- split3
| | +- split3_nobg
| | +- split4
| | +- split4_nobg
+- data.py
+- exp_classification.py
+- model.py
+- util.py

```

## Do Experiment
You can try the following command
```
python3 exp_classification.py
```
It might have an error at line 79 of data.py<br>
If it's possible, please do Farthest Point Sampling there<br>
Or just delete it to do random sampling
