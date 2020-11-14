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

## Result
Train | Test on 256 Points | Test on 512 Points | Test on 1024 Points | Test on 2048 Points
------------- | ------------- | ------------- | ------------- | -------------
256 Points | 0.833046 | 0.826162 | 0.652324 | 0.495697 
512 Points | 0.753873 | 0.864028 | 0.833046 | 0.741824
1024 Points | 0.506024 | 0.788296 | 0.857143 | 0.848537
2048 Points | 0.421687 | 0.583477 | 0.791738 | 0.839931
Mix | 0.841652 | 0.850258 | 0.874355 | 0.872633



