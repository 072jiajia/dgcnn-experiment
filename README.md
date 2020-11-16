# dgcnn-experiment

## Setup

* Intstall ``python3.6``
* Install ``torch1.5`` with CUDA
* Install dependencies

    ``` bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

Download the scanobjectnn by yourself and put it into the following directory

```bash
dgcnn-experiment
+- data
| +- h5_files
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
+- pointnet2_ops_lib
+- data.py
+- exp_classification.py
+- model.py
+- util.py

```

## Do Experiment

You can try the following command

```bash
python exp_classification.py
```

## Result

### Random sample

Train | Test on 256 Points | Test on 512 Points | Test on 1024 Points | Test on 2048 Points
------------- | ------------- | ------------- | ------------- | -------------
256 Points | 0.833046 | 0.826162 | 0.652324 | 0.495697 
512 Points | 0.753873 | 0.864028 | 0.833046 | 0.741824
1024 Points | 0.506024 | 0.788296 | 0.857143 | 0.848537
2048 Points | 0.421687 | 0.583477 | 0.791738 | 0.839931
Mix | 0.841652 | 0.850258 | 0.874355 | 0.872633

### Farest point sampling

Train | Test on 256 Points | Test on 512 Points | Test on 1024 Points | Test on 2048 Points
------------- | ------------- | ------------- | ------------- | -------------
256 Points | 0.843373 | 0.791738 | 0.602410 | 0.469880
512 Points | 0.745267 | 0.846816 | 0.833046 | 0.722892 
1024 Points | 0.512909 | 0.807229 | 0.857143 | 0.839931
2048 Points | 0.330465 | 0.621343 | 0.812392 | 0.848537
Mix | 0.884441 | 0.860585 | 0.876076 | 0.862306

### Farest point sampling - Random sample

Train | Test on 256 Points | Test on 512 Points | Test on 1024 Points | Test on 2048 Points
------------- | ------------- | ------------- | ------------- | -------------
256 Points | <font color="green">0.010</font>|<font color="red">-0.034</font>|<font color="red">-0.050</font>|<font color="red">-0.026</font>
512 Points |  <font color="red">-0.009</font>|<font color="red">-0.017</font>|<font color="green">0.000</font>|<font color="red">-0.019</font>
1024 Points |  <font color="green">0.007</font>|<font color="green">0.019</font>|<font color="green">0.000</font>|<font color="red">-0.009</font>
2048 Points | <font color="red">-0.091</font>|<font color="green">0.038</font>|<font color="green">0.021</font>|<font color="green">0.009</font>
Mix |  <font color="green">0.043</font>|<font color="green">0.010</font>|<font color="green">0.002</font>|<font color="red">-0.010</font>
