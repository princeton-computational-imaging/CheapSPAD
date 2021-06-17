"""
Supplemental code for SIGGRAPH 2021 paper "Low-Cost SPAD Sensing for Non-Line-Of-Sight Tracking, Material Classification and Depth Imaging"

Script loader for all NLOS tracking approaches. For details on the methods, 
please consult the main paper. 
Select the used method, data and 
"""

from subprocess import run, PIPE


###################### set parameters here ######################

### method
method = "direct_position"                          # direct prediction of coordinates
# method = "distance_multilateration                # prediction of distance + multilateration
# method = "histshift_distance_multilateration"     # as above, but with shifted histogram
# method = "peak_finding_multilateration"                  # multilateration without neural network


### with or without mirror
# mirror = True
mirror = False


### train or test? 
traintest = "test"
# traintest = "train"


### data source
# mat_file = "./data/bigtarget_mirrors.mat"
# mat_file = "./data/bigtarget_nomirrors.mat"
# mat_file = "./data/smalltarget_mirrors.mat"
mat_file = "./data/smalltarget_nomirrors.mat"


### checkpoint
# ckpt_path = "./checkpoint/mirror_big_coordinate"
# ckpt_path = "./checkpoint/mirror_big_distance"
# ckpt_path = "./checkpoint/mirror_big_histshift_distance"
# ckpt_path = "./checkpoint/mirror_small_coordinate"
# ckpt_path = "./checkpoint/mirror_small_distance"
# ckpt_path = "./checkpoint/mirror_small_histshift_distance"
# ckpt_path = "./checkpoint/nomirror_big_coordinate"
# ckpt_path = "./checkpoint/nomirror_big_distance"
# ckpt_path = "./checkpoint/nomirror_big_histshift_distance"
ckpt_path = "./checkpoint/nomirror_small_coordinate"
# ckpt_path = "./checkpoint/nomirror_small_distance"
# ckpt_path = "./checkpoint/nomirror_small_histshift_distance"


##################################################################



## run script

c = "python %s.py %s --mat_file %s --ckpt_path %s --with_mirror %s" % (method, traintest, mat_file, ckpt_path, mirror)
p = run(c, stdout=PIPE, stderr=PIPE, universal_newlines=True, check=True)
print(p.stdout)