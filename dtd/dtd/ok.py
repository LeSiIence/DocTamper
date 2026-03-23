import os
nums = [16,15,14,13,12,11,17,18,19,20,21,22,23,24]*8
for n in nums:
    os.system('python systrain.py --py cfpns_cov_mulc_75_load.py --model_name DocTamperV1-TrainingSet --num %d'%n)
