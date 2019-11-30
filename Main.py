from Model import Lgb_model
import numpy as np
import lightgbm as lgb
import pandas as pd
import os 
import time
Start_time = time.time()

Lgb_model.Cv_train()

print((time.time() - Start_time) / 60, '分')