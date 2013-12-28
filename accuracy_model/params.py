"""
Parameters (weights) for logistic regression.

Training specs
--------------
Trained in 2012
Dataset: last 1 million problems logs since Sept. 2
Using history: only last 20 problems
Predicting on: All problems done except first
"""

INTERCEPT = -1.2229719
EWMA_3 = 0.8393673
EWMA_10 = 2.1262489
CURRENT_STREAK = 0.0153545
LOG_NUM_DONE = 0.4135883
LOG_NUM_MISSED = -0.5677724
PERCENT_CORRECT = 0.6284309
