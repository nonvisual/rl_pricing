# rl_pricing

This repository contains a study on how one can apply reinforcement learning algorithm for dynamic pricing problem under certain financial KPIs goal. 
The goal of the study is to understand if and how reinforcement learning algorithm can substitute typical forecast + optimizer + planner combinations.

## Experiments

### Stable Baseline agent
Run 
```
jupyter notebook
```

and then navigate to notebook StableBaseline3.ipynb. 

To see the tensorboard, run 
```
tensorboard --logdir ./a2c_pricing_tensorboard --port 6007
```



## Links
Some links which were useful for this work (at least for initial information):
* [Dynamic pricing blog post](https://towardsdatascience.com/dynamic-pricing-using-reinforcement-learning-and-neural-networks-cc3abe374bf5)
* https://tryolabs.com/blog/2019/05/14/can-you-beat-this-pricing-algorithm
* https://priceoptimization.ai/
* https://towardsdatascience.com/advantage-actor-critic-tutorial-mina2c-7a3249962fc8
* https://github.com/DLR-RM/stable-baselines3
