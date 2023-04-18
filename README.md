# TD3BC++ Implementation

Welcome to the implementation of the TD3BC++ algorithm, as proposed in the paper "Robust Offline Reinforcement Learning from Contaminated Demonstrations", available on [arXiv](https://arxiv.org/pdf/2210.10469.pdf).

This code proposes plugins for policy-constrained offline RL method that face performance degradation on contaminated datasets. Specifically, when training with datasets that combine expert and medium-level demonstrations, state-of-the-art offline RL methods can perform poorly, as shown by the results on the D4RL medium-expert dataset.
<p align="center">
    <img src="https://github.com/cangcn/TD3BCpp/blob/main/Figure_PerformanceDegradation.png" width="500px"/>
</p>


This paper show that when learning from the dataset contaminated by low level demonstrations, e,g., expert-random and expert-cloned, they may face catastrophic failures. 
<p align="center">
    <img src="https://github.com/cangcn/TD3BCpp/blob/main/Figure_CatastrophicFailure.png" width="500px"/>
</p>

To overcome this challenge, we use (1) the conservative policy improvement to reduce the impact of gradient operation on unstable Q-predictions, and then relax the closeness constraint toward non-expert dataset decisions by the polished Q-function. These modifications are highly impactful and do not significantly affect the runtime of the original algorithms.
<p align="center">
    <img src="https://github.com/cangcn/TD3BCpp/blob/main/Figure_TrainingTime.png" width="500px"/>
</p>


To reproduce the paper results, please run the bash script `run.sh`. 
