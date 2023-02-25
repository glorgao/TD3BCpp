# Implenmentation of TD3BC++

This is the code for the TD3BC++ algorithm proposed in the [arxiv paper](https://arxiv.org/pdf/2210.10469.pdf) **Robust Offline Reinforcement Learning from Contaminated Demonstrations** 

Policy constraint offline RL methods often struggle to learn effectively from the contaminated dataset. For instance, for the D4RL medium-expert dataset, a combination of expert and medium-level decisions, several state-of-the-art offline RL methods exhibit lower scores compared to those achieved on the expert datasets.
<p align="center">
    <img src="https://github.com/cangcn/TD3BCpp/blob/main/Figure_PerformanceDegradation.png" width="500px"/>
</p>


This paper show that when learning from the dataset contaminated by low level demonstrations, e,g., expert-random and expert-cloned, they may face catastrophic failures. 
<p align="center">
    <img src="https://github.com/cangcn/TD3BCpp/blob/main/Figure_CatastrophicFailure.png" width="500px"/>
</p>


To recover the performance on such contaminated datasets, we use (1) the conservative policy improvement to reduce the impact of gradient operation on unstable Q-predictions, and then relax the closeness constraint toward non-expert dataset decisions by the polished Q-function. The modifications we made are highly impactful and do not significantly alter the runtime of the original algorithms.
<p align="center">
    <img src="https://github.com/cangcn/TD3BCpp/blob/main/Figure_TrainingTime.png" width="500px"/>
</p>


To reproduce the paper results, please run the bash script `run.sh`. 
