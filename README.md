# Implenmentation of TD3BC++

This is the code for the TD3BC++ algorithm proposed in arxiv paper **Robust Offline Reinforcement Learning from Contaminated Demonstrations**. 

Policy constraint offline RL methods often struggle to learn effectively from the contaminated dataset. For instance, for the D4RL medium-expert dataset, a combination of expert and medium-level decisions, several state-of-the-art offline RL methods exhibit lower scores compared to those achieved on the expert datasets.

![Image text](https://github.com/cangcn/TD3BCpp/blob/main/Figure_PerformanceDegradation.png)

This paper show that when learning from the dataset contaminated by low level demonstrations, e,g., expert-random and expert-cloned, they may face catastrophic failure. 

![Image text](https://github.com/cangcn/TD3BCpp/blob/main/Figure_CatastrophicFailure.png)


To recover the performance on such contaminated datasets, we use (1) the conservative policy improvement to reduce the impact of gradient operation on unstable Q-predictions, and then relax the closeness constraint toward non-expert dataset decisions by the polished Q-function. The modifications we made are highly impactful and do not significantly alter the runtime of the original algorithm.

![Image text](https://github.com/cangcn/TD3BCpp/blob/main/Figure_TrainingTime.png)

 



For the algorithmic details, please read the white paper https://arxiv.org/pdf/2210.10469.pdf
For the conda environment, please see the .yaml file.
For reproducing the paper results, please run the bash script. 
