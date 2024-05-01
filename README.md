
### TD3BC++ Implementation

Welcome to the implementation of the TD3BC++ algorithm, as proposed in the paper "Robust Offline Reinforcement Learning from Contaminated Demonstrations", available on [arXiv](https://arxiv.org/pdf/2210.10469.pdf).

### Details

- Observation and motivation. We observed performance degradation of many state-of-the-art offline RL algorithms on heterogeneous datasets, which contain both expert and non-expert behaviors. This observation motivated our work.
- Our methods. We identified two key issues in policy constraint offline RL: (1) risky policy improvement on non-expert states that makes use of unstable Q-gradients and (2) harmful policy constraint towards non-expert dataset actions.
- Implementation. We proposed two solutions for each issue: (1) conservative policy improvement to reduce unstable Q-function gradients with respect to actions and (2) closeness constraint relaxation to loosen the constraint on non-expert actions. These solutions are simple but effective. (For the impressive results, see our paper)
- Rerun. To reproduce the results presented in the paper, please run the bash script `run.sh`.
