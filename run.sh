#!/bin/bash
# reproducing the results in table 1 and 2, please run the code ./run.sh 
# we provide our implementation on the top of TD3+BC.

for ((i=0;i<5;i+=1))
do
python main.py --env 'walker2d-expert-v0' --seed $i --policy 'PExp-' --coef_grad_random_action 1 --gpu 0 > /dev/null 2>&1 &
python main.py --env 'walker2d-medium-expert-v0' --seed $i --policy 'PEM-' --coef_grad_random_action 1 --gpu 0 > /dev/null 2>&1 &
python main.py --env 'walker2d-expert-v0' --mix_env 'walker2d-random-v0' --mix_ratio 0.1 --seed $i --policy 'ER1-' --coef_grad_random_action 1 --gpu 0 > /dev/null 2>&1 &
python main.py --env 'walker2d-expert-v0' --mix_env 'walker2d-random-v0' --mix_ratio 0.3 --seed $i --policy 'ER3-' --coef_grad_random_action 1 --gpu 0 > /dev/null 2>&1 &
python main.py --env 'walker2d-expert-v0' --mix_env 'walker2d-random-v0' --mix_ratio 0.5 --seed $i --policy 'ER5-' --coef_grad_random_action 1 --gpu 0 > /dev/null 2>&1 &
python main.py --env 'walker2d-expert-v0' --mix_env 'walker2d-random-v0' --mix_ratio 0.7 --seed $i --policy 'ER7-' --coef_grad_random_action 1 --gpu 0 > /dev/null 2>&1 &

python main.py --env 'hopper-expert-v0' --seed $i --policy 'PExp-' --coef_grad_random_action 1 --gpu 1 > /dev/null 2>&1 &
python main.py --env 'hopper-medium-expert-v0' --seed $i --policy 'PEM-' --coef_grad_random_action 1 --gpu 1 > /dev/null 2>&1 &
python main.py --env 'hopper-expert-v0' --mix_env 'hopper-random-v0' --mix_ratio 0.1 --seed $i --policy 'ER1-' --coef_grad_random_action 1 --gpu 1 > /dev/null 2>&1 &
python main.py --env 'hopper-expert-v0' --mix_env 'hopper-random-v0' --mix_ratio 0.3 --seed $i --policy 'ER3-' --coef_grad_random_action 1 --gpu 1 > /dev/null 2>&1 &
python main.py --env 'hopper-expert-v0' --mix_env 'hopper-random-v0' --mix_ratio 0.5 --seed $i --policy 'ER5-' --coef_grad_random_action 1 --gpu 1 > /dev/null 2>&1 &
python main.py --env 'hopper-expert-v0' --mix_env 'hopper-random-v0' --mix_ratio 0.7 --seed $i --policy 'ER7-' --coef_grad_random_action 1 --gpu 1 > /dev/null 2>&1 &

python main.py --env 'halfcheetah-expert-v0' --seed $i --policy 'PExp-' --coef_grad_random_action 1 --gpu 2 > /dev/null 2>&1 &
python main.py --env 'halfcheetah-medium-expert-v0' --seed $i --policy 'PEM-' --coef_grad_random_action 1 --gpu 2 > /dev/null 2>&1 &
python main.py --env 'halfcheetah-expert-v0' --mix_env 'halfcheetah-random-v0' --mix_ratio 0.1 --seed $i --policy 'ER1-' --coef_grad_random_action 1 --gpu 2 > /dev/null 2>&1 &
python main.py --env 'halfcheetah-expert-v0' --mix_env 'halfcheetah-random-v0' --mix_ratio 0.3 --seed $i --policy 'ER3-' --coef_grad_random_action 1 --gpu 2 > /dev/null 2>&1 &
python main.py --env 'halfcheetah-expert-v0' --mix_env 'halfcheetah-random-v0' --mix_ratio 0.5 --seed $i --policy 'ER5-' --coef_grad_random_action 1 --gpu 2 > /dev/null 2>&1 &
python main.py --env 'halfcheetah-expert-v0' --mix_env 'halfcheetah-random-v0' --mix_ratio 0.7 --seed $i --policy 'ER7-' --coef_grad_random_action 1 --gpu 2 > /dev/null 2>&1 &

python main.py --env 'door-expert-v0' --seed $i --policy 'PExp-' --coef_grad_random_action 1 --alpha 0.5 --gpu 3 > /dev/null 2>&1 &
python main.py --env 'door-expert-v0' --mix_env 'door-cloned-v0' --mix_ratio 0.3 --seed $i --policy 'EC3-' --coef_grad_random_action 1 --alpha 0.5 --gpu 3 > /dev/null 2>&1 &
python main.py --env 'door-expert-v0' --mix_env 'door-cloned-v0' --mix_ratio 0.5 --seed $i --policy 'EC5-' --coef_grad_random_action 1 --alpha 0.5 --gpu 3 > /dev/null 2>&1 &
python main.py --env 'door-expert-v0' --mix_env 'door-cloned-v0' --mix_ratio 0.7 --seed $i --policy 'EC7-' --coef_grad_random_action 1 --alpha 0.5 --gpu 3 > /dev/null 2>&1 &

python main.py --env 'hammer-expert-v0' --seed $i --policy 'PExp-' --coef_grad_random_action 1 --alpha 0.2 --gpu 4 > /dev/null 2>&1 &
python main.py --env 'hammer-expert-v0' --mix_env 'hammer-cloned-v0' --mix_ratio 0.3 --seed $i --policy 'EC3-' --coef_grad_random_action 1 --alpha 0.2 --gpu 4 > /dev/null 2>&1 &
python main.py --env 'hammer-expert-v0' --mix_env 'hammer-cloned-v0' --mix_ratio 0.5 --seed $i --policy 'EC5-' --coef_grad_random_action 1 --alpha 0.2 --gpu 4 > /dev/null 2>&1 &
python main.py --env 'hammer-expert-v0' --mix_env 'hammer-cloned-v0' --mix_ratio 0.7 --seed $i --policy 'EC7-' --coef_grad_random_action 1 --alpha 0.2 --gpu 4 > /dev/null 2>&1 &

python main.py --env 'pen-expert-v0' --seed $i --policy 'PExp-' --coef_grad_random_action 1 --alpha 0.5 --gpu 5 > /dev/null 2>&1 &
python main.py --env 'pen-expert-v0' --mix_env 'pen-cloned-v0' --mix_ratio 0.3 --seed $i --policy 'EC3-' --coef_grad_random_action 1 --gpu 5 --alpha 0.5 > /dev/null 2>&1 &
python main.py --env 'pen-expert-v0' --mix_env 'pen-cloned-v0' --mix_ratio 0.5 --seed $i --policy 'EC5-' --coef_grad_random_action 1 --gpu 5 --alpha 0.5 > /dev/null 2>&1 &
python main.py --env 'pen-expert-v0' --mix_env 'pen-cloned-v0' --mix_ratio 0.7 --seed $i --policy 'EC7-' --coef_grad_random_action 1 --gpu 5 --alpha 0.5 > /dev/null 2>&1 &

python main.py --env 'relocate-expert-v0' --seed $i --policy 'PExp-' --coef_grad_random_action 0.1 --alpha 0.02 --gpu 6 > /dev/null 2>&1 &
python main.py --env 'relocate-expert-v0' --mix_env 'relocate-cloned-v0' --mix_ratio 0.3 --seed $i --policy 'EC3-' --coef_grad_random_action 0.1 --alpha 0.02 --gpu 6 > /dev/null 2>&1 &
python main.py --env 'relocate-expert-v0' --mix_env 'relocate-cloned-v0' --mix_ratio 0.5 --seed $i --policy 'EC5-' --coef_grad_random_action 0.1 --alpha 0.02 --gpu 6 > /dev/null 2>&1 &
python main.py --env 'relocate-expert-v0' --mix_env 'relocate-cloned-v0' --mix_ratio 0.7 --seed $i --policy 'EC7-' --coef_grad_random_action 0.1 --alpha 0.02 --gpu 6
sleep 20m 
done 