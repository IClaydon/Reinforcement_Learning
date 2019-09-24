# Hotel Pricing Simulation and Reinforcement Learning Agent

Reproduction of the work done for Pace Revenue Management during an internship. 

fatstest_simulation contains a simulation of a hotel booking environment, generating customers who want to book at a hotel over one year. 
Compatible with python version >3.4 

# # Requires
```bash
pip install numpy
pip install pandas
```

# # Usage
```bash
python run_simulation.py
```


# Reinforcement Learning

reinforcement_learning contains two RL algorithms monte_carlo and sarsa(L). These are run in the same way as fastest_simulation, 
but contains an RL agent which learns using epsilon-greedy algorithm. Therefore revenue will increase each year simulated as the agent learns and increasingly picks the optimal action.

Currently the output is not saved.


