# File structure
```
|-- envs
    |-- 2Dpusher_env.py # Implementation of the pusher environment 
    |-- __init__.py # Register two pusher environmnets
|-- run.py # Entry of all the experiments. Also check out the commented code in this file.
|-- mpc.py # MPC to optimize model; Call CEM from this file
|-- model.py # Creat and train the ensemble dynamics model
|-- agent.py # An agent that can be used to interact with the environment
             # and a random policy
|-- util.py # Some utility functions we kindly provide. You do not have to use them.
|-- opencv_draw.py # Contains features for rendering the Box2D environments. You don't have to do anything to this file.
```

# Prerequisite
Run `pip install opencv-python`
 
# Guidelines for Implementation
### Q1.1.1
* When implementing the `MPC` class, use the `mpc_params` that is passed into this class.
* One tip is to write a separate `CEMOptimizer` and `RandomOptimizer`, which optimize a cost function over 
action sequences.
* We provide a state based cost function. To get the cost of a trajectory, you need to sum/average over the cost of all
the states.
* Sometimes the cost function can give `nan` values. Make sure to deal with such cases in your code.

### Q1.1.2
* If you have written a `CEMOptimizer` class, in this question, you should write another `RandomOptimizer` class.

### Q1.2.2
* Running time of this question for the TAs: 6 hours

### Q1.2.7
* Read though all the questions before you start running this experiment:)
* Running time of this question for the TAs: 6 hours 