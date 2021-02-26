# adaptive_agent
This repository implements [stable_baselines](https://github.com/hill-a/stable-baselines) training on the environment [gym_novel_gridworlds](https://github.com/gtatiya/gym-novel-gridworlds)

## Installation

### Conda environment
```conda create --name <env> --file requirements.txt```

```conda activate <env> ```

## RL-agent


### Train & Evaluate

#### Base script
``` python train.py -E <name-of-env> -N <novelty-name> -D <novelty-difficulty> -N1 <novelty-arg1> -N2 <novelty-arg2> -I <timestep-to-inject-novelty> -T <number-of-tests> -M <number-of-models-to-save>```

#### Breakincrease novelty
``` python train.py -N breakincrease -N1 stick```

#### Remap action novelty
``` python train.py -N remapaction```

#### firewall novelty
``` python train.py -N firewall```

### Plot

```python plot_results.py```

-------------------------------------

## Planning agent

### PDDLS

PDDL files for all the environments are found [here](https://github.com/goelshivam1210/adaptive_agent/tree/master/PDDLs)

### CSVs
CSV files from the planning architecture used for evaluations in the paper are found [here](https://github.com/goelshivam1210/adaptive_agent/tree/master/CSVs)

<!-- http://github.com - automatic!
[GitHub](http://github.com) -->
<!-- 
To run evaluations

```python evaluate.py -N breakincrease -N1 stick```

```python evaluate.py -N remapaction```

```python evaluate.py -N firewall``` -->

