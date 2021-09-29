<!--
*** Based on template: https://github.com/othneildrew/Best-README-Template
-->

<br />
<p align="center">
  <!-- <a href="https://github.com/enriqueboyafalcon/EPSU1445">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <a href="https://github.com/enriqueboyafalcon/EPSU1445">
    <h3 align="center">RL algorithm and policy comparison in Google Research Football</h3>
  </a>
  
  <!-- <p align="center">
    Set of programs implemented to 
  </p> -->
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
	<li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains a set of programs developed for my bachelor's thesis. It's objective is to evaluate and compare different reinfocement learning (RL) algorithms and policies in order to determine which are best suited for the [Google Research Football](https://github.com/google-research/football) enviroment. The comparison will be based on the agent's accuracy, which is the ratio of scored goals in a set ammount of episodes.

The python scripts featured in this repository streamline and automate the agent learning process, the obtaining, processing and interpretation of the generated data, as well as the creation of graphs to represent it. It also includes the custom scenarios used in the tests and the resulting data.

Policies tested:
* [Scoring](https://arxiv.org/pdf/1907.11180.pdf#page=4)
* [Checkpoints](https://arxiv.org/pdf/1907.11180.pdf#page=4)

RL Algorithms tested:
* [A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)
* [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
* [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

### Built With

* [Python](https://www.python.org/)
* [Google Research Football](https://github.com/google-research/football)
* [OpenAI Gym](https://gym.openai.com/)
* [Stable Baselines](https://github.com/DLR-RM/stable-baselines3)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Due to the version of pygame the enviroment was built in, the latest version of python it works on is 3.7.

Installing the GPU versions of these packages is recommended if possible.

1. Add the deadsnakes repository and install python 3.7
   ```sh
   sudo add-apt-repository ppa:fkrull/deadsnakes
   sudo apt-get update
   sudo apt-get install python3.7
   ```

2. Install Google Research Football and it's required packages
   ```sh
   sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
   libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
   libdirectfb-dev libst-dev mesa-utils xvfb x11vnc
   ```
   ```sh
   python3.7 -m pip install gfootball
   ```
   ```sh
   python3.7 -m pip install --upgrade pip setuptools wheel
   ```
   ```sh
   python3.7 -m pip install tensorflow==1.15.*  #CPU
   # or
   python3.7 -m pip install tensorflow-gpu==1.15.* #GPU
   ```
   ```sh
   python3.7 -m pip install dm-sonnet==1.* psutil
   ```
   
3. Install PytorchCPU or PytorchGPU+CUDA (Nvidia gpus)
   ```sh
   python3.7 -m pip install torch==1.9.0+cpu torchvision==0.10.0+cpu \
   torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html #CPU
   # or
   sudo apt install nvidia-cuda-toolkit
   python3.7 -m pip install torch torchvision torchaudio #GPU
   ```

4. Install Stable Baselines
   ```sh
   python3.7 -m pip install stable-baselines
   python3.7 -m pip install stable-baselines3
   ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/enriqueboyafalcon/EPSU1445.git
   ```

2. If you wish to use the custom scenarios
   ```sh
   cp -a ./EPSU1445/Code/Scenarios/ /home/<USERNAME>/.local/lib/python3.7/site-packages/gfootball/scenarios
   ```


<!-- USAGE EXAMPLES -->
## Usage

The training process saves the model with best accuracy and the logs generated during training in Code/Models and Code/Logs respectively. In order to avoid accidental overwrites, these files should me moved to Data/Models and Data/Logs using the following commands:

```sh
mv ./Code/Models/* ./Data/Models/
mv ./Code/Logs/* ./Data/Logs/
```

All scripts that use the generated data expect it to be in that path.

Generated data has the following file name format:

```
<scenario> <algorithm> <politic> <agent id>.csv/zip
```

### train.py

The script [train.py](Code/train.py) trains 30 agents with the desired politic and algorithm in a specific scenario. Modify these variables in it's main to choose the the characteristics of the training, which and how many agents to train.

```python
scenario = 'MyScenarios_Penalty' # MyScenarios_Penalty/Counterattack/Iniesta
model = 'PPO'                    # A2C/DQN/PPO
checkpoints = False              # False/True

first_agent = 1
last_agent  = 30
```

<!-- Example training 2 agents with PPO and checkpoints in MyScenarios_Counterattack:

```sh
output
``` -->

### test.py

The script [test.py](Code/test.py) loads a given trained model and runs it for the desired ammount of episodes, with or without rendering. 

Usage example:

```sh
python3.7 test.py -p Data/Models/MyScenarios_Penalty\ PPO\ scoring\ 01.zip -e 10 -v -r
```

Use -h to see option list.

### evaluate.py

The script [evaluate.py](Code/evaluate.py) loads trained models 1-30 of a specific scenario and writes down their accuracies after 1000 episodes. Output is a csv file, stored in Data/Results.

### sort.py

The script [sort.py](Code/sort.py) generates a copy the of output of evaluate.py where the values are ordered, for better readability and easier interpretation.

### plot.py

The script [plot.py](Code/plot.py) generates a series of graphs that visualize the output of evaluate.py and the logs from the training process.

![Penalti PPO Training logs](https://github.com/enriqueboyafalcon/EPSU1445/blob/master/Data/Graphs/plotall%2Bminmaxavg%20Penalty%20PPO%20scoring.png)

### results.py

The script [results.py](Code/results.py) performs a series of statistical tests in order to formally compare and draw conclusions from the output of evaluate.py.

<!-- CONTACT -->
## Contact

Enrique Boya Falcón - enriqueboyafalcon@gmail.com


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/enriqueboyafalcon/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/enriqueboyafalcon/EPSU1445/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/enriqueboyafalcon/repo.svg?style=for-the-badge
[forks-url]: https://github.com/enriqueboyafalcon/EPSU1445/network/members
[stars-shield]: https://img.shields.io/github/stars/enriqueboyafalcon/repo.svg?style=for-the-badge
[stars-url]: https://github.com/enriqueboyafalcon/EPSU1445/stargazers
[issues-shield]: https://img.shields.io/github/issues/enriqueboyafalcon/repo.svg?style=for-the-badge
[issues-url]: https://github.com/enriqueboyafalcon/EPSU1445/issues
[license-shield]: https://img.shields.io/github/license/enriqueboyafalcon/repo.svg?style=for-the-badge
[license-url]: https://github.com/enriqueboyafalcon/EPSU1445/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/enrique-boya-falcon/
