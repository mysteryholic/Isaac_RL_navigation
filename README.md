RL_Navigation
Research Log 🔗
How to Set:

cd && mkdir -p rl_navigation/src && cd rl_navigation
git clone https://github.com/j-wye/RL_Navigation src/
sudo rm -rf src/runs/* && sudo rm -rf src/checkpoints/*
colcon build && source install/setup.bash
Register below alias at ~/.bashrc:

alias cbt='cd ~/rl_navigation && colcon build --symlink-install --parallel-workers 16 --cmake-args -DCMAKE_BUILD_TYPE=Release && sb && cd src/ && ~/isaacsim2/python.sh scripts/train.py'
alias path='python3 ~/rl_navigation/src/scripts/path.py'
alias ts='tensorboard --logdir=~/rl_navigation/src/runs'
Just for update Several or Specific Code Examples:

cd ~/rl_navigation/src/scripts && curl -L -O https://raw.githubusercontent.com/j-wye/RL_Navigation/refs/heads/main/scripts/TD_CBAM.py
