# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.






from . import *


def build_scenario(builder):
  builder.config().game_duration = 100
  builder.config().deterministic = False
  builder.config().offsides = True
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True
  builder.SetBallPosition(0.68, -0.07)


  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)     # Casillas
  builder.AddPlayer(0.69, -0.077, e_PlayerRole_CM)   # Fabregas
  builder.AddPlayer(0.75, 0.08, e_PlayerRole_CF)    # Iniesta

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-0.98, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(-0.75, 0.0, e_PlayerRole_LB)
  builder.AddPlayer(-0.75, 0.065, e_PlayerRole_CB)
  builder.AddPlayer(-0.75, 0.095, e_PlayerRole_RB)
  builder.AddPlayer(-0.62, 0.085, e_PlayerRole_CM)
  builder.AddPlayer(-0.62, -0.06, e_PlayerRole_LM)
  builder.AddPlayer(-0.62, 0.2, e_PlayerRole_RM)
