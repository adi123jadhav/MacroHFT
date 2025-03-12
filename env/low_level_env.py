from logging import raiseExceptions
import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
import argparse
import os
import torch
import sys
import pathlib
import pdb

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from MacroHFT.tools.demonstration import make_q_table_reward

tech_indicator_list = np.load('../../data/feature_list/single_features.npy', allow_pickle=True).tolist()
tech_indicator_list_trend = np.load('../../data/feature_list/trend_features.npy', allow_pickle=True).tolist()



transcation_cost = 0.0002
back_time_length = 1
max_holding_number = 0.01
alpha = 0

class Testing_Env(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        initial_action=1,  # Changed default to 1 (hold)
    ):
        self.tech_indicator_list = tech_indicator_list
        self.tech_indicator_list_trend = tech_indicator_list_trend
        self.df = df
        self.initial_action = initial_action
        self.action_space = spaces.Discrete(3)  # Modified to 3 actions: 0 (sell), 1 (hold), 2 (buy)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(back_time_length * len(self.tech_indicator_list), ))
        self.terminal = False
        self.stack_length = back_time_length
        self.m = back_time_length
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = initial_action
        self.comission_fee = transcation_cost
        self.max_holding_number = max_holding_number
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        
        # Set initial position based on action: 0=sell (0 holdings), 1=hold (current/0), 2=buy (max)
        if initial_action == 0:
            self.previous_position = 0
            self.position = 0
        elif initial_action == 1:
            self.previous_position = 0  # Initial hold means 0 since we're just starting
            self.position = 0
        elif initial_action == 2:
            self.previous_position = max_holding_number
            self.position = max_holding_number
        
        self.initial_action = initial_action

    def calculate_value(self, price_information, position):
        return price_information["close"] * position

    def reset(self):
        self.terminal = False
        self.m = back_time_length
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.initial_reward = 0
        self.reward_history = [self.initial_reward]
        self.previous_action = self.initial_action
        price_information = self.data.iloc[-1]
        
        # Set position based on initial action
        if self.initial_action == 0:
            self.previous_position = 0
            self.position = 0
        elif self.initial_action == 1:
            self.previous_position = 0  # Initial hold means 0 since we're just starting
            self.position = 0
        elif self.initial_action == 2:
            self.previous_position = self.max_holding_number
            self.position = self.max_holding_number
            
        self.needed_money_memory = []
        self.sell_money_memory = []
        self.comission_fee_history = []
        self.needed_money_memory.append(self.position *
                                        self.data.iloc[-1]["close"])
        self.sell_money_memory.append(0)
        return self.single_state, self.trend_state, {
            "previous_action": self.initial_action,
        }

    def step(self, action):
        # Convert action (0=sell, 1=hold, 2=buy) to position
        if action == 0:  # Sell
            position = 0
        elif action == 1:  # Hold
            position = self.position  # Keep current position
        elif action == 2:  # Buy
            position = self.max_holding_number
        
        self.terminal = (self.m >= len(self.df.index.unique()) - 1)
        previous_position = self.previous_position
        previous_price_information = self.data.iloc[-1]
        self.m += 1
        self.data = self.df.iloc[self.m - self.stack_length:self.m]
        current_price_information = self.data.iloc[-1]
        self.single_state = self.data[self.tech_indicator_list].values
        self.trend_state = self.data[self.tech_indicator_list_trend].values
        self.previous_position = previous_position
        self.position = position
        self.changing = (self.position != self.previous_position)
        
        if previous_position > position:
            self.sell_size = previous_position - position

            cash = self.sell_size * previous_price_information['close'] * (1 - self.comission_fee)
            self.comission_fee_history.append(self.comission_fee * self.sell_size * previous_price_information['close'])

            self.sell_money_memory.append(cash)
            self.needed_money_memory.append(0)
            self.position = position
            previous_value = self.calculate_value(previous_price_information,
                                                  self.previous_position)
            current_value = self.calculate_value(current_price_information,
                                                 self.position)
            self.reward = current_value + cash - previous_value
            if previous_value == 0:
                return_rate = 0
            else:
                return_rate = (current_value + cash -
                               previous_value) / previous_value
            self.return_rate = return_rate
            self.reward_history.append(self.reward)

        elif previous_position < position:
            self.buy_size = position - previous_position
            needed_cash = self.buy_size * previous_price_information['close'] * (1 + self.comission_fee)
            self.comission_fee_history.append(self.comission_fee * self.buy_size * previous_price_information['close'])

            self.needed_money_memory.append(needed_cash)
            self.sell_money_memory.append(0)

            self.position = position
            previous_value = self.calculate_value(previous_price_information,
                                                  self.previous_position)
            current_value = self.calculate_value(current_price_information,
                                                 self.position)
            self.reward = current_value - needed_cash - previous_value
            return_rate = (current_value - needed_cash -
                           previous_value) / (previous_value + needed_cash)

            self.reward_history.append(self.reward)
            self.return_rate = return_rate
        else:  # Hold case (position unchanged)
            previous_value = self.calculate_value(previous_price_information,
                                                 self.previous_position)
            current_value = self.calculate_value(current_price_information,
                                                self.position)
            self.reward = current_value - previous_value
            if previous_value == 0:
                return_rate = 0
            else:
                return_rate = (current_value - previous_value) / previous_value
            self.return_rate = return_rate
            self.reward_history.append(self.reward)
            self.needed_money_memory.append(0)
            self.sell_money_memory.append(0)
            
        self.previous_position = self.position

        if self.terminal:
            return_margin, pure_balance, required_money, commission_fee = self.get_final_return_rate(
            )
            self.pured_balance = pure_balance
            self.final_balance = self.pured_balance + self.calculate_value(
                current_price_information, self.position)
            self.required_money = required_money
            print("the portfit margine is ",
                  self.final_balance / self.required_money)

        return self.single_state, self.trend_state, self.reward, self.terminal, {
            "previous_action": action,
        }

    def get_final_return_rate(self, slient=False):
        sell_money_memory = np.array(self.sell_money_memory)
        needed_money_memory = np.array(self.needed_money_memory)
        true_money = sell_money_memory - needed_money_memory
        final_balance = np.sum(true_money)
        balance_list = []
        for i in range(len(true_money)):
            balance_list.append(np.sum(true_money[:i + 1]))
        required_money = -np.min(balance_list)
        commission_fee = np.sum(self.comission_fee_history)
        return final_balance / required_money, final_balance, required_money, commission_fee


class Training_Env(Testing_Env):
    def __init__(
        self,
        df: pd.DataFrame,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        transcation_cost=transcation_cost,
        back_time_length=back_time_length,
        max_holding_number=max_holding_number,
        initial_action=1,  # Changed default to 1 (hold)
        alpha=alpha,
    ):
        super(Training_Env,
              self).__init__(df, tech_indicator_list, tech_indicator_list_trend, transcation_cost,
                             back_time_length, max_holding_number, initial_action)
        self.q_table = make_q_table_reward(df,
                                           num_action=3,  # Modified to 3 actions
                                           max_holding=max_holding_number,
                                           commission_fee=0.001,
                                           reward_scale=1,
                                           gamma=0.99,
                                           max_punish=1e12)
        self.initial_action = initial_action


    def reset(self):
        single_state, trend_state, info = super(Training_Env, self).reset()
        self.previous_action = self.initial_action
        
        # Set position based on initial action, consistent with parent class
        if self.initial_action == 0:
            self.previous_position = 0
            self.position = 0
        elif self.initial_action == 1:
            self.previous_position = 0
            self.position = 0
        elif self.initial_action == 2:
            self.previous_position = self.max_holding_number
            self.position = self.max_holding_number
            
        info['q_value'] = self.q_table[self.m - 1][self.previous_action][:]
        return single_state, trend_state, info

    def step(self, action):
        single_state, trend_state, reward, done, info = super(Training_Env, self).step(action)
        info['q_value'] = self.q_table[self.m - 1][action][:]
        return single_state, trend_state, reward, done, info