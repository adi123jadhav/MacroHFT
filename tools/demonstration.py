import pandas as pd
import numpy as np

def make_q_table_reward(df: pd.DataFrame,
                        num_action=3,  # Default to 3 actions: SELL, HOLD, BUY
                        max_holding=100,
                        reward_scale=1000,
                        gamma=0.999,
                        commission_fee=0.001,
                        max_punish=1e12):
    
    # Define action constants
    SELL = 0
    HOLD = 1
    BUY = 2
    
    # Initialize Q-table with zeros
    q_table = np.zeros((len(df), num_action, num_action))
    
    def calculate_value(price_information, position):
        return price_information["close"] * position
    
    for t in range(2, len(df) + 1):
        current_price_information = df.iloc[-t]
        future_price_information = df.iloc[-t + 1]
        
        for previous_action in range(num_action):
            # Determine the position for previous action
            if previous_action == SELL:
                previous_position = 0
            elif previous_action == BUY:
                previous_position = max_holding
            else:  # HOLD - This is a placeholder value since true HOLD depends on context
                previous_position = max_holding / 2  # Middle position for initialization
            
            for current_action in range(num_action):
                # Determine position for current action
                if current_action == SELL:
                    current_position = 0
                elif current_action == BUY:
                    current_position = max_holding
                elif current_action == HOLD:
                    # For HOLD, we maintain the previous position
                    current_position = previous_position
                
                # Calculate reward based on action transition
                if current_action == previous_action:
                    # No position change (including a true HOLD)
                    current_value = calculate_value(current_price_information, previous_position)
                    future_value = calculate_value(future_price_information, current_position)
                    reward = future_value - current_value  # No transaction costs
                    reward = reward_scale * reward
                    q_table[len(df) - t][previous_action][current_action] = reward + gamma * np.max(
                        q_table[len(df) - t + 1][current_action][:])
                
                elif (current_action == BUY and previous_action != BUY) or \
                     (current_action == HOLD and previous_action == SELL):
                    # Buying action (from SELL or HOLD to BUY) or (from SELL to HOLD)
                    position_change = current_position - previous_position
                    buy_money = position_change * current_price_information['close'] * (1 + commission_fee)
                    current_value = calculate_value(current_price_information, previous_position)
                    future_value = calculate_value(future_price_information, current_position)
                    reward = future_value - (current_value + buy_money)
                    reward = reward_scale * reward
                    q_table[len(df) - t][previous_action][current_action] = reward + gamma * np.max(
                        q_table[len(df) - t + 1][current_action][:])
                
                elif (current_action == SELL and previous_action != SELL) or \
                     (current_action == HOLD and previous_action == BUY):
                    # Selling action (from BUY or HOLD to SELL) or (from BUY to HOLD)
                    position_change = previous_position - current_position
                    sell_money = position_change * current_price_information['close'] * (1 - commission_fee)
                    current_value = calculate_value(current_price_information, previous_position)
                    future_value = calculate_value(future_price_information, current_position)
                    reward = future_value + sell_money - current_value
                    reward = reward_scale * reward
                    q_table[len(df) - t][previous_action][current_action] = reward + gamma * np.max(
                        q_table[len(df) - t + 1][current_action][:])
    
    return q_table