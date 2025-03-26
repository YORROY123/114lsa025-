import numpy as np
from decimal import Decimal, getcontext
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from collections import Counter
from collections import namedtuple
import copy
import matplotlib.pyplot as plt
import warnings



# Blackjack環境（實現所有要求）
class BlackjackEnv:
    SUITS = ["♠", "♥", "♦", "♣"]  # 黑桃、紅心、方塊、梅花
    RANKS = {1: "A", 11: "J", 12: "Q", 13: "K"}  # 牌面對應

    def __init__(self, num_decks=1):
        self.num_decks = num_decks  # 使用多少副牌
        self.reset_deck()
        self.reset()

    def reset_deck(self):
        """ 初始化牌堆並洗牌 """
        self.deck = [(rank, suit) for rank in range(1, 14) for suit in self.SUITS] * self.num_decks
        random.shuffle(self.deck)

    def draw_card(self):
        """ 發牌，並自動補充牌堆 """
        if not self.deck:
            self.reset_deck()
        
        rank, suit = self.deck.pop()  # 從牌堆中抽取一張
        value = min(rank, 10)  # J, Q, K 都算 10
        return (value, suit)

    def reset(self):
        self.player = [self.draw_card(), self.draw_card()]  # 玩家兩張牌
        self.dealer = [self.draw_card(), self.draw_card()]  # 莊家兩張牌
        self.done = False
        self.actions_history = []
        return self.get_state()

    def calculate_total(self, hand):
        """ 計算手牌總和，A 可為 1 或 11 """
        total = 0
        aces = 0
        for value, _ in hand:  # 只取數值，不管花色
            if value == 1:
                aces += 1
            else:
                total += value
        for _ in range(aces):
            if total + 11 <= 21:
                total += 11
            else:
                total += 1
        return total

    def get_state(self):
        """ 回傳狀態：莊家總和 (隱藏第一張)，玩家第二張牌 """
        return (self.calculate_total(self.dealer), self.player[1][0])

    def step(self, action, verbose=False):
        if action == 1:  # 莊家要牌
            new_card = self.draw_card()
            self.dealer.append(new_card)
            self.actions_history.append(f"莊家要牌: {new_card}")
            if verbose:
                print(f"莊家抽到: {new_card[0]} {new_card[1]}")
                print(f"莊家當前牌: {self.dealer} (總和: {self.calculate_total(self.dealer)})")
            if self.calculate_total(self.dealer) > 21:
                self.done = True
                return self.get_state(), -1, True  # 莊家爆牌，輸
            return self.get_state(), 0, False
        
        if action == 0:
            self.actions_history.append("莊家停止")
            self.done = True
            player_total = self.calculate_total(self.player)
            dealer_total = self.calculate_total(self.dealer)
            
            if dealer_total > 21 or player_total > dealer_total:
                return self.get_state(), -1, True  # 莊家輸
            elif player_total < dealer_total:
                return self.get_state(), 1, True  # 莊家贏
            else:
                return self.get_state(), 0, True  # 平手
                
        return self.get_state(), 0, False

    def show_state(self, reveal_dealer=False):
        """ 顯示目前牌面狀況 """
        player_hand = " ".join(f"{rank}{suit}" for rank, suit in self.player)
        print(f"玩家的牌: {player_hand} (總和: {self.calculate_total(self.player)})")
        
        if reveal_dealer:
            dealer_hand = " ".join(f"{rank}{suit}" for rank, suit in self.dealer)
            print(f"莊家的牌: {dealer_hand} (總和: {self.calculate_total(self.dealer)})")
        else:
            print(f"莊家的第一張牌: {self.dealer[0][0]}{self.dealer[0][1]}")


# DQN神經網路
class DQN(nn.Module):
    def __init__(self, input_size=2, hidden_size=24, output_size=2):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# 優先經驗回放緩衝區
from collections import deque
import numpy as np
from decimal import Decimal, getcontext

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.alpha = Decimal(alpha)  # 使用 Decimal 儲存 alpha
        self.pos = 0
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # 初始化優先權為1，保證它不會是零
        max_priority = max(self.priorities, default=Decimal(1.0))  # 使用 Decimal
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)
        self.pos += 1
    
    def sample(self, batch_size, beta=0.4):
        # 先將所有優先權轉換為 float，確保能進行運算
        priorities = np.array([float(priority) if isinstance(priority, Decimal) else priority for priority in self.priorities])

        # 使用浮點數計算冪
        priorities = np.array([priority**float(self.alpha) for priority in priorities], dtype=object)

        
        # 打印優先權，檢查是否有問題
        # print(f"Priorities before filtering: {priorities}")

        # 保證所有優先權都是有效的
        priorities = np.nan_to_num(priorities, nan=1.0)  # 將 NaN 替換為 1
        priorities = np.maximum(priorities, 1e-6)  # 保證優先權不為零
        # print(f"Priorities after filtering: {priorities}")
        
        # 計算機率，將計算結果從 Decimal 轉回 float
        priorities = np.array([float(priority) for priority in priorities])  # 轉換為 float 類型
        probabilities = priorities / priorities.sum()

        # 進一步檢查機率是否正確
        # print(f"Probabilities: {probabilities}")

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # 計算權重
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # 正規化權重
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        # 更新優先權，轉為 Decimal
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = Decimal(float(priority))  # 更新為高精度


# DQN智能體（莊家）
class DQNAgent:
    def __init__(self):
        # 檢查是否使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            print(f"使用GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("使用CPU")

        self.policy_net = DQN().to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.losses = []
        self.target_update_freq = 10  # 設定每10次更新一次目標網絡
        self.update_count = 0
        self.alpha = 0.6  # 優先經驗回放的超參數
        self.beta = 0.4  # 優先經驗回放的beta值（初始值）

        # 初始學習率，後續會根據訓練進程調整
        self.initial_lr = 0.001
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.memory.buffer) < self.batch_size:
            return 0
        batch, indices, weights = self.memory.sample(self.batch_size, self.beta)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # 使用雙重Q學習來計算目標Q值
        next_actions = self.policy_net(next_states).argmax(1)  # 策略網絡選擇動作
        next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # 目標網絡估算Q值
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 計算加權MSE損失
        loss = (weights * (current_q - target_q).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 更新學習率
        self.lr_scheduler.step()

        # 優先更新經驗回放中的優先權
        td_errors = (current_q - target_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        self.losses.append(loss.item())
        return loss.item()

    def update_target(self):
        # 每隔一定步數更新目標網絡
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.update_count += 1

# 交互模式（實現所有要求）
def play_with_human(agent):
    env = BlackjackEnv()
    agent.epsilon = 0
    round_count = 0
    
    while True:
        round_count += 1
        print(f"\n=== 第 {round_count} 回合 ===")
        state = env.reset()
        
        # 檢查玩家是否上手21點 (A + J/Q/K)
        player_total = env.calculate_total(env.player)
        if len(env.player) == 2 and player_total == 21:
            print(f"玩家的牌: {env.player} (總和: {player_total})")
            print(f"莊家的牌: {env.dealer} (總和: {env.calculate_total(env.dealer)})")
            print("玩家上手21點，你贏了!")
            play_again = input("再玩一局? (y/n): ")
            if play_again.lower() != 'y':
                print(f"遊戲結束，共玩了 {round_count} 回合")
                break
            continue
        
        # 顯示初始牌面
        print(f"玩家的牌: {env.player} (總和: {player_total})")
        print(f"莊家的第一張牌: {env.dealer[0]}")
        
        # 玩家回合
        while True:
            choice = input("請選擇行動 (0=停止, 1=要牌, q=退出): ")
            if choice == 'q':
                print("遊戲結束")
                return
            try:
                action = int(choice)
                if action not in [0, 1]:
                    print("請輸入0或1")
                    continue
                if action == 1:
                    new_card = env.draw_card()
                    env.player.append(new_card)
                    env.actions_history.append(f"玩家要牌: {new_card}")
                    print(f"你抽到: {new_card}")
                    player_total = env.calculate_total(env.player)
                    print(f"玩家的牌: {env.player} (總和: {player_total})")
                    # 檢查五張牌規則
                    if len(env.player) >= 5 and player_total <= 21:
                        print(f"玩家抽滿五張牌且總和不超過21，你贏了!")
                        break
                    if player_total > 21:
                        break
                if action == 0:
                    env.actions_history.append("玩家停止")
                    break
            except ValueError:
                print("請輸入有效數字或q")
        
        # 檢查玩家是否爆牌或五張牌勝利
        player_total = env.calculate_total(env.player)
        if len(env.player) >= 5 and player_total <= 21:
            print(f"\n最終結果:")
            print(f"玩家的牌: {env.player} (總和: {player_total})")
            print(f"莊家的牌: {env.dealer} (總和: {env.calculate_total(env.dealer)})")
            print("\n本局動作歷史:")
            for i, action in enumerate(env.actions_history, 1):
                print(f"{i}. {action}")
            play_again = input("再玩一局? (y/n): ")
            if play_again.lower() != 'y':
                print(f"遊戲結束，共玩了 {round_count} 回合")
                break
            continue
        
        if player_total > 21:
            print(f"\n最終結果:")
            print(f"玩家的牌: {env.player} (總和: {player_total})")
            print(f"莊家的牌: {env.dealer} (總和: {env.calculate_total(env.dealer)})")
            print("你爆牌，莊家贏!")
            print("\n本局動作歷史:")
            for i, action in enumerate(env.actions_history, 1):
                print(f"{i}. {action}")
            play_again = input("再玩一局? (y/n): ")
            if play_again.lower() != 'y':
                print(f"遊戲結束，共玩了 {round_count} 回合")
                break
            continue
        
        # 莊家回合
        print("\n輪到莊家行動:")
        print(f"莊家看到: 玩家第二張牌 {env.player[1]}, 莊家牌 {env.dealer}")
        print(f"莊家當前牌: {env.dealer} (總和: {env.calculate_total(env.dealer)})")
        while not env.done:
            action = agent.get_action(state)
            state, reward, done = env.step(action, verbose=True)
            if not done:
                input("按Enter繼續觀看莊家行動...")
        
        # 顯示結果
        print(f"\n最終結果:")
        print(f"玩家的牌: {env.player} (總和: {env.calculate_total(env.player)})")
        print(f"莊家的牌: {env.dealer} (總和: {env.calculate_total(env.dealer)})")
        if reward > 0:
            print("莊家贏!")
        elif reward < 0:
            print("你贏了!")
        else:
            print("平手!")
        
        print("\n本局動作歷史:")
        for i, action in enumerate(env.actions_history, 1):
            print(f"{i}. {action}")
        
        play_again = input("再玩一局? (y/n): ")
        if play_again.lower() != 'y':
            print(f"遊戲結束，共玩了 {round_count} 回合")
            break



#在 在 train_agent 內部呼叫，用於繪製訓練曲線

def plot_training_progress(episodes, avg_rewards, win_rates, avg_losses, epsilons):
    
    plt.ion()  # 打開交互模式
    plt.figure(1, figsize=(16, 12))
    plt.clf() #清除舊圖  
    warnings.filterwarnings("ignore",message=r".*figure with num: 1 already exists.*")  # 忽略警告

    episodes_range = list(range(1000, (len(avg_rewards) + 1) * 1000, 1000))

    plt.subplot(2, 2, 1)
    plt.plot(episodes_range, avg_rewards, label="Avg Reward", color="b")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Training Progress: Average Reward")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(episodes_range, win_rates, label="Win Rate (%)", color="g")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate (%)")
    plt.title("Training Progress: Win Rate")
    plt.ylim(0, 100)
    # 設置刻度位置和標籤
    plt.yticks(ticks=range(0, 101, 5), labels=[f"{i}%" for i in range(0, 101, 5)]  )  # 刻度位置：0, 5, 10, ..., 100 # 標籤格式：0%, 5%, 10%, ...
    
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(episodes_range, avg_losses, label="Avg Loss", color="r")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Training Progress: Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(episodes_range, epsilons, label="Epsilon", color="purple")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.title("Training Progress: Epsilon Decay")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.pause(0.005)  # 短暫暫停，確保畫面更新
# 訓練函數
def train_agent(episodes=1000000):
    env = BlackjackEnv()
    agent = DQNAgent()
    rewards = []
    wins = []
    avg_rewards = []
    win_rates = []
    epsilons = []
    avg_losses = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        # 檢查玩家是否上手21點 (A + J/Q/K)
        player_total = env.calculate_total(env.player)
        if len(env.player) == 2 and player_total == 21:
            rewards.append(-1)  # 玩家勝，莊家輸
            wins.append(0)
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(rewards[-1000:])
                win_rate = np.mean(wins[-1000:]) * 100
                avg_loss = np.mean(agent.losses[-1000:]) if agent.losses else 0
                avg_rewards.append(avg_reward)
                win_rates.append(win_rate)
                epsilons.append(agent.epsilon)
                avg_losses.append(avg_loss)
                agent.update_target()
                print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.3f}, Win Rate: {win_rate:.1f}%, Avg Loss: {avg_loss:.3f}, Epsilon: {agent.epsilon:.3f}")
                if avg_rewards:  #確保有數據
                    plot_training_progress(episode, avg_rewards, win_rates, avg_losses, epsilons)
                    warnings.filterwarnings("ignore",message=r".*figure with num: 1 already exists.*")  # 忽略警告
            continue
        
        # 玩家行動（模擬玩家策略：低於17要牌）
        while True:
            player_total = env.calculate_total(env.player)
            if player_total < 17:
                new_card = env.draw_card()
                env.player.append(new_card)
                env.actions_history.append(f"玩家要牌: {new_card}")
            else:
                env.actions_history.append("玩家停止")
                break
            player_total = env.calculate_total(env.player)
            # 檢查五張牌規則
            if len(env.player) >= 5 and player_total <= 21:
                rewards.append(-1)  # 玩家勝，莊家輸
                wins.append(0)
                break
            if player_total > 21:
                rewards.append(1)  # 玩家爆牌，莊家勝
                wins.append(1)
                break
        
        if len(env.player) >= 5 and env.calculate_total(env.player) <= 21:
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(rewards[-1000:])
                win_rate = np.mean(wins[-1000:]) * 100
                avg_loss = np.mean(agent.losses[-1000:]) if agent.losses else 0
                avg_rewards.append(avg_reward)
                win_rates.append(win_rate)
                epsilons.append(agent.epsilon)
                avg_losses.append(avg_loss)
                agent.update_target()
                print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.3f}, Win Rate: {win_rate:.1f}%, Avg Loss: {avg_loss:.3f}, Epsilon: {agent.epsilon:.3f}")
                if avg_rewards:  #確保有數據
                    plot_training_progress(episode, avg_rewards, win_rates, avg_losses, epsilons)
                    warnings.filterwarnings("ignore",message=r".*figure with num: 1 already exists.*")  # 忽略警告
            continue
        
        if env.calculate_total(env.player) > 21:
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(rewards[-1000:])
                win_rate = np.mean(wins[-1000:]) * 100
                avg_loss = np.mean(agent.losses[-1000:]) if agent.losses else 0
                avg_rewards.append(avg_reward)
                win_rates.append(win_rate)
                epsilons.append(agent.epsilon)
                avg_losses.append(avg_loss)
                agent.update_target()
                print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.3f}, Win Rate: {win_rate:.1f}%, Avg Loss: {avg_loss:.3f}, Epsilon: {agent.epsilon:.3f}")
                if avg_rewards:  #確保有數據
                    plot_training_progress(episode, avg_rewards, win_rates, avg_losses, epsilons)
                    warnings.filterwarnings("ignore",message=r".*figure with num: 1 already exists.*")  # 忽略警告
            continue
        
        # 莊家行動
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train()
            state = next_state
            episode_reward += reward
        
        rewards.append(episode_reward)
        wins.append(1 if episode_reward > 0 else 0)


        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards[-1000:])
            win_rate = np.mean(wins[-1000:]) * 100
            avg_loss = np.mean(agent.losses[-1000:]) if agent.losses else 0
            avg_rewards.append(avg_reward)
            win_rates.append(win_rate)
            epsilons.append(agent.epsilon)
            avg_losses.append(avg_loss)
            agent.update_target()
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.3f}, Win Rate: {win_rate:.1f}%, Avg Loss: {avg_loss:.3f}, Epsilon: {agent.epsilon:.3f}")
            if avg_rewards:  #確保有數據
                plot_training_progress(episode, avg_rewards, win_rates, avg_losses, epsilons)
                warnings.filterwarnings("ignore",message=r".*figure with num: 1 already exists.*")  # 忽略警告

    plt.savefig('dealer_training_curves.png')
    print("訓練曲線已保存為 'dealer_training_curves.png'")
    
    return agent
# 主程式
if __name__ == "__main__":
    trained_agent = None
    model_path = "dealer_blackjack_dqn_model.pth"
    
    while True:
        mode = input("選擇模式 (1=訓練新模型作為莊家, 2=載入已有模型, 3=與AI莊家對玩, 4=退出): ")
        
        if mode == '1':
            print("開始訓練AI莊家...")
            trained_agent = train_agent(episodes=100000)
            print("\n訓練完成! 正在保存模型...")
            torch.save(trained_agent.policy_net.state_dict(), model_path)
            print(f"模型已保存至 {model_path}")
        
        elif mode == '2':
            if trained_agent is None:
                trained_agent = DQNAgent()
                try:
                    trained_agent.policy_net.load_state_dict(torch.load(model_path))
                    trained_agent.policy_net.eval()
                    print(f"已從 {model_path} 載入模型")
                except FileNotFoundError:
                    print(f"找不到模型檔案 {model_path}，請先訓練模型")
            else:
                print("記憶體中已有模型，若要重新載入請先退出程式")
        
        elif mode == '3':
            if trained_agent is None:
                print("請先訓練或載入模型")
            else:
                print("\n開始與AI莊家對戰!")
                play_with_human(trained_agent)
        
        elif mode == '4':
            print("程式結束")
            break
        
        else:
            print("請輸入1, 2, 3或4")