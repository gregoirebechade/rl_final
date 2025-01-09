# from gymnasium.wrappers import TimeLimit
# from env_hiv import HIVPatient
# from evaluate import evaluate_agent, evaluate_HIV_population, evaluate_HIV
# import time
# import numpy as np
# from scipy.optimize import minimize


# env = TimeLimit(
#     env=HIVPatient(domain_randomization=False), max_episode_steps=200
# )  # The time wrapper limits the number of steps in an episode at 200.
# # Now is the floor is yours to implement the agent and train it.


# # You have to implement your own agent.
# # Don't modify the methods names and signatures, but you can add methods.
# # ENJOY!
# class ProjectAgent:
#     def act(self, observation, use_random=False):
#         # observation = state. c'est un vecteur de taille 6 : [T1, T1*, T2, T2*, V, E]
#         # mon but : minimiser V et maximiser E et donner le moins de médocs possibles
#         # mon action = un entier entre 0 et 3. C'est l'indice de l'action dans l'ensemble d'actions : 
#         # [[0.0, 0.0], [0.0, 0.3], [0.7, 0.0], [0.7, 0.3]]
        
#         probas = self.poids @ np.array(observation)
#         probas -= np.max(probas)  # Subtract max value to prevent overflow
#         probas = np.exp(probas)
#         probas /= np.sum(probas)
#         return np.random.choice([0, 1, 2, 3], p=probas)
#     def save(self, path):
#         pass

#     def load(self):
#         pass
#     def update_poids(self, poids): 
#         self.poids=poids



# if __name__=='__main__':
#     print('beginning training')
#     scores=[]
#     def evaluate_weights(poids):
#         poids = np.reshape(poids, (4, 6))
#         agent.update_poids(poids)
#         sc=-evaluate_HIV(agent=agent, nb_episode=1)  # Négatif car on veut maximiser
#         scores.append(sc)
#         print('current score : ', sc)
#         return sc

#     agent = ProjectAgent()
#     agent.load()
    
#     # score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)  # sur 15 patients tirés au hasard
#     T1 = 163573.0  # healthy type 1 cells concentration (cells per mL)
#     T1star = 11945.0  # infected type 1 cells concentration (cells per mL)
#     T2 = 5.0  # healthy type 2 cells concentration (cells per mL)
#     T2star = 46.0  # infected type 2 cells concentration (cells per mL)
#     V = 63919.0  # free virus (copies per mL)
#     E = 24.0  # immune effector 

#     if True : 
#         env_begin=np.array([T1, T1star, T2, T2star, V, E])
#         poids=np.ones((4, 6))
#         for i in range(len(env_begin)): 
#             poids[:, i] = (1/6)/env_begin[i] 
        
#         poids=np.array([[9.439274309737346276e-07,5.715083019394448050e-05,3.333333333333333287e-02,3.623188405797101060e-03,3.013253753448374212e-06,6.944444444444444059e-03],
#         [1.630260902879244741e-07,7.359272533835634628e-06,4.181333333333333430e-02,3.623188405797101060e-03,2.607466741761708749e-06,1.877777777777777908e-03],
#         [2.608417444606791586e-06,2.358029859076322027e-05,3.333333333333333287e-02,4.544927536231883371e-03,2.607466741761708749e-06,6.944444444444444059e-03],
#         [1.177481409931141792e-06,1.395283940281847274e-05,3.333333333333333287e-02,3.623188405797101060e-03,2.607466741761708749e-06,6.944444444444444059e-03]])

        
#         agent=ProjectAgent()
#         agent.update_poids(poids) 
#         test=[evaluate_HIV(agent=agent, nb_episode=1) for _ in range(10)]
#         current_score=np.mean(test)
#         if np.std(test) < 0.001 * np.mean(test) : 
#             print('euh, bizarre')
#             print(test)
#         with open('scores_bis.txt', 'w') as f : 
#             f.write('scooooores:' +'\n')
#             f.write(str(current_score)+'\n')
#         for l in range(20): 
#             update=1.2
#             scores=[]
#             for i in range(5): 
#                 update=update/2
#                 for j in range(len(poids)): 
#                     for k in range(len(poids[j])):
#                         print(l, i, j, k )
#                         smaller = np.copy(poids)
#                         smaller[j, k] *= (1-update)
#                         agent.update_poids(smaller)
#                         small_score=np.mean([evaluate_HIV(agent=agent, nb_episode=1) for _ in range(10)])
#                         bigger = np.copy(poids)
#                         bigger[j, k] *= (1+update)
#                         agent.update_poids(bigger)
#                         big_score=np.mean([evaluate_HIV(agent=agent, nb_episode=1) for _ in range(10)])
#                         if big_score > current_score : 
#                             count=0
#                             while big_score > current_score and count<10:
#                                 count+=1
#                                 poids = bigger
#                                 current_score = big_score
#                                 with open('scores.txt', 'a') as f : 
#                                     f.write(str(current_score)+'\n')
#                                 scores.append(current_score)
#                                 bigger[j, k] *= (1+update)
#                                 agent.update_poids(bigger)
#                                 big_score=np.mean([evaluate_HIV(agent=agent, nb_episode=1) for _ in range(10)])
#                         elif small_score > current_score :
#                             count=0
#                             while small_score > current_score and count<10:
#                                 count+=1
#                                 poids = smaller
#                                 current_score = small_score
#                                 with open('scores.txt', 'a') as f : 
#                                     f.write(str(current_score)+'\n')
#                                 scores.append(current_score)
#                                 smaller[j, k] *= (1-update)
#                                 agent.update_poids(smaller)
#                                 small_score=np.mean([evaluate_HIV(agent=agent, nb_episode=1) for _ in range(10)])
#                 np.savetxt('./poids_opti'+str(l)+str(i)+'_bis.txt', poids, delimiter=",")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV
import os 
import torch.nn.functional as F  # Optional for more concise activation functions


# Define the Q-Network
class QNetwork(nn.Module): # approxime Q(s,a); A un état s, on associe Q(s,a) pour chaque action a
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, output_dim)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x 

# Replay Buffer for Experience Replay
class ReplayBuffer: # sert à stocker les expériences passées pour les réutiliser
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = [state  if len(state)==6 else state[0] for state in states]
        for state in states : 
            if len(state)!=6 : 
                print('state', state)   
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class ProjectAgent:
    def __init__(self):
        # self.criterion =  nn.SmoothL1Loss() # essayer avec mse
        self.criterion = nn.MSELoss()
        self.file_path = os.path.join(os.path.dirname(__file__), "max_so_far.pth")
        self.state_dim = 6
        self.action_dim = 4
        self.q_network = QNetwork(self.state_dim, self.action_dim)
        self.target_network = QNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer(10000)
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.9977
        self.epsilon_min = 0.1 # passer à 0.001 pour voir 

    def act(self, state):
        if state[1]=={}: 
            state=state[0]  
        # if np.random.rand() < self.epsilon:
        #     return np.random.choice(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item() # renvoie le max de [Q(s,a) pour chaque a]

    # def train(self): # GREGOIRE
    #     if len(self.replay_buffer) < self.batch_size:
    #         return

    #     # Sample a batch from the replay buffer
    #     states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

    #     states = torch.FloatTensor(states)
    #     actions = torch.LongTensor(actions)
    #     rewards = torch.FloatTensor(rewards)
    #     next_states = torch.FloatTensor(next_states)
    #     dones = torch.FloatTensor(dones)

    #     # Compute Q-values and targets
    #     q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    #     next_q_values = self.target_network(next_states).max(1)[0]
    #     targets = rewards + self.gamma * next_q_values * (1 - dones)

    #     # Compute loss
    #     loss = nn.MSELoss()(q_values, targets)

    #     # Backpropagation
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()


    def train(self): # ROMANE
        if len(self.replay_buffer) < self.batch_size:
            return 
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        targets = torch.addcmul(rewards, 1 - dones, next_q_values, value=self.gamma)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        loss = self.criterion(current_q_values, targets.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # dessous = code romane

        # X, A, R, Y, D = self.replay_buffer.sample(self.batch_size)


        # X = torch.FloatTensor(X)
        # A = torch.LongTensor(A)
        # R = torch.FloatTensor(R)
        # Y = torch.FloatTensor(Y)
        # D = torch.FloatTensor(D)


        # QYmax = self.target_network(Y).max(1)[0].detach()
        # update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
        # QXA = self.q_network(X).gather(1, A.to(torch.long).unsqueeze(1))
        # loss = self.criterion(QXA, update.unsqueeze(1))
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()


    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path = "dqn_agent.pth"):
        torch.save(self.q_network.state_dict(), path)

    def load(self):
        # path = "dqn_agent.pth"
        self.q_network.load_state_dict(torch.load(self.file_path))

# Training Loop
if __name__ == "__main__":
    training = True
    if training : 
        print('train greg, mse ')
        env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = ProjectAgent()
        num_episodes = 2000
        target_update_freq = 10
        perf_max=0
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done, trunc = False, False
            while not done and not trunc :                 
                if np.random.rand() < agent.epsilon:
                    action =  np.random.choice(agent.action_dim)
                else : 
                    action = agent.act(state)
                
                next_state, reward, done, trunc, _ = env.step(action)
                agent.replay_buffer.add(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                agent.train()
                if done:
                    break

            # Update the target network
            if episode % target_update_freq == 0:
                agent.update_target_network()

            # Decay epsilon
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            perf = evaluate_HIV(agent=agent, nb_episode=5)
            if perf > perf_max : 
                print('new perf max', f"{perf:.2e}")
                perf_max = perf
                agent.save(path= 'max_so_far'+str(episode)+'.pth')

            # print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2e}, perf: {perf:.2e} Epsilon: {agent.epsilon:.3e}") # nvlle méthode 

        # Save the trained model
        agent.save()
        print('finally, the best loss reached is', str(perf_max))
        


# essayer avec L1 loss
# faire varier : train de romane, mse loss, epsilon decay


                

