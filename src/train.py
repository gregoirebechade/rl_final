from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_agent, evaluate_HIV_population, evaluate_HIV
import time
import numpy as np
from scipy.optimize import minimize


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def act(self, observation, use_random=False):
        # observation = state. c'est un vecteur de taille 6 : [T1, T1*, T2, T2*, V, E]
        # mon but : minimiser V et maximiser E et donner le moins de médocs possibles
        # mon action = un entier entre 0 et 3. C'est l'indice de l'action dans l'ensemble d'actions : 
        # [[0.0, 0.0], [0.0, 0.3], [0.7, 0.0], [0.7, 0.3]]
        
        probas = self.poids @ np.array(observation)
        probas -= np.max(probas)  # Subtract max value to prevent overflow
        probas = np.exp(probas)
        probas /= np.sum(probas)
        return np.argmax(probas)
    def save(self, path):
        pass

    def load(self):
        pass
    def update_poids(self, poids): 
        self.poids=poids



if __name__=='__main__':
    scores=[]
    def evaluate_weights(poids):
        poids = np.reshape(poids, (4, 6))
        agent.update_poids(poids)
        sc=-evaluate_HIV(agent=agent, nb_episode=1)  # Négatif car on veut maximiser
        scores.append(sc)
        print('current score : ', sc)
        return sc

    agent = ProjectAgent()
    agent.load()
    
    # score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=15)  # sur 15 patients tirés au hasard
    T1 = 163573.0  # healthy type 1 cells concentration (cells per mL)
    T1star = 11945.0  # infected type 1 cells concentration (cells per mL)
    T2 = 5.0  # healthy type 2 cells concentration (cells per mL)
    T2star = 46.0  # infected type 2 cells concentration (cells per mL)
    V = 63919.0  # free virus (copies per mL)
    E = 24.0  # immune effector 

    if True : 
        env_begin=np.array([T1, T1star, T2, T2star, V, E])
        poids=np.ones((4, 6))
        for i in range(len(env_begin)): 
            poids[:, i] = (1/6)/env_begin[i]
        
        # result = minimize(evaluate_weights, poids.flatten(), method='L-BFGS-B')
        # poids_optimal = result.x.reshape((4,6))
        # agent.update_poids(poids_optimal)
        # # save poids_optimal
        # # agent.save('poids_optimal.npy')
        # np.savetxt('./poids_opti.txt', poids_optimal, delimiter=",")
        
    
        # score_agent = evaluate_HIV(agent=agent, nb_episode=1) # sur un patient, 7.5 secondes 


        # print(score_agent)
            
        update=0.2
        scores=[]
        agent=ProjectAgent()
        agent.update_poids(poids) 
        current_score=evaluate_HIV(agent=agent, nb_episode=1)
        with open('scores.txt', 'w') as f : 
            f.write(str(current_score)+'\n')
        scores.append(current_score)
        for i in range(10): 
            update=update/2
            for j in range(len(poids)): 
                for k in range(len(poids[j])):
                    print(i, j, k )

                    smaller = np.copy(poids)
                    smaller[j, k] *= (1-update)
                    agent.update_poids(smaller)
                    small_score=evaluate_HIV(agent=agent, nb_episode=1)

                    bigger = np.copy(poids)
                    bigger[j, k] *= (1+update)
                    agent.update_poids(bigger)
                    big_score=evaluate_HIV(agent=agent, nb_episode=1)

                    if big_score > current_score : 
                        count=0
                        while big_score > current_score and count<10:
                            count+=1
                            poids = bigger
                            current_score = big_score
                            with open('scores.txt', 'a') as f : 
                                f.write(str(current_score)+'\n')
                            scores.append(current_score)
                            bigger[j, k] *= (1+update)
                            agent.update_poids(bigger)
                            big_score=evaluate_HIV(agent=agent, nb_episode=1)

                    elif small_score > current_score :
                        count=0
                        while small_score > current_score and count<10:
                            count+=1
                            poids = smaller
                            current_score = small_score
                            with open('scores.txt', 'a') as f : 
                                f.write(str(current_score)+'\n')
                            scores.append(current_score)
                            smaller[j, k] *= (1-update)
                            agent.update_poids(smaller)
                            small_score=evaluate_HIV(agent=agent, nb_episode=1)
            np.savetxt('./poids_opti'+str(i)+'.txt', poids, delimiter=",")







    
