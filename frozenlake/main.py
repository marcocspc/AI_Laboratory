import numpy as np
import gym
import random
import time
import sys
from agent import agent

print("Inicializando ambiente...")
env = gym.make("FrozenLake-v0")


print("Inicializando agente...")
action_size = env.action_space.n
state_size = env.observation_space.n
gamma = 0.96
learning_rate = 0.81 
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.005

x = agent(learning_rate, gamma, action_size, state_size, decay_rate,
    epsilon = epsilon, max_epsilon = max_epsilon, min_epsilon =  min_epsilon)

try:
    input_usuario = int(input("Olá! Insira a quantidade de episodios que o agente irá treinar:" +
                            " [Padrão 10000] "))
except ValueError:
    input_usuario = 0

total_episodes = 0

if input_usuario == 10000 or input_usuario == 0:
    #numero maximo de episodios
    total_episodes = 10000
else:
    total_episodes = input_usuario

try:
    input_usuario = int(input("Olá! Insira a quantidade de acoes que o agente irá realizar " + 
                            "por episódio: [Padrão 100] "))
except ValueError:
    input_usuario = 0

max_steps = 0

if input_usuario == 99 or input_usuario == 0:
    #numero maximo de acoes por episodio
    max_steps = 100
else:
    max_steps = input_usuario



# List of rewards
rewards = []

print("Iniciando treinamento do agente.")
time.sleep(1)
print("Pode apertar Ctrl-C durante o treinamento para interrompê-lo")
time.sleep(1)
print("Treinando...")

try:
# 2 For life or until learning is stopped
    for episode in range(total_episodes):

        print("Episódio " + str(episode) + " de " + str(total_episodes), end = "\r")

        # Reset the environment
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
        
        
        for step in range(max_steps):

            numero_aleatorio = random.uniform(0, 1)

            #se nosso numero aleatorio for maior que epsilon
            #devemos aproveitar o conhecimento adquirido
            #se nao, explorar
            if numero_aleatorio > x.epsilon:
                action = x.act(state)
            else:
                action = env.action_space.sample()

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)

            x.learn(state, action, new_state, reward)

            # Our new state is state
            state = new_state

            total_rewards += reward
        
            # If done (if we're dead) : finish episode
            if done == True: 
                break

        rewards.append(total_rewards)
        x.decay_epsilon(episode)

except KeyboardInterrupt:
    print()

print()
print("Treino finalizado!")
print("Recompensa acumulada: " + str(sum(rewards)/total_episodes))
print("Q-Table: ")
print(x.qtable)
print()

try:
    input("Aperte ENTER para ver o agente jogando. Ctrl-C para cancelar.")
except KeyboardInterrupt:
    print("\nTchau!")
    sys.exit(0)

############### O AGENTE VAI JOGAR DAQUI PRA BAIXO ############

#redefinir ambiente
env.reset()

try:
    input_usuario = int(input("Olá! Insira a quantidade de episodios que o agente irá jogar:" +
                         " [Padrão 5] "))
except ValueError:
    input_usuario = 0

if input_usuario == 0:
    #numero maximo de episodios
    total_episodes = 5
else:
    total_episodes = input_usuario

try:
    input_usuario = int(input("Olá! Insira a quantidade de acoes que o agente irá realizar " + 
                            "por episódio: [Padrão 100] "))
except ValueError:
    input_usuario = 0

max_steps = 0

if input_usuario == 0:
    #numero maximo de acoes por episodio
    max_steps = 100
else:
    max_steps = input_usuario

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISÓDIO ", episode)

    for step in range(max_steps):
        # Take the action (index) that have the maximum expected future reward given that state
        action = x.act(state)
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()
            
            # We print the number of step it took.
            print("Número de passos: ", step)
            break

        state = new_state

env.close()

input("Concluído! Aperte ENTER para finalizar.")

