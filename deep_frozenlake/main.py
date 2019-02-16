import numpy as np
import gym
import random
import time
import sys
from agent import Agent
from memory import Memory
from matplotlib import pyplot as plt
import os

#como o agente agora utiliza uma rede neural, esta funcao ira ajudar na conversao
#do estado  do jogo em um nparray para que seja compativel com a camada de entrada 
#da rede
def convert_state(state):
    if state != None:
        index = state
        state = np.zeros((1, 16))
        state[0][index] = 1
        return state
    else:
        return None

def immediate_reward (reward):
    if reward == 1:
        return 10
    elif reward == 0:
        return -1
    else:
        return -900

def get_int_input (message, default_value):
    try:
        return int(input(message + " [Padrão " + str(default_value) + "] "))
    except ValueError:
        return default_value

        


def main():
    #impedir o pyplot de congelar meu processo atual
    plt.ion()

    #silenciar tensorflow https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

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

    x = Agent(learning_rate, gamma, action_size, state_size, decay_rate,
        epsilon = epsilon, max_epsilon = max_epsilon, min_epsilon =  min_epsilon)

    memo_size = get_int_input("Olá! Insira o tamanho da memória utilizada para treinar o agente:", 50000)
    memo = Memory(memo_size)
    
    batch_size = get_int_input("Insira o tamanho do batch (quantos slots de memória serão utilizados por " +
        "vez para treinar o agente:", 50)

    total_episodes = get_int_input("Insira a quantidade de episodios que o agente irá treinar:", 10000)

    max_steps = get_int_input("Insira a quantidade de acoes que o agente irá realizar " +
        "por episódio:", 100)

    # List of rewards
    rewards = []

    #lista das medias de recompensa para imprimir o grafico
    reward_mean = []

    #lista do numero de vitorias
    victories = []
    victory_percentage = []

    print("Iniciando treinamento do agente.")
    print("Pode apertar Ctrl-C durante o treinamento para interrompê-lo")
    print("Treinando...")

    try:
    # 2 For life or until learning is stopped
        for episode in range(total_episodes):

            print("Episódio " + str(episode + 1) + " de " + str(total_episodes), end = "\r")

            # Reset the environment
            state = convert_state(env.reset())
            step = 0
            done = False
            total_rewards = 0
            victory = False
            
            
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
                new_state = convert_state(new_state)

                if reward == 1:
                    victory = True


                reward = immediate_reward(reward)

                #guardar tupla para inseri-la na memoria
                sample = np.array([state, action, new_state, reward])

                #inserir tupla na memoria
                memo.add_sample(sample)

                #fazer o agente aprender com um exemplo da memoria
                x.learn_batch(memo.sample(batch_size))

                # Our new state is state
                state = new_state

                total_rewards += reward
            
                # If done (if we're dead) : finish episode
                if done == True: 
                    break

            rewards.append(total_rewards)
            reward_mean.append(sum(rewards)/(episode + 1))

            if victory:
                victories.append(1)
            else:
                victories.append(0)
            victory_percentage.append(sum(victories)/(episode + 1))

            x.decay_epsilon(episode)

            plt.figure("Média de Vitórias")
            plt.plot(victory_percentage)

            plt.figure("Média de Recompensas")
            plt.plot(reward_mean)

            plt.show()
            plt.pause(0.0000000000001)
            plt.clf()

    except KeyboardInterrupt:
        print()

    print()
    print("Treino finalizado!")
    print("Recompensa acumulada: " + str(sum(rewards)/total_episodes))
    print()

    try:
        input("Aperte ENTER para ver o agente jogando. Ctrl-C para cancelar.")
    except KeyboardInterrupt:
        print("\nTchau!")
        sys.exit(0)

    ############### O AGENTE VAI JOGAR DAQUI PRA BAIXO ############

    #redefinir ambiente
    env.reset()

    total_episodes = get_int_input("Olá! Insira a quantidade de episodios que o agente irá jogar:", 5)

    max_steps = get_int_input("Olá! Insira a quantidade de acoes que o agente irá realizar " + 
        "por episódio:", 100)

    for episode in range(total_episodes):
        state = convert_state(env.reset())
        step = 0
        done = False
        print("****************************************************")
        print("EPISÓDIO ", episode)

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = x.act(state)
            
            new_state, reward, done, info = env.step(action)
            new_state = convert_state(new_state)
            
            if done:
                # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
                env.render()
                
                # We print the number of step it took.
                print("Número de passos: ", step)
                break

            state = new_state

    env.close()

    input("Concluído! Aperte ENTER para finalizar.")

main()