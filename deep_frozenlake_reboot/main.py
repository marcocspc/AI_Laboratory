# -*- coding: utf-8 -*-

import numpy as np
import gym
import random
import time
import sys
from agent import Agent
#from memory import Memory
from matplotlib import pyplot as plt
import pickle
from datetime import datetime


from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

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

#utilizar uma melhor distribuição de recompensas
def immediate_reward (reward):
    if reward == 1:
        return 1000
    elif reward == 0:
        return 1
    else:
        return -1000

#função auxiliar para obter input int do usuário
def get_int_input (message, default_value):
    try:
        return int(input(message + " [Padrão " + str(default_value) + "] "))
    except ValueError:
        return default_value    

#função auxiliar para obter input string do usuário
def get_string_input (message, default_value):
    try:
        string = str(input(message + " [Padrão " + str(default_value) + "] "))
        return string if string != "" else default_value
    except ValueError:
        return default_value

def train(agent, env):
    #memo_size = get_int_input("Olá! Insira o tamanho da memória utilizada para treinar o agente:", 50000)
    #memo = Memory(memo_size)
    
    #batch_size = get_int_input("Insira o tamanho do batch (quantos slots de memória serão utilizados por " +
    #    "vez para treinar o agente:", 50)

    total_episodes = get_int_input("Insira a quantidade de episodios que o agente irá treinar:", 10000)

    max_steps = get_int_input("Insira a quantidade de acoes que o agente irá realizar " +
        "por episódio:", 100)

    graphs = get_string_input("Deseja ver os gráficos? S/N", "N")

    # List of rewards
    rewards = []

    #lista das medias de recompensa para imprimir o grafico
    reward_mean = [0]

    #lista do numero de vitorias
    victories = [0]
    victory_percentage = [0]

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
                if numero_aleatorio > agent.epsilon:
                    action = agent.act(state)
                else:
                    action = env.action_space.sample()

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, done, info = env.step(action)
                new_state = convert_state(new_state)

                if reward == 1:
                    victory = True

                reward = immediate_reward(reward)


                #guardar tupla para inseri-la na memoria
                #sample = np.array([state, action, new_state, reward])

                #inserir tupla na memoria
                #memo.add_sample(sample)

                #fazer o agente aprender com um exemplo da memoria
                #agent.learn_batch(memo.sample(batch_size))
                agent.learn(state, action, new_state, reward)

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

            agent.decay_epsilon(episode)

            if graphs == "S":
                plt.figure("Média de Vitórias")
                plt.plot(victory_percentage)

                plt.figure("Média de Recompensas")
                plt.plot(reward_mean)

                plt.show()

                plt.pause(1e-10)

                plt.clf()
    except KeyboardInterrupt:
        print()

    print()
    print("Treino finalizado!")
    print("Recompensa acumulada: " + str(sum(rewards)/total_episodes))
    print("Taxa de vitórias: " + str(victory_percentage[-1]))
    print()

    string = get_string_input("Deseja salvar o agente? [S/N]", "N")
    
    if string == "S":
        name = get_string_input("Insira um nome para o arquivo:", "agent.pkl")

        try:
            with open(name, "wb") as agent_file:
                pickle.dump(agent, agent_file)
        except IOError as e:
            print(e)

def play(agent, env):
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
            action = agent.act(state)
            
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

    string = get_string_input("Deseja salvar o agente? [S/N]", "S")

    if string == "S":
        name = get_string_input("Insira um nome para o arquivo:", "agent.pkl")

        try:
            with open(name, "wb") as agent_file:
                pickle.dump(agent, agent_file)
        except IOError as e:
            print(e)


def main():
    #impedir o pyplot de congelar meu processo atual
    plt.ion()

    opcao = get_int_input("Qual ambiente desejaria utilizar?\n" + 
        "1 - FrozenLake-v0\n" +
        "2 - FrozenLakeNotSlippery-v0\n", 1)

    print("Inicializando ambiente...")
    if opcao == 1:
        #criar ambiente frozenlake estocástico
        env = gym.make("FrozenLake-v0")
    else:
        #criar ambiente frozenlake determinístico
        env = gym.make("FrozenLakeNotSlippery-v0")

    string = get_string_input("Deseja carregar algum arquivo serializado de um agente " +
        "já treinado? [S/N]", "N")

    x = None

    if string == "S":
        nome = get_string_input("Insira o nome do arquivo a ser carregado:", "agent.pkl")

        print("Inicializando agente...")
        with open(nome, "rb") as ai:
            x = pickle.load(ai)
    else:
        print("Inicializando agente...")
        action_size = env.action_space.n
        state_size = env.observation_space.n
        gamma = 0.96
        learning_rate = 0.9
        epsilon = 1.0                 # Exploration rate
        max_epsilon = 1.0             # Exploration probability at start
        min_epsilon = 0.01            # Minimum exploration probability 
        decay_rate = 0.0005

        x = Agent(learning_rate, gamma, action_size, state_size, decay_rate,
            epsilon = epsilon, max_epsilon = max_epsilon, min_epsilon =  min_epsilon)

    if string == "S":
        string = get_string_input("Deseja treinar o agente carregado? [S/N]", "N")

        if string == "S":
            train(x, env)
    else:
        #treinar agente
        train(x, env)

    string = get_string_input("Deseja ver o agente jogando? [S/N]", "S")

    if string == "S":
        play(x, env)
        
    input("Concluído! Aperte ENTER para finalizar.")

main()