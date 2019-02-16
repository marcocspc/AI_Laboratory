import random
import numpy as np

class agent:

    def __init__(self, alpha_learn_rate, gamma_discount_rate, 
        action_size, state_size, epsilon_decay_rate, epsilon = 1.0, 
        max_epsilon = 1.0, min_epsilon = 0.01):
        
        #fator de aprendizado alfa
        self.alpha = alpha_learn_rate;
        #fator de desconto para o proximo estado gamma
        self.gamma = gamma_discount_rate;
        
        #fator de exploracao epsilon
        self.epsilon = epsilon;
        #valor max para epsilon
        self.max_epsilon = max_epsilon;
        #valor min para epsilon
        self.min_epsilon = min_epsilon;
        #fator de decaimento de epsilon
        self.epsilon_decay =  epsilon_decay_rate;

        #quantidade de acoes possiveis
        self.action_size = action_size
        #quantidade de estados possiveis
        self.state_size = state_size

        #preencher qtable com zeros qtd_estdos x qtd_acoes
        self.qtable = np.zeros((state_size, action_size))

        #agente inicializado!!!!

    def act(self, state):
        #pegar os valores de cada acao armazenados na qtable
        values = self.qtable[state,:]
        #retornar a acao com o maior valor, ou seja, a posicao com o maior valor
        #action = np.argmax(values)

        values = values.tolist()
        action = values.index(max(values))

        return action

    def maxq(self, state):
        #pegar os valores de cada acao armazenados na qtable
        values = self.qtable[state,:]
        #retornar a acao com o maior valor, ou seja, a posicao com o maior valor
        #action = np.argmax(values)

        values = values.tolist()
        mxq = max(values)

        return mxq

    def learn(self, state, action, next_state, reward):
        #para facilitar a leitura da formula abaixo, atribuir o valor atual de q 
        #a uma variavel que representa Q(s,a)
        qsa = self.qtable[state, action]
        
        #atualizar o valor de Q(s, a) de acordo com a Equação de Bellman
        # Q(s, a) = Q(s, a) + α(R + γmaxQ(a', s') - Q(s, a))
        self.qtable[state, action] = qsa + self.alpha * (reward + self.gamma * self.maxq(next_state) - qsa)

    def decay_epsilon(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-(self.epsilon_decay) * episode)
