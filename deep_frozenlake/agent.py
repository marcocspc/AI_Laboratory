import random
import numpy as np
import tensorflow as tf

class Agent:

    def __init__(self, alpha_learn_rate, gamma_discount_rate, 
        action_size, state_size, epsilon_decay_rate_lambda, epsilon = 1.0, 
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
        self.epsilon_decay_rate_lambda =  epsilon_decay_rate_lambda;

        #quantidade de acoes possiveis
        self.action_size = action_size
        #quantidade de estados possiveis
        self.state_size = state_size

        ######### rede neural ##########

        #criar uma camada de entrada para ter como shape
        #None e o tamanho do estado
        # o número 1 poderia ser colocado no lugar de None
        #mas isto faria com que a rede so pudesse ser
        #treinada com 1 estado por vez, deixando como
        #None permite a insercao de multiplos estados
        #e o tensor vai se ajustando a entrada
        self.input_layer = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)

        #criar duas camadas ocultas de 50 nos (esta quantidade foi tirada do tutorial linkado abaixo)
        #https://adventuresinmachinelearning.com/reinforcement-learning-tensorflow/
        #porem o numero de camadas e nos pode variar e existem algumas
        #formulas para este calculo, vide:
        #https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        #uma informacao importante eh que a quantidade de camadas altera a forma da onda que descreve
        #o modelo matematico criado pela rede neural, quanto mais camadas, mais complexo o padrao que
        #a rede pode descrever, mas camadas de mais pode gerar um overfitting, vide o link:
        #https://www.mathworks.com/help/deeplearning/ug/improve-neural-network-generalization-and-avoid-overfitting.html;jsessionid=91e4560f25d7d89c69206a11fee1
        self.hidden_layer_1 = tf.layers.dense(self.input_layer, 50, activation=tf.nn.relu)
        self.hidden_layer_2 = tf.layers.dense(self.hidden_layer_1, 50, activation=tf.nn.relu)

        #criar uma camada de saida que corresponde a acao que o agente ira realizar
        self.output_layer = tf.layers.dense(self.hidden_layer_2, self.action_size)

        #a proxima parte eh um pouco mais complicada de se entender, e 
        #eh necessaria pela estrutura do tensorflow
        #mas basicamente vamos definir como a rede sera treinada 
        #aqui utilizaremos a media quadratica como funcao de perda
        #em seguida passamos essa perda para o otimizador que
        #ajustara os pesos da rede neural para que ela possa
        #aproximar melhor os valores de saida
        #tambem definimos aqui um placeholder dentro do tensorflow
        #para guardar o valor de Q(s, a) real (dado pela equaçãp de Bellman), que sera utilizado
        #mais tarde no treinamento
        self.tf_qsa = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)
        #self.tf_qsa = tf.placeholder(shape=[1, self.action_size], dtype=tf.float32)
        loss = tf.losses.mean_squared_error(self.tf_qsa, self.output_layer)
        self.optimizer = tf.train.AdamOptimizer().minimize(loss)

        ######## fim da rede neural ####

        #criar uma session do tensorflow para executar a rede neural
        #quando for necessario
        self.session = tf.Session()
        #pedir ao tensorflow para logar se o processamento eh feito em CPU ou GPU
        #self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))


        #inicializar variaveis do TensorFlow
        self.session.run(tf.initializers.global_variables())


        #agente inicializado!!!!

    def act(self, state):
        #perguntar a rede neural os valores para cada acao do ambiente
        values = self.session.run(self.output_layer, feed_dict={self.input_layer: state})

        #caso np.argmax estiver retornando valores errados (foi o meu caso)
        #comente a linha do np.argmax e descomente as duas abaixo
        #values = values.tolist()
        #action = values.index(max(values))

        #retornar a acao com o maior valor, ou seja, a posicao com o maior valor
        action = np.argmax(values[0])

        return action

    def maxq(self, state):
        #perguntar a rede neural os valores para cada acao do ambiente
        values = self.session.run(self.output_layer, feed_dict={self.input_layer: state})

        #retornar o maior Q dentre os valores retornados 
        index = np.argmax(values[0])
        mxq = values[0, index]

        return mxq

    def learn(self, state, action, next_state, reward):
        qsa_values = self.session.run(self.output_layer, feed_dict={self.input_layer: state})

        #atualizar o valor de Q utilizando a equacao:
        # Y(s,a,r,s′)  =r + γ*maxarg*Q(s′,a′)
        current_q = 0
        #verificar se o proximo estado eh nulo
        if next_state is None:
            #se sim, quer dizer que estamos no ultimo estado, desta forma
            #o termo a direita da equacao Y zera
            #e o valor Q REAL sera a propria recompensa
            current_q = reward
        else:
            #se nao, o valor tera que ser calculado usando a equacao de bellman
            #destacando que aqui nao ha utilizacao do alpha, pois nao estamos
            #atualizando um valor em uma tabela, a alteracao eh dada pela funcao de
            #perda e pelo otimizador
            current_q = reward + self.gamma * self.maxq(next_state)
        
        #agora que temos um valor Q 'real' para esse estado
        #podemos atualizar o valor da previsao feita pela rede
        #note que apenas o valor Q da acao escolhida pela rede
        #foi alterado, pois queremos diminuir ou aumentar a importancia
        #desta acao dependendo da recompensa
        qsa_values[0, action] = current_q

        #agora que temos o valor Q atualizado, vamos informar ao otimizador qual
        #o estado deve jogar na rede neural e qual os valores 'reais' 
        #que devem ser produzidos para esse estado.
        #com base na funcao de perda, ele comparara a saida da rede com os valores 'reais'
        #informados e assim alterara os pesos da rede neural para que a saida seja,
        #da proxima vez, mais proxima dos valores 'reais'
        self.session.run(self.optimizer, feed_dict={self.input_layer: state, self.tf_qsa: qsa_values})

    def learn_batch (self, batch):
        #eh necessario primeiro extrair todos os dados 
        #do batch

        states = np.zeros((len(batch), self.state_size))
        actions = np.zeros((len(batch), self.action_size))
        new_states = np.zeros((len(batch), self.state_size))
        rewards = []

        for i in range(len(batch)):
            states[i], actions[i], new_states[i], reward = batch[i]
            rewards.append(reward)
        
        #agora eh necessario obter os Q(s', a') para cada next_state
        qsa_ = []

        for new_state in new_states:
            qsa_.append(self.maxq(new_state.reshape(1, self.state_size)))
        
        qsa = self.session.run(self.output_layer, feed_dict={self.input_layer: states})
        #qsa = np.zeros((len(batch), self.action_size))
        #atualizar todos os os valores Q(s,a) para cada estado utilizando os Q(s',a')
        for i in range(len(states)):
            if new_states[i] is None:
                qsa[np.argmax(actions[i])] = rewards[i]
            else:
                qsa[np.argmax(actions[i])] = rewards[i] + self.gamma * qsa_[i]
        
        states = np.array(states)

        self.session.run(self.optimizer, feed_dict={self.input_layer: states, self.tf_qsa: qsa})
        


    def decay_epsilon(self, number_of_episodes):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-(self.epsilon_decay_rate_lambda) * number_of_episodes)

    def close_session(self):
        self.session.close()
