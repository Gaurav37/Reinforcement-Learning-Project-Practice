# Policy Gradient on Cartpole-1


![](/PolicyGradientCartpole.png)


```
    env=gym.make('CartPole-v1')
    agent=Agent(lr=0.001, state_dim=[4], gamma=0.99, num_layers=3, hidden_dim=256,num_actions=2)
    score_history=[]
    score=0
    n_episodes=2500
        
    for i in range(n_episodes):
        print('episode: ',i, 'score %.3f' % score)
        done=False
        score = 0
        observation = env.reset()
        while not done:
            action=agent.choose_action(observation)
            observation_new,reward,done,info=env.step(action)
            agent.reward_memory.append(reward)
            observation=observation_new
            score += reward
        score_history.append(score)
        agent.learn()
```

The above code block is highest level block in whole code where it calls other methods to choose actions based on observation records and then just like most other RL algorithms appends the reward of that action in the episode. After each episode scores are appended into score list and learn method is called.

```
        for g, logprob in zip(G,self.action_memory):   #log probability=action_memory
            loss+= -g*logprob
            
        loss.backward()
        self.policy.optimizer.step()
```
This part of code makes sure that the losses are backpropagated. Action_memory contains values of logarithmic action probabilities and G is normalized gain or result of stepwise action. In short, it is implementation of formula θ ← θ+αγ​t​G(​∇​lnπ(At|St,θ)) where we are backpropagating losses by taking into consideration the differentiation of logarithmic action probabilities.

The policy gradient algorithm uses categorical distribution over softmax action probabilities of output from neural network used.
```
        probabilities=F.softmax(self.policy.forward(observation))
        action_probabilities=T.distributions.Categorical(probabilities)
        action=action_probabilities.sample()
        log_probs=action_probabilities.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()
```
