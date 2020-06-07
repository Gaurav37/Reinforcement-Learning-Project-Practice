#Policy Gradient


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

