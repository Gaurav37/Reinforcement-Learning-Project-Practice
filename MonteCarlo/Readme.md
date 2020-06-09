Picture shows the results we received from Monte Carlo implementation to solve frozen lake environment.

![](/MonteCarloEnvFrozenLake)

Let us now begin exploring the main method of Monte Carlo which is as follows.
```
    for i in pbar: # for i in num_episodes
        G = 0.0
        trans=sample_episode(env, Q, eps=eps)
        trans.reverse()
        for count1 in trans:
            G=G+count1[2]
            C[count1[0]][count1[1]]+=1
            Q[count1[0]][count1[1]]=Q[count1[0]][count1[1]]+((G-Q[count1[0]][count1[1]])/(C[count1[0]][count1[1]]))
        returns[i]=G
        G_queue.append(G)
        pbar.set_description(f'Episodes G={sum(G_queue) / len(G_queue)}')
        
    return Q, returns
```
For each episode we are calling sample_episode method which as we can see below returns transactions of the form S0, A0, R1. For each transaction, we are increment the count whenever we encounter the state again and then we are adjusting Q values as per those counts by the formula provided in Sutton Barto book.
```
    state = env.reset()
    while True:
        Qs = Q[state]
        choosen_action = select_action_epsilon_greedy(Q_array = Qs, eps = eps)
        next_state, reward, done, info = env.step(choosen_action)
        transitions.append( (state, choosen_action, reward) )
        state = next_state
        if done==True:
            break
    return transitions
```
Now the sample_episode method calls select_action_epsilon_greedy method which selects e-greedy action based on the current Q-values and breaks tie between equal Q-values randomly.
```
    probE=np.random.random()
    if eps<probE:
        Qmax= np.max(Q_array)
        Qmaxpos=np.where(Q_array==Qmax)
        action=np.random.choice(Qmaxpos[0])
    else :
        action=np.random.choice(len(Q_array))
    return action
```
