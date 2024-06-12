## JAX Noughts and crosses bot

This is a quick JAX project to find out how easy it is to use (very easy!) by making a noughts and crosses bot. 
The bot uses a policy model to predict win rate based a given board, and then checks each of its options and picks the move which creates the most favourable board. The policy model is a simple linear model, and I've played around with some gradient optimisers (momentum, RMSProp) for fun.

It's pretty short, so hopefully fairly easy to understand. As of 2024-12-06, it's only:
```sh
$ (cd src && ls | grep "\.py" | xargs cat | wc -l)
280
```
lines long.