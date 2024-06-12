from model import (
    make_random_training_data,
    random_policy_weights,
    loss,
    momentum_rmsprop_update_weights,
    fight_random,
    show_fight_random
)
import jax
SEED = 0

testX, testY = make_random_training_data(SEED - 1, 100)
test_loss = lambda weights: sum([loss(x, y, weights) for x, y in zip(testX, testY)])

# Simple policy - Each space contributes to win percentage randomly
original_policy_weights = random_policy_weights(SEED)

policy_weights = original_policy_weights.copy()

losses : list = []
for i in range(10):
    X, Y = make_random_training_data(SEED + i, 100)
    policy_weights = momentum_rmsprop_update_weights(policy_weights, X, Y)
    cur_loss = test_loss(policy_weights)
    losses.append(cur_loss)
    print(f"Current Loss: {cur_loss}")


print(f"Win rate: {fight_random(100, policy_weights)}")

show_fight_random(policy_weights, 2)
print("=========")
show_fight_random(policy_weights, 5)
print("=========")
