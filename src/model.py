from game import *

LEARNING_RATE = 0.01

@jit
def random_policy_weights(seed, shape=[9,20]) -> jax.Array:
    return random.normal(key=random.key(seed=seed), shape=shape, dtype=jnp.float32)

@jit
def policy_model(board : jax.Array, policy_weights : jax.Array):
    return logistic(jnp.sum(jnp.matmul(board, policy_weights)))

@jit
def logistic(x): # A logistic function mapping to [0, 1]
    return 1 / (1 + jnp.exp(-x))

@jit
def loss(board, is_win, policy_weights):
    # Win rate is the output of the policy model
    
    diff = is_win - policy_model(board, policy_weights)
    return diff * diff

def make_random_training_data(seed, iters):
    # Generate random boards
    key = random.PRNGKey(seed)
    X = []
    Y = []

    for _ in range(iters):
        board = init_board()
        boards = []
        winner = BLANK
        while winner == BLANK and not full_board(board):
            key, subkey = random.split(key)
            board = play(board, random_strategy(board, subkey), NOUGHT)
            boards.append(board.copy())
            
            winner = board_win(board)
            if full_board(board) or winner != BLANK:
                break

            key, subkey = random.split(key)
            board = play(board, random_strategy(board, subkey), CROSS)
            winner = board_win(board)

        for board in boards:
            X.append(board)
            Y.append(0.5 if winner == BLANK else 1 if winner == NOUGHT else 0)
    
    return X, Y


@jit
def update_weights(weights, x, y):
    dw = -jax.grad(lambda w: loss(x, y, w))(weights)
    return weights + LEARNING_RATE * dw

@jit
def average_loss(weights, X, Y):
    return sum([loss(x, y, weights) for x, y in zip(X, Y)]) / len(X)

@jit
def bulk_update_weights(weights, X, Y):
    dw = -jax.grad(lambda w: average_loss(w, X, Y))(weights)
    return weights + len(X)*LEARNING_RATE * dw

@jit
def dl_dw(x,y,weights) -> jax.Array:
    return jax.grad(loss, argnums=2)(x,y,weights)

def momentum_update_weights(weights, X, Y, beta=0.9) -> jax.Array:
    last_update = jnp.zeros_like(weights)

    for i in range(len(X)):
        x = X[i]
        y = Y[i]

        next_update = beta*last_update + LEARNING_RATE * -dl_dw(x, y, weights)
        weights = weights + next_update
        last_update = next_update
    return weights

def momentum_rmsprop_update_weights(weights, X, Y, beta=0.0, gamma=0.99) -> jax.Array:
    last_update = jnp.zeros_like(weights)
    avg_squares = 1
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        next_grad = -dl_dw(x, y, weights)
        grad_norm = jnp.linalg.matrix_norm(next_grad)
        next_squares = (1-gamma)*grad_norm**2 + gamma * avg_squares

        next_update = beta*last_update + LEARNING_RATE * next_grad
        weights = weights + next_update/(jnp.sqrt(next_squares) + 1e-8)
        last_update = next_update
        avg_squares = next_squares
    return weights

def best_policy_move(board, policy_weights) -> int:
    avail_moves = available_moves(board)
    best_idx = -1
    best_score = -1
    for idx in avail_moves:
        score = policy_model(play(board, idx, NOUGHT), policy_weights)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx

def make_training_data_from_model(n, policy_weights):
    X = []
    Y = []
    for i in range(n):
        board = init_board()
        boards = []
        winner = BLANK
        while winner == BLANK and not full_board(board):
            board = play(board, best_policy_move(board, policy_weights), NOUGHT)
            boards.append(board.copy())
            
            winner = board_win(board)
            if full_board(board) or winner != BLANK:
                break

            board = play(board, random_strategy(board, random.PRNGKey(i)), CROSS)
            winner = board_win(board)

        for board in boards:
            X.append(board)
            Y.append(0.5 if winner == BLANK else 1 if winner == NOUGHT else 0)

    return X, Y

def fight_random(n, policy_weights):
    wins = 0
    for i in range(n):
        board = init_board()
        winner = BLANK
        while winner == BLANK and not full_board(board):
            board = play(board, best_policy_move(board, policy_weights), NOUGHT)
            winner = board_win(board)
            if full_board(board) or winner != BLANK:
                break

            board = play(board, random_strategy(board, random.PRNGKey(i)), CROSS)
            winner = board_win(board)

        if winner == NOUGHT:
            wins += 1
    return wins / n

def show_fight_random(policy_weights, seed = 0):
    board = init_board()
    winner = BLANK
    while winner == BLANK and not full_board(board):
        board = play(board, best_policy_move(board, policy_weights), NOUGHT)
        print(show_board(board))
        print("=========")

        winner = board_win(board)
        if full_board(board) or winner != BLANK:
            break

        board = play(board, random_strategy(board, random.PRNGKey(seed)), CROSS)
        print(show_board(board))
        print("=========")

        winner = board_win(board)

    if winner == NOUGHT:
        print("Policy wins")
    elif winner == CROSS:
        print("Random wins")
    else:
        print("Draw")
