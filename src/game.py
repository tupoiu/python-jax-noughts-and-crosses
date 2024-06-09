from jax import jit, random, grad
import jax
import jax.numpy as jnp


NOUGHT = 1
CROSS = -1
BLANK = 0

def init_board():
    return jnp.zeros(9, dtype=jnp.int32)

def to_space(space):
    match space:
        case 1: return "O"
        case -1: return "X"
        case _: return " "

def show_board(board):
    for i in range(3):
        row = "|".join(map(to_space, board[3*i:3*i+3]))
        print(f"[{row}]")

def board_win(board):
    for i in range(3):
        # Columns
        if jnp.all(board[i] == board[i+3] == board[i+6]):
            return board[i]
        # Rows
        if jnp.all(board[3*i] == board[3*i+1] == board[3*i+2]):
            return board[3*i]
    # TL to BR diagonal
    if jnp.all(board[0] == board[4] == board[8]):
        return board[0]
    # TR to BL diagonal
    if jnp.all(board[2] == board[4] == board[6]):
        return board[2]
    return BLANK

def available_moves(board: jax.Array) -> list[int]:
    out = jnp.empty(0, dtype=jnp.int32)
    for i in range(9):
        if board[i] == BLANK:
            out = jnp.append(out, i)

    return out

# Playing against random opponents
# input : board, output: win_percentage
def play(board : jax.Array, spot_to_play, space):
    available = available_moves(board)
    if spot_to_play not in available:
        return None # raise ValueError("Invalid move")
    next_board = board.at[spot_to_play].set(space)
    return next_board

def ml_strategy(board, f):
    # Let f be the player model which takes in a board and outputs a vector of 9
    # floats in [0,1] with ratings of each move.
    available_moves = jnp.where(board == BLANK)
    # Get the index of the move with the highest rating
    scores = f(board)
    return jnp.argmax(scores[available_moves])

def random_strategy(board, rand):
    return random.choice(rand, available_moves(board))

@jit
def random_policy_weights(seed) -> jax.Array:
    return random.normal(key=random.key(seed=SEED), shape=[9], dtype=jnp.float32)

@jit
def policy_model(board : jax.Array, policy_weights : jax.Array):
    return logistic(jnp.dot(board, policy_weights))

@jit
def logistic(x): # A logistic function mapping to [0, 1]
    return 1 / (1 + jnp.exp(-x))

@jit
def loss(board, is_win, policy_weights):
    # Win rate is the output of the policy model
    
    diff = is_win - policy_model(board, policy_weights)
    return diff * diff

simple_board = play(init_board(), 1, NOUGHT)
show_board(simple_board)

# print(policy_model(simple_board, policy_weights))

@jit
def full_board(board : jax.Array) -> bool:
    return board.dot(board) == 9

def make_training_data(seed, iters):
    # Generate random boards
    key = random.PRNGKey(seed)
    X = []
    Y = []

    for _ in range(iters):
        board = init_board()
        boards = []
        winner = BLANK
        while winner == BLANK:
            key, subkey = random.split(key)
            board = play(board, random_strategy(board, subkey), NOUGHT)
            boards.append(board.copy())
            
            winner = board_win(board)
            if full_board(board):
                break

            key, subkey = random.split(key)
            board = play(board, random_strategy(board, subkey), CROSS)
            winner = board_win(board)
            if full_board(board):
                break
        for board in boards:
            X.append(board)
            Y.append(0.5 if winner == BLANK else 1 if winner == NOUGHT else 0)
    
    return X, Y

LEARNING_RATE = 0.01

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

def adam_update_weights(weights, X, Y, beta=0.0, gamma=0.99) -> jax.Array:
    last_update = jnp.zeros_like(weights)
    avg_squares = 1
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        next_grad = -dl_dw(x, y, weights)
        next_squares = (1-gamma)*jnp.dot(next_grad, next_grad) + gamma * avg_squares

        next_update = beta*last_update + LEARNING_RATE * next_grad
        weights = weights + next_update/(jnp.sqrt(next_squares) + 1e-8)
        last_update = next_update
        avg_squares = next_squares
    return weights

SEED = 0

testX, testY = make_training_data(SEED - 1, 100)
test_loss = lambda weights: sum([loss(x, y, weights) for x, y in zip(testX, testY)])

# Simple policy - Each space contributes to win percentage randomly
original_policy_weights = random_policy_weights(SEED)

policy_weights = original_policy_weights.copy()

losses : list = []
for i in range(10):
    X, Y = make_training_data(SEED + i, 100)
    policy_weights = adam_update_weights(policy_weights, X, Y)
    cur_loss = test_loss(policy_weights)
    losses.append(cur_loss)
    print(cur_loss)
print(policy_weights.tolist())