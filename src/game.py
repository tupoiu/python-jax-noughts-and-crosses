from jax import jit, random
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

def show_board(board) -> str:
    outrows = []
    for i in range(3):
        row = "|".join(map(to_space, board[3*i:3*i+3]))
        outrows.append(f"[{row}]")
    return "\n".join(outrows)

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

def random_strategy(board, rand):
    return random.choice(rand, available_moves(board))

simple_board = play(init_board(), 1, NOUGHT)
assert show_board(simple_board) == """
[ |O| ]
[ | | ]
[ | | ]
""".strip()

@jit
def full_board(board : jax.Array) -> bool:
    return board.dot(board) == 9