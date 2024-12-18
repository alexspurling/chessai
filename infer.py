import chess
from chess import Board, Move

from tokenisation.decoder import PAWN, KING, FIRST_POSITION, LAST_POSITION, decode, decode_token, FIRST_NUM, LAST_NUM, \
    encode, SHORT_CASTLE, LONG_CASTLE
from model import ChessModel


def get_legal_move(board, piece, position):
    # We have a move, let's see if it's valid
    legal_moves = [move for move in board.legal_moves if move.to_square == position]
    for m in legal_moves:
        start_piece = board.piece_at(m.from_square)
        if start_piece.symbol().upper() == piece:
            return m
    return None


# Returns the list of tokens that make up a valid move on the given board
def find_next_move(board, tokens, token_start_idx) -> (int, Move):
    piece = None
    position = None
    for i in range(token_start_idx, len(tokens)):
        token = tokens[i]
        # If this is a move token then ensure it represents the current move num
        if FIRST_NUM <= token <= LAST_NUM:
            if token - FIRST_NUM != board.fullmove_number:
                return [], None
        elif token == SHORT_CASTLE:
            # True if white
            piece = "K"
            if board.turn:
                position = chess.G1
            else:
                position = chess.G8
        elif token == LONG_CASTLE:
            # True if white
            piece = "K"
            if board.turn:
                position = chess.C1
            else:
                position = chess.C8
        # Keep track of the last piece
        # Keep track of the last position
        elif PAWN <= token <= KING:
            piece = decode_token(token)
            # position = None
        elif FIRST_POSITION <= token <= LAST_POSITION:
            # Gives a number for the square between 0 and 63 which just happens to also be how the chess library
            # encodes positions
            position = token - FIRST_POSITION

        if piece is not None and position is not None:
            valid_move = get_legal_move(board, piece, position)
            if valid_move is not None:
                # We consumed some valid tokens up until i
                return i + 1, valid_move
    return None


def get_next_human_move(board: Board):
    while True:
        try:
            print(board)
            print("Your move:")
            move = input()
            return board.parse_san(move)
        except ValueError as e:
            print(e)


def play(play_as):
    b = Board()

    game_tokens = [33]

    if play_as == "white":
        # We need to ask the human for the first move
        move = get_next_human_move(b)
        move_tokens = encode(b, move)
        game_tokens.extend(move_tokens)
        b.push(move)

    m = ChessModel("savedmodel.pt")

    while not b.is_checkmate():
        token_idx = len(game_tokens)  # start evaluating the generated tokens from this index

        print("Continuing game:")
        print(decode(game_tokens))
        tokens = m.generate(game_tokens)

        print("Tokens: ", tokens)
        print("Decoded: ", decode(tokens))

        print("Board is:", b.fen())
        print("Board full move num: ", b.fullmove_number)

        (valid_token_idx, new_move) = find_next_move(b, tokens, token_idx)

        if new_move is not None:
            b.push(new_move)
            # Reset the game tokens to only include those we deemed to be valid
            game_tokens = tokens[:valid_token_idx]

            print("Cur state:", decode(game_tokens))

            # Ask the human for the next move
            human_move = get_next_human_move(b)
            move_tokens = encode(b, human_move)

            b.push(human_move)

            if play_as == "white":
                # Add a move num token if playing as white
                game_tokens.append(FIRST_NUM + b.fullmove_number)
            game_tokens.extend(move_tokens)

        else:
            print(f"Invalid move at {token_idx}. Regenerating...")


# play("black")


class OnlineGame:

    def __init__(self):
        self.b = Board()
        self.m = ChessModel("savedmodel.pt")
        self.game_tokens = [33]

    def make_move(self, uci_move):
        move = self.b.parse_uci(uci_move)
        move_tokens = encode(self.b, move)
        self.game_tokens.extend(move_tokens)
        self.b.push(move)

        token_idx = len(self.game_tokens)

        while True:

            print("Generating moves from position: ", decode(self.game_tokens))
            tokens = self.m.generate(self.game_tokens)
            (valid_token_idx, new_move) = find_next_move(self.b, tokens, token_idx)

            if new_move is not None:
                print("Engine played:", new_move)
                self.b.push(new_move)
                # Reset the game tokens to only include those we deemed to be valid
                self.game_tokens = tokens[:valid_token_idx]

                # if play_as == "white":
                # Add a move num token if playing as white
                self.game_tokens.append(FIRST_NUM + self.b.fullmove_number)

                print("Current game tokens")
                print(decode(self.game_tokens))

                return new_move.uci()
            else:
                print("Generated invalid move:", tokens[token_idx:])

    def get_fen(self):
        return self.b.fen()

    def reset(self):
        self.b.reset()
        self.game_tokens = [33]

