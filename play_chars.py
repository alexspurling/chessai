import chess
from chess import Board, Move
import time

from model import ChessModel
from tokenisation.charactertokeniser import decode, encode


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
    decoded_tokens = decode(tokens)
    delim_pos = decoded_tokens.find(" ", token_start_idx)
    if delim_pos == -1:
        delim_pos = decoded_tokens.find("#", token_start_idx)
    if delim_pos == -1:
        delim_pos = decoded_tokens.find("\n", token_start_idx)
    if delim_pos == -1:
        delim_pos = len(tokens)
    next_move = decoded_tokens[token_start_idx:delim_pos]

    try:
        valid_move = board.parse_san(next_move)
        return delim_pos, valid_move
    except ValueError:
        return [], None


def get_next_human_move(board: Board):
    while True:
        try:
            print(board)
            print("Your move:")
            return input()
        except ValueError as e:
            print(e)


# Play on the command line
def play(play_as):
    b = Board()

    game_tokens = encode("\n1.")

    if play_as == "white":
        # We need to ask the human for the first move
        move = get_next_human_move(b)
        move_san = b.san(move)
        move_tokens = encode(move_san)
        game_tokens.extend(move_tokens)
        b.push(move)

    m = ChessModel("stockfishmodel.pt")

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
            move_tokens = encode(human_move)

            b.push_san(human_move)

            if play_as == "white":
                # Add a move num token if playing as white
                game_tokens.extend(encode(f"{b.fullmove_number}."))
            game_tokens.extend(move_tokens)

        else:
            print(f"Invalid move at {token_idx}. Regenerating...")


# play("black")


class OnlineGameChars:

    def __init__(self, board, model_state_file, params, device, play_as="white"):
        start_time = time.time()
        self.b = board
        model_load_start_time = time.time()
        self.m = ChessModel(model_state_file, vocab_size=44, params=params, device=device)
        print("Model load time was", (time.time() - model_load_start_time))
        self.game_tokens = encode("\n1.")
        self.play_as = play_as  # play_as is the colour of the human player
        self.reset(self.play_as)
        self.time_per_move = 1
        print("Start up time was", (time.time() - start_time))

    def get_next_move(self):
        token_idx = len(self.game_tokens)
        print("Generating moves from position:", decode(self.game_tokens))

        move_start_time = time.time()
        new_move = None
        while new_move is None:
            print(time.time(), "Generating tokens")
            generate_start_time = time.time()
            tokens = self.m.generate(self.game_tokens, num_moves_to_generate=6)
            new_tokens = tokens[len(self.game_tokens):]
            generate_time = time.time() - generate_start_time
            print(time.time(), "Generated new tokens", new_tokens, "in", generate_time)
            print(time.time(), "New tokens are:", decode(new_tokens))
            print(time.time(), "Finding next move")
            (valid_token_idx, new_move) = find_next_move(self.b, tokens, token_idx)
            print(time.time(), "Found move", new_move)

            if new_move is not None:
                # Reset the game tokens to only include those we deemed to be valid
                self.game_tokens = tokens[:valid_token_idx]
                self.game_tokens.extend(encode(" "))
            else:
                print("Generated invalid move")
                # print("Generated invalid move:", tokens[token_idx:])
                if time.time() - move_start_time > self.time_per_move:
                    for new_move in self.b.generate_legal_moves():
                        # Add the random move to our encoded tokens list
                        move_tokens = encode(self.b.san(new_move))
                        self.game_tokens.extend(move_tokens)
                        self.game_tokens.extend(encode(" "))
                        break
                    print("No move found in time limit. Generated random move:", new_move)

        if self.play_as == "white":
            # Add a move num token if playing as white
            self.game_tokens.extend(encode(f"{self.b.fullmove_number + 1}."))

        print(time.time(), "Current game tokens")
        print(decode(self.game_tokens))
        return new_move

    def push_uci_move(self, uci_move):
        print(time.time(), "Parsing move", uci_move)
        move = self.b.parse_uci(uci_move)
        print(time.time(), "Checking game over state")
        self.update_tokens(move)
        print(time.time(), "Pushing move to board")
        self.b.push(move)

    def update_tokens(self, move):
        print(time.time(), "Encoding move", move)
        move_tokens = encode(f"{self.b.san(move)} ")
        print(time.time(), "Extending tokens")
        self.game_tokens.extend(move_tokens)
        if self.play_as == "black":
            self.game_tokens.extend(encode(f"{self.b.fullmove_number + 1}."))

    def play(self):
        if self.b.is_game_over():
            print(f"Game over", self.b.outcome())
            return None
        print(time.time(), "Requesting engine move")
        engine_move = self.get_next_move()
        print(time.time(), "Engine played:", engine_move)
        self.b.push(engine_move)
        if self.b.is_game_over():
            print(f"Game over", self.b.outcome())
            winner = "black" if self.play_as == "white" else "black"
            print(f"Checkmate. {winner} wins.")
        return engine_move

    def get_fen(self):
        return self.b.fen()

    def reset(self, play_as):
        self.play_as = play_as
        self.b.reset()
        self.game_tokens = encode("\n1.")

        # if the human player is black, then the bot gets to make the first move
        if self.play_as == "black":
            return self.play()

        return None

