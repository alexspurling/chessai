import chess.engine
from chess.engine import PlayResult
import chess.pgn
from chess.pgn import StringExporter

from model import Small, Medium
from play import OnlineGame

stockfish_path = "C:\\Users\\alexs\\Downloads\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"

# Load engines
stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
# Configure Stockfish to Level 0 (weakest)
stockfish_engine.configure({"Skill Level": 0})


# Set up a chess board
board = chess.Board()
game = chess.pgn.Game()
node = game

white_bot = OnlineGame(board, "savedmodel.pt", params=Small(), device="cpu", play_as="black")
black_bot = OnlineGame(board, "stockfishmodel.pt", params=Medium(), device="cpu")

# Play the game
while not board.is_game_over():
    if board.turn:  # White to move
        move = white_bot.get_next_move()
        if move is not None:
            black_bot.update_tokens(move)
    else:  # Black to move
        move = black_bot.get_next_move()
        if move is not None:
            white_bot.update_tokens(move)

    if move is not None:
        board.push(move)
        node = node.add_variation(move)
        print(game.accept(StringExporter(columns=None, headers=False, comments=False, variations=False)))


# Display the result
print("Game Over!")
print(board.result())
# print(game)

# Quit engines
stockfish_engine.quit()
