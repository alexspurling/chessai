import chess.engine
from chess.engine import PlayResult
import chess.pgn
from chess.pgn import StringExporter

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

blunder_bot = OnlineGame(board, "stockfishmodel.pt", "cpu")

# Play the game
while not board.is_game_over():
    if board.turn:  # White to move
        result = stockfish_engine.play(board, chess.engine.Limit(time=0.1))
        # Update the internal state of the bot with stockfish's move
        if result.move is not None:
            blunder_bot.update_tokens(result.move)
    else:  # Black to move
        result = PlayResult(move=blunder_bot.get_next_move(), ponder=None, draw_offered=False, resigned=False)

    if result.move is not None:
        board.push(result.move)
        node = node.add_variation(result.move)
        print(game.accept(StringExporter(columns=None, headers=False, comments=False, variations=False)))


# Display the result
print("Game Over!")
print(board.result())
# print(game)

# Quit engines
stockfish_engine.quit()
