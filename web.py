from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import chess
import chess.pgn

# Import your `find_next_move` function
from infer import find_next_move, OnlineGame

app = Flask(__name__)
CORS(app)

# Initialize a chess board
online_game = OnlineGame()


@app.route('/get_board', methods=['GET'])
def get_board():
    """
    Returns the current board state in FEN notation.
    """
    global online_game
    return jsonify({'fen': online_game.get_fen()})


@app.route('/make_move', methods=['POST'])
def make_move():
    """
    Player makes a move. This function updates the board and gets the engine's response.
    """
    global online_game
    player_move = request.json.get('move')  # Move in UCI format, e.g., "e2e4"
    print("Got move", player_move)

    try:
        # Apply the player's move
        engine_move = online_game.make_move(player_move)

        return jsonify({'fen': online_game.get_fen(), 'engine_move': engine_move})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/reset', methods=['POST'])
def reset_board():
    """
    Resets the board to the initial state.
    """
    global online_game
    online_game.reset()
    return Response(status=200)


if __name__ == '__main__':
    app.run(debug=True)
