from pydantic import BaseModel

# Import your `find_next_move` function
from infer import OnlineGame

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost:4000",
    "https://chessai-gaj8.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# Initialize a chess board
online_game = OnlineGame()


# Serve the index.html for the root path
@app.get("/", include_in_schema=False)
def read_root():
    return FileResponse("static/index.html")


@app.get('/get_board')
def get_board():
    """
    Returns the current board state in FEN notation.
    """
    global online_game
    return {'fen': online_game.get_fen(), 'playAs': online_game.play_as}


class MoveRequest(BaseModel):
    move: str


@app.post('/make_move')
def make_move(move_request: MoveRequest):
    """
    Player makes a move. This function updates the board and gets the engine's response.
    """
    global online_game
    print("Got move", move_request.move)

    # Apply the player's move in UCI format e.g. e2e4
    engine_move = online_game.make_move(move_request.move)

    return {'fen': online_game.get_fen(), 'engine_move': engine_move}


class ResetRequest(BaseModel):
    playAs: str


@app.post('/reset')
def reset_board(reset_request: ResetRequest):
    """
    Resets the board to the initial state.
    """
    global online_game
    engine_move = online_game.reset(reset_request.playAs)
    return {'fen': online_game.get_fen(), 'engine_move': engine_move}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web:app", host="0.0.0.0", port=10000, reload=True)

# Preferred way to start the app is:
# uvicorn web:app --host 0.0.0.0 --port 10000
# Not sure why but when running in IntelliJ, it causes the app to start twice
