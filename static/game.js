var config = {
  pieceTheme: 'static/images/chesspieces/wikipedia/{piece}.png',
  position: 'start',
  draggable: true,
  onDrop: handleMove,
  moveTime: 1000
}
const board = Chessboard('board', config);
const game = new Chess();
let playAs = "white";

function getBoard() {
    fetch('/get_board')
    .then((response) => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    }).then((body) => {
        playAs = body['playAs'];
        board.orientation(playAs);
        game.load(body['fen']);
        board.position(body['fen']);
        checkGameOver();
    });
}

function handleMove(source, target) {
    let move = source + target;
    // Automatically promote to Queen if needed
    if (game.get(source).type == "p") {
        if (game.turn() == "w" && target[1] == "8") {
            move += "q";
        } else if (target[1] == "1") {
            move += "q";
        }
    }

    const gamemove = game.move({ from: source, to: target, promotion: 'q' });
    if (!gamemove) {
        return 'snapback'; // Invalid move
    }
    console.log("Game move", gamemove);

    document.getElementById('thinking').style = "display: block";

    fetch('/make_move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ move })
    }).then((response) => {
        document.getElementById('thinking').style = "display: none";
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    }).then((body) => {
        console.log("Engine played", body['engineMove']);
        game.load(body['fen']);
        board.position(body['fen']);
        checkGameOver();
    }).catch((e) => {
        console.error("Error", e);
    });
}

function reset() {
    // Reset and switch sides as well
    playAs = playAs == "white" ? "black" : "white";
    game.reset();
    board.orientation(playAs);
    board.start();
    document.getElementById('outcome').innerHTML = "";

    fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ playAs })
    })
    .then((response) => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    }).then((body) => {
        game.load(body['fen']);
        board.position(game.fen());
    });
}

function checkGameOver() {
    if (game.game_over()) {
        if (game.in_checkmate()) {
            outcome = "Checkmate, " + ((game.turn() == "w" ? "black" : "white") + " wins.");
        } else if (game.isStaleMate()) {
            outcome = "Stalemate.";
        } else if (game.isThreefoldRepetition()) {
            outcome = "Threefold repetition.";
        } else if (game.isInsufficientMaterial()) {
            outcome = "Insufficient material.";
        }
        document.getElementById('outcome').innerHTML = outcome;
    }
}

document.getElementById('reset').addEventListener('click', reset);

getBoard();