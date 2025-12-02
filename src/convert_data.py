import os
import random
import argparse
import chess.pgn
import bagz as bagz
from apache_beam import coders
from typing import Tuple, Optional


def next_user(current_user: chess.Color) -> chess.Color:
    if current_user == chess.WHITE:
        return chess.BLACK
    return chess.WHITE


def get_user_color(pgn_headers: chess.pgn.Headers, username: str) -> Optional[chess.Color]:
    if pgn_headers.get("White", "").lower() == username.lower():
        return chess.WHITE
    elif pgn_headers.get("Black", "").lower() == username.lower():
        return chess.BLACK
    return None


def parse_games(pgn_path: str, username: str, data_accumulator: list[Tuple[str, str]]):
    pgn = open(pgn_path)

    game = chess.pgn.read_game(pgn)
    with open(pgn_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            variant = game.headers.get("Variant", "Standard")
            if variant != "Standard":
                continue

            user_color = get_user_color(game.headers, username)
            # Skip games where the specified user is not found
            if user_color is None:
                continue

            board = game.board()

            for move in game.mainline_moves():
                if board.turn == user_color:
                    fen = board.fen(en_passant="fen")  # Train input
                    move_uci = move.uci()  # Train target
                    data_accumulator.append((fen, move_uci))
                board.push(move)


def save_data(data: list[Tuple[str, str]], username: str, split: Tuple[float, float] = (0.8, 0.2)):
    assert sum(split) == 1, "sum of data split is not 1."
    os.makedirs("../data/train/", exist_ok=True)
    os.makedirs("../data/test/", exist_ok=True)

    bc_coder = coders.TupleCoder((
        coders.StrUtf8Coder(),  # For FEN
        coders.StrUtf8Coder()   # For Move
    ))

    train_frac, _ = split

    random.shuffle(data)

    train_items = round(len(data) * train_frac)
    train_data = data[:train_items]
    test_data = data[train_items:]

    train_output_path = f"../data/train/{username}_personal_cloning_data.bag"
    with bagz.BagWriter(train_output_path) as train_writer:
        for fen, move_uci in train_data:
            encoded_bytes = bc_coder.encode((fen.strip(), move_uci.strip()))
            train_writer.write(encoded_bytes)

    test_output_path = f"../data/test/{username}_personal_cloning_data.bag"
    with bagz.BagWriter(test_output_path) as test_writer:
        for fen, move_uci in test_data:
            encoded_bytes = bc_coder.encode((fen.strip(), move_uci.strip()))
            test_writer.write(encoded_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, required=True, help="The username you played under")
    parser.add_argument("--split", nargs=2, type=float, default=[0.8, 0.2], help="Train/Test split ratio")
    parser.add_argument("--remove_duplicates", action="store_true")
    parser.add_argument("--data", nargs='+', type=str, required=True, help="The path to the .pgn file containing your played games")
    args = parser.parse_args()

    all_data: list[Tuple[str, str]] = []
    for pgn_file in args.data:
        if os.path.exists(pgn_file):
            parse_games(pgn_file, args.username, all_data)
        else:
            print(f"'{pgn_file}' not found. Skipping.")

    if args.remove_duplicates:
        all_data = list(set(all_data))

    save_data(all_data, args.username, tuple(args.split))
    print(f"Saved {len(all_data)} positions")
