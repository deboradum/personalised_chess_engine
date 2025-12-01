import os
import re
import json
import argparse
import urllib.error
import urllib.request

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_chesscom_games(username):
    print(f"--- Processing Chess.com for: {username} ---")

    base_dir = os.path.join(username, "chesscom")
    ensure_dir(base_dir)

    base_api_url = f"https://api.chess.com/pub/player/{username}/games/"
    archives_url = base_api_url + "archives"

    try:
        with urllib.request.urlopen(archives_url) as f:
            data = json.loads(f.read().decode("utf-8"))

        archives_list = data.get("archives", [])

        if not archives_list:
            print("No archives found for this user.")
            return

        for archive_url in archives_list:
            match = re.search(r'/(\d{4})/(\d{2})$', archive_url)
            if match:
                year, month = match.groups()
                filename = f"{year}-{month}.pgn"
                filepath = os.path.join(base_dir, filename)

                pgn_url = archive_url + "/pgn"

                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(pgn_url, filepath)

        print("Chess.com downloads complete.")

    except urllib.error.HTTPError as e:
        print(f"Error connecting to Chess.com: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def download_lichess_games(username):
    print(f"--- Processing Lichess for: {username} ---")

    base_dir = os.path.join(username, "lichess")
    ensure_dir(base_dir)

    url = f"https://lichess.org/api/games/user/{username}"

    try:
        print("Connecting to Lichess stream (this may take a moment)...")
        with urllib.request.urlopen(url) as f:

            game_buffer = []
            current_month_file = None

            for line in f:
                decoded_line = line.decode("utf-8")
                game_buffer.append(decoded_line)

                # Look for the Date tag: [Date "YYYY.MM.DD"]
                if decoded_line.startswith("[Date"):
                    # Regex to find YYYY.MM
                    match = re.search(r'"(\d{4})\.(\d{2})\.\d{2}"', decoded_line)
                    if match:
                        year, month = match.groups()
                        current_month_file = f"{year}-{month}.pgn"

                if decoded_line.strip() == "" and len(game_buffer) > 1:
                    if current_month_file:
                        filepath = os.path.join(base_dir, current_month_file)
                        # Append mode 'a'
                        with open(filepath, "a", encoding="utf-8") as out_f:
                            out_f.write("".join(game_buffer))
                            # Ensure separation between games in the file
                            out_f.write("\n\n")

                    game_buffer = []
                    current_month_file = None

            if game_buffer and current_month_file:
                filepath = os.path.join(base_dir, current_month_file)
                with open(filepath, "a", encoding="utf-8") as out_f:
                    out_f.write("".join(game_buffer))

        print("Lichess downloads complete.")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"User '{username}' not found on Lichess.")
        else:
            print(f"Error connecting to Lichess: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download chess games from Chess.com and Lichess.")

    parser.add_argument("--username", help="Username to use for both sites (unless overridden)")
    parser.add_argument("--chesscom", help="Specific Chess.com username")
    parser.add_argument("--lichess", help="Specific Lichess username")

    args = parser.parse_args()

    chesscom_user = args.chesscom if args.chesscom else args.username
    lichess_user = args.lichess if args.lichess else args.username

    if not chesscom_user and not lichess_user:
        print("Error: You must provide a username. Use --username, --chesscom, or --lichess.")
        exit(1)

    if chesscom_user:
        download_chesscom_games(chesscom_user)
        print("-" * 20)

    if lichess_user:
        download_lichess_games(lichess_user)
