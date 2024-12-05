import re
import time

import zstandard as zstd
import io
import gzip


total_games = 90847982
zst_file = r"C:\Users\alexs\Downloads\lichess_db_standard_rated_2024-11.pgn.zst"
output_file = "bestgames5.gz"


def main():

    clean_up_time = 0
    write_time = 0

    dctx = zstd.ZstdDecompressor()
    with (open(zst_file, 'rb') as f,
          gzip.open(output_file, "wt", encoding="utf-8", newline="\n") as w):

        with dctx.stream_reader(f) as reader:
            i = 0
            games = 0
            games_written = 0

            # Wrap the reader in a TextIOWrapper for line-by-line reading
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            white_elo = 0
            black_elo = 0
            # Read and process the file line by line
            for line in text_stream:
                # print(line.strip())  # Process each line

                if line.startswith("[WhiteElo "):
                    start_idx = line.index('"') + 1
                    end_idx = line.index('"', start_idx)
                    white_elo = int(line[start_idx:end_idx])
                elif line.startswith("[BlackElo "):
                    start_idx = line.index('"') + 1
                    end_idx = line.index('"', start_idx)
                    black_elo = int(line[start_idx:end_idx])
                elif line.startswith("1."):
                    games += 1
                    if white_elo > 1500 and black_elo > 1500:

                        start_time = time.time()
                        cleaned_game = cleanup_game(line)
                        clean_up_time += (time.time() - start_time)

                        start_time = time.time()
                        w.write(cleaned_game)
                        write_time += (time.time() - start_time)
                        games_written += 1

                i += 1
                if i % 1000000 == 0:
                    percent = (100 * games / total_games)
                    print(f"{games_written} out of {games} games written ({percent:.2f}%)")
                if games > 1:
                    break

    print("Clean up time: ", clean_up_time)
    print("Write time: ", write_time)


cleanup_pattern = re.compile(r"({[^}]*})|( [0-9]+\.\.\. )")
padding_pattern = re.compile(r"([^ ])( [0-9]+[.-/*])")
pad_unfinished_games = re.compile(r"([^ ]) ([*])$")
pad_drawn_games = re.compile(r"([^ ]) (1/2-1/2)$")
pad_normal_games = re.compile(r"([^ ]) (1/2-1/2)$")

# Clean up time:  7.96174168586731
# Write time:  24.34288239479065
# Time taken: 37.37824988365173


def cleanup_game(game):
    comments_and_half_moves_removed = re.sub(cleanup_pattern, "", game)
    second_move_idx = comments_and_half_moves_removed.find("2.")
    if second_move_idx != -1 and comments_and_half_moves_removed[second_move_idx - 2] == " ":
        return comments_and_half_moves_removed
    # Some games don't have double spaces between moves. Let's add that back
    padded = re.sub(padding_pattern, "\\1 \\2", comments_and_half_moves_removed)

    # Add padding to the end of some games
    padded = re.sub(pad_unfinished_games, "\\1  \\2", padded)
    padded = re.sub(pad_drawn_games, "\\1  \\2", padded)
    padded = re.sub(pad_normal_games, "\\1  \\2", padded)
    return padded


if __name__ == "__main__":
    start_time = time.time()
    main()
    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken}")
