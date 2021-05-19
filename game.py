from typing import Any, Dict, Optional, Tuple, Union
from aenum import MultiValueEnum
import numpy as np


class Move(MultiValueEnum):
    LEFT = "left", "l"
    RIGHT = "right", "r"
    UP = "up", "u"
    DOWN = "down", "d"


class GameStatus(MultiValueEnum):
    WIN = "win", "w"
    LOSE = "lose", "l"
    ONGOING = "ongoing", "o"


class Game:
    DEFAULT_SIZE = 4
    DEFAULT_SIZE_M = 4
    DEFAULT_SIZE_N = 4

    DEFAULT_RANDOM_CONFIG = {
        "count": 1,
        "probs": {
            0: 0.5,
            2: 0.5,
        },
    }

    def __init__(self, board_shape: Optional[Union[int, Tuple[int]]] = None, random_config: Optional[Dict[str, Any]] = None) -> None:
        self.board_shape: Tuple[int, int] = self._parse_board_shape(board_shape)
        self.board = self._get_starting_board(self.board_shape)
        self.random_config = self._parse_random_config(random_config)
        self.sum_score = 0

    @property
    def max_score(self) -> int:
        return np.max(self.board)

    @property
    def game_status(self) -> GameStatus:
        if self.max_score >= 1028:
            return GameStatus.WIN
        elif True: #  TODO: make condition
            return GameStatus.LOSE
        else:
            return GameStatus.ONGOING

    def _parse_board_shape_element(self, board_shape_element: Optional[int], default_value: int) -> int:
        if board_shape_element is None or board_shape_element < 2:
            return default_value
        return board_shape_element

    def _parse_board_shape(self, board_shape: Optional[Union[int, Tuple[int]]]) -> int:
        if board_shape is None:
            m, n = self.DEFAULT_SIZE_M, self.DEFAULT_SIZE_N
        elif isinstance(board_shape, int):
            m = n = self._parse_board_shape_element(board_shape, self.DEFAULT_SIZE)
        elif isinstance(board_shape, tuple) and len(board_shape) == 2:
            m = self._parse_board_shape_element(board_shape[0], self.DEFAULT_SIZE_M)
            n = self._parse_board_shape_element(board_shape[1], self.DEFAULT_SIZE_N)
        else:
            raise ValueError("Cannot parse board_shape")
        
        return m, n

    def _parse_random_config_count(self, count: Optional[int]) -> int:
        if isinstance(count, int):
            return count if count > 0 else self.DEFAULT_RANDOM_CONFIG["count"]
        else:
            return self.DEFAULT_RANDOM_CONFIG["count"]

    def _parse_random_config_probs(self, probs: Dict[Optional[int], float]) -> int:
        def _is_power_of_two_or_zero(n: int) -> bool:
            return (n & (n-1) == 0)
        def _normalize_to_sum_one(values: np.ndarray) -> np.ndarray:
            return values / np.sum(values)

        if isinstance(probs, dict):
            if (
                all([isinstance(k, int) and _is_power_of_two_or_zero(k) for k in probs.keys()]) and
                all([isinstance(v, (int, float)) and v >= 0 for v in probs.values()]) and
                np.sum(list(probs.keys())) > 0
            ):
                keys = probs.keys()
                values = _normalize_to_sum_one(list(probs.values()))
                return dict(zip(keys, values))
            else:
                return self.DEFAULT_RANDOM_CONFIG["probs"]
        else:
            return self.DEFAULT_RANDOM_CONFIG["probs"]

    def _parse_random_config(self, random_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not isinstance(random_config, dict):
            return self.DEFAULT_RANDOM_CONFIG
        else:
            count = self._parse_random_config_count(random_config.get("count", None))
            probs = self._parse_random_config_probs(random_config.get("probs", None))
            return {
                "count": count,
                "probs": probs,
            }

    def _get_empty_board(self, board_shape: Tuple[int, int]) -> np.ndarray:
        return np.zeros(board_shape, dtype=np.uint32)

    def _set_empty_board(self) -> None:
        self.board = self._get_empty_board(self.board_shape)

    def _get_starting_board(self, board_shape: Tuple[int, int]) -> np.ndarray:
        board = np.zeros(board_shape, dtype=np.uint32)
        board = self._set_random_empty_cell_to_value(board, value=2)
        board = self._set_random_empty_cell_to_value(board, value=2)
        return board
    
    def _set_starting_board(self) -> None:
        self.board = self._get_starting_board(self.board_shape)

    def _get_random_empty_cell(self, board: np.ndarray) -> Tuple[int, int]:
        board_shape = board.shape
        while True:
            r = np.random.randint(low=0, high=board_shape[0])
            c = np.random.randint(low=0, high=board_shape[1])
            if board[r, c] == 0:
                return (r, c)
        
    def _set_random_empty_cell_to_value(self, board: np.ndarray, value: int) -> None:
        r, c = self._get_random_empty_cell(board)
        board[r, c] = value
        return board

    def __str__(self) -> str:
        board_str = "\n"
        for row in range(self.board_shape[0]):
            row_numbers = self.board[row, :]
            row_str = " ".join([str(e) for e in row_numbers])
            board_str += row_str
            board_str += "\n"
        return board_str
    
    def compress(self) -> bool:
        changed = False
        board_new = self._get_empty_board(self.board_shape)
        for r in range(self.board_shape[0]):
            c_new = 0
            for c in range(self.board_shape[1]):
                if self.board[r, c] != 0:
                    board_new[r, c_new] = self.board[r, c]
                    if c != c_new:
                        changed = True
                    c_new += 1
        self.board = board_new
        return changed

    def merge(self) -> bool:
        changed = False
        for r in range(self.board_shape[0]):
            for c in range(self.board_shape[1]-1):
                if self.board[r, c] != 0 and self.board[r, c] == self.board[r, c+1]:
                    self.sum_score += 2 * self.board[r, c]
                    self.board[r, c] *= 2
                    self.board[r, c+1] = 0
                    changed = True
        return changed

    def add_board_randomization(self) -> None:
        count = self.random_config["count"]
        probs_values, probs_weights = list(self.random_config["probs"].keys()), list(self.random_config["probs"].values())
        random_values = np.random.choice(probs_values, size=count, p=probs_weights)
        for v in random_values:
            self.board = self._set_random_empty_cell_to_value(self.board, value=v)

    def play_move_left(self) -> bool:
        changed1 = self.compress()
        changed2 = self.merge()
        changed3 = self.compress()
        return changed1 or changed2 or changed3
    
    def play_move_right(self) -> bool:
        self.board = np.flip(self.board, axis=1)
        changed = self.play_move_left()
        self.board = np.flip(self.board, axis=1)
        return changed

    def play_move_up(self) -> bool:
        self.board = np.rot90(self.board, k=1, axes=(0,1))
        self.board_shape = (self.board_shape[1], self.board_shape[0])
        changed = self.play_move_left()
        self.board = np.rot90(self.board, k=1, axes=(1,0))
        self.board_shape = (self.board_shape[1], self.board_shape[0])
        return changed

    def play_move_down(self) -> bool:
        self.board = np.rot90(self.board, k=1, axes=(1,0))
        self.board_shape = (self.board_shape[1], self.board_shape[0])
        changed = self.play_move_left()
        self.board = np.rot90(self.board, k=1, axes=(0,1))
        self.board_shape = (self.board_shape[1], self.board_shape[0])
        return changed

    def play_move(self, move: Move) -> bool:
        if move == Move.LEFT:
            changed = self.play_move_left()
        elif move == Move.RIGHT:
            changed = self.play_move_right()
        elif move == Move.UP:
            changed = self.play_move_up()
        elif move == Move.DOWN:
            changed = self.play_move_down()
        return changed

    def parse_move(self, move_str: str) -> Optional[Move]:
        move_str = move_str.lower()
        try:
            move = Move(move_str)
        except ValueError:
            return None
        return move

    def ask_move(self) -> Move:
        possible_moves_str = " ".join(["/".join(key.values) for key in Move])
        question_str = f"Your move? [{possible_moves_str}]\n"
        while True:
            answer = input(question_str)
            parsed_move = self.parse_move(answer)
            if parsed_move is not None:
                return parsed_move

    def get_score_str(self) -> str:
        return f"max-score = {self.max_score}\nsum-score = {self.sum_score}"

    #  TODO: add game status check
    def start(self) -> None:
        while True:
            print(self.get_score_str())
            print(self)
            move = self.ask_move()
            changed = self.play_move(move)
            if changed:
                self.add_board_randomization()


if __name__ == "__main__":
    g = Game(board_shape=4, random_config={"count": 1, "probs": {2: 1}})
    g.start()
