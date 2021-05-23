from typing import Any, Dict, Optional, Tuple, Union
from aenum import MultiValueEnum
import numpy as np


class Move(MultiValueEnum):
    """Move class denoting possible moves in 2048 game.
    """
    LEFT = "left", "l"
    RIGHT = "right", "r"
    UP = "up", "u"
    DOWN = "down", "d"


class GameStatus(MultiValueEnum):
    """GameStatus class denoting possible game status of 2048 game.
    """
    WIN = "win", "w"
    LOSE = "lose", "l"
    ONGOING = "ongoing", "o"


class Game:
    """Game class which handles all the logic of 2048 game.
    """

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
        """Constructor for Game class

        Args:
            board_shape (Optional[Union[int, Tuple[int]]], optional): shape of game board. Defaults to None.
            random_config (Optional[Dict[str, Any]], optional): config for randomization of board. Defaults to None.
        """
        self.board_shape: Tuple[int, int] = self._parse_board_shape(board_shape)
        self.random_config: Dict[str, Any] = self._parse_random_config(random_config)
        self.board: np.ndarray = self._get_starting_board(self.board_shape, count=2)
        self.sum_score: int = 0
        self.continue_game_if_win: Optional[bool] = None

    def reset(self) -> None:
        self.board: np.ndarray = self._get_starting_board(self.board_shape, count=2)
        self.sum_score: int = 0
        self.continue_game_if_win: Optional[bool] = None

    @property
    def max_score(self) -> int:
        """Maximum value in the board

        Returns:
            int: max score
        """
        return np.max(self.board)

    @property
    def game_status(self) -> GameStatus:
        """Current game status

        Returns:
            GameStatus: game status
        """
        if self.max_score >= 1028:
            return GameStatus.WIN
        elif not self._is_board_movable():
            return GameStatus.LOSE
        else:
            return GameStatus.ONGOING

    def is_game_over(self, game_status: Optional[GameStatus] = None) -> bool:
        """Check if game is over. For game to be over, either game status is `LOSE`,
        or game status is `WIN` and user does not want to continue to play.

        Args:
            game_status (Optional[GameStatus], optional): current game status. Defaults to None.

        Returns:
            bool: 
        """
        game_status = self.game_status if game_status is None else game_status
        game_over = game_status == GameStatus.LOSE or (game_status == GameStatus.WIN and self.continue_game_if_win == False)
        return game_over

    def _is_board_movable(self) -> bool:
        """Checks if there is a valid move in the board. Returns False if no '0' cell in the board,
        and every cell value is different than its neighbours.

        Returns:
            bool: if move exist in board
        """
        if np.any(self.board == 0):
            return True

        movable = False
        m, n = self.board_shape
        for r in range(m):
            for c in range(n):
                if r < m - 1 and self.board[r, c] == self.board[r+1, c]:
                    movable = True
                    break
                if c < n - 1 and self.board[r, c] == self.board[r, c+1]:
                    movable = True
                    break
            if movable:
                break
        return movable

    def _parse_board_shape_element(self, board_shape_element: Optional[int], default_value: int) -> int:
        """Parses element of given `board_shape`

        Args:
            board_shape_element (Optional[int]): element of board shape
            default_value (int): default value if non-valid value is passed

        Returns:
            int: [description]
        """
        if board_shape_element is None or board_shape_element < 2:
            return default_value
        return board_shape_element

    def _parse_board_shape(self, board_shape: Optional[Union[int, Tuple[int]]]) -> Tuple[int, int]:
        """Parses given `board_shape`

        Args:
            board_shape (Optional[Union[int, Tuple[int]]]): board shape

        Raises:
            ValueError: cannot parse given `board_shape`

        Returns:
            Tuple[int, int]: board shape in (num of rows, num of columns) format
        """
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
        """Parses `count` field in `random_config`

        Args:
            count (Optional[int]): given count value

        Returns:
            int: number of zero valued cells to randomly fill with non-zero value
        """
        if isinstance(count, int):
            return count if count > 0 else self.DEFAULT_RANDOM_CONFIG["count"]
        else:
            return self.DEFAULT_RANDOM_CONFIG["count"]

    def _parse_random_config_probs(self, probs: Dict[Optional[int], float]) -> int:
        """Parses `probs` field in `random_config`

        Args:
            probs (Dict[Optional[int], float]): given probs value

        Returns:
            int: cell values and corresponding probabilities for board randomization
        """
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
        """Parses `random_config`

        Args:
            random_config (Optional[Dict[str, Any]], optional): given random config value. Defaults to None.

        Returns:
            Dict[str, Any]: config for board randomization
        """
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
        """Returns empty board with given `board_shape`

        Args:
            board_shape (Tuple[int, int]): board shape

        Returns:
            np.ndarray: empty board
        """
        return np.zeros(board_shape, dtype=np.uint32)

    def _set_empty_board(self) -> None:
        """Set `board` attribute to empty board
        """
        self.board = self._get_empty_board(self.board_shape)

    def _get_starting_board(self, board_shape: Tuple[int, int], count: int) -> np.ndarray:
        """Returns starting board

        Args:
            board_shape (Tuple[int, int]): board shape
            count (int): number of non-zero cells

        Returns:
            np.ndarray: starting board
        """
        board = np.zeros(board_shape, dtype=np.uint32)
        for _ in range(count):
            board = self._set_random_empty_cell_to_value(board, value=2)
        return board
    
    def _set_starting_board(self) -> None:
        """Set `board` attribute to starting board
        """
        self.board = self._get_starting_board(self.board_shape, count=2)

    def _get_random_empty_cell(self, board: np.ndarray) -> Optional[Tuple[int, int]]:
        """Returns indices of empty cell if exists, otherwise None

        Args:
            board (np.ndarray): board

        Returns:
            Optional[Tuple[int, int]]: indices of empty cell
        """
        zero_cell_indices = np.argwhere(board == 0)
        if len(zero_cell_indices) > 0:
            random_int = np.random.randint(low=0, high=len(zero_cell_indices))
            random_indices = tuple(zero_cell_indices[random_int])
            return random_indices
        else:
            return None

    def _set_random_empty_cell_to_value(self, board: np.ndarray, value: int) -> np.ndarray:
        """Sets empty cell of `board` to given `value` if exists, otherwise does nothing

        Args:
            board (np.ndarray): board
            value (int): value to set

        Returns:
            [np.ndarray]: new board
        """
        indices = self._get_random_empty_cell(board)
        if indices is None:
            return board
        r, c = indices
        board[r, c] = value
        return board

    def get_board_str(self) -> str:
        """Returns `board` attribute in str format

        Returns:
            str: board attribute in str format
        """
        cell_len = len(str(np.max(self.board)))
        board_str = "\n"
        for row in range(self.board_shape[0]):
            row_numbers = self.board[row, :]
            row_str = " ".join([f"{str(e).center(cell_len)}" for e in row_numbers])
            board_str += row_str
            board_str += "\n"
        return board_str
    
    def compress(self) -> bool:
        """Compresses cells. Always assumes `LEFT` move, when compression occurs
        every non-zero cells in row will be on the left of their row.

        Returns:
            bool: whether anything has changed in board
        """
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
        """Merges cells. Always assumes `LEFT` move, when compression occurs neighbour
        cells with the same value are merged, cell values and score are updated.

        Returns:
            bool: whether anything has changed in board
        """
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
        """Add randomization to board according to random_config
        """
        count = self.random_config["count"]
        probs_values, probs_weights = list(self.random_config["probs"].keys()), list(self.random_config["probs"].values())
        random_values = np.random.choice(probs_values, size=count, p=probs_weights)
        for v in random_values:
            self.board = self._set_random_empty_cell_to_value(self.board, value=v)

    def play_move_left(self) -> bool:
        """Plays `LEFT` move

        Returns:
            bool: whether anything has changed in board
        """
        changed1 = self.compress()
        changed2 = self.merge()
        changed3 = self.compress()
        return changed1 or changed2 or changed3
    
    def play_move_right(self) -> bool:
        """Plays `RIGHT` move. Uses `play_move_left` by flipping board.

        Returns:
            bool: whether anything has changed in board
        """
        self.board = np.flip(self.board, axis=1)
        changed = self.play_move_left()
        self.board = np.flip(self.board, axis=1)
        return changed

    def play_move_up(self) -> bool:
        """Plays `UP` move. Uses `play_move_left` by rotating board.

        Returns:
            bool: whether anything has changed in board
        """
        self.board = np.rot90(self.board, k=1, axes=(0,1))
        self.board_shape = (self.board_shape[1], self.board_shape[0])
        changed = self.play_move_left()
        self.board = np.rot90(self.board, k=1, axes=(1,0))
        self.board_shape = (self.board_shape[1], self.board_shape[0])
        return changed

    def play_move_down(self) -> bool:
        """Plays `DOWN` move. Uses `play_move_left` by rotating board.

        Returns:
            bool: whether anything has changed in board
        """
        self.board = np.rot90(self.board, k=1, axes=(1,0))
        self.board_shape = (self.board_shape[1], self.board_shape[0])
        changed = self.play_move_left()
        self.board = np.rot90(self.board, k=1, axes=(0,1))
        self.board_shape = (self.board_shape[1], self.board_shape[0])
        return changed

    def play_move(self, move: Move) -> bool:
        """Play given `move` 

        Args:
            move (Move): move to play

        Returns:
            bool: whether anything has changed in board
        """
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
        """Parses given move str

        Args:
            move_str (str): move str

        Returns:
            Optional[Move]: move if parse is successful, None otherwise
        """
        move_str = move_str.lower()
        try:
            move = Move(move_str)
        except ValueError:
            return None
        return move

    def ask_move(self) -> Move:
        """Asks user a move to play

        Returns:
            Move: move to play, given by user
        """
        possible_moves_str = " ".join(["/".join(key.values) for key in Move])
        question_str = f"Your move? [{possible_moves_str}]:\n"
        while True:
            answer = input(question_str)
            parsed_move = self.parse_move(answer)
            if parsed_move is not None:
                return parsed_move
    
    def ask_continue_game_if_win(self) -> None:
        """Asks if user wants to continue to play although 2048 is achieved in the game,
        updates `continue_game_if_win` attribute accordingly
        """
        question_str = "2048 is achieved! Continue to play? [y/yes n/no]:\n"
        while True:
            answer = input(question_str)
            answer = answer.lower()
            if answer in ["y", "yes"]:
                self.continue_game_if_win = True
                break
            elif answer in ["n", "no"]:
                self.continue_game_if_win = False
                break

    def get_score_str(self) -> str:
        """Returns score in str format

        Returns:
            str: score str
        """
        return f"max-score = {self.max_score}\nsum-score = {self.sum_score}"

    def get_game_status_str(self, game_status: Optional[GameStatus] = None) -> str:
        """Returns game status in str format

        Args:
            game_status (Optional[GameStatus], optional): current game status. Defaults to None.

        Returns:
            str: game status str
        """
        game_status = self.game_status if game_status is None else game_status
        if game_status == GameStatus.WIN:
            return "You won!"
        elif game_status == GameStatus.LOSE:
            return "You lost..."
        elif game_status == GameStatus.ONGOING:
            return "Game is ongoing."

    def start(self) -> None:
        """Start the game.

        1) Print current score
        2) Print current board
        3) Check if game is over
            - if game is over
                4a) print game status
                5a) exit
            - if game is not over
                4b) check game status
                    - if game status is `WIN` and `continue_game_if_win` is None
                        5ba) `ask_continue_game_if_win`
                        6ba) check if game is over
                            - if True
                                7baa) print game status
                                8baa) exit
                            - otherwise
                                7bab) go to 5bb
                    - otherwise
                        5bb) ask user a move to play
                        6bb) play move
                            - if change in board
                                7bba) add randomization to board
                                    8bba) go to 1
                            - if no change in board
                                7bbb) go to 1
        """
        while True:
            print(self.get_score_str())
            print(self.get_board_str())

            game_status = self.game_status
            game_over = self.is_game_over(game_status)
            if game_over:
                print(self.get_game_status_str(game_status))
                exit(0)
            if self.continue_game_if_win is None and game_status == GameStatus.WIN:
                self.ask_continue_game_if_win()
                if self.is_game_over(game_status):
                    print(self.get_game_status_str(game_status))
                    exit(0)

            move = self.ask_move()
            changed = self.play_move(move)
            if changed:
                self.add_board_randomization()


if __name__ == "__main__":
    g = Game(board_shape=4, random_config={"count": 20, "probs": {2: 1}})
    g.start()