"""Tic-Tac-Toe with a Q-learning opponent.

Run this file directly, or run ``python play.py``.  The program uses only
the Python standard library, so no package installation is needed.
"""

from __future__ import annotations

import json
import random
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional


class TicTacToeGame:
    """The game board and all game-ending checks.

    Empty cells are represented by a space.  ``winner`` is ``"X"``, ``"O"``,
    ``"draw"``, or ``None`` while the game is still being played.
    """

    WIN_LINES = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.board = [" "] * 9
        self.winner: Optional[str] = None

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    def available_moves(self) -> list[int]:
        return [index for index, cell in enumerate(self.board) if cell == " "]

    def make_move(self, position: int, player: str) -> bool:
        """Place a mark, check the result immediately, and return success."""
        if self.is_over or position not in self.available_moves() or player not in ("X", "O"):
            return False
        self.board[position] = player
        self._check_game_end()
        return True

    def _check_game_end(self) -> None:
        for a, b, c in self.WIN_LINES:
            if self.board[a] != " " and self.board[a] == self.board[b] == self.board[c]:
                self.winner = self.board[a]
                return
        if not self.available_moves():
            self.winner = "draw"

    def state(self) -> str:
        """A hashable state used as the Q-table key."""
        return "".join(self.board)


class QLearningAgent:
    """Q-learning player.  It learns values for (board state, legal action)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 0.95, epsilon: float = 0.15) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q: dict[str, list[float]] = {}

    def _values(self, state: str) -> list[float]:
        if state not in self.q:
            self.q[state] = [0.0] * 9
        return self.q[state]

    def choose_action(self, game: TicTacToeGame, explore: bool = True) -> int:
        moves = game.available_moves()
        if explore and random.random() < self.epsilon:
            return random.choice(moves)
        values = self._values(game.state())
        best = max(values[move] for move in moves)
        return random.choice([move for move in moves if values[move] == best])

    def learn(self, state: str, action: int, reward: float, next_state: str, terminal: bool) -> None:
        current = self._values(state)[action]
        # Only moves into empty cells are valid in the next board state.
        legal_next = [index for index, cell in enumerate(next_state) if cell == " "]
        future = 0.0 if terminal else max(self._values(next_state)[move] for move in legal_next)
        self._values(state)[action] = current + self.alpha * (reward + self.gamma * future - current)

    def train(self, episodes: int, progress=None) -> None:
        """Train against a random X player.  The agent always plays O."""
        old_epsilon = self.epsilon
        for episode in range(episodes):
            game = TicTacToeGame()
            # A transition starts at O's decision and ends after X responds.
            game.make_move(random.choice(game.available_moves()), "X")
            while not game.is_over:
                state = game.state()
                action = self.choose_action(game, explore=True)
                game.make_move(action, "O")
                if game.is_over:
                    reward = 1.0 if game.winner == "O" else 0.2
                    self.learn(state, action, reward, game.state(), True)
                    break

                game.make_move(random.choice(game.available_moves()), "X")
                self.learn(state, action, -1.0 if game.winner == "X" else 0.0, game.state(), game.is_over)

            # Gradually reduce exploration, while retaining a little variety.
            self.epsilon = max(0.02, self.epsilon * 0.9995)
            if progress and (episode + 1) % max(1, episodes // 20) == 0:
                progress(episode + 1, episodes)
        self.epsilon = min(old_epsilon, max(0.02, self.epsilon))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.q), encoding="utf-8")

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        loaded = json.loads(path.read_text(encoding="utf-8"))
        self.q = {state: [float(v) for v in values] for state, values in loaded.items()}
        return True


class TicTacToeApp:
    """Tkinter interface: the person plays X and the learned agent plays O."""

    MODEL_PATH = Path(__file__).with_name("q_table.json")

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Tic-Tac-Toe | Q-Learning")
        self.root.geometry("800x1000")
        self.root.resizable(False, False)
        self.root.configure(bg="#eef2f7")
        self.game = TicTacToeGame()
        self.agent = QLearningAgent()
        loaded = self.agent.load(self.MODEL_PATH)
        self.mode = tk.StringVar(value="Vs Computer")
        self.first_mark = tk.StringVar(value="X")
        self.current_player = "X"
        self.player_one_mark = "X"
        self.computer_mark = "O"
        self.status = tk.StringVar(value="You are X — choose a square" if loaded else "Train the model first, then play as X")
        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#ffffff")
        style.configure("Card.TFrame", background="#f7f9fc")
        style.configure("Title.TLabel", background="#ffffff", foreground="#172b4d", font=("Arial", 25, "bold"))
        style.configure("Subtitle.TLabel", background="#ffffff", foreground="#607086", font=("Arial", 11))
        style.configure("Field.TLabel", background="#f7f9fc", foreground="#263b5e", font=("Arial", 11, "bold"))
        style.configure("Info.TLabel", background="#f7f9fc", foreground="#455b78", font=("Arial", 11))
        style.configure("Status.TLabel", background="#e7f0ff", foreground="#174a8b", font=("Arial", 12, "bold"), padding=(14, 11))
        style.configure("Primary.TButton", font=("Arial", 11, "bold"), padding=(13, 9), background="#2264d1", foreground="white")
        style.map("Primary.TButton", background=[("active", "#174fa8")])
        style.configure("Secondary.TButton", font=("Arial", 11, "bold"), padding=(13, 9), background="#e7edf6", foreground="#1c3f74")
        style.map("Secondary.TButton", background=[("active", "#d6e1f2")])
        style.configure("Reset.TButton", font=("Arial", 12, "bold"), padding=(16, 10), background="#142b4c", foreground="white")
        style.map("Reset.TButton", background=[("active", "#0b1e38")])

        frame = ttk.Frame(self.root, style="App.TFrame", padding=(42, 34))
        frame.pack(fill="both", expand=True, padx=24, pady=24)
        for column in range(3):
            frame.columnconfigure(column, weight=1, uniform="board")

        ttk.Label(frame, text="Tic-Tac-Toe", style="Title.TLabel").grid(row=0, column=0, columnspan=3, pady=(0, 2))
        ttk.Label(frame, text="Play a friend or train an opponent with Q-Learning", style="Subtitle.TLabel").grid(row=1, column=0, columnspan=3, pady=(0, 22))

        game_options = ttk.Frame(frame, style="Card.TFrame", padding=(18, 14))
        game_options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 18))
        game_options.grid_anchor("center")
        ttk.Label(game_options, text="Game mode:", style="Field.TLabel").grid(row=0, column=0, padx=(0, 8), pady=(0, 8), sticky="e")
        mode_box = ttk.Combobox(game_options, textvariable=self.mode, values=("Vs Computer", "Two Players"), state="readonly", width=16, font=("Arial", 11))
        mode_box.grid(row=0, column=1, padx=(0, 20), pady=(0, 8), sticky="w")
        mode_box.bind("<<ComboboxSelected>>", self.update_player_labels)
        ttk.Label(game_options, text="Player one:", style="Field.TLabel").grid(row=1, column=0, padx=(0, 8), sticky="e")
        mark_box = ttk.Combobox(game_options, textvariable=self.first_mark, values=("X", "O"), state="readonly", width=6, font=("Arial", 11))
        mark_box.grid(row=1, column=1, sticky="w")
        mark_box.bind("<<ComboboxSelected>>", self.update_player_labels)
        self.other_player_label = ttk.Label(game_options, text="", style="Info.TLabel")
        self.other_player_label.grid(row=0, column=2, rowspan=2, padx=(0, 6), sticky="w")

        board_frame = tk.Frame(frame, bg="#ffffff")
        board_frame.grid(row=3, column=0, columnspan=3, pady=(2, 8))
        self.buttons: list[tk.Button] = []
        for i in range(9):
            button = tk.Button(board_frame, text="", width=7, height=2, font=("Arial", 31, "bold"), command=lambda pos=i: self.human_move(pos), relief="flat", bd=0, cursor="hand2", bg="#f2f6fc", activebackground="#dbe9ff", disabledforeground="#172b4d")
            button.grid(row=i // 3, column=i % 3, padx=6, pady=6)
            self.buttons.append(button)

        ttk.Label(frame, textvariable=self.status, style="Status.TLabel", anchor="center", wraplength=600).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(20, 14))
        controls = ttk.Frame(frame, style="Card.TFrame", padding=(14, 12))
        controls.grid(row=5, column=0, columnspan=3, sticky="ew")
        controls.grid_anchor("center")
        ttk.Label(controls, text="Training episodes:", style="Field.TLabel").grid(row=0, column=0, padx=(0, 8))
        self.episodes = tk.StringVar(value="30000")
        ttk.Entry(controls, textvariable=self.episodes, width=10, font=("Arial", 11)).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(controls, text="Train Model", command=self.train_model, style="Primary.TButton").grid(row=0, column=2, padx=(0, 5))
        ttk.Button(controls, text="Train Model from Scratch", command=lambda: self.train_model(from_scratch=True), style="Secondary.TButton").grid(row=1, column=0, columnspan=3, pady=(10, 0), sticky="ew")
        ttk.Button(frame, text="↻  New Game", command=self.new_game, style="Reset.TButton").grid(row=6, column=0, columnspan=3, sticky="ew", pady=(16, 0))
        self.update_player_labels()

    def update_player_labels(self, _event=None) -> None:
        # The agent is trained as O, so keep the computer game consistent with
        # its learned board representation.  The X/O selector is for two-player
        # games, where either person may take the first turn.
        if self.mode.get() == "Vs Computer" and self.first_mark.get() != "X":
            self.first_mark.set("X")
        other_mark = "O" if self.first_mark.get() == "X" else "X"
        if self.mode.get() == "Two Players":
            self.other_player_label.config(text=f"Player two: {other_mark}")
        else:
            self.other_player_label.config(text=f"Computer: {other_mark}")

    def train_model(self, from_scratch: bool = False) -> None:
        try:
            episodes = int(self.episodes.get())
            if episodes < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid value", "Enter a positive whole number for training episodes.")
            return

        if from_scratch:
            # Start with no learned values and replace the persisted model.
            self.agent = QLearningAgent()
            if self.MODEL_PATH.exists():
                self.MODEL_PATH.unlink()
            self.status.set("Previous training cleared — training from scratch…")
        else:
            self.status.set("Training in progress…")
        self.root.update_idletasks()

        def show_progress(done: int, total: int) -> None:
            self.status.set(f"Training: {done:,} / {total:,}")
            self.root.update_idletasks()

        self.agent.train(episodes, show_progress)
        self.agent.save(self.MODEL_PATH)
        self.status.set(f"Training complete ({episodes:,} episodes). You are X — choose a square")
        self.new_game(keep_status=True)

    def human_move(self, position: int) -> None:
        if self.mode.get() == "Vs Computer" and self.current_player == self.computer_mark:
            return
        if not self.game.make_move(position, self.current_player):
            return
        self.refresh_board()
        if self.finish_if_needed():
            return
        if self.mode.get() == "Vs Computer":
            self.current_player = self.computer_mark
            self.status.set("Computer is thinking…")
            self.root.update_idletasks()
            self.root.after(200, self.ai_move)
        else:
            self.current_player = "O" if self.current_player == "X" else "X"
            self.status.set(f"Player {'one' if self.current_player == self.player_one_mark else 'two'}'s turn — {self.current_player}")

    def ai_move(self) -> None:
        action = self.agent.choose_action(self.game, explore=False)
        self.game.make_move(action, self.computer_mark)
        self.refresh_board()
        if not self.finish_if_needed():
            self.current_player = self.player_one_mark
            self.status.set(f"Your turn — you are {self.player_one_mark}")

    def finish_if_needed(self) -> bool:
        if not self.game.is_over:
            return False
        if self.game.winner == "draw":
            message = "It's a draw."
        elif self.mode.get() == "Two Players":
            message = f"Player {'one' if self.game.winner == self.player_one_mark else 'two'} wins! 🎉"
        elif self.game.winner == self.player_one_mark:
            message = "You win! 🎉"
        else:
            message = "The computer wins."
        self.status.set(message)
        self.refresh_board()
        return True

    def refresh_board(self) -> None:
        for i, button in enumerate(self.buttons):
            mark = self.game.board[i]
            color = "#dc3e5f" if mark == "X" else "#2264d1"
            button.config(
                text="" if mark == " " else mark,
                fg=color if mark != " " else "#172b4d",
                bg="#ffffff" if mark != " " else "#f2f6fc",
                state="disabled" if self.game.is_over or mark != " " else "normal",
            )

    def new_game(self, keep_status: bool = False) -> None:
        self.game.reset()
        self.player_one_mark = self.first_mark.get()
        self.computer_mark = "O" if self.player_one_mark == "X" else "X"
        self.current_player = "X"
        self.refresh_board()
        if not keep_status:
            if self.mode.get() == "Two Players":
                self.status.set(f"Player {'one' if self.current_player == self.player_one_mark else 'two'}'s turn — {self.current_player}")
            elif self.current_player == self.computer_mark:
                self.status.set("Computer starts…")
                self.root.after(200, self.ai_move)
            else:
                self.status.set(f"Your turn — you are {self.player_one_mark}")


def main() -> None:
    root = tk.Tk()
    TicTacToeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
