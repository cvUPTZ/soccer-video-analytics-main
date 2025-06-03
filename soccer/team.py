# In soccer/team.py

from typing import List, Tuple, Optional

# Forward reference for Pass, assuming it's defined in pass_event.py or similar
if False:
    from soccer.pass_event import Pass


class Team:
    def __init__(
        self,
        name: str,
        color: Tuple[int, int, int] = (0, 0, 0), # RGB
        abbreviation: str = "NNN",
        board_color: Optional[Tuple[int, int, int]] = None, # RGB for scoreboard
        text_color: Tuple[int, int, int] = (0, 0, 0), # RGB for text on board
    ):
        if not (len(abbreviation) == 3 and abbreviation.isupper()):
            raise ValueError("Abbreviation must be 3 uppercase characters.")
            
        self.name: str = name
        self.possession: int = 0  # In frames
        self.passes: List["Pass"] = [] 
        self.color: Tuple[int, int, int] = color
        self.abbreviation: str = abbreviation
        self.text_color: Tuple[int, int, int] = text_color
        self.board_color: Tuple[int, int, int] = board_color if board_color is not None else color

    def get_percentage_possession(self, total_match_duration_frames: int) -> float:
        if total_match_duration_frames == 0:
            return 0.0
        return round(self.possession / total_match_duration_frames, 2) # Percentage as float (0.0 to 1.0)

    def get_time_possession(self, fps: int) -> str:
        if fps <= 0: return "00:00" # Avoid division by zero
        
        total_seconds = round(self.possession / fps)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Team):
            return NotImplemented
        return self.name == other.name
    
    def __hash__(self) -> int: # Add hash if __eq__ is defined, for use in sets/dicts
        return hash(self.name)
    # The erroneous ``` line was here

    @staticmethod
    def from_name(teams: List["Team"], name: str) -> Optional["Team"]:
        for team in teams:
            if team.name == name:
                return team
        # print(f"Warning: Team with name '{name}' not found.")
        return None