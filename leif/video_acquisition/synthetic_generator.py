"""
Synthetic Controlled Environment Video Generator.
Generates image-sequence video datasets depicting people, objects, locations,
and observable state changes (office / laboratory / kitchen setting).
Allows immediate end-to-end testing of the complete LEIF pipeline without external downloads.
"""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SyntheticVideoGenerator:
    """
    Renders a realistic 2D simulated office/lab environment with:
    - Persons (P1, P2) moving across regions
    - Objects (cup, laptop, bottle, book, door)
    - Surfaces / Regions (desk_zone, coffee_station, door_zone)
    - Interactions (holding cup, opening door, placing laptop, moving bottle)
    - Observable environmental changes (laptop closed -> opened, cup moved from desk to table)
    """

    def __init__(
        self,
        output_dir: str,
        fps: float = 10.0,
        duration_seconds: float = 20.0,
        resolution: Tuple[int, int] = (640, 360),
    ):
        self.output_dir = output_dir
        self.fps = fps
        self.duration_seconds = duration_seconds
        self.total_frames = int(fps * duration_seconds)
        self.width, self.height = resolution
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self) -> Dict[str, Any]:
        """
        Renders the frames to disk and returns ground truth metadata.
        """
        ground_truth_events = []

        # Color palette for entities
        BG_COLOR = (240, 243, 246)
        DESK_COLOR = (180, 160, 140)
        COFFEE_STATION_COLOR = (200, 180, 160)
        DOOR_COLOR = (120, 140, 160)
        PERSON_COLOR = (41, 128, 185)
        CUP_COLOR = (231, 76, 60)
        LAPTOP_COLOR = (52, 73, 94)

        # Static regions
        desk_region = (50, 150, 250, 320)
        coffee_region = (400, 150, 590, 320)
        door_region = (550, 40, 620, 150)

        # Dynamic state variables
        # Event 1: Person P1 enters at t=1.0s, moves to desk, uses laptop (t=3s to 8s)
        # Event 2: Person P1 picks up cup (holding) at t=9s, moves to coffee station at t=12s, places cup
        # Event 3: Person P1 opens door at t=16s to 18s
        cup_pos = [140.0, 200.0]  # Initially on desk
        laptop_pos = [80.0, 180.0] # On desk
        laptop_state = "closed"
        door_state = "closed"

        for f_idx in range(self.total_frames):
            t = f_idx / self.fps
            img = Image.new("RGB", (self.width, self.height), BG_COLOR)
            draw = ImageDraw.Draw(img)

            # 1. Draw Environmental Regions & Surfaces
            # Desk
            draw.rectangle(desk_region, fill=DESK_COLOR, outline=(140, 120, 100), width=2)
            draw.text((60, 160), "DESK ZONE", fill=(60, 60, 60))

            # Coffee Station
            draw.rectangle(coffee_region, fill=COFFEE_STATION_COLOR, outline=(160, 140, 120), width=2)
            draw.text((410, 160), "COFFEE STATION", fill=(60, 60, 60))

            # Door
            door_fill = (90, 110, 130) if door_state == "opened" else DOOR_COLOR
            draw.rectangle(door_region, fill=door_fill, outline=(80, 100, 120), width=2)
            draw.text((555, 50), f"DOOR ({door_state.upper()})", fill=(255, 255, 255))

            # 2. Simulate Person Trajectory & Interactions
            person_visible = False
            person_pos = [0.0, 0.0]
            holding_cup = False

            if 1.0 <= t < 19.0:
                person_visible = True
                if t < 3.0:
                    # Walking towards desk
                    progress = (t - 1.0) / 2.0
                    person_pos = [100.0 + progress * 20.0, 50.0 + progress * 150.0]
                elif t < 8.0:
                    # Standing at desk, interacting with laptop
                    person_pos = [120.0, 200.0]
                    laptop_state = "opened"
                elif t < 12.0:
                    # Pick up cup, walk towards coffee station
                    holding_cup = True
                    progress = (t - 8.0) / 4.0
                    person_pos = [120.0 + progress * 320.0, 200.0]
                    cup_pos = [person_pos[0] + 15, person_pos[1] + 20]
                elif t < 15.0:
                    # At coffee station, cup placed down
                    holding_cup = False
                    person_pos = [450.0, 200.0]
                    cup_pos = [460.0, 210.0]  # Moved to coffee station!
                elif t < 18.0:
                    # Walking to door, opens door
                    progress = (t - 15.0) / 3.0
                    person_pos = [450.0 + progress * 100.0, 200.0 - progress * 100.0]
                    if t >= 16.5:
                        door_state = "opened"
                else:
                    # Exiting
                    person_pos = [580.0, 80.0]

            # 3. Draw Laptop
            l_x, l_y = laptop_pos
            draw.rectangle((l_x, l_y, l_x + 35, l_y + 25), fill=LAPTOP_COLOR)
            draw.text((l_x + 2, l_y + 5), f"Lptp ({laptop_state[:3]})", fill=(255, 255, 255))

            # 4. Draw Cup
            c_x, c_y = cup_pos
            draw.ellipse((c_x, c_y, c_x + 18, c_y + 18), fill=CUP_COLOR, outline=(150, 20, 20))
            draw.text((c_x - 2, c_y - 12), "Cup", fill=(100, 20, 20))

            # 5. Draw Person
            if person_visible:
                px, py = person_pos
                # Head
                draw.ellipse((px - 12, py - 35, px + 12, py - 11), fill=(245, 190, 160))
                # Body
                draw.rectangle((px - 15, py - 10, px + 15, py + 30), fill=PERSON_COLOR)
                draw.text((px - 10, py - 28), "P1", fill=(0, 0, 0))

            # 6. Render HUD info
            draw.text((10, 10), f"LEIF Controlled Benchmark | Time: {t:05.2f}s | Frame: {f_idx:04d}", fill=(20, 20, 20))

            # Save frame
            frame_filename = os.path.join(self.output_dir, f"frame_{f_idx:05d}.png")
            img.save(frame_filename)

        metadata_dict = {
            "output_dir": self.output_dir,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
        }
        try:
            import json
            with open(os.path.join(self.output_dir, "metadata.json"), "w") as mf:
                json.dump(metadata_dict, mf, indent=2)
        except Exception:
            pass

        return {
            "output_dir": self.output_dir,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "simulated_events": [
                {
                    "interaction": "using",
                    "subject": "person_P1",
                    "object": "laptop_01",
                    "location": "desk_zone",
                    "t_start": 3.0,
                    "t_end": 8.0,
                    "change": "laptop opened",
                },
                {
                    "interaction": "holding",
                    "subject": "person_P1",
                    "object": "cup_01",
                    "location": "desk_zone",
                    "t_start": 8.0,
                    "t_end": 12.0,
                },
                {
                    "interaction": "placing",
                    "subject": "person_P1",
                    "object": "cup_01",
                    "location": "coffee_station",
                    "t_start": 12.0,
                    "t_end": 15.0,
                    "change": "cup moved to coffee_station",
                },
                {
                    "interaction": "opening",
                    "subject": "person_P1",
                    "object": "door_01",
                    "location": "door_zone",
                    "t_start": 16.5,
                    "t_end": 18.0,
                    "change": "door opened",
                },
            ],
        }
