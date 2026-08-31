import os
import time
import json
from langchain_core.tools import tool

@tool
def spawn_geometry(commands: list[dict]):
    """
    Trigger this tool strictly when the user explicitly commands the system to draw, create, or spawn a geometric shape in the viewport.
    
    Rules:
    - NEVER guess or hallucinate dimensions if they are not provided.
    - Map user synonyms to the exact allowed shape strings below.
    - All angles are ABSOLUTE with respect to the x-axis (0.0 = Right/Horizontal, 90.0 = Up/Vertical, 180.0 = Left, 270.0 = Down).

    Args:
        commands (list[dict]): A chronological list of drawing steps to execute. Each dictionary in the list MUST contain:
            - 'shape' (str): Exactly "line", "polygon", or "circle". Do not capitalize. (Map "square", "rectangle", "triangle" to "polygon" or break them into a series of "line" commands).
            - 'size' (float): The numerical length or radius. MUST be a positive float. Default to 100.0 if unspecified.
            - 'angle' (float): The angle of the shape with respect to the x-axis in an anticlockwise direction. Extract directly from the user's prompt. If the user does not specify an angle, you MUST default to 0.0. Use standard notation like if prompt says vertical then it will be 90.0 if said horizontal then it will be 0.0 
            
    Examples:
        User: "Draw a square of size 200"
        commands = [
            {"shape": "line", "size": 200.0, "angle": 0.0},
            {"shape": "line", "size": 200.0, "angle": 90.0},
            {"shape": "line", "size": 200.0, "angle": 180.0},
            {"shape": "line", "size": 200.0, "angle": 270.0}
        ]
        
        User: "Draw a spiral starting from 500, reducing by 50 each time until it reaches 50."
        commands = [
            {"shape": "line", "size": 500.0, "angle": 0.0},
            {"shape": "line", "size": 450.0, "angle": 90.0},
            {"shape": "line", "size": 400.0, "angle": 180.0},
            {"shape": "line", "size": 350.0, "angle": 270.0},
            {"shape": "line", "size": 300.0, "angle": 0.0},
            {"shape": "line", "size": 250.0, "angle": 90.0},
            {"shape": "line", "size": 200.0, "angle": 180.0},
            {"shape": "line", "size": 150.0, "angle": 270.0},
            {"shape": "line", "size": 100.0, "angle": 0.0},
            {"shape": "line", "size": 50.0, "angle": 90.0}
        ]
    """
    mailbox_dir = r"G:/litho/Content"
    mailbox_path = os.path.join(mailbox_dir, "unreal_mailbox.json")

    if not os.path.exists(mailbox_dir):
        os.makedirs(mailbox_dir)
        
    command_packet = {
        "commands": commands,
        "command_id": str(time.time())
    }

    with open(mailbox_path, "w") as json_file:
        json.dump(command_packet, json_file, indent=4)
        
    log_message = f"Successfully dispatched a sequence of {len(commands)} drawing commands to Unreal Engine to draw your desired shape."
    return log_message
