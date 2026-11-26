# Geometry failures

_MOCK run; not benchmark data._

| Input | Gold | Prediction | Replay diff |
| :--- | :--- | :--- | ---: |
| Draw a spiral of lines that get shorter and rotate 30 degrees each time, starting with a length of 100. | `[{"shape": "line", "size": 100, "angle": 0}, {"shape": "line", "size": 90, "angle": 30}, {"shape": "line", "size": 80, "angle": 60}, {"shape": "line", "size": 70, "angle": 90}, {"shape": "line", "size": 60, "angle": 120}, {"shape": "line", "size": 50, "angle": 150}, {"shape": "line", "size": 40, "angle": 180}, {"shape": "line", "size": 30, "angle": 210}, {"shape": "line", "size": 20, "angle": 240}, {"shape": "line", "size": 10, "angle": 270}]` | `[{"shape": "line", "size": 155.0, "angle": 0}, {"shape": "line", "size": 90, "angle": 30}, {"shape": "line", "size": 80, "angle": 60}, {"shape": "line", "size": 70, "angle": 90}, {"shape": "line", "size": 60, "angle": 120}, {"shape": "line", "size": 50, "angle": 150}, {"shape": "line", "size": 40, "angle": 180}, {"shape": "line", "size": 30, "angle": 210}, {"shape": "line", "size": 20, "angle": 240}, {"shape": "line", "size": 10, "angle": 270}]` | 55.0 |
| Draw a spiral of lines that get shorter and rotate 45 degrees each time, starting with a length of 100. | `[{"shape": "line", "size": 100, "angle": 0}, {"shape": "line", "size": 50, "angle": 45}, {"shape": "line", "size": 25, "angle": 90}]` | `[{"shape": "line", "size": 155.0, "angle": 0}, {"shape": "line", "size": 50, "angle": 45}, {"shape": "line", "size": 25, "angle": 90}]` | 55.0 |
| Draw a horizontal line of length 50. | `[{"shape": "line", "size": 50, "angle": 0}]` | `[{"shape": "line", "size": 80.0, "angle": 0}]` | 30.0 |
