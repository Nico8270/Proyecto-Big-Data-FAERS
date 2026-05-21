from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    exec(open(PROJECT_ROOT / "src" / "main.py").read())
    from main import main
    main()