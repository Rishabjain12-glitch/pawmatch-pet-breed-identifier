"""
cli_demo.py
-----------

Command-line demo for the Pet Breed Identifier.

Usage:
python cli_demo.py compare path/to/pet1.jpg path/to/pet2.jpg
python cli_demo.py identify path/to/pet.jpg
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

from breed_identifier import BreedIdentifier

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def bar(value: float, width: int = 30, char: str = "█") -> str:
filled = int(value * width)
return f"{char * filled}{DIM}{'░' * (width - filled)}{RESET}"

def print_prediction(pred, label: str = "") -> None:
header = f" {BOLD}{CYAN}{label}{RESET}" if label else ""
icon = ""
print(header)
print(f"  {icon}  {BOLD}{pred.breed}{RESET} {DIM}({pred.pet_type}){RESET}")
print(f"  Confidence: {YELLOW}{pred.confidence * 100:.1f}%{RESET} {bar(pred.confidence)}")
print(f"\n  {DIM}Top-3 predictions:{RESET}")
for rank, (breed, score) in enumerate(pred.top_3, 1):
print(f"    {rank}. {breed:<35s} {score * 100:5.1f}% {bar(score, 20)}")
print()

def main() -> None:
parser = argparse.ArgumentParser(description="PawMatch — Pet Breed Identifier CLI")
subparsers = parser.add_subparsers(dest="cmd")

```
cmp = subparsers.add_parser("compare", help="Compare two pet images")
cmp.add_argument("image1", help="Path to first image")
cmp.add_argument("image2", help="Path to second image")

idf = subparsers.add_parser("identify", help="Identify breed in a single image")
idf.add_argument("image", help="Path to image")

parser.add_argument("pos1", nargs="?", help=argparse.SUPPRESS)
parser.add_argument("pos2", nargs="?", help=argparse.SUPPRESS)

args = parser.parse_args()

if args.cmd == "compare" or (args.pos1 and args.pos2):
    img1_path = Path(args.image1 if args.cmd == "compare" else args.pos1)
    img2_path = Path(args.image2 if args.cmd == "compare" else args.pos2)

    for p in (img1_path, img2_path):
        if not p.exists():
            print(f"{RED}File not found: {p}{RESET}")
            sys.exit(1)

    print(f"\n{BOLD}Loading model …{RESET}")
    identifier = BreedIdentifier()

    print(f"\n{BOLD}Analysing images …{RESET}\n")
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    result = identifier.compare(img1, img2)

    color = GREEN if result.same_breed else RED
    headline = "SAME BREED" if result.same_breed else "DIFFERENT BREEDS"

    sep = "─" * 55
    print(f"{sep}")
    print(f"{color}{BOLD} {headline}{RESET} confidence {YELLOW}{result.confidence * 100:.1f}%{RESET}")
    print(f"{DIM} Cosine similarity: {result.similarity_score:.4f}{RESET}")
    print(f"{sep}\n")

    print_prediction(result.image1_prediction, label=f"Image 1 · {img1_path.name}")
    print_prediction(result.image2_prediction, label=f"Image 2 · {img2_path.name}")
    print(f"{sep}")
    print(f" {result.verdict}")
    print(f"{sep}\n")

elif args.cmd == "identify" or args.pos1:
    img_path = Path(args.image if args.cmd == "identify" else args.pos1)
    if not img_path.exists():
        print(f"{RED}File not found: {img_path}{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}Loading model …{RESET}")
    identifier = BreedIdentifier()
    img = Image.open(img_path)
    pred = identifier.identify_breed(img)

    sep = "─" * 45
    print(f"\n{sep}")
    print_prediction(pred, label=img_path.name)
    print(f"{sep}\n")

else:
    parser.print_help()
```

if **name** == "**main**":
main()
