"""
main_v3.py — Punto de entrada para Civilization-ABM v3.

Uso:
    python main_v3.py                          # todas las condiciones
    python main_v3.py --condition baseline     # solo baseline
    python main_v3.py --jobs 16                # 16 workers
    python main_v3.py --no-ray                 # fuerza multiprocessing
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from experiments.ray_runner import main

if __name__ == "__main__":
    main()
