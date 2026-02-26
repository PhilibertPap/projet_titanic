import argparse

from grande_echelle.main import (
    lancer_comparaison_rivets_rapide,
    lancer_comparaison_rivets_screening,
)


def main():
    parser = argparse.ArgumentParser(description="Comparaison A/B rivets (avec/sans)")
    parser.add_argument(
        "--mode",
        choices=("screening", "fast"),
        default="fast",
        help="screening = tri rapide, fast = discretisation plus fine",
    )
    args = parser.parse_args()

    if args.mode == "screening":
        lancer_comparaison_rivets_screening()
    else:
        lancer_comparaison_rivets_rapide()


if __name__ == "__main__":
    main()
