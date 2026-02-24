from grande_echelle.main import (
    config_etude_rivets_rapide,
    lancer_calcul,
)


def main():
    print("=== Cas 1 : avec effet des rivets ===")
    cfg_avec = config_etude_rivets_rapide(with_rivets=True)
    lancer_calcul(cfg_avec)

    print("=== Cas 2 : sans effet des rivets ===")
    cfg_sans = config_etude_rivets_rapide(with_rivets=False)
    lancer_calcul(cfg_sans)

    print("Comparaison terminee.")
    print("Comparer les fichiers monitor.csv et les champs de dommage dans results/.")


if __name__ == "__main__":
    main()
