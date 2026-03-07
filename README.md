# Projet Titanic - Impact coque / rivets / endommagement (FEM)

## Objectif du projet

Ce projet sert a etudier, avec des modeles elements finis simplifies, l'influence des zones rivetees sur la reponse d'un segment de coque soumis a un impact localise (iceberg).

L'idee n'est pas de reconstruire le navire complet ni le contact reel iceberg-coque, mais de comparer des scenarios de maniere coherente :

- deformation de coque,
- localisation de l'endommagement,
- sensibilite a la presence (ou non) d'un effet rivets,
- cout de calcul / robustesse numerique.

## Vision physique (ce que represente le modele)

### 1) Grande echelle : coque mince avec endommagement diffuse

Le dossier `grande_echelle/` contient un modele de coque (surface maillée) avec :

- cinematique de coque (deplacement + rotation),
- materiaux heterogenes (coque + zones rivets homogenisees),
- endommagement global de type phase-field (irreversible),
- solveur quasi-statique (alternance mecanique / phase-field).

Le chargement iceberg est actuellement impose de facon simple :

- **Dirichlet** (deplacement impose),
- applique sur un patch localise,
- **dans la direction normale locale** de la coque (`e3`),
- avec une amplitude gaussienne se deplacant le long de la zone d'impact.

C'est un choix de pilotage numeriquement robuste pour des problemes avec perte de rigidite (post-pic / endommagement).

### 2) Rivets en grande echelle : homogeneisation en bandes

Les rivets ne sont pas modelises individuellement dans `grande_echelle/`.
A la place, on represente leur effet par des **bandes verticales homogenisees** sur la coque (dans la zone de passage de l'iceberg).

Chaque bande peut modifier localement :

- `E` (raideur elastique) via `facteur_E`,
- l'epaisseur via `facteur_epaisseur`,
- `Gc` (tenacite de rupture du phase-field) via `facteur_Gc`.

Cette approche est pratique pour :

- les etudes parametriques,
- les comparaisons A/B (avec/sans effet rivets),
- les temps de calcul raisonnables.

### 3) Modele local/intermediaire : `rivet/`

Le dossier `rivet/` fournit un **modele local 3D** (plaque percee, phase-field AT1) qui sert de proxy pour calibrer des facteurs effectifs utilises ensuite en grande echelle.

Ce n'est pas encore un assemblage rivete complet (deux toles + contact + rivet detaille), mais c'est un bon **modele intermediaire** pour extraire des tendances :

- seuil/traction de rupture locale,
- endommagement final,
- rigidite effective (via reponse force-deplacement, a enrichir si besoin).

Le script `grande_echelle/scripts/calibrer_bandes_depuis_rivet.py` fait le pont :

- calcule (ou lit) un resume local `rivet`,
- convertit les resultats en facteurs homogenises,
- ecrit un preset JSON de bandes pour `grande_echelle`.

## Organisation du depot (etat actuel)

- `grande_echelle/mesh.py` : generation du maillage coque (Gmsh)
- `grande_echelle/main.py` : config + lecture maillage + lancement du calcul global
- `grande_echelle/shell.py` : modele de coque (cinematique, CL, champs materiaux)
- `grande_echelle/quasi_static.py` : solveur quasi-statique + phase-field global
- `grande_echelle/scripts/calibrer_bandes_depuis_rivet.py` : calibration bandes depuis `rivet/`
- `rivet/rivet.py` : modele local/intermediaire phase-field + export preset bandes
- `vis_rivet/vis_rivet.py` : wrapper local minimal pour lancer le modele `rivet/` avec une CLI simple

## Workflow recommande (multi-echelles)

### Chemin TD (3 commandes)

```bash
python grande_echelle/mesh.py
python grande_echelle/scripts/calibrer_bandes_depuis_rivet.py --run-local
python grande_echelle/main.py
```

### A. Generer / verifier le maillage grande echelle

```bash
python grande_echelle/mesh.py
```

### B. Calibrer des bandes rivets depuis le modele local `rivet/`

Lancer un calcul local puis generer un preset :

```bash
python grande_echelle/scripts/calibrer_bandes_depuis_rivet.py --run-local
```

Version plus rapide (utile pour tests) :

```bash
python grande_echelle/scripts/calibrer_bandes_depuis_rivet.py --run-local --local-steps 30 --local-max-traction-mpa 220
```

Le script produit typiquement :

- un preset bandes (`.json`) pour `grande_echelle`,
- un rapport de calibration (`.calibration.json`).

### C. Lancer le calcul grande echelle (preset calibre charge automatiquement)

Si le fichier standard existe :

- `rivet/bandes_rivets_grande_echelle_calibre.json`

alors `grande_echelle/main.py` le charge automatiquement.

Commande la plus simple :

```bash
python grande_echelle/main.py
```

Le preset peut toujours etre force explicitement depuis Python si besoin.

```python
from grande_echelle.main import creer_config, lancer_calcul

cfg = creer_config(
    fichier_preset_bandes_rivets="rivet/bandes_rivets_grande_echelle_calibre.json"
)
lancer_calcul(cfg)
```

### D. Comparaison A/B avec / sans effet rivets

```python
from grande_echelle.main import lancer_comparaison_rivets_rapide
lancer_comparaison_rivets_rapide()
```

## Parametres importants (grande echelle)

Note compatibilite:

- les anciens noms anglais des cles de configuration restent acceptes automatiquement,
- mais le code utilise maintenant les noms francais comme reference.

### Contact / chargement iceberg

- `iceberg_centre_y`
- `flottaison_z`
- `iceberg_profondeur_sous_flottaison_m`
- `iceberg_zone_x_debut_m`, `iceberg_zone_x_fin_m`
- `deplacement_pic_iceberg`, `signe_deplacement_iceberg`
- `sigma` (largeur spatiale du patch gaussien)
- `iceberg_contact_t_debut`, `iceberg_contact_t_fin`

Remarque importante :

- la position verticale de l'appui iceberg est pilotee par `iceberg_profondeur_sous_flottaison_m`
- plus cette valeur est grande, plus l'appui descend sur la coque
- la configuration par defaut a ete ajustee pour charger plus bas (eviter la zone trop bombee)

### Phase-field global

- `enable_global_phase_field`
- `phase_field_gc_j_m2`, `phase_field_l0_m`
- `phase_field_raideur_residuelle`
- `phase_field_scinder_traction_compression`
- `phase_field_seuil_nucleation_j_m3`
- `phase_field_mise_a_jour_tous_les_n_pas`

### Rivets homogenises (bandes)

- `utiliser_bandes_rivets_z`
- `bandes_rivets_z`
- `fichier_preset_bandes_rivets`

## Sorties importantes

### Grande echelle (`results/<nom_cas>/`)

- `run_metadata.json` : config du run + metadonnees
- `quasi_static/monitor.csv` : indicateurs (deplacement max, dommage, temps de calcul)
- `quasi_static/*.pvd` : champs (deplacement, rotation, dommage)
- `local_frame/*.pvd` : diagnostics base locale / champs materiaux

### Visualisation ParaView (pratique)

Pour voir deformation + endommagement en meme temps :

1. ouvrir `results/<nom_cas>/quasi_static/displacement.pvd`
2. ouvrir `results/<nom_cas>/quasi_static/damage.pvd`
3. afficher `damage` sur la coque
4. appliquer `Warp By Vector` avec le champ de deplacement (optionnel)

Pour verifier les bandes rivets :

- ouvrir `results/<nom_cas>/local_frame/material_fields.pvd`
- regarder `GcFactorBandes`, `RivetBandsMask`, `RivetBandsMaskViz`

Note :

- si `RivetBandsMaskViz` parait "bancal" (taches/discontinuites), c'est souvent un probleme de resolution de maillage
- le maillage a ete raffine localement dans les bandes rivets dans `grande_echelle/mesh.py`, mais il faut regenerer le maillage pour en beneficier

### Modele local `rivet/`

- `run_summary.json` : resume local (rupture, traction de rupture, dommage final, etc.)
- fichiers `.bp` pour visualisation ParaView

## Hypotheses / limites (important)

- Le chargement iceberg est **impose** (pas de contact mecanique explicite iceberg-coque).
- Le chargement iceberg est impose en **Dirichlet sur la normale locale `e3`** (patch gaussien mobile), choix robuste numeriquement.
- Le modele `grande_echelle` utilise des **bandes homogenisees** pour les rivets (pas de rivets discrets).
- Le modele `rivet/` actuel est un **proxy local** (plaque percee AT1), pas encore un assemblage rivete complet.
- Les resultats sont donc surtout utiles pour des **comparaisons relatives** et du **screening physique/numerique**.

## Evolution naturelle du projet

Si on veut monter en fidelite sans perdre la structure actuelle :

1. enrichir le modele local `rivet/` (assemblage rivete plus realiste),
2. extraire des lois/equivalents homogenises plus robustes,
3. conserver `grande_echelle` en bandes homogenisees pour le screening,
4. reserver des modeles plus fins (CZM/contact detaille) a des sous-problemes cibles.
