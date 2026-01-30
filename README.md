## Quickstart

```bash
uv sync
``` 

pour récupérer les dépendances


## Download le dataset:

```sh
uv run python -m scripts.download
```

Cette commande ne télécharge que 5% des données, soit un peu plus de 10 go.

pour télécharger le dataset en entier, il faut faire

```sh
uv run python -m scripts.download --full
```

Celui ci en fait 250, donc ce sera pour le processing complet.

