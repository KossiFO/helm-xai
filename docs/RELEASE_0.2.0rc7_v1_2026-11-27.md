# HELM 0.2.0rc7 — parcours testeur et retours explicites

Candidate de recherche. Le moteur d’explication, les paramètres expérimentaux
et la portée scientifique ne changent pas dans cette version.

- Guide d’essai Windows/macOS/Linux, environnements isolés et dépannage.
- Onglet « Tester et faire un retour », accessible avant une première analyse.
- Rapport texte téléchargeable et relisible, contenant les versions et, si disponible,
  le modèle, le profil, les méthodes et les notes de la dernière analyse affichée.
- Aucun ajout automatique du commentaire analysé, des journaux, des chemins locaux
  ou de l’historique au rapport. Le texte libre saisi par le testeur reste sous sa responsabilité.
- Liens GitHub/courriel sans transmission automatique du rapport.
- Formulaires GitHub pour les problèmes reproductibles et les retours d’essai.
- Commande `python -m helm.ui --diagnostic` sans démarrage serveur ni création d’historique.
- Notes clairement identifiées comme locales ; export complet marqué comme contenant le texte saisi.

Le modèle de démonstration reste entraîné sur 12 phrases synthétiques. Les tests
du logiciel et les retours volontaires ne remplacent pas une étude utilisateur contrôlée.
La page de retour est locale ; aucun service de collecte ou hébergement public n’est ajouté.
