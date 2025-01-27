# App streamlit pour présenter notre projet et nos résultats obtenus avec les différents modèles

import streamlit as st

st.set_page_config()

# Titre de la page
st.title("Classification de catalogues de musique")

st.divider()

# Introduction
st.header("Présentation du projet")
st.markdown("""[Mewo](https://www.mewo.io/) est une plateforme de gestion de catalogues pour les bibliothèques de musique de production utilisée par de grands acteurs français des médias, de la publicité et de la musique. Mewo ont déjà un système de labellisation automatique qui prédit des scores numériques indiquant la pertinence d’un morceau pour différents tags qui sont organisés en trois grandes catégories : genres, instruments, humeurs (“mood”). Cependant, pour exploiter ces prédictions, il est nécessaire de les convertir en décisions binaires : associer ou non un tag à une piste. 

L’approche présentée dans le challenge utilise un seuil par tag pour maximiser un F1-score individuel. Bien que performante, cette méthode ne prend pas en compte les relations entre tags, catégories ou classes. Bien sûr le challenge date de 2021 et leur processus de labellisation a certainement un meilleur benchmark aujourd’hui. L’objectif donné par le benchmark est un weighted F1-Score de **54%** sur le **train set** et **46%** sur le **test set**.

**Le challenge consiste donc à concevoir une méthode plus avancée pour améliorer la qualité globale de la classification des labellisations.**&#x20;
""")
