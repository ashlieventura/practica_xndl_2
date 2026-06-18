# Laboratori XNDL final notat

L'objectiu és crear un model des de zero (sense utilitzar cap pes pre-entrenat) capaç de classificar imatges de 32x32 píxels en una de les 14 categories següents: poma, pilota de bàsquet, cervell, cercle, rellotge, brúixola, galeta, donut, cara, lluna, patata, sol, síndria, roda

dades/
├── train/          ← imatges del dataset original
├── val/     
└── test/           ← ocult, però idèntic a val en procedència

Totes les imatges són de 32x32 i en escala de grisos.

## Què esperem que treballeu

L’esquelet que us donem té una xarxa fully connected molt senzilla. Funciona, però tracta cada píxel de forma independent i ignora completament l’estructura espacial de la imatge. El primer que hauríeu de preguntar-vos és: **té sentit aquesta arquitectura per a imatges?**

Penseu en com funcionen les xarxes convolucionals: detecten patrons locals (vores, textures, formes) que apareixen en qualsevol posició de la imatge, i els combinen en capes successives fins a reconèixer objectes complets. Per a un problema de visió com aquest, és el punt de partida natural.

A partir d’aquí, hi ha molts factors que influeixen en el resultat final: la profunditat i amplada de la xarxa, l’ús de normalització, tècniques de regularització per evitar overfitting, o augmentació de dades per fer el model més robust. Entendre per què cada canvi ajuda (o no) és el que us permetrà millorar l’accuracy de forma sistemàtica en comptes d’anar a cegues.

## Instruccions

- Entreneu el vostre model des de zero. No es permet cap pes preentrenat.
- Rebreu les credencials d’accés a Boada per correu electrònic. L’entrenament serà executat en un clúster (Boada o similar; us adjunto una guia sobre com utilitzar-lo), amb una GPU RTX 3080, o una altra GPU disponible en el moment de l’avaluació, i ha de durar com a màxim 5 minuts. Podeu desenvolupar i provar el codi al vostre ordinador personal si ho preferiu, però tingueu en compte que els temps d’execució seran diferents als de la màquina d’avaluació. Això no hauria de ser un problema: si un model convergeix bé en una màquina, en general també ho fa en una altra, tot i que no és garantit.
- El fitxer final ha de ser un únic .py, amb codi lliure però llegible. Comenteu el codi explicant les decisions que heu pres, no només el que fa cada línia.
- Es proporciona un esquelet amb els elements bàsics. Es pot modificar tot, sempre i quan compleixi les restriccions.
- Es fa en parelles (s’assumeixen els grups de la pràctica anterior).
- Data límit: 22 de juny a les 23:59.

## Avaluació

Per aprovar (nota ≥ 5) necessiteu superar el 70% d'accuracy (definit com micro F1) en test (mateixa distribució que el valid proporcionat). El 10 el marcarà qui aconsegueixi treure la millor puntuació. La resta de notes seran proporcionals entre el 5 (70%) i el 10 (millor puntuació de la classe).

Ànim i bona feina!
