# CV3-Proiect

## Sarcina: 
Detectarea automată a defectelor din asfalt și estimarea gravității gropilor.

## Setul de date: 
Subset al RDD2022 (Road Damage Detection), cu 4 clase de defecte: fisuri longitudinale, fisuri transversale, fisuri aligator și gropi. Pentru severitate, crop-urile cu gropi sunt adnotate manual în 3 niveluri.

## Metrici:

* Detecție: mAP@0.5:0.95 și F1
* Severitatea gropilor: acuratețe și F1

## Fezabilitatea pe hardware-ul disponibil: 
Imaginile sunt redimensionate la 640×640 pentru YOLO, ceea ce permite antrenarea pe un GPU T4 în Colab sau Kaggle. CNN-ul de severitate lucrează pe crop-uri mici și este ușor de antrenat.

## Flow: 
Un model YOLO detectează defectele și determină clasa lor. Pentru detecțiile de clasă „groapă”, severitatea este estimată de un CNN antrenat pe date adnotate manual.

Reguli de adnotare a severității:
* Scăzută: groapa are adâncime mică, iar mașinile pot trece peste ea cu ușurință.
* Medie: trecerea peste ea poate provoca daune mașinii.
* Ridicată: se recomandă ocolirea ei.
