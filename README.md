# Moral Misalignment Pipeline

Analyse von Nutzerkommentaren zu Elon Musk auf X.com im Kontext moralischer Dissonanz  
**Bachelorarbeit Max Dorfmann · Juni 2025**

---

## Projektüberblick

Diese Pipeline extrahiert, filtert und analysiert Reply-Kommentare zu ausgewählten Tweets von Elon Musk. Ziel ist es, Muster moralischer Rechtfertigung trotz kritischer Haltung gegenüber dem Absender zu identifizieren.

Die Anwendung kombiniert:
- API-basiertes Reply-Sampling über `tweepy`
- Moral-Filtrierung anhand des Moral Foundations Dictionary (MFD)
- Themenclustering mit BERTopic
- qualitative Clusterinterpretation mittels GPT-4o

Die Ausgabe umfasst für jedes Event:
- eine CSV-Datei mit den gefilterten Replies
- JSON-Zusammenfassungen für alle Cluster
- ein serialisiertes Topic-Modell

---

## Aufbau der Pipeline

| Modul                             | Funktion                                                              |
|----------------------------------|-----------------------------------------------------------------------|
| `moral_misalignment_pipeline.py` | Hauptskript: Datenerhebung, Filterung, Clustering, Zusammenfassung    |
| `moral_foundations_dictionary.dic` | Enthält 156 Begriffe aus dem Moral Foundations Dictionary (Graham, Haidt & Nosek, 2009); für akademische Nutzung beigelegt |
| `requirements.txt`               | Listet alle Python-Abhängigkeiten zur Reproduktion                    |

---

## Vorbereitung

1. Projekt klonen:
   ```bash
   git clone https://github.com/maxdorfmann/moral-misalignment-pipeline.git
   cd moral-misalignment-pipeline
    ```

2. Abhängigkeiten installieren:

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. API-Schlüssel setzen (erforderlich):

   ```bash
   export OPENAI_API_KEY="sk-..."     # GPT-Zugang
   export TW_BEARER_TOKEN="..."       # Twitter API Bearer Token
   ```

---

## Ausführung

Zum Ausführen aller hinterlegter Events:
```bash
python moral_misalignment_pipeline.py
```
Oder zum Ausführen von nur einem bestimmten Events:
```bash
python moral_misalignment_pipeline.py --events CE1
```

---

## Ergebnisstruktur

```text
results/
 ├─ CE1/
 │   ├─ replies.csv                # 150 gefilterte Replies
 │   ├─ cluster_summaries.json    # GPT-4o-gestützte Clusterinterpretationen
 │   └─ bertopic_model/           # gespeichertes BERTopic-Modell
 └─ CE2/ …
```

---

## Methodische Eckpunkte

* **Datenquelle**: X.com (ehemals Twitter), Tweets von Elon Musk
* **Filterung**:

  * ≥ 5 Wörter
  * max. 1 URL
  * moralischer Gehalt gemäß MFD v1.0
* **Zielgröße**: exakt 150 valide Replies pro Event
* **Clustering**: BERTopic (`all-MiniLM-L6-v2`)
* **Interpretation**: GPT-4o (2–3 Sätze pro Cluster, theorie-neutral)

---

## Systemumgebung

Die Pipeline wurde entwickelt und getestet unter folgender Konfiguration:

- **Python-Version**: 3.11.9
- **Betriebssystem**: Ubuntu 24.04 LTS

Andere Umgebungen (z. B. Windows oder macOS) können abweichendes Verhalten zeigen und wurden nicht evaluiert.

---

## Externe Ressourcen

Dieses Projekt verwendet das *Moral Foundations Dictionary* von Graham, Haidt und Nosek (2009) zur Analyse des moralischen Gehalts von Texten auf Basis der Moral Foundations Theory.

Die Datei [`moral_foundations_dictionary.dic`](moral_foundations_dictionary.dic) basiert auf der offiziellen Version, die unter folgender Adresse verfügbar ist:

[https://moralfoundations.org/wp-content/uploads/files/downloads/moral%20foundations%20dictionary.dic](https://moralfoundations.org/wp-content/uploads/files/downloads/moral%20foundations%20dictionary.dic)

Ursprüngliche Quelle:

Graham, J., Haidt, J., & Nosek, B. A. (2009).
*Liberals and conservatives rely on different sets of moral foundations.*
*Journal of Personality and Social Psychology, 96(5), 1029–1046.*
[https://doi.org/10.1037/a0015141](https://doi.org/10.1037/a0015141)

Hinweis: Die Datei ist urheberrechtlich geschützt © Graham, Haidt & Nosek (2009) und wird ausschließlich für akademische Zwecke im Rahmen dieses Projekts verwendet. Sie ist nicht Teil der MIT-Lizenz dieses Repositories. Für weitergehende Nutzung ist auf die Bestimmungen unter [moralfoundations.org](https://moralfoundations.org) zu verweisen.

---

## Lizenz

Dieses Repository steht unter der **MIT License** (siehe `LICENSE`).

> © 2025 Max Dorfmann

---

## Hinweis

Die Nutzung der Twitter API unterliegt den geltenden Nutzungsbedingungen von X. Dieses Projekt wurde ausschließlich für nicht-kommerzielle, wissenschaftliche Zwecke im Rahmen einer universitären Abschlussarbeit entwickelt.
