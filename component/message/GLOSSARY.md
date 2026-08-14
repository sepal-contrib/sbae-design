# Translation glossary

The recurring accuracy-assessment and sampling vocabulary, fixed once per
language and used consistently across `es/locale.json` and `fr/locale.json`.
The terminology follows Olofsson et al. (2014), *Good practices for estimating
area and assessing accuracy of land change*.

**Rows marked FLAG need a reviewer's decision.** Everything else follows the
established rendering in the Spanish/French remote-sensing literature. French
has had no native review at all and should be read end to end.

## Core terms

| English | Spanish | French | Note |
| --- | --- | --- | --- |
| accuracy | exactitud | exactitude | Deliberately not `precisión`/`précision`, which are reserved for *precision* below |
| precision | precisión | précision | |
| accuracy assessment | evaluación de la exactitud | évaluation de l'exactitude | **FLAG (fr)** — `évaluation de la précision` is common in the wild but conflates accuracy with precision |
| overall accuracy (OA) | exactitud global | exactitude globale | |
| user's accuracy (UA) | exactitud del usuario | exactitude de l'utilisateur | |
| producer's accuracy (PA) | exactitud del productor | exactitude du producteur | |
| expected user's accuracy (EUA) | exactitud esperada del usuario (EUA) | exactitude attendue de l'utilisateur (EUA) | **FLAG (es)** — `EUA` also abbreviates *Estados Unidos de América* in parts of Latin America. Kept because the acronym appears in the on-screen formulas |
| error matrix | matriz de error | matrice d'erreur | Olofsson's preferred term; kept distinct from *confusion matrix* below, as in the English |
| confusion matrix | matriz de confusión | matrice de confusion | |
| error-adjusted area | área ajustada por error | superficie ajustée pour l'erreur | **FLAG (fr)** — I could not confirm the conventional francophone rendering. Alternatives: `superficie corrigée des erreurs`, `estimation de superficie corrigée du biais` |
| area estimate | estimación de área | estimation de superficie | French uses `superficie` for land area throughout, never `aire` |
| area proportion | proporción de área | proportion de superficie | |
| map area | área del mapa | superficie cartographiée | **FLAG (es)** — `área cartografiada` is arguably clearer than the literal `área del mapa`; kept literal because the column sits next to *Adj. area* |
| SRS area | área SRS | superficie SRS | **FLAG (both)** — kept the English acronym. Localised forms exist (`MAS` / `EAS`) but are less recognisable to this audience than `SRS` |

## Sampling

| English | Spanish | French | Note |
| --- | --- | --- | --- |
| sampling | muestreo | échantillonnage | |
| sample | muestra | échantillon | |
| sample point | punto de muestreo | point d'échantillonnage | |
| sample size | tamaño de muestra | taille d'échantillon | |
| sample design | diseño de muestreo | plan d'échantillonnage | French `plan` is the statistical term; `conception` is used only for the UI tab, see below |
| stratified sampling | muestreo estratificado | échantillonnage stratifié | |
| simple random sampling | muestreo aleatorio simple | échantillonnage aléatoire simple | |
| systematic sampling | muestreo sistemático | échantillonnage systématique | |
| stratum / strata | estrato / estratos | strate / strates | |
| sample allocation | asignación de la muestra | répartition de l'échantillon | **FLAG (fr)** — `allocation` is also used; `répartition` reads as more idiomatic statistical French |
| proportional allocation | asignación proporcional | répartition proportionnelle | |
| design effect (DEFF) | efecto de diseño (DEFF) | effet de plan (DEFF) | **FLAG (fr)** — `effet de sondage` is the alternative |
| seed (RNG) | semilla | graine | |

## Uncertainty

| English | Spanish | French | Note |
| --- | --- | --- | --- |
| margin of error (MOE) | margen de error (MOE) | marge d'erreur (MOE) | Acronym kept, see *Acronyms* below |
| standard error (SE) | error estándar (EE) | erreur type (ET) | **FLAG (fr)** — `erreur standard` is also seen; `erreur type` is the formal term |
| confidence level (CL) | nivel de confianza (NC) | niveau de confiance (NC) | |
| confidence interval (CI) | intervalo de confianza (IC) | intervalle de confiance (IC) | `IC` is established in both languages |
| Z-score | valor Z | score Z | |

## Data and maps

| English | Spanish | French | Note |
| --- | --- | --- | --- |
| reference data | datos de referencia | données de référence | |
| validation | validación | validation | |
| reference class | clase de referencia | classe de référence | |
| mapped class / map class | clase del mapa | classe cartographiée | |
| predicted class | clase predicha | classe prédite | |
| classification map | mapa de clasificación | carte de classification | |
| land cover | cobertura del suelo | occupation du sol | French `occupation du sol` is land *cover*; `utilisation des sols` would be land *use* |
| area of interest (AOI) | área de interés (AOI) | zone d'intérêt (AOI) | Acronym kept, matching pysepal's own catalogs |
| raster | raster | raster | Unaccented, matching pysepal's own catalogs (`ráster` also valid in Spanish) |
| nodata | sin datos | sans données | Does not currently appear in the catalog; recorded for future strings |
| CRS | SRC | SCR | **FLAG (both)** — the QGIS conventions. Reverting to `CRS` in both is defensible |
| upload (verb) | cargar | téléverser | **FLAG (fr)** — `importer` and `charger` are more common in France; `téléverser` is the official term |
| download (verb) | descargar | télécharger | |
| tile | mosaico | tuile | **FLAG (es)** — `tile` is often left untranslated in Spanish GIS usage |

## Acronyms kept in English

`SBAE`, `SEPAL`, `GeoTIFF`, `CSV`, `GeoJSON`, `Olofsson`, `rasterio`, `AOI`,
`EUA`, `DEFF`, `SRS`, `MOE`, and the formula symbols (`n`, `n_h`, `p_h`, `N_h`,
`Z`, `OA`, `Wi`, `Si`, `SE`).

**FLAG — `MOE` in particular.** It appears in compact chart axes, chips and
tooltips (`MOE (%)`, `Max MOE`, `MOE vs Sample Size`). Localised acronyms
(`ME` / `M.E.`) are not established, and spelling the term out breaks the chip
layout, so the English acronym is used in both languages with the full term
given in the neighbouring tooltip. `SE` and `CL` are localised (`EE`/`ET`,
`NC`) because they only appear where a tooltip defines them.

## Register

- Spanish: *usted* throughout (`Cargue`, `Seleccione`), impersonal where
  natural. pysepal's own Spanish catalog mixes *tú* and *usted*; this catalog
  is internally consistent on *usted*.
- French: *vouvoiement* throughout, with the conventional space before `:`,
  `;`, `!` and inside `« »`.
- Neither language adds an exclamation mark the English does not have. The two
  places the English uses one (`allocation_changed`, `calculator.complete`) keep it.

## Two deliberate non-translations

- **The paper title.** `design.help.body` and the Olofsson citation keep
  *Good practices for estimating area and assessing accuracy of land change*
  in English. Translating the title of a published work would invent a
  citation that does not exist.
- **Decimal separators.** `0.5`, `1.0` and `≈1.0` keep the decimal point in
  both languages, because the app renders its own numbers with a point
  (Python formatting) and a mixed convention on the same screen reads worse
  than an anglicised one. **FLAG (fr)** — French normally uses `0,5`.

## UI terms where the English is ambiguous

| English | Reading taken | Note |
| --- | --- | --- |
| `upload.sample_map` "Use Sample Map" | *example* map, not *sampling* map | It loads the bundled demo GeoTIFF. Rendered `mapa de ejemplo` / `carte d'exemple` |
| `design.tab` / `landing.step_design` "Design" | short for *sample design* | `Diseño` / `Conception`. **FLAG (fr)** — `Conception` is bland; `Plan` collides with the map sense and `Échantillonnage` collides with the *Sampling* workflow button |
| `upload.preview.features` "Features" | vector feature count | The upload only accepts rasters, so this label is already odd in English. Rendered `Entidades` / `Entités` |
| `design.workflow.advanced` "Sampling" | generic sampling, as opposed to the Olofsson AA design | `Muestreo` / `Échantillonnage` |
