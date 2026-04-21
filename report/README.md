# Reporte en Quarto

Este directorio contiene un borrador del informe en Quarto Markdown.

## Archivos

- `report.qmd`: borrador principal del informe.
- `references.bib`: base BibTeX para futuras citas automatizadas.
- `_quarto.yml`: configuracion del proyecto Quarto.
- `images/`: figuras exportadas desde el notebook y ya enlazadas en el borrador.

## Estado actual

- El informe ya incluye una seccion de referencias redactada manualmente en estilo APA.
- `references.bib` queda listo para automatizar citas mas adelante.
- Las figuras del notebook ya fueron exportadas al directorio `images/` y enlazadas dentro de `report.qmd`.
- Ya existe una version renderizada en PDF: `report/report.pdf`.

## Renderizado

En este entorno ya quedaron instalados `quarto` y TinyTeX, por lo que el renderizado local a PDF funciona.

El comando usado para generar el PDF es:

```bash
quarto render report.qmd --to pdf
```

Tambien se puede generar una version HTML con:

```bash
quarto render report.qmd --to html
```

## Nota sobre APA

El archivo `references.bib` sirve como insumo bibliografico, pero el borrador actual mantiene una seccion final de referencias en formato APA escrita manualmente para no depender de una hoja CSL externa en esta etapa.
