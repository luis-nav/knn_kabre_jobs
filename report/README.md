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
- Todavia no se han exportado las figuras definitivas del notebook; por eso el borrador usa marcadores de posicion para imagenes.

## Renderizado

En este entorno no estan instalados `quarto`, `pandoc` ni una distribucion LaTeX, asi que el renderizado a PDF no se puede ejecutar todavia desde aqui.

Cuando la herramienta este instalada, los comandos esperados son:

```bash
quarto render report.qmd --to pdf
```

o bien:

```bash
quarto render report.qmd --to html
```

## Nota sobre APA

El archivo `references.bib` sirve como insumo bibliografico, pero el borrador actual mantiene una seccion final de referencias en formato APA escrita manualmente para no depender de una hoja CSL externa en esta etapa.
