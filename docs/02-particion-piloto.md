# 02 · Partición provisional del piloto

Se aplicaron las exclusiones temporales de la [auditoría](01-auditoria-datos.md): 23 archivos de fotos idénticas con máscaras en conflicto y tres máscaras de cobertura extrema. Las listas incluyen 280 parejas y conservan las 34 máscaras vacías tal como vienen en `vt`. Su condición de negativos revisados sigue pendiente.

| Partición | Imágenes | Porcentaje | Máscaras vacías | Grupos |
|---|---:|---:|---:|---:|
| Entrenamiento | 196 | 70 % | 24 | 20 |
| Validación | 42 | 15 % | 5 | 5 |
| Prueba | 42 | 15 % | 5 | 6 |

Se agruparon los nombres `…_Best.png` por el UUID de 36 caracteres que aparece después del prefijo de ocho caracteres. Todos los archivos que comparten ese UUID quedaron juntos. Las 35 imágenes sin un UUID reconocible se dejaron juntas en entrenamiento. Los 20 grupos de entrenamiento incluyen ese bloque y 19 UUID.

La agrupación usa los nombres como aproximación porque falta un mapa de placas, pozos, días o muestras. La validación y la prueba describen solo los grupos con UUID. Dos UUID distintos podrían proceder de un mismo experimento. La partición sirve para comparar métodos en el piloto, con esa limitación al interpretar generalización.

El script usa la semilla **2026** y selecciona una combinación que se aproxima al tamaño y al número de máscaras vacías previstos en cada partición. Se verificó que las listas no se cruzan, que ningún grupo por nombre aparece en dos particiones y que las listas más las exclusiones cubren las 306 parejas. También se comprobó que la regeneración produce los mismos archivos.

```powershell
python auditar.py "$env:USERPROFILE\Downloads\vt\vt" resultados/auditoria.json
python particionar.py resultados/auditoria.json particiones/piloto
```

Para el siguiente paso, usar entrenamiento para construir la referencia clásica y validación para elegir ajustes. Reservar prueba hasta que los ajustes estén fijados. Cualquier recorte debe heredar la partición de su imagen original.
