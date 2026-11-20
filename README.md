# Daras — compañero personal

Prototipo portable con interfaz gráfica azul oscuro y negra. El ejecutable incluido está preparado para **Linux de 64 bits (x86_64)**.

## Abrir

1. Descarga y extrae toda la carpeta en la USB.
2. Abre la carpeta que contiene el ejecutable `Daras`.
3. Haz doble clic en `Daras` y elige **Ejecutar** si el escritorio lo pregunta.

No hace falta instalar Python, NumPy ni paquetes de la aplicación. El runtime y el modelo van dentro del ejecutable. Para comprobar el modelo desde una terminal: `./Daras --verificar-modelo` (debe mostrar `2 + 2 = 4`). Linux proporciona el cargador y entorno gráfico habituales. La USB debe permitir ejecutar programas y escribir archivos.

Daras crea `biblioteca.sqlite3`, `config.json` y `daras_cerebro_memoria.json` junto al ejecutable. La carpeta debe permitir escritura.

## Ejecutar la versión actual del código

Los cambios de consentimiento y conservación de datos descritos aquí
están en el código fuente. El binario histórico `Daras` no se reconstruyó.
Para usarlos, instala Python con Tkinter y NumPy, y ejecuta desde el repositorio.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy
.venv/bin/python -m unittest discover -s tests
.venv/bin/python companiona.py
```

## Qué hace

- Usa el modelo Cerebro que compartiste: transformer pequeño en NumPy, 190.528 parámetros, ejecutado en tu ordenador. No envía la conversación a un proveedor de IA.
- Usa el modelo para saludos y conversación cotidiana. Las matemáticas sencillas van a una calculadora segura; `2 mas 2` debe devolver `2 + 2 = 4` aunque la búsqueda de Internet esté activada.
- Para curiosidades, responde con lo que aprendió Cerebro. En preguntas sobre historia, sucesos, Roblox y otros temas que el modelo no conoce, reconoce el límite en vez de fingir que investigó.
- Internet empieza apagado en cada sesión. Para enviar una consulta, activa **Dar acceso a Internet para investigar** y escribe `busca en internet: …`. Daras consulta DuckDuckGo, Wikipedia y Wikidata; añade Crossref y OpenAlex cuando la consulta pida estudios o información académica. Muestra los extractos y enlaces de origen. No presenta esos extractos como conclusiones generadas por el modelo.
- Para buscar vídeos, especifica el tema en el mismo comando. El historial no se reutiliza para consultas externas.
- Escribe `recuerda: ...` para guardar un recuerdo y `olvida: ...` para borrar los que coincidan.

## Límite del modelo

Cerebro es un modelo experimental muy pequeño, entrenado desde cero con un corpus corto. Según las métricas que publica su repositorio, acierta respuestas exactas en el 47 % de reformulaciones de validación y generaliza mal a temas que no vio. Puede responder de forma corta, extraña o equivocada fuera de su corpus; por eso Daras deriva cuentas a la calculadora y temas externos a fuentes. No tiene la capacidad abierta de ChatGPT ni puede resumir fuentes web con comprensión fiable. Para mejorar conversaciones abiertas habría que ampliar y volver a entrenar el corpus o conectar otro modelo.

La búsqueda no rastrea toda la web: consulta los conectores anteriores, guarda extractos breves y respeta los límites de esos servicios. No ofrece diagnóstico médico ni sustituye ayuda profesional. Las recomendaciones de vídeo son resultados de búsqueda, no una clasificación de calidad.

Si quedan menos de 512 MB libres, limpia resultados de búsqueda de más de 90 días y conserva conversaciones y recuerdos.

El ejecutable es solo para Linux x86_64; Windows y macOS necesitan paquetes distintos. El código y el checkpoint de Cerebro proceden de [tu repositorio](https://github.com/Dirix24/cerebro).
# Daras-IA

## Licencia y estado

El código original de la aplicación Daras se distribuye bajo la [licencia MIT](LICENSE).
La licencia de este cambio cubre el código propio de Daras. El modelo, el checkpoint,
el código de Cerebro incluido y las dependencias del binario conservan sus
atribuciones y condiciones de origen.

Codex, una herramienta de IA de OpenAI, asistió con la implementación, las pruebas
y esta documentación. Consulta la [auditoría de estabilidad](docs/STABILITY_AUDIT.md)
para ver los límites de verificación y los requisitos pendientes de 1.0.
