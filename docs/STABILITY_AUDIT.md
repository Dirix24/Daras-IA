# Diagnóstico y primer incremento de estabilización

La base contenía la aplicación Tkinter, un binario Linux, Cerebro y el
checkpoint. No incluía una suite de pruebas ejecutable en este repositorio.
Se cargó el checkpoint desde Python y la calculadora respondió `2 + 2 = 4`
usando un directorio temporal. Tkinter 8.6 se importó correctamente. La
prueba de ventana queda pendiente porque el entorno carece de pantalla y Xvfb.

| Prioridad | Hallazgo | Estado |
| --- | --- | --- |
| P1 | Una frase ambigua podía activar Internet y las preguntas normales podían salir por la red con el interruptor activo. La búsqueda de vídeos reutilizaba el historial. | Consentimiento por sesión, comando explícito y tema de vídeo en la consulta actual. La recopilación automática conserva su propio interruptor y los temas configurados. |
| P1 | La limpieza de disco eliminaba conversaciones antiguas sin confirmación. | Limpieza limitada a caché de investigación. Prueba de conservación de conversaciones y recuerdos. |
| P1 | Los conectores admitían redirecciones y leían respuestas sin límite de tamaño. | Lista de endpoints HTTPS, redirecciones bloqueadas y límite de 2 MiB por respuesta. |
| P2 | Los trabajadores llamaban a Tkinter directamente y el timeout de investigación podía esperar al cierre del ejecutor. | Cola de eventos para el hilo principal, plazo por lote y descarte de resultados tras revocar permiso. Las solicitudes ya iniciadas pueden terminar en segundo plano. |
| P2 | Las preferencias se reemplazaban mediante escritura directa. | Escritura temporal sincronizada y reemplazo atómico. |
| P2 | Faltan pruebas con ventana, borrado sincronizado entre bases, migraciones, empaquetado y verificación de binarios. | Pendiente. El ejecutable Linux incluido es histórico y el alcance del cambio se verifica desde código fuente. |

Las pruebas locales simulan los casos de red y usan una base temporal.
Los proveedores reales, Windows y macOS quedan sin verificar. El siguiente
incremento debe verificar la sincronización y eliminación definitiva de
recuerdos entre SQLite y Cerebro, incluida la recuperación después de reiniciar.
