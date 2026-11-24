# Demo en vivo de NewsBot 2.0

**Duración:** unos 4 minutos. Abre NewsBot antes de presentar y espera a que aparezca el corpus en la barra lateral (**300 Wikinews** en la demo pública o **1.440 BBC** en la versión local). Para la parte multilingüe, usa una conexión a internet.

| Tiempo | Pantalla | Acción y resultado que mostrar |
| --- | --- | --- |
| 0:00–0:50 | **Analyze News** | Pega el texto sobre Apple y pulsa **Analyze article**. Señala la categoría predicha, el gráfico de probabilidades y el tono VADER. Abre **Named entities** para mostrar Apple y California y **Topics** para señalar LDA y NMF. |
| 0:50–1:45 | **Summarize & Search** | Pega el mismo texto, elige dos frases y pulsa **Generate summary**. Muestra el resumen TextRank y abre una noticia relacionada. En la pestaña de búsqueda del corpus activo, busca `elections and voters` y abre el primer resultado. |
| 1:45–2:40 | **Multilingual Analysis** | Pega la frase en español y la versión inglesa de abajo en sus casillas respectivas. Pulsa **Analyze languages**. Muestra la detección, el texto traducido y las dos filas de clasificación. Si el proveedor está temporalmente caído, señala el mensaje de indisponibilidad y muestra cómo se analiza la versión inglesa. |
| 2:40–3:50 | **Chat with NewsBot** | Envía `Show me positive tech articles`, luego `More` y finalmente `What about the sentiment?`. Señala que el segundo turno trae otros artículos de la misma búsqueda y el tercero habla del sentimiento de esos resultados. Si hay tiempo, pide `Summarize article 5`. |

**Texto sobre Apple**

> Apple announced a new iPhone at its headquarters in California. The new iPhone offers better battery life and a faster processor. Apple said the iPhone will reach stores next week. The smartphone launch brought good news for technology investors.

**Español**

> El gobierno anunció nuevas medidas para ayudar a las familias que necesitan apoyo económico.

**Inglés**

> The government announced new measures to help families in need of financial support.
