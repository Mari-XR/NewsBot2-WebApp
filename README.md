# NewsBot Intelligence System 2.0

Proyecto final de **ITAI 2373 – Natural Language Processing**. Equipo **Purple Berries**.

1. Marisabel Morales
2. Khalida Bestani
3. Sadea Saeed

NewsBot explora artículos periodísticos con cuatro módulos de PLN y una interfaz Streamlit. Los modelos se cargan o entrenan al iniciar el proceso y se comparten en caché. Cada sesión del navegador tiene su propio historial y contexto de conversación. La interfaz y los comandos del chat están en inglés.

## Instalación local

La instalación pública incluye un corpus de Wikinews. Para ejecutar la versión académica local con BBC News, conserva tu copia de `data/BBC News Train.csv`. Si el archivo existe, NewsBot lo usa automáticamente.

Con Python 3.12 instalado, desde la raíz del proyecto:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abre [NewsBot local](http://localhost:8501). El primer arranque puede tardar mientras se preparan los recursos y se entrenan los modelos. La instalación descarga el modelo inglés `en_core_web_sm`. En el primer arranque se descargan los recursos NLTK que falten en `.nltk_data/`. Clasificación, sentimiento, entidades, temas, búsqueda y chat local funcionan con esos recursos instalados. La traducción requiere acceso a Google Translate o MyMemory.

## Páginas

| Página | Funciones |
| --- | --- |
| Analyze News | Probabilidades de las cinco categorías, tono y puntuación VADER, entidades spaCy, pesos y términos de LDA y NMF. |
| Summarize & Search | Resumen extractivo TextRank, búsqueda por similitud coseno sobre TF-IDF, expansión opcional, artículos relacionados, ficha de contexto y reporte descargable. |
| Multilingual Analysis | Detección con langdetect, traducción al inglés, destino adicional y comparación de dos versiones. Incluye inglés, español, francés, alemán e italiano. |
| Chat with NewsBot | Motor de reglas del notebook, búsqueda y paginación, seguimiento de sentimiento, resúmenes y traducción de resúmenes por referencia. |

Las referencias internas comienzan en 0 y corresponden al corpus activo. La barra lateral identifica el corpus y los resultados de Wikinews enlazan a sus fuentes y contribuyentes. El chat acepta “More” después de listados o búsquedas y “What about the sentiment?” después de un conjunto de resultados o un artículo. Las consultas no requieren un LLM.

Para traducción, Google es el primer proveedor y MyMemory el respaldo cuando se conoce un idioma de origen compatible. La app envía el texto a estos servicios solo al solicitar traducción. Admite hasta 4.500 caracteres por artículo y divide el texto en fragmentos según los límites de los proveedores. La app muestra un estado de error si ambos proveedores fallan, devuelven un resultado incompleto o repiten el texto de origen. La clasificación multilingüe se ejecuta sobre el texto en inglés.

## Código y datos

- `app.py` contiene la navegación y los formularios.
- `src/` adapta las funciones de las celdas finales del notebook. Contiene preprocesamiento, análisis, recuperación, traducción, chat e inicialización.
- `data/wikinews-demo.csv` contiene el corpus público. `data/BBC News Train.csv` se reserva para el proyecto académico local. Los datos procesados viven en memoria.
- `assets/` incluye la imagen de referencia y el tema de la interfaz.
- `.streamlit/config.toml`, `requirements.txt` y `.gitignore` contienen la configuración de ejecución y publicación.
- `tests/test_smoke.py` comprueba el motor y las cuatro páginas con AppTest.

### Datos

**Demo pública.** La app usa 300 extractos de artículos de Wikinews, 60 por categoría. Se incluyen los enlaces al original y al historial de contribuyentes. Los extractos adaptados están distribuidos bajo CC BY 2.5. Los modelos se ajustan con este corpus y sus resultados corresponden a esa muestra. [Atribución y licencia](data/WIKINEWS.md)

**Proyecto local.** El corpus BBC procede de la competición [BBC News Classification en Kaggle](https://www.kaggle.com/competitions/learn-ai-bbc/data). Kaggle lo sujeta a las [reglas de la competición](https://www.kaggle.com/competitions/learn-ai-bbc/rules). El permiso de redistribución pública del CSV no se pudo confirmar. Se conserva en el equipo y queda excluido de Git.

El ZIP contenía únicamente el CSV y no incluía artefactos de modelos. El notebook y el documento técnico guiaron la adaptación. Se descartaron 50 duplicados exactos por texto de las 1.490 filas originales. Se reprodujeron el preprocesamiento NLTK y la partición 80/20 estratificada con semilla 42. TF-IDF y regresión logística se ajustan dentro de la partición de entrenamiento, con validación cruzada de tres pliegues y C en {1, 5, 10, 20}. El conjunto reservado de 288 artículos se usa solo para evaluar el clasificador. El índice de búsqueda, el resumen de sentimiento del corpus y los modelos temáticos exploratorios usan las 1.440 noticias.

La ejecución local con BBC eligió **C=10** y logró **96,53 % de exactitud** en la partición reservada (278/288), coincidiendo con la salida de Colab. Ese resultado corresponde a la clasificación del corpus BBC.

## Pruebas

Con el entorno activado:

```sh
python -m unittest discover -s tests -v
NEWSBOT_TEST_TRANSLATION=1 python -m unittest discover -s tests -v
python -m pip check
```

La segunda ejecución incluye traducción real inglés↔español y depende de la conectividad. Las pruebas comunes también simulan fallos de red para verificar que NewsBot informa del error y evita clasificar texto sin traducir.

## Publicar en GitHub y Streamlit Community Cloud

1. En GitHub crea un repositorio **público y vacío** llamado `NewsBot2-WebApp`. Deja sin marcar la creación de README, licencia y archivo de exclusión. Verifica que la imagen de referencia y los nombres del equipo pueden compartirse públicamente.
2. Abre Terminal y ve a la carpeta local. Prepara el primer commit con la lista explícita de archivos. Antes de hacer el commit, revisa el listado y verifica que no contiene `BBC News Train.csv`, el ZIP, el notebook, el PDF, entornos locales ni credenciales.

   ```sh
   cd "$HOME/Documents/ChatGPT/NewsBot2-WebApp"
   git init -b main
   git add -- .gitignore .gitattributes README.md DEMO.md app.py requirements.txt .streamlit/config.toml assets/theme.css assets/newsbot_background.png data/wikinews-demo.csv data/WIKINEWS.md src/*.py tests/*.py
   git diff --cached --check
   git diff --cached --name-only
   git commit -m "Prepare NewsBot 2.0 public academic demo"
   git remote add origin https://github.com/TU_USUARIO/NewsBot2-WebApp.git
   git push -u origin main
   ```

3. En [Streamlit Community Cloud](https://share.streamlit.io), conecta tu cuenta de GitHub. En **Create app**, elige `TU_USUARIO/NewsBot2-WebApp`, rama `main` y archivo `app.py`.
4. En **Advanced settings**, elige **Python 3.12** y deja **Secrets** vacío. Pulsa **Deploy**.
5. Espera a que el panel indique **300 Wikinews articles**. Repite la [demo](DEMO.md). En **App settings → Sharing**, verifica que la app es pública y abre la URL `streamlit.app` en una ventana privada antes de compartirla con tu profesor y compañeros.

Consulta la [guía oficial de despliegue](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) y la [configuración de acceso de GitHub](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-account/manage-your-github-connection).

## Demostración para clase

El [guion de cuatro minutos](DEMO.md) incluye entradas listas para pegar, resultados esperados y una alternativa si la traducción externa falla durante la presentación.
