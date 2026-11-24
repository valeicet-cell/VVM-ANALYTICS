# VVM Analytics

Aplicación de Control Estadístico de Procesos construida con Python y Streamlit.
La aplicación incluye una pantalla de bienvenida para usuarios nuevos, carga de un archivo completo de proceso y conserva las opciones independientes de carga de variables y atributos. Los módulos de **Gráficos por Variables** y **Gráficos por Atributos** siguen separados.

## Requisitos
- Python 3.10 o superior (recomendado 3.11/3.12).
- Windows, macOS o Linux.

## Instalación en Windows
1. Descomprima la carpeta en una ubicación de trabajo.
2. Abra una terminal dentro de la carpeta `VVM Analytics`.
3. Cree un entorno virtual: `py -m venv venv`
4. Actívelo: `.venv\\Scripts\\activate`
5. Instale dependencias: `python -m pip install -r requirements.txt`
6. Ejecute: `streamlit run app.py`

También puede ejecutar `INICIAR_CEP.bat` después de instalar las dependencias.

## Instalación en macOS / Linux
```bash
python3 -m venv .venv
source venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Navegación
- Inicio: bienvenida y guía de primeros pasos cuando aún no hay datos; al cargar información muestra un resumen del análisis.
- Datos: permite cargar un archivo completo del proceso (mediciones y atributos en una misma base) o cargar variables y atributos por separado.
- Gráficos por Variables: gráficos de control para mediciones continuas.
- Gráficos por Atributos: gráficos p, np, c, u, Pareto e Ishikawa.
- Capacidad, Supuestos, Diseño de Gráficos, Muestreo de Aceptación y Reportes.
- Configuración: información y limpieza de datos temporales de sesión.

## Notas
- La interfaz utiliza una identidad visual oscura en negro y azul. Los módulos incorporan interpretaciones y conclusiones dinámicas según los resultados calculados; el informe PDF consolida los hallazgos de los módulos ejecutados.
- En la pestaña **Proceso completo**, cargue un único Excel/CSV y asigne las columnas de medición, subgrupo, conteo y tamaño de muestra. El programa activa los datos seleccionados para los módulos de análisis.
- Si desea trabajar solo con mediciones o solo con conteos, use las pestañas independientes **Solo variables** y **Solo atributos**.
- La sesión de Streamlit guarda datos temporalmente en memoria. Guarde los archivos exportados desde los módulos correspondientes.
- No incluya contraseñas, tokens ni credenciales en archivos que vaya a compartir.
