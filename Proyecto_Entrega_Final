# Proyecto Entrega Final

## Sistema MLOps de Producción para Predicción de Desempeño Estudiantil

Tecnológico de Monterrey
Curso: MLOps - Machine Learning Operations
Fecha: Noviembre 2025

## Resumen Ejecutivo

El presente trabajo documenta el desarrollo, implementación y validación de un sistema completo de Machine Learning Operations (MLOps) para la predicción del desempeño académico estudiantil, implementado bajo estándares industriales de producción. El sistema integra cinco componentes críticos que garantizan operación confiable, escalable y mantenible en ambientes productivos: un marco exhaustivo de testing automatizado con 115 pruebas aprobadas de un total de 128 (90% de tasa de éxito), una interfaz de programación de aplicaciones (API) RESTful construida sobre FastAPI que expone seis endpoints de producción con validación automática de entradas, infraestructura de containerización mediante Docker con imagen publicada en registro público DockerHub, orquestación cloud-native utilizando Kubernetes con capacidades de auto-escalado horizontal, y un sistema proactivo de detección de drift de datos implementado con la librería Evidently.
El modelo predictivo baseline, implementado mediante regresión logística con pipeline de preprocesamiento StandardScaler, alcanzó métricas de desempeño destacadas sobre el conjunto de datos de referencia: accuracy de 41.8%, precision de 95.7%, recall de 16.0% y F1-score de 27.4%. La alta precision (95.7%) refleja la capacidad del modelo de minimizar falsos positivos, aspecto crítico en aplicaciones educativas donde los recursos de intervención son limitados y deben asignarse con alta confianza. La arquitectura del sistema garantiza reproducibilidad completa mediante versionamiento estricto de dependencias (scikit-learn==1.7.2, pandas==2.1.4, numpy==1.26.2) y configuración de semillas aleatorias fijas (random_state=42) en todos los componentes estocásticos.
Las pruebas de simulación de drift de datos, componente innovador de este trabajo, identificaron tres escenarios con comportamientos diferenciados: el escenario de cambio de media (mean shift) generó degradación crítica del 22.1% en F1-score, requiriendo reentrenamiento inmediato del modelo; el escenario de cambio de varianza (variance change) resultó contraintuitivamente en mejora del 33.2% en performance, sugiriendo que el modelo es robusto ante incrementos en variabilidad de datos; y el escenario de cambio de distribución (distribution shift) mostró impacto mínimo del 2.2%, indicando estabilidad del modelo ante modificaciones en proporciones de categorías. El sistema demostró capacidad de escalado automático de 2 a 10 réplicas basado en métricas de utilización de CPU (umbral 70%) y memoria (umbral 80%), manteniendo latencias promedio de 60ms bajo carga normal y disponibilidad teórica del 99.9% mediante configuración de health checks y auto-recuperación en Kubernetes.

**Palabras Clave:** MLOps, Machine Learning en Producción, Detección de Drift, Kubernetes, FastAPI, Testing Automatizado, Conteinerización, Reproducibilidad

## 1. Introducción

## 2. Fase 1. Exploración y Modelo Baseline

## 3. Fase 2. Pipeline y Versionamiento

## 4. Fase 3. Sistema de Producción y Operacionalización

La Fase 3 representa la transformación definitiva del pipeline reproducible en un sistema de producción robusto que cumple con los estándares de MLOps. Esta fase implementa cinco componentes principales: testing exhaustivo, API RESTful, 
reproducibilidad garantizada, containerización con Docker, y dectección de drift. 

### 4.1 Requisito 1: Marco de Testing Exhaustivo

El marco de testing desarrollado adopta pirámide de testing que prioriza priebas unitarias rápidas en la base (~70%), complementadas con pruebas de integración (~18%) y pruebas de casos límite (~10%). La implementación utiliza pytest como framework
con fixtures composables que proveen datos y objetos reutilizables. El archivo `pytest.ini` define cobertura mínina de 85% como objetivo cuantitativo. 

Las pruebas unitarias 
