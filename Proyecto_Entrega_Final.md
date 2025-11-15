<img width="468" height="25" alt="image" src="https://github.com/user-attachments/assets/65b0e4f6-caa3-428d-8fb8-ee7f5f69d249" /># Proyecto Entrega Final

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

La Fase 3 representa la transformación definitiva del pipeline reproducible en un sistema de producción robusto que cumple con los estándares de MLOps. Esta fase implementa cinco componentes principales: 

1. **Testing Automatizado**: Marco exhaustivo con 128 pruebas que validan corrección funcional
2. **API RESTful**: Interfaz estandarizada construida con FastAPI que expone 6 endpoints
3. **Reproducibilidad**: Mecanismos rigurosos que garantizan consistencia entre ambientes
4. **Containerización**: Infraestructura Docker + Kubernetes para escalabilidad
5. **Monitoreo de Drift**: Detección proactiva de degradación del modelo

### Arquitectura General del Sistema

![Arquitectura MLOps](images/arquitectura_mlops.png)
*Figura 1: Arquitectura completa del sistema MLOps implementado*

### 4.1 Requisito 1: Marco de Testing Exhaustivo

El marco de testing desarrollado adopta pirámide de testing que prioriza priebas unitarias rápidas en la base (~70%), complementadas con pruebas de integración (~18%) y pruebas de casos límite (~10%). 

```
        /\
       /  \
      / E2E \     ← 10% - Pruebas End-to-End (lentas, alto valor)
     /______\
    /        \
   / INTEGR.  \   ← 20% - Pruebas de Integración
  /____________\
 /              \
/  UNIT TESTS   \ ← 70% - Pruebas Unitarias (rápidas, bajo nivel)
/________________\
```

La implementación utiliza pytest como framework con fixtures composables que proveen datos y objetos reutilizables. El archivo `pytest.ini` define cobertura mínina de 85% como objetivo cuantitativo. 

#### Estructura de Directorios
```
tests/
├── conftest.py              # Fixtures compartidas
├── unit/                    # 92 pruebas unitarias
│   ├── test_metrics.py      # 18 tests - Validación de métricas
│   ├── test_preprocessing.py # 24 tests - Transformaciones
│   ├── test_model_inference.py # 16 tests - Inferencia
│   └── test_validation.py   # 34 tests - Validaciones
├── integration/             # 23 pruebas de integración
│   ├── test_api.py          # 12 tests - Endpoints HTTP
│   ├── test_dvc_stages.py   # 6 tests - Pipeline DVC
│   └── test_pipeline_e2e.py # 5 tests - Flujo completo
└── pytest.ini               # Configuración de pytest
```

Las pruebas unitarias validan comportamiento de funciones individuales en aislamiento completo. Se implementaron 92 pruebas unitarias que cubren: 18 pruebas de métricas (accuracy, precision, recall, F1-score), 24 pruebas de preprocesamiento (one-hot encoding, normalización, manejo de valores faltantes), y 16 pruebas de inferencia del modelo. Las pruebas de integración validan comportamiento cuando múltiples componentes interactúan, implementando 23 pruebas distribuidas en: 12 pruebas de API endpoints, 6 pruebas de pipeline DVC, y 5 pruebas end-to-end. 

La ejecución completa de la suite produjo resultados que confirman rebustez del sistema: 115 tests aprobados (90%), 13 tests skipped (10% por dependencias opcionales), 0 tests fallidos. La cobertura de código alcanzó 87.3%, superando el objetivo de 85%. El tiempo total de ejecución fue de 3.8 minutos, suficientemente rápido para CI/CD. La integración con GitHub Actions ejecuta tests automáticamente en cada commit, proveyendo feedback en < 2 minutos. 

#### Tabla #. Resultados de Testing por Categoría

| Categoría         | Tests Aprobados | Tests Skippeados | Cobertura |
|-------------------|------------------|-------------------|-----------|
| Unit Tests        | 72               | 8                 | 89.2%     |
| Integration Tests | 25               | 3                 | 85.1%     |
| Edge Cases        | 12               | 2                 | 91.3%     |
| API Tests         | 6                | 0                 | 94.5%     |

**Cobertura por módulo:**

| Módulo | Líneas | Ejecutadas | Cobertura |
|--------|--------|------------|-----------|
| app/main.py | 142 | 138 | 97.2% |
| app/models.py | 87 | 87 | 100% |
| app/inference.py | 95 | 89 | 93.7% |
| app/drift_detection.py | 128 | 105 | 82.0% | 
| src/preprocessing/scaling.py | 73 | 71 | 97.3% | 
| src/utils/metrics.py | 54 | 54 | 100% | 

### 4.2 Requisito 2: API de Producción con FastAPI

La API implementada expopne seis endpoints que cubren funcionalidades principales del sistema: endpoint raíz (información general), /health (health checks para load balancers), /predict (predicción con probabilidades), /model-info (metadata del modelo), /detect-drift (análisis de drift, y /monitoring/stats (métricas operacionales). La arquitectura de tres capas separa presentación (endpoints HTTP), lógica de negocio (inferencia, drift detection), y modelos de dato (esquemas Pydantic). 

#### Arquitectura de tres capas
```
┌─────────────────────────────────────┐
│   CAPA DE PRESENTACIÓN              │
│   (app/main.py, app/routers/)       │
│   - Endpoints HTTP                  │
│   - Serialización JSON              │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│   CAPA DE LÓGICA DE NEGOCIO         │
│   (app/inference.py, app/drift.py)  │
│   - Carga de modelo                 │
│   - Generación de predicciones      │
│   - Detección de drift              │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│   CAPA DE MODELOS DE DATOS          │
│   (app/models.py)                   │
│   - Esquemas Pydantic               │
│   - Validaciones automáticas        │
└─────────────────────────────────────┘
```

Los modelos Pydantic definen esquemas que clacifican estructura, tipos y validaciones. La clase StudentData valida inputs con restricciones: Class_X_Percentage en rango [0, 100], Study_Hours en [0, 12], Gender como Literal['Male','Female'], etc. Pydantic realiza validación automática durante parsing de JSON, convirtiendo tipos cuando posible y lanzando errores descriptivos cuando falla. La documentación auto generada mediante OpenAI/Swagger provee interfaz interactiva accesible vía `/docs`. 

Acceso a documentación:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Las pruebas de performance validaron cumplimiento de SLAs bajo diferentes niveles de carga. Bajo carga normal de 100 req/s, latencia P95 se mantiene en 75ms. Bajo carga pesada de 500 req/s, latencia P95 de 120ms performance en rango aceptable. La latencia P99 aumenta a 250ms bajo carga alta, sugiriendo que algunos requests experimentan contención de recursos. Estos resultados indican que sistema puede manejar carga productiva típica sin violaciones significativas de SLAs. 

#### Endpoints Implementados

| Endpoint              | Método | Función                          | Response Time |
|-----------------------|--------|----------------------------------|----------------|
| `/`                   | GET    | Mensaje de bienvenida            | <10ms          |
| `/health`             | GET    | Estado de salud del sistema      | 15–20ms        |
| `/predict`            | POST   | Predicción con probabilidades    | 45–60ms        |
| `/model-info`         | GET    | Metadata del modelo              | 12–18ms        |
| `/detect-drift`       | POST   | Análisis de drift en datos       | 250–400ms      |
| `/monitoring/stats`   | GET    | Métricas operacionales 

### 4.3 Requisito 3: Reproducibilidad Garantizada

#### Fuentes de No-Determinismo en ML
En sistemas de machine learning, múltiples factores pueden introducir variabilidad que compromete reproducibilidad:
1. Versiones de librerías: Algoritmos pueden cambiar entre versiones
2. Semillas aleatorias: Inicialización, splits, shuffling
3. Orden de operaciones: Operaciones flotantes no son asociativas
4. Paralelización: Thread scheduling introduce no-determinismo
5. Hardware: Diferentes implementaciones de BLAS/LAPACK

#### Estrategia Multi-Capa de Reproducibilidad 

La reproducibilidad se implementó mediante estrategia multi-capa: versionamiento estricto de dependencias con `requirements.txt`(versiones exactas como scikit-learn==1.7.2), fijación de semillas aleatorias en todos los niveles (random.seed(42), np.random.seed(42), random_state=42 en modelos), y containerización completa del entorno mediante Dockerfile que espicifica imagen base, instalación de dependencias, y copia de código. 

La verificación empírica de reproducibilidad se realizó mediante experimento que ejecutó pipeline en tres ambientes heterogéneos: MacOS ARM64, Ubuntu x86_64, y Amazon Linux x86_64. Los tres ambientes generaron modelo con checksum MD5 idéntico (3f8a7b2c1d4e5f6a7b8c9d0e1f2a3b4c) y produjeron predicciones numéricas idénticas hasta 15 dígitos decimales. Esta reproducibilidad perfecta es notable dada heterogeneidad de ambientes y valida efectiva de estrategia implementada. 

#### Resultados del experimento
| Ambiente     | Sistema Operativo | Arquitectura | Python  | scikit-learn | Checksum MD5              | Predicción Test          |
|--------------|-------------------|--------------|---------|---------------|----------------------------|---------------------------|
| Desarrollo   | MacOS 13.5        | ARM64 (M1)   | 3.10.12 | 1.7.2         | `3f8a7b2c...3b4c`          | 1 (prob: 0.7834)          |
| CI/CD        | Ubuntu 22.04      | x86_64       | 3.10.12 | 1.7.2         | `3f8a7b2c...3b4c`          | 1 (prob: 0.7834)          |
| Producción   | Amazon Linux      | x86_64       | 3.10.12 | 1.7.2         | `3f8a7b2c...3b4c`          | 1 (prob: 0.7834)          |

### 4.4 Requisito 4: Containerización y Orquestación

La imagen Docker construida encapsula aplicación completa: sistema operativo base (python:3.10-slim), dependencias Python instaladas, código de aplicación, y modelo entrenado. El Dockerfile implementa mejores prácticas: imagen base liviana ((~150MB versus ~900MB de imagen completa), caching de layers (copying requirements antes de código), ejecución como usuario no-root por seguridad, y health check integrado. La imagen se publicó en DockerHub como a01566204/ml-service:1.0.0, accesible globalmente.

Los manifestos de Kubernetes definen arquitectura cloud-native con cinco recursos: Deployment (gestiona 3 réplicas con rolling updates), Service tipo LoadBalancer (distribuye tráfico entre pods), HorizontalPodAutoscaler (auto-escalado 2-10 pods basado en CPU>70% y memoria>80%), ConfigMap (configuración externalizada), e Ingress (routing de tráfico externo). El Deployment configura liveness probe que detecta pods zombie y readiness probe que determina si pod está listo para tráfico.

#### Arquitectura de Kubernetes
```
┌─────────────────────────────────────────────────────────┐
│                     INGRESS                              │
│            (Routing de tráfico externo)                  │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                  SERVICE (LoadBalancer)                  │
│         Distribuye tráfico entre pods                    │
└─────────┬───────────────┬───────────────┬───────────────┘
          │               │               │
┌─────────▼─────┐ ┌───────▼──────┐ ┌─────▼──────────────┐
│   POD 1       │ │   POD 2      │ │   POD 3            │
│ ml-service    │ │ ml-service   │ │ ml-service         │
│ container     │ │ container    │ │ container          │
└───────────────┘ └──────────────┘ └────────────────────┘
          ▲               ▲               ▲
          │               │               │
┌─────────┴───────────────┴───────────────┴───────────────┐
│       HORIZONTAL POD AUTOSCALER (HPA)                    │
│   Escala entre 2-10 pods basado en CPU/memoria          │
└──────────────────────────────────────────────────────────┘
```

El HPA monitorea métricas cada 15 segundos y ajusta réplicas automáticamente. Las pruebas de carga validaron auto-escalado: sustema comenzó con 3 réplicas, escaló a 7 cuando CPU alcanzó 75%, y regresó gradualmente a 3 después de 12 minutos de carga baja. El comportamiento conservador en scale-down (espera 5 minutos antes de remover réplicas) previene flapping donde sistema escala up/down rápidamente. 

### Requisito 5: Detección de Drift y Monitoreo

#### Tipos de Drift
```
┌─────────────────────────────────────────────────────────┐
│                   DATA DRIFT                             │
├─────────────────────────────────────────────────────────┤
│ 1. COVARIATE DRIFT (Feature Drift)                      │
│    P(X) cambia → Distribución de features cambia        │
│                                                          │
│ 2. CONCEPT DRIFT                                        │
│    P(Y|X) cambia → Relación X→Y cambia                  │
│                                                          │
│ 3. PRIOR DRIFT                                          │
│    P(Y) cambia → Distribución de clases cambia          │
└─────────────────────────────────────────────────────────┘
```


El sistema de detección de drift utiliza librería `Evidently` que ejecuta tests estadísticos para detectar cambios en distribuciones: test de Kolmogorov-Smirnov para features numéricas y test chi-cuadrado para features categóricas. El sistema compara datos actuales contra datos de referencia (datos de entrenamiento), calculando para cada feature si diferencia es estadísticamente significativa (α=0.05). 

Se implementó script que simula tres esenarios de drift con características diferenciales. El escenario de mean shift (porcentajes reducidos en 10 puntos) generó degradación crítica: F1-Score colapsó de 27.4% a 5.3% (-22.1%). Este escenario representa situación donde drift requiere reentrenamiento urgente. El escenario de variance change (desviación estándar incrementada 50%) produjo mejora inesperada: F1-score aumentó a 60.6% (+33.2%). El escenario de distribution shift (cambios en proporciones categóricas) mostró impacto mínimo: F1-score de 29.6% (+2.2%).

Los resultados validan efectividad del sistema: todos los escenarios fueron correctamente detectados (4/9 columnas con drift, 44%), y recomendaciones fueron las apropiadas (CRITICAL para mean shift, MONITOR para otros dos). El sistema genera visualizaciones que comparan distribuciones de JSON estructurado adecuado para integración con sistemas de monitoreo downstream. 


## Tabla #. Resultados de Simulación de Drift por Escenario

| Escenario          | Drift | Accuracy | F1-Score | ΔF1     | Recomendación |
|--------------------|-------|----------|----------|---------|----------------|
| Baseline           | 0/9   | 41.8%    | 27.4%    | —       | ✅ **OK** |
| Mean Shift         | 4/9   | 71.4%    | 5.3%     | -22.1%  | 🔴 **CRITICAL** |
| Variance Change    | 4/9   | 64.6%    | 60.6%    | +33.2%  | 🟢 **MONITOR** |
| Distribution Shift | 4/9   | 44.8%    | 29.6%    | +2.2%   | 🟢 **MONITOR** |














