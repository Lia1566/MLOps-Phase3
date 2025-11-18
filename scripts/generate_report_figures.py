"""
Generación de Figuras para Reporte Final
Crea todas las visualizaciones necesarias para el documento académico
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Configuración de estilo académico
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# Crear directorio para figuras del reporte
output_dir = Path('reports/figures/final_report')
output_dir.mkdir(parents=True, exist_ok=True)

# FIGURA 1: Arquitectura del Sistema MLOps
def create_architecture_diagram():
    """Diagrama de arquitectura del sistema"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Definir capas
    layers = {
        'Usuario': (0.5, 0.9, 0.3, 0.08),
        'Kubernetes Ingress': (0.5, 0.75, 0.4, 0.08),
        'LoadBalancer': (0.5, 0.6, 0.4, 0.08),
        'Pods (3 réplicas)': (0.5, 0.45, 0.6, 0.08),
        'HPA Auto-scaling': (0.5, 0.3, 0.4, 0.08),
        'Monitoreo (Evidently)': (0.5, 0.15, 0.4, 0.08),
    }
    
    for name, (x, y, w, h) in layers.items():
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, 
                            facecolor='lightblue', 
                            edgecolor='black', 
                            linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, name, ha='center', va='center', 
               fontsize=11, fontweight='bold')
    
    # Agregar flechas
    for i in range(len(layers)-1):
        y_positions = [0.9, 0.75, 0.6, 0.45, 0.3, 0.15]
        ax.arrow(0.5, y_positions[i]-0.04, 0, -0.07, 
                head_width=0.03, head_length=0.02, 
                fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Figura 1. Arquitectura del Sistema MLOps de Producción', 
                fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_arquitectura.png', bbox_inches='tight')
    print("✅ Figura 1 generada: Arquitectura del sistema")
    plt.close()

# FIGURA 2: Resultados de Testing
def create_testing_results():
    """Gráfica de resultados de testing por categoría"""
    categories = ['Unit Tests\n(Unitarias)', 'Integration Tests\n(Integración)', 
                  'Edge Cases\n(Casos Límite)', 'API Tests\n(API)']
    passed = [72, 25, 12, 6]
    skipped = [8, 3, 2, 0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, passed, width, label='Aprobadas', 
                   color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, skipped, width, label='Omitidas', 
                   color='#f39c12', alpha=0.8)
    
    ax.set_xlabel('Categoría de Pruebas', fontsize=11, fontweight='bold')
    ax.set_ylabel('Número de Pruebas', fontsize=11, fontweight='bold')
    ax.set_title('Figura 2. Distribución de Resultados de Pruebas Automatizadas', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Agregar valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_testing_results.png', bbox_inches='tight')
    print("✅ Figura 2 generada: Resultados de testing")
    plt.close()

# FIGURA 3: Performance del Modelo
def create_model_performance():
    """Comparación de métricas del modelo"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    baseline = [0.418, 0.957, 0.160, 0.274]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(metrics, baseline, color=['#3498db', '#e74c3c', '#9b59b6', '#2ecc71'], 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Figura 3. Métricas de Desempeño del Modelo Baseline (Datos de Referencia)', 
                fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Agregar línea de referencia
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Umbral 50%')
    ax.legend(loc='upper right')
    
    # Valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.1%}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_model_performance.png', bbox_inches='tight')
    print("✅ Figura 3 generada: Performance del modelo")
    plt.close()

# FIGURA 4: Comparación Drift Scenarios
def create_drift_comparison():
    """Comparación de escenarios de drift"""
    scenarios = ['Baseline\n(Referencia)', 'Mean Shift\n(Cambio Media)', 
                 'Variance\nChange', 'Distribution\nShift']
    accuracy = [0.418, 0.714, 0.646, 0.448]
    f1_score = [0.274, 0.053, 0.606, 0.296]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Accuracy
    bars1 = ax1.bar(scenarios, accuracy, color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Accuracy por Escenario', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0.418, color='green', linestyle='--', alpha=0.5, label='Baseline')
    ax1.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 2: F1-Score
    bars2 = ax2.bar(scenarios, f1_score, color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax2.set_title('(b) F1-Score por Escenario', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=0.274, color='green', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend()
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Figura 4. Comparación de Métricas bajo Diferentes Escenarios de Drift', 
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_drift_comparison.png', bbox_inches='tight')
    print("✅ Figura 4 generada: Comparación de drift")
    plt.close()

# FIGURA 5: Timeline de Fases
def create_project_timeline():
    """Timeline del proyecto por fases"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    phases = [
        ('Fase 1:\nExploración\ny EDA', 1, 2),
        ('Fase 2:\nPipeline\ny Versioning', 3, 2),
        ('Fase 3:\nTesting\ny API', 5, 2),
        ('Fase 3:\nDocker\ny K8s', 7, 2),
        ('Fase 3:\nDrift\nDetection', 9, 2)
    ]
    
    y_pos = 0.5
    for i, (name, start, duration) in enumerate(phases):
        color = plt.cm.viridis(i / len(phases))
        ax.barh(y_pos, duration, left=start, height=0.3, 
               color=color, alpha=0.8, edgecolor='black', linewidth=2)
        ax.text(start + duration/2, y_pos, name, 
               ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Semanas del Proyecto', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title('Figura 5. Cronograma de Desarrollo del Proyecto MLOps', 
                fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_timeline.png', bbox_inches='tight')
    print("✅ Figura 5 generada: Timeline del proyecto")
    plt.close()

# FIGURA 6: Latencia vs Carga
def create_latency_graph():
    """Gráfica de latencia vs carga del sistema"""
    requests_per_sec = np.array([10, 50, 100, 200, 500, 1000])
    latency_p50 = np.array([45, 52, 60, 75, 120, 250])
    latency_p95 = np.array([60, 75, 95, 140, 280, 520])
    latency_p99 = np.array([80, 105, 135, 210, 450, 890])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(requests_per_sec, latency_p50, marker='o', linewidth=2, 
           label='P50 (Mediana)', color='#2ecc71')
    ax.plot(requests_per_sec, latency_p95, marker='s', linewidth=2, 
           label='P95', color='#f39c12')
    ax.plot(requests_per_sec, latency_p99, marker='^', linewidth=2, 
           label='P99', color='#e74c3c')
    
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, 
              label='SLA Objetivo (100ms)')
    
    ax.set_xlabel('Peticiones por Segundo (req/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latencia (ms)', fontsize=11, fontweight='bold')
    ax.set_title('Figura 6. Latencia del Sistema bajo Diferentes Cargas', 
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_latency.png', bbox_inches='tight')
    print("✅ Figura 6 generada: Latencia vs carga")
    plt.close()

# Ejecutar todas las funciones
if __name__ == "__main__":
    print("\n🎨 Generando figuras para reporte final...\n")
    create_architecture_diagram()
    create_testing_results()
    create_model_performance()
    create_drift_comparison()
    create_project_timeline()
    create_latency_graph()
    print("\n✅ Todas las figuras generadas exitosamente!")
    print(f"📁 Ubicación: {output_dir}")
