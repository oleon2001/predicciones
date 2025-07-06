# HERRAMIENTA DE MIGRACIÓN DE CÓDIGO LEGACY
"""
Herramienta para migrar código del monolito a la nueva arquitectura modular
Analiza dependencias, extrae funcionalidad y genera código refactorizado
"""

import ast
import inspect
import re
from typing import Dict, List, Set, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import logging

from config.system_config import SystemConfig
from core.interfaces import *
from core.dependency_container import DependencyContainer

logger = logging.getLogger(__name__)

@dataclass
class FunctionAnalysis:
    """Análisis de función"""
    name: str
    module: str
    dependencies: List[str]
    external_calls: List[str]
    complexity: int
    lines_of_code: int
    can_be_extracted: bool
    suggested_interface: Optional[str] = None

@dataclass
class ClassAnalysis:
    """Análisis de clase"""
    name: str
    module: str
    methods: List[FunctionAnalysis]
    attributes: List[str]
    inheritance: List[str]
    dependencies: List[str]
    coupling_score: float
    cohesion_score: float
    suggested_refactoring: List[str]

@dataclass
class ModuleAnalysis:
    """Análisis de módulo"""
    name: str
    path: str
    classes: List[ClassAnalysis]
    functions: List[FunctionAnalysis]
    imports: List[str]
    lines_of_code: int
    complexity_score: float
    maintainability_index: float
    suggested_splits: List[str]

class CodeAnalyzer:
    """Analizador de código para migración"""
    
    def __init__(self):
        self.ast_parser = ASTParser()
        self.dependency_analyzer = DependencyAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer()
    
    def analyze_file(self, file_path: str) -> ModuleAnalysis:
        """Analiza archivo de código"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parsear AST
            tree = ast.parse(content)
            
            # Analizar componentes
            classes = self._analyze_classes(tree, file_path)
            functions = self._analyze_functions(tree, file_path)
            imports = self._extract_imports(tree)
            
            # Calcular métricas
            lines_of_code = len(content.splitlines())
            complexity_score = self.complexity_analyzer.calculate_complexity(tree)
            maintainability_index = self._calculate_maintainability(
                lines_of_code, complexity_score
            )
            
            return ModuleAnalysis(
                name=Path(file_path).stem,
                path=file_path,
                classes=classes,
                functions=functions,
                imports=imports,
                lines_of_code=lines_of_code,
                complexity_score=complexity_score,
                maintainability_index=maintainability_index,
                suggested_splits=self._suggest_module_splits(classes, functions)
            )
            
        except Exception as e:
            logger.error(f"Error analizando archivo {file_path}: {e}")
            raise
    
    def _analyze_classes(self, tree: ast.AST, file_path: str) -> List[ClassAnalysis]:
        """Analiza clases en el AST"""
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = self._analyze_class_methods(node, file_path)
                attributes = self._extract_class_attributes(node)
                inheritance = [base.id for base in node.bases if isinstance(base, ast.Name)]
                
                # Calcular métricas
                coupling_score = self._calculate_coupling(node)
                cohesion_score = self._calculate_cohesion(node)
                
                class_analysis = ClassAnalysis(
                    name=node.name,
                    module=file_path,
                    methods=methods,
                    attributes=attributes,
                    inheritance=inheritance,
                    dependencies=self._extract_class_dependencies(node),
                    coupling_score=coupling_score,
                    cohesion_score=cohesion_score,
                    suggested_refactoring=self._suggest_class_refactoring(node)
                )
                
                classes.append(class_analysis)
        
        return classes
    
    def _analyze_functions(self, tree: ast.AST, file_path: str) -> List[FunctionAnalysis]:
        """Analiza funciones en el AST"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Solo funciones de nivel superior (no métodos)
                parent = getattr(node, 'parent', None)
                if parent and isinstance(parent, ast.ClassDef):
                    continue
                
                dependencies = self._extract_function_dependencies(node)
                external_calls = self._extract_external_calls(node)
                complexity = self.complexity_analyzer.calculate_function_complexity(node)
                lines_of_code = len(node.body)
                
                function_analysis = FunctionAnalysis(
                    name=node.name,
                    module=file_path,
                    dependencies=dependencies,
                    external_calls=external_calls,
                    complexity=complexity,
                    lines_of_code=lines_of_code,
                    can_be_extracted=self._can_function_be_extracted(node),
                    suggested_interface=self._suggest_interface_for_function(node)
                )
                
                functions.append(function_analysis)
        
        return functions
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extrae imports del AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return imports
    
    def _calculate_maintainability(self, loc: int, complexity: float) -> float:
        """Calcula índice de mantenibilidad"""
        # Fórmula simplificada del índice de mantenibilidad
        if loc == 0:
            return 100.0
        
        maintainability = max(0, 171 - 5.2 * complexity - 0.23 * loc)
        return min(100, maintainability)
    
    def _suggest_module_splits(self, classes: List[ClassAnalysis], 
                              functions: List[FunctionAnalysis]) -> List[str]:
        """Sugiere divisiones del módulo"""
        suggestions = []
        
        # Agrupar por responsabilidad
        responsibility_groups = defaultdict(list)
        
        for cls in classes:
            # Determinar responsabilidad por nombre y métodos
            responsibility = self._determine_responsibility(cls.name, [m.name for m in cls.methods])
            responsibility_groups[responsibility].append(f"class {cls.name}")
        
        for func in functions:
            responsibility = self._determine_responsibility(func.name, [])
            responsibility_groups[responsibility].append(f"function {func.name}")
        
        # Generar sugerencias
        for responsibility, items in responsibility_groups.items():
            if len(items) > 1:
                suggestions.append(f"Crear módulo '{responsibility}' con: {', '.join(items)}")
        
        return suggestions
    
    def _determine_responsibility(self, name: str, methods: List[str]) -> str:
        """Determina responsabilidad basada en nombre y métodos"""
        name_lower = name.lower()
        all_names = [name_lower] + [m.lower() for m in methods]
        
        # Patrones de responsabilidad
        patterns = {
            'data': ['data', 'fetch', 'get', 'obtain', 'download', 'load'],
            'analysis': ['analyze', 'calculate', 'compute', 'process', 'indicator'],
            'prediction': ['predict', 'forecast', 'model', 'train', 'ml'],
            'visualization': ['plot', 'graph', 'chart', 'visual', 'render'],
            'risk': ['risk', 'var', 'volatility', 'drawdown'],
            'sentiment': ['sentiment', 'news', 'social', 'emotion'],
            'macro': ['macro', 'economic', 'fed', 'inflation'],
            'backtest': ['backtest', 'test', 'simulate', 'validate']
        }
        
        # Contar coincidencias
        scores = defaultdict(int)
        for responsibility, keywords in patterns.items():
            for keyword in keywords:
                for name in all_names:
                    if keyword in name:
                        scores[responsibility] += 1
        
        # Retornar responsabilidad con mayor score
        if scores:
            return max(scores, key=scores.get)
        else:
            return 'utility'

class MigrationPlan:
    """Plan de migración"""
    
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []
        self.dependencies: Dict[str, List[str]] = {}
        self.estimated_effort: Dict[str, int] = {}
    
    def add_step(self, step_type: str, description: str, 
                 dependencies: List[str] = None, effort_hours: int = 1):
        """Añade paso al plan"""
        step = {
            'type': step_type,
            'description': description,
            'dependencies': dependencies or [],
            'effort_hours': effort_hours,
            'status': 'pending'
        }
        self.steps.append(step)
    
    def get_execution_order(self) -> List[Dict[str, Any]]:
        """Obtiene orden de ejecución respetando dependencias"""
        # Implementación simplificada - en producción usaríamos topological sort
        return sorted(self.steps, key=lambda x: len(x['dependencies']))

class MigrationExecutor:
    """Ejecutor de migración"""
    
    def __init__(self, target_architecture: DependencyContainer):
        self.target_architecture = target_architecture
        self.code_generator = CodeGenerator()
        self.file_organizer = FileOrganizer()
    
    def execute_migration(self, analysis: ModuleAnalysis, 
                         plan: MigrationPlan) -> Dict[str, Any]:
        """Ejecuta migración"""
        results = {
            'files_created': [],
            'files_modified': [],
            'interfaces_created': [],
            'tests_created': [],
            'errors': []
        }
        
        try:
            # Ejecutar pasos del plan
            for step in plan.get_execution_order():
                if step['type'] == 'extract_interface':
                    self._extract_interface(step, analysis, results)
                elif step['type'] == 'create_implementation':
                    self._create_implementation(step, analysis, results)
                elif step['type'] == 'generate_tests':
                    self._generate_tests(step, analysis, results)
                elif step['type'] == 'update_dependencies':
                    self._update_dependencies(step, analysis, results)
                
                step['status'] = 'completed'
        
        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"Error ejecutando migración: {e}")
        
        return results
    
    def _extract_interface(self, step: Dict[str, Any], 
                          analysis: ModuleAnalysis, 
                          results: Dict[str, Any]):
        """Extrae interfaz"""
        # Implementación específica para extraer interfaces
        pass
    
    def _create_implementation(self, step: Dict[str, Any], 
                             analysis: ModuleAnalysis, 
                             results: Dict[str, Any]):
        """Crea implementación"""
        # Implementación específica para crear implementaciones
        pass

class CodeGenerator:
    """Generador de código"""
    
    def generate_interface(self, class_analysis: ClassAnalysis) -> str:
        """Genera interfaz a partir de análisis de clase"""
        interface_name = f"I{class_analysis.name}"
        
        # Generar métodos abstractos
        methods = []
        for method in class_analysis.methods:
            if not method.name.startswith('_'):  # Solo métodos públicos
                methods.append(f"    @abstractmethod\n    def {method.name}(self, ...):\n        pass")
        
        interface_code = f"""
from abc import ABC, abstractmethod

class {interface_name}(ABC):
    \"\"\"{class_analysis.name} interface\"\"\"
    
{chr(10).join(methods)}
"""
        return interface_code
    
    def generate_implementation(self, class_analysis: ClassAnalysis, 
                              interface_name: str) -> str:
        """Genera implementación que usa interfaz"""
        implementation_code = f"""
from {interface_name.lower()} import {interface_name}

class {class_analysis.name}({interface_name}):
    \"\"\"Implementation of {interface_name}\"\"\"
    
    def __init__(self, config: SystemConfig):
        self.config = config
        # TODO: Migrar inicialización
    
    # TODO: Migrar métodos de la clase original
"""
        return implementation_code

class FileOrganizer:
    """Organizador de archivos"""
    
    def __init__(self):
        self.structure = {
            'core': 'interfaces and base classes',
            'models': 'ML models and algorithms',
            'analysis': 'analysis components',
            'backtesting': 'backtesting framework',
            'utils': 'utility functions',
            'tests': 'test files'
        }
    
    def organize_file(self, file_content: str, file_type: str) -> str:
        """Organiza archivo en estructura correcta"""
        if file_type == 'interface':
            return f"core/{file_content.lower().replace('i', '', 1)}.py"
        elif file_type == 'model':
            return f"models/{file_content.lower()}.py"
        elif file_type == 'analysis':
            return f"analysis/{file_content.lower()}.py"
        elif file_type == 'test':
            return f"tests/test_{file_content.lower()}.py"
        else:
            return f"utils/{file_content.lower()}.py"

class MigrationTool:
    """Herramienta principal de migración"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.analyzer = CodeAnalyzer()
        self.container = DependencyContainer()
        self.executor = MigrationExecutor(self.container)
    
    def migrate_file(self, file_path: str, 
                    target_architecture: str = "modular") -> Dict[str, Any]:
        """Migra archivo completo"""
        logger.info(f"Iniciando migración de {file_path}")
        
        # Analizar archivo
        analysis = self.analyzer.analyze_file(file_path)
        
        # Generar plan de migración
        plan = self._generate_migration_plan(analysis)
        
        # Ejecutar migración
        results = self.executor.execute_migration(analysis, plan)
        
        # Generar reporte
        report = self._generate_migration_report(analysis, plan, results)
        
        return {
            'analysis': analysis,
            'plan': plan,
            'results': results,
            'report': report
        }
    
    def _generate_migration_plan(self, analysis: ModuleAnalysis) -> MigrationPlan:
        """Genera plan de migración"""
        plan = MigrationPlan()
        
        # Extraer interfaces para clases complejas
        for class_analysis in analysis.classes:
            if class_analysis.coupling_score > 0.7:
                plan.add_step(
                    'extract_interface',
                    f"Extraer interfaz para {class_analysis.name}",
                    effort_hours=2
                )
                
                plan.add_step(
                    'create_implementation',
                    f"Crear implementación para {class_analysis.name}",
                    dependencies=[f"extract_interface_{class_analysis.name}"],
                    effort_hours=4
                )
        
        # Migrar funciones independientes
        for func in analysis.functions:
            if func.can_be_extracted:
                plan.add_step(
                    'extract_function',
                    f"Migrar función {func.name}",
                    effort_hours=1
                )
        
        # Generar tests
        plan.add_step(
            'generate_tests',
            "Generar tests para componentes migrados",
            dependencies=['create_implementation'],
            effort_hours=3
        )
        
        return plan
    
    def _generate_migration_report(self, analysis: ModuleAnalysis, 
                                 plan: MigrationPlan, 
                                 results: Dict[str, Any]) -> str:
        """Genera reporte de migración"""
        report = f"""
# REPORTE DE MIGRACIÓN

## Archivo Original
- **Nombre**: {analysis.name}
- **Líneas de código**: {analysis.lines_of_code}
- **Complejidad**: {analysis.complexity_score:.2f}
- **Mantenibilidad**: {analysis.maintainability_index:.2f}

## Componentes Analizados
- **Clases**: {len(analysis.classes)}
- **Funciones**: {len(analysis.functions)}
- **Imports**: {len(analysis.imports)}

## Plan de Migración
- **Pasos totales**: {len(plan.steps)}
- **Esfuerzo estimado**: {sum(s['effort_hours'] for s in plan.steps)} horas

## Resultados
- **Archivos creados**: {len(results['files_created'])}
- **Archivos modificados**: {len(results['files_modified'])}
- **Interfaces creadas**: {len(results['interfaces_created'])}
- **Tests creados**: {len(results['tests_created'])}
- **Errores**: {len(results['errors'])}

## Recomendaciones
{chr(10).join(analysis.suggested_splits)}
"""
        return report

# Clases auxiliares para análisis AST
class ASTParser:
    """Parser AST especializado"""
    
    def parse_with_metadata(self, content: str) -> ast.AST:
        """Parsea con metadata adicional"""
        tree = ast.parse(content)
        
        # Añadir metadata
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
        
        return tree

class DependencyAnalyzer:
    """Analizador de dependencias"""
    
    def analyze_dependencies(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Analiza dependencias en AST"""
        dependencies = defaultdict(list)
        
        # Implementar análisis de dependencias
        # TODO: Implementar análisis completo
        
        return dict(dependencies)

class ComplexityAnalyzer:
    """Analizador de complejidad"""
    
    def calculate_complexity(self, tree: ast.AST) -> float:
        """Calcula complejidad ciclomática"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
        
        return complexity
    
    def calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calcula complejidad de función específica"""
        complexity = 1
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
        
        return complexity

def main():
    """Función principal para ejecutar migración"""
    config = SystemConfig()
    tool = MigrationTool(config)
    
    # Migrar archivo principal
    results = tool.migrate_file('prediccion_avanzada.py')
    
    print("=== REPORTE DE MIGRACIÓN ===")
    print(results['report'])
    
    return results

if __name__ == "__main__":
    main() 