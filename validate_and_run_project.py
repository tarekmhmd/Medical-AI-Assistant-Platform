"""
Project Validation and Execution Script
========================================
Scans, validates dependencies, and executes project scripts safely.
"""

import os
import json
import sys
import subprocess
import importlib.util
from datetime import datetime
from collections import defaultdict
import re

# Project root
PROJECT_ROOT = r"D:\project 2"
VENV_PYTHON = r"D:\project 2\venv\Scripts\python.exe"

# Output
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "project_run_report.json")

class ProjectValidator:
    def __init__(self):
        self.project_scripts = []
        self.model_files = []
        self.dataset_files = []
        self.dependencies = {}
        self.execution_results = []
        self.warnings = []
        self.errors = []
        
    def scan_project(self):
        """Scan project for all relevant files."""
        print("=" * 60)
        print("PROJECT SCAN")
        print("=" * 60)
        
        # Scan Python scripts in project root
        root_scripts = []
        for f in os.listdir(PROJECT_ROOT):
            if f.endswith('.py'):
                root_scripts.append(os.path.join(PROJECT_ROOT, f))
        
        # Scan backend scripts
        backend_scripts = []
        backend_path = os.path.join(PROJECT_ROOT, 'backend')
        if os.path.exists(backend_path):
            for f in os.listdir(backend_path):
                if f.endswith('.py'):
                    backend_scripts.append(os.path.join(backend_path, f))
        
        # Scan models
        models_scripts = []
        models_path = os.path.join(PROJECT_ROOT, 'models')
        if os.path.exists(models_path):
            for f in os.listdir(models_path):
                if f.endswith('.py'):
                    models_scripts.append(os.path.join(models_path, f))
        
        # Combine all scripts
        self.project_scripts = root_scripts + backend_scripts + models_scripts
        
        # Scan model files
        checkpoints_path = os.path.join(PROJECT_ROOT, 'checkpoints')
        if os.path.exists(checkpoints_path):
            for f in os.listdir(checkpoints_path):
                if f.endswith(('.pth', '.pt', '.h5')):
                    self.model_files.append(os.path.join(checkpoints_path, f))
        
        models_pretrained = os.path.join(PROJECT_ROOT, 'models_pretrained')
        if os.path.exists(models_pretrained):
            for f in os.listdir(models_pretrained):
                if f.endswith(('.pth', '.pt', '.h5')):
                    self.model_files.append(os.path.join(models_pretrained, f))
        
        # Scan datasets (JSON and CSV in data folder)
        data_path = os.path.join(PROJECT_ROOT, 'data')
        if os.path.exists(data_path):
            for root, dirs, files in os.walk(data_path):
                # Skip large directories
                if 'venv' in root or '__pycache__' in root:
                    continue
                for f in files:
                    if f.endswith(('.json', '.csv')):
                        full_path = os.path.join(root, f)
                        # Skip very large files for reporting
                        try:
                            size = os.path.getsize(full_path)
                            if size < 100 * 1024 * 1024:  # < 100MB
                                self.dataset_files.append(full_path)
                        except:
                            pass
        
        print(f"\nFound:")
        print(f"  Python scripts: {len(self.project_scripts)}")
        print(f"  Model files: {len(self.model_files)}")
        print(f"  Dataset files: {len(self.dataset_files)}")
        
        return True
    
    def analyze_script_dependencies(self, script_path):
        """Analyze a script's imports and dependencies."""
        deps = {
            'imports': [],
            'data_dependencies': [],
            'model_dependencies': [],
            'output_files': []
        }
        
        try:
            with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Find imports
            import_pattern = r'^(?:from|import)\s+([a-zA-Z0-9_\.]+)'
            for match in re.finditer(import_pattern, content, re.MULTILINE):
                deps['imports'].append(match.group(1).split('.')[0])
            
            # Find file references
            file_patterns = [
                r'["\']([^"\']+\.(?:json|csv|pth|pt|h5|txt))["\']',
                r'["\']([^"\']+\.(?:jpg|png|wav|mp3))["\']',
            ]
            for pattern in file_patterns:
                for match in re.finditer(pattern, content):
                    deps['data_dependencies'].append(match.group(1))
            
            # Find output file writes
            output_patterns = [
                r'(?:open|save|dump)\s*\(\s*["\']([^"\']+\.(?:json|csv|pth|pt|h5|txt))["\']',
            ]
            for pattern in output_patterns:
                for match in re.finditer(pattern, content):
                    deps['output_files'].append(match.group(1))
            
            deps['imports'] = list(set(deps['imports']))
            deps['data_dependencies'] = list(set(deps['data_dependencies']))
            deps['model_dependencies'] = list(set(deps['model_dependencies']))
            deps['output_files'] = list(set(deps['output_files']))
            
        except Exception as e:
            self.warnings.append(f"Could not analyze {script_path}: {str(e)}")
        
        return deps
    
    def verify_dependencies(self):
        """Verify all dependencies between scripts and files."""
        print("\n" + "=" * 60)
        print("DEPENDENCY VERIFICATION")
        print("=" * 60)
        
        for script in self.project_scripts:
            deps = self.analyze_script_dependencies(script)
            script_name = os.path.basename(script)
            self.dependencies[script_name] = deps
            
            # Check for missing data files
            for data_dep in deps['data_dependencies']:
                # Check if it's an absolute path or relative
                if not os.path.isabs(data_dep):
                    # Check relative to script location
                    script_dir = os.path.dirname(script)
                    full_path = os.path.join(script_dir, data_dep)
                    if not os.path.exists(full_path):
                        # Check relative to project root
                        full_path = os.path.join(PROJECT_ROOT, data_dep)
                
                if 'data_dep' in locals() and not os.path.exists(full_path if 'full_path' in locals() else data_dep):
                    self.warnings.append(f"{script_name}: Missing data file - {data_dep}")
        
        print(f"\nAnalyzed {len(self.dependencies)} scripts")
        print(f"Found {len(self.warnings)} warnings")
        
        return True
    
    def categorize_scripts(self):
        """Categorize scripts by their function."""
        categories = {
            'setup': [],
            'data_preparation': [],
            'training': [],
            'evaluation': [],
            'viewing': [],
            'utility': [],
            'backend': [],
            'models': []
        }
        
        for script_path in self.project_scripts:
            script_name = os.path.basename(script_path).lower()
            
            if 'setup' in script_name or 'install' in script_name:
                categories['setup'].append(script_path)
            elif 'prepare' in script_name or 'download' in script_name or 'organize' in script_name:
                categories['data_preparation'].append(script_path)
            elif 'train' in script_name or 'finetune' in script_name or 'merge' in script_name:
                categories['training'].append(script_path)
            elif 'evaluate' in script_name or 'test' in script_name:
                categories['evaluation'].append(script_path)
            elif 'view' in script_name:
                categories['viewing'].append(script_path)
            elif 'backend' in script_path.lower():
                categories['backend'].append(script_path)
            elif 'models' in script_path.lower() and 'models\\' in script_path.replace('/', '\\'):
                categories['models'].append(script_path)
            else:
                categories['utility'].append(script_path)
        
        return categories
    
    def get_execution_order(self):
        """Determine safe execution order based on dependencies."""
        categories = self.categorize_scripts()
        
        order = []
        
        # 1. Setup scripts (usually don't need to run automatically)
        # order.extend(categories['setup'])
        
        # 2. Data preparation scripts
        order.extend(categories['data_preparation'])
        
        # 3. Training scripts
        order.extend(categories['training'])
        
        # 4. Evaluation scripts
        order.extend(categories['evaluation'])
        
        # 5. Utility scripts (optional)
        # order.extend(categories['utility'])
        
        return order
    
    def execute_script(self, script_path, timeout=120):
        """Execute a single script safely."""
        script_name = os.path.basename(script_path)
        result = {
            'script': script_name,
            'path': script_path,
            'status': 'pending',
            'output': '',
            'error': '',
            'duration': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n  Executing: {script_name}...")
        
        try:
            start_time = datetime.now()
            
            # Run with virtual environment Python
            process = subprocess.run(
                [VENV_PYTHON, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.dirname(script_path) or PROJECT_ROOT
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result['duration'] = duration
            result['output'] = process.stdout[-2000:] if len(process.stdout) > 2000 else process.stdout
            result['error'] = process.stderr[-1000:] if len(process.stderr) > 1000 else process.stderr
            
            if process.returncode == 0:
                result['status'] = 'success'
                print(f"    ✓ Success ({duration:.1f}s)")
            else:
                result['status'] = 'failed'
                print(f"    ✗ Failed (exit code {process.returncode})")
                self.errors.append(f"{script_name}: {process.stderr[:500]}")
        
        except subprocess.TimeoutExpired:
            result['status'] = 'timeout'
            result['error'] = f'Script timed out after {timeout} seconds'
            print(f"    ⏱ Timeout ({timeout}s)")
            self.warnings.append(f"{script_name}: Execution timed out")
        
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            print(f"    ✗ Error: {str(e)[:100]}")
            self.errors.append(f"{script_name}: {str(e)}")
        
        self.execution_results.append(result)
        return result
    
    def check_critical_outputs(self):
        """Check if critical output files exist."""
        critical_outputs = {
            'models': [
                'checkpoints/chatbot_model.pth',
                'checkpoints/skin_model.pth',
                'checkpoints/lab_model.pth',
                'checkpoints/sound_model.pth',
            ],
            'datasets': [
                'processed_data/databases/disease_database.json',
                'data/chatbot_training_combined/chatbot_training_combined.json',
            ],
            'reports': [
                'models_summary/training_summary.json',
            ]
        }
        
        results = {}
        
        print("\n" + "=" * 60)
        print("OUTPUT VERIFICATION")
        print("=" * 60)
        
        for category, files in critical_outputs.items():
            results[category] = {}
            for file_path in files:
                full_path = os.path.join(PROJECT_ROOT, file_path)
                exists = os.path.exists(full_path)
                results[category][file_path] = exists
                status = "✓" if exists else "✗"
                print(f"  {status} {file_path}")
        
        return results
    
    def generate_report(self):
        """Generate comprehensive project report."""
        print("\n" + "=" * 60)
        print("GENERATING REPORT")
        print("=" * 60)
        
        # Categorize scripts
        categories = self.categorize_scripts()
        
        # Check outputs
        output_status = self.check_critical_outputs()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': PROJECT_ROOT,
            'scan_results': {
                'total_scripts': len(self.project_scripts),
                'total_models': len(self.model_files),
                'total_datasets': len(self.dataset_files),
            },
            'script_categories': {k: [os.path.basename(v) for v in vals] for k, vals in categories.items()},
            'dependencies': {k: v for k, v in list(self.dependencies.items())[:20]},  # Limit for size
            'execution_results': self.execution_results,
            'output_verification': output_status,
            'model_files': [os.path.basename(f) for f in self.model_files],
            'warnings': self.warnings[:50],  # Limit
            'errors': self.errors[:20],  # Limit
            'summary': {
                'scripts_executed': len([r for r in self.execution_results if r['status'] in ['success', 'failed']]),
                'scripts_succeeded': len([r for r in self.execution_results if r['status'] == 'success']),
                'scripts_failed': len([r for r in self.execution_results if r['status'] == 'failed']),
                'scripts_timeout': len([r for r in self.execution_results if r['status'] == 'timeout']),
                'total_warnings': len(self.warnings),
                'total_errors': len(self.errors),
            }
        }
        
        # Save report
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nReport saved to: {OUTPUT_FILE}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        print(f"  Scripts found: {report['scan_results']['total_scripts']}")
        print(f"  Models found: {report['scan_results']['total_models']}")
        print(f"  Datasets found: {report['scan_results']['total_datasets']}")
        print(f"  Scripts executed: {report['summary']['scripts_executed']}")
        print(f"  Succeeded: {report['summary']['scripts_succeeded']}")
        print(f"  Failed: {report['summary']['scripts_failed']}")
        print(f"  Timeout: {report['summary']['scripts_timeout']}")
        print(f"  Warnings: {report['summary']['total_warnings']}")
        print(f"  Errors: {report['summary']['total_errors']}")
        
        return report


def main():
    """Main execution function."""
    print("\n" + "=" * 60)
    print("PROJECT VALIDATION AND EXECUTION")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    
    validator = ProjectValidator()
    
    # 1. Scan project
    validator.scan_project()
    
    # 2. Verify dependencies
    validator.verify_dependencies()
    
    # 3. Get execution order
    execution_order = validator.get_execution_order()
    
    print("\n" + "=" * 60)
    print("SAFE EXECUTION ORDER")
    print("=" * 60)
    
    if execution_order:
        print("\nScripts to execute (in order):")
        for i, script in enumerate(execution_order, 1):
            print(f"  {i}. {os.path.basename(script)}")
        
        print(f"\nTotal: {len(execution_order)} scripts")
        
        # Execute scripts
        print("\n" + "=" * 60)
        print("EXECUTING SCRIPTS")
        print("=" * 60)
        
        for script in execution_order:
            # Skip scripts that might take too long or modify critical data
            script_name = os.path.basename(script).lower()
            
            # Skip training scripts for safety (they take too long)
            if 'train' in script_name:
                print(f"\n  Skipping: {os.path.basename(script)} (training script - run manually)")
                continue
            
            # Skip download scripts (require internet)
            if 'download' in script_name:
                print(f"\n  Skipping: {os.path.basename(script)} (download script - run manually)")
                continue
            
            # Execute other scripts
            validator.execute_script(script, timeout=60)
    else:
        print("\nNo scripts require automatic execution.")
    
    # 4. Check outputs
    output_status = validator.check_critical_outputs()
    
    # 5. Generate report
    report = validator.generate_report()
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    main()
