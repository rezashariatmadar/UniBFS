import subprocess
import os
import shutil

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
exp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'experiments'))

tuning_py = os.path.join(src_dir, 'tuning.py')
fitness_py = os.path.join(src_dir, 'fitness.py')
run_tuned_py = os.path.join(exp_dir, 'run_tuned.py')

def backup():
    shutil.copy(tuning_py, tuning_py + '.bak')
    shutil.copy(fitness_py, fitness_py + '.bak')
    shutil.copy(run_tuned_py, run_tuned_py + '.bak')

def restore():
    shutil.copy(tuning_py + '.bak', tuning_py)
    shutil.copy(fitness_py + '.bak', fitness_py)
    shutil.copy(run_tuned_py + '.bak', run_tuned_py)

def patch_mo():
    with open(tuning_py, 'r') as f:
        code = f.read()
    code = code.replace('return result["mean_acc"]', 'return result["mean_acc"], result["mean_nfeats"]')
    code = code.replace('study = optuna.create_study(direction="maximize")', 'study = optuna.create_study(directions=["maximize", "minimize"])')
    code = code.replace('best_params = study.best_params', 'best_trial = sorted(study.best_trials, key=lambda t: (-t.values[0], t.values[1]))[0]\n    best_params = best_trial.params')
    with open(tuning_py, 'w') as f:
        f.write(code)

def patch_svm():
    # Patch fitness.py default evaluator
    with open(fitness_py, 'r') as f:
        code = f.read()
    code = code.replace('evaluator: str = "knn"', 'evaluator: str = "svm"')
    with open(fitness_py, 'w') as f:
        f.write(code)
        
    # Patch tuning.py evaluator
    with open(tuning_py, 'r') as f:
        code = f.read()
    code = code.replace('from sklearn.neighbors import KNeighborsClassifier\n        model = KNeighborsClassifier(n_neighbors=1, metric="euclidean")', 
                        'from sklearn.svm import LinearSVC\n        model = LinearSVC(dual=False)')
    with open(tuning_py, 'w') as f:
        f.write(code)
        
    # Patch run_tuned.py evaluator
    with open(run_tuned_py, 'r') as f:
        code = f.read()
    code = code.replace('from sklearn.neighbors import KNeighborsClassifier\n        model = KNeighborsClassifier(n_neighbors=1, metric="euclidean")', 
                        'from sklearn.svm import LinearSVC\n        model = LinearSVC(dual=False)')
    with open(run_tuned_py, 'w') as f:
        f.write(code)

# Make sure we start clean
backup()

try:
    print("\n\n" + "="*50)
    print("=== RUN 1: Multi-Objective (Accuracy + Features) ===")
    print("="*50)
    patch_mo()
    subprocess.run([".venv/Scripts/python.exe", "experiments/run_tuned.py"], check=True)
    
    print("\n\n" + "="*50)
    print("=== RUN 2: SVM Evaluator (Single Objective) ===")
    print("="*50)
    restore()
    patch_svm()
    subprocess.run([".venv/Scripts/python.exe", "experiments/run_tuned.py"], check=True)

    print("\n\n" + "="*50)
    print("=== RUN 3: Multi-Objective + SVM ===")
    print("="*50)
    restore()
    patch_mo()
    patch_svm()
    subprocess.run([".venv/Scripts/python.exe", "experiments/run_tuned.py"], check=True)
finally:
    restore()
    print("Restored original files.")
