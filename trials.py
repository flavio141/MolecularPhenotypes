import subprocess

for i in range(10):
    try:
        subprocess.run(['python', 'src/train_test.py', f'--trials={i}', f'--prepare_data={False}', f'--difference={True}', f'--global_metrics={False}', f'--fold_mapping={True}', f'--epochs={100}'], check=True) 
        #subprocess.run(f'python src/train_test.py --trials={i} --difference=True --global_metrics=True --fold_mapping=True --epochs=200')
    except subprocess.CalledProcessError as e:
        print(f"Errore nell'esecuzione del comando: {e}") 
