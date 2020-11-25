all:
	@echo "Choices are BP, AG, DE, PSO"

BP:
	@echo "Running BP set"
	python3 -u main.py BP cancer | tee results/cancerBP.txt 
	python3 -u main.py BP glass | tee results/glassBP.txt 
	python3 -u main.py BP soybean | tee results/soybeanBP.txt 
	python3 -u main.py BP abalone | tee results/abaloneBP.txt 
	python3 -u main.py BP hardware | tee results/hardwareBP.txt 
	python3 -u main.py BP forestfires | tee results/forestfiresBP.txt 
	@echo "done"
	
DE:
	@echo "Running DE set"
	python3 main.py DE cancer | tee results/cancerDE.txt 
	python3 main.py DE glass | tee results/glassDE.txt 
	python3 main.py DE soybean | tee results/soybeanDE.txt 
	python3 main.py DE abalone | tee results/abaloneDE.txt 
	python3 main.py DE hardware | tee results/hardwareDE.txt 
	python3 main.py DE forestfires | tee results/forestfiresDE.txt 
	@echo "done"

GA:
	@echo "Running GA set"
	python3 -u main.py GA cancer | tee results/cancerGA.txt 
	python3 -u main.py GA glass | tee results/glassGA.txt 
	python3 -u main.py GA soybean | tee results/soybeanGA.txt 
	python3 -u main.py GA abalone | tee results/abaloneGA.txt 
	python3 -u main.py GA hardware | tee results/hardwareGA.txt 
	python3 -u main.py GA forestfires | tee results/forestfiresGA.txt 
	@echo "done"

PSO:
	@echo "Running PSO set"
	python3 -u main.py PSO cancer | tee results/cancerPSO.txt
	python3 -u main.py PSO glass | tee results/glassPSO.txt
	python3 -u main.py PSO soybean | tee results/soybeanPSO.txt
	python3 -u main.py PSO abalone | tee results/abalonePSO.txt
	python3 -u main.py PSO hardware | tee results/hardwarePSO.txt
	python3 -u main.py PSO forestfires | tee results/forestfiresPSO.txt
	@echo "done"