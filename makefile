all:
	@echo "Choices are BP, AG, DE, PSO"

BP:
	@echo "Running BP set"
	python3 main.py BP cancer > results/cancerBP.txt 
	python3 main.py BP glass > results/glassBP.txt 
	python3 main.py BP soybean > results/soybeanBP.txt 
	python3 main.py BP abalone > results/abaloneBP.txt 
	python3 main.py BP hardware > results/hardwareBP.txt 
	python3 main.py BP forestfires > results/forestfiresBP.txt 
	@echo "done"
	
DE:
	@echo "Running DE set"
	python3 main.py DE cancer > results/cancerDE.txt 
	python3 main.py DE glass > results/glassDE.txt 
	python3 main.py DE soybean > results/soybeanDE.txt 
	python3 main.py DE abalone > results/abaloneDE.txt 
	python3 main.py DE hardware > results/hardwareDE.txt 
	python3 main.py DE forestfires > results/forestfiresDE.txt 
	@echo "done"

GA:
	@echo "Running GA set"
	python3 main.py GA cancer > results/cancerGA.txt 
	python3 main.py GA glass > results/glassGA.txt 
	python3 main.py GA soybean > results/soybeanGA.txt 
	python3 main.py GA abalone > results/abaloneGA.txt 
	python3 main.py GA hardware > results/hardwareGA.txt 
	python3 main.py GA forestfires > results/forestfiresGA.txt 
	@echo "done"

PSO:
	@echo "Running PSO set"
	python3 main.py PSO cancer > results/cancerPSO.txt 
	python3 main.py PSO glass > results/glassPSO.txt 
	python3 main.py PSO soybean > results/soybeanPSO.txt 
	python3 main.py PSO abalone > results/abalonePSO.txt 
	python3 main.py PSO hardware > results/hardwarePSO.txt 
	python3 main.py PSO forestfires > results/forestfiresPSO.txt 
	@echo "done"