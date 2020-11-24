# Authors:
# Kemal Turksonmez
# Arash Ajam
from models import Models
import sys
programType = sys.argv[1]
fileName = sys.argv[2]
print(programType, fileName)
md = Models()
if fileName == "cancer":
    md.cancer(programType)
elif fileName == "glass":
    md.glass(programType)
elif fileName == "soybean":
    md.soybean(programType)
elif fileName == "abalone":
    md.abalone(programType)
elif fileName == "hardware":
    md.hardware(programType)
elif fileName == "forestfires":
    md.fires(programType)