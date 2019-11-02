#Author: Mihai Avram - mihai.v.avram@gmail.com
#Date: 06/27/2017

import sys
import argparse
import pandas as pd
import numpy as np
import string

def main():
	parser = argparse.ArgumentParser()
	#Required parameters inputFile1 and inputFile2 which denote the paths to the two input algorithm files to be compared (print output to the console)
	parser.add_argument("inputFile1", type=str, help="the relative or absolute location and name of the first input file")
	parser.add_argument("inputFile2", type=str, help="the relative or absolute location and name of the second input file")
	#Optional parameters -oc, -o1, -o2 which denote output paths for the comparisons (print output to these files)
	parser.add_argument("-oc", "--outputcomparison", type=str, help="the relative or absolute location and name of the file 1 vs. file 2 comparison output file (.txt extension recommended)", required=False, default="CompOutputDefault.txt")
	parser.add_argument("-o1", "--file1Confusion", type=str, help="the relative or absolute location and name of the first confusion result output file (.csv extension recommended)", required=False, default="File1OutConfDefault.csv")
	parser.add_argument("-o2", "--file2Confusion", type=str, help="the relative or absolute location and name of the second confusion result output file (.csv extension recommended)", required=False, default="File2OutConfDefault.csv")
	args = parser.parse_args()
	
	computeAndPrint(args.inputFile1, args.inputFile2, args.outputcomparison, args.file1Confusion, args.file2Confusion)
	
def computeAndPrint(inputFile1, inputFile2, outputcomparison, file1Confusion, file2Confusion):
    #Importing algorithms to be compared	
	try:
		alg1 = pd.read_csv(inputFile1)
		alg2 = pd.read_csv(inputFile2)
	except:
		print("Error reading the alg1 and alg2 input files, please ensure file path and file format is correct.", file=sys.stderr)
		sys.exit(1)
	
	#Alg1 checking for null values in pertinent columns
	if any(pd.isnull(alg1['sid'])) or any(pd.isnull(alg1['oid'])) or any(pd.isnull(alg1['class'])) or any(pd.isnull(alg1['score'])):
		print("There are missing column values in the sid, oid, class, or score columns in alg1", file=sys.stderr)
		sys.exit(1)
		
	#Alg2 checking for null values in pertinent columns
	if any(pd.isnull(alg2['sid'])) or any(pd.isnull(alg2['oid'])) or any(pd.isnull(alg2['class'])) or any(pd.isnull(alg2['score'])):
		print("There are missing column values in the sid, oid, class, or score columns in alg2", file=sys.stderr)
		sys.exit(1)

	#Algorithm comparison
	#Preparatory work
	alg1 = alg1.sort_values(['sid','score'], axis=0, ascending=False)
	alg2 = alg2.sort_values(['sid','score'], axis=0, ascending=False)

	alg1DistinctObjNum = len(alg1.oid.unique())
	alg2DistinctObjNum = len(alg2.oid.unique())

	idColLocation = alg1.columns.get_loc('sid')
	subjectColLocation = alg1.columns.get_loc('sub')
	objectIDColLocation = alg1.columns.get_loc('oid')
	objectColLocation = alg1.columns.get_loc('obj')
	classColLocation = alg1.columns.get_loc('class')
	scoreColLocation = alg1.columns.get_loc('score')

	correctPredAlg1 = {}
	incorrectPredAlg1 = {}

	correctPredAlg2 = {}
	incorrectPredAlg2 = {}

	#Creating Dictionary of id/subject matches
	idSubjectMapping = {}
	for index, row in alg1.iterrows():
		idSubjectMapping[row[subjectColLocation]] = row[idColLocation]

	#Comparisons
	for sample in range(0, alg1.shape[0], alg1DistinctObjNum):
		if alg1.iloc[sample, classColLocation] == 1:
			correctPredAlg1[str(alg1.iloc[sample, subjectColLocation])] = alg1.iloc[sample, scoreColLocation]
		elif alg1.iloc[sample, classColLocation] == 0:
			incorrectPredAlg1[str(alg1.iloc[sample, subjectColLocation])] = alg1.iloc[sample, scoreColLocation]


	for sample in range(0, alg2.shape[0], alg2DistinctObjNum):
		if alg2.iloc[sample, classColLocation] == 1:
			correctPredAlg2[str(alg2.iloc[sample, subjectColLocation])] = alg2.iloc[sample, scoreColLocation]
		elif alg2.iloc[sample, classColLocation] == 0:
			incorrectPredAlg2[str(alg2.iloc[sample, subjectColLocation])] = alg2.iloc[sample, scoreColLocation]


	#if the 3 output files were not specified, print results to console, if only one was specified, use that one and default locations for the others
	if outputcomparison == "CompOutputDefault.txt" and file1Confusion == "File1OutConfDefault.csv" and file2Confusion == "File2OutConfDefault.csv":
		#Printing Results (comparisonPrint function used for this)
		def comparisonPrint(algorithm):
			for subject, score in algorithm.items():
				print("ID:", idSubjectMapping[subject], "- Subject:", subject.replace("dbr:",""), "with score", score)

		print("\nALG1 CORRECTLY PREDICTED THE FOLLOWING")
		comparisonPrint(correctPredAlg1)
		
		print("\nALG1 INCORRECTLY PREDICTED THE FOLLOWING")
		comparisonPrint(incorrectPredAlg1)
		
		print("\nALG2 CORRECTLY PREDICTED THE FOLLOWING")
		comparisonPrint(correctPredAlg2)

		print("\nALG2 INCORRECTLY PREDICTED THE FOLLOWING")
		comparisonPrint(incorrectPredAlg2)

		bothCorrect = set(correctPredAlg1.keys()) & set(correctPredAlg2.keys())
		bothIncorrect = set(incorrectPredAlg1.keys()) & set(incorrectPredAlg2.keys())

		alg1Correctalg2Incorrect = set(correctPredAlg1.keys()) & set(incorrectPredAlg2.keys())
		alg2Correctalg1Incorrect = set(correctPredAlg2.keys()) & set(incorrectPredAlg1.keys())

		print("\nINSTANCES WHERE ALG1 PREDICTED CORRECTLY BUT ALG2 DIDN'T")
		for instance in alg1Correctalg2Incorrect:
			print("ID:",idSubjectMapping[instance], "- Subject:", instance.replace("dbr:",""))

		print("\nINSTANCES WHERE ALG2 PREDICTED CORRECTLY BUT ALG1 DIDN'T")
		for instance in alg2Correctalg1Incorrect:
			print("ID:",idSubjectMapping[instance], "- Subject:", instance.replace("dbr:",""))

		#Comparison of algorithms for when both predicted correctly
		print("\nCOMPARISON OF ALGS WHEN BOTH PREDICTED CORRECTLY")
		for subject in bothCorrect:

			alg1CompSegment = alg1[alg1['sub'] == subject]
			alg2CompSegment = alg2[alg2['sub'] == subject]

			#Same prediction level, so must compare ratios between our first correct prediction, and the next prediction
			alg1Ratio = alg1CompSegment.iloc[0,scoreColLocation]/alg1CompSegment.iloc[1,scoreColLocation]
			alg2Ratio = alg2CompSegment.iloc[0,scoreColLocation]/alg2CompSegment.iloc[1,scoreColLocation]

			#The greater the ratio the better the prediction, i.e. 
			#0.5 (our prediction) 0.04 (next prediction) vs.
			#0.5 (our prediction) 0.4 (next prediction)
			if alg1Ratio > alg2Ratio:
				print("Alg1 > Alg2 for relation:", "ID:", idSubjectMapping[subject], "- Subject:", subject.replace("dbr:",""))
			elif alg1Ratio < alg2Ratio:
				print("Alg1 < Alg2 for relation:", "ID:", idSubjectMapping[subject], "- Subject:", subject.replace("dbr:",""))



		print("\nCOMPARISON OF ALGS WHEN BOTH PREDICTED INCORRECTLY")
		#Comparison of algorithms for when both predicted incorrectly
		for subject in bothIncorrect:
			#Alg1 Analysis
			alg1CompSegment = alg1[alg1['sub'] == subject]
			alg1Sample = 0
			while alg1CompSegment.iloc[alg1Sample,classColLocation] != 1:
				alg1Sample = alg1Sample + 1

			#Alg2 Analysis
			alg2CompSegment = alg2[alg2['sub'] == subject]
			alg2Sample = 0
			while alg2CompSegment.iloc[alg2Sample,classColLocation] != 1:
				alg2Sample = alg2Sample + 1

			if alg1Sample > alg2Sample:
				print("Alg1 < Alg2 for relation:", "ID:", idSubjectMapping[subject], "- Subject:", subject.repalce("dbr:",""))
			elif alg1Sample < alg2Sample:
				print("Alg1 > Alg2 for relation:", "ID:", idSubjectMapping[subject], "- Subject:", subject.replace("dbr:",""))
			elif alg1Sample == alg2Sample:
				#Same prediction level, so must compare ratios between highest weight prediction and this wrong preiction
				alg1Ratio = alg1CompSegment.iloc[0,scoreColLocation]/alg1CompSegment.iloc[alg1Sample,scoreColLocation]
				alg2Ratio = alg2CompSegment.iloc[0,scoreColLocation]/alg2CompSegment.iloc[alg2Sample,scoreColLocation]
				#The bigger the ratio the worse the prediction: i.e. 0.5 (first), 0.3, 0.001, 0.0001 (our prediction) vs.
				#0.5 (first), 0.2 (our prediction)
				if alg1Ratio > alg2Ratio:
					print("Alg1 < Alg2 for relation:", "ID:", idSubjectMapping[subject], "- Subject:", subject.repalce("dbr:",""))
				elif alg1Ratio < alg2Ratio:
					print("Alg1 > Alg2 for relation:", "ID:", idSubjectMapping[subject], "- Subject:", subject.replace("dbr:",""))


		
		print("\n\nALGORITHM 1: CALCULATING CONFUSION METRICS I.E. TRUE POSITIVE, FALSE POSITIVE, ETC...\n\n")

		for sample in range(0, alg1.shape[0]):
			if sample % alg1DistinctObjNum == 0:
				if alg1.iloc[sample, classColLocation] == 1:
					print("SID:", alg1.iloc[sample, idColLocation], "- SUBJECT:", alg1.iloc[sample, subjectColLocation].replace("dbr:",""), "- OID:", alg1.iloc[sample, objectIDColLocation], "- OBJECT:", alg1.iloc[sample, objectColLocation].replace("dbr:",""), "- CONFUSION:", "TP")
				else:
					print("SID:", alg1.iloc[sample, idColLocation], "- SUBJECT:", alg1.iloc[sample, subjectColLocation].replace("dbr:",""), "- OID:", alg1.iloc[sample, objectIDColLocation], "- OBJECT:", alg1.iloc[sample, objectColLocation].replace("dbr:",""), "- CONFUSION:", "FP")
			else:
				if alg1.iloc[sample, classColLocation] == 1:
					print("SID:", alg1.iloc[sample, idColLocation], "- SUBJECT:", alg1.iloc[sample, subjectColLocation].replace("dbr:",""), "- OID:", alg1.iloc[sample, objectIDColLocation], "- OBJECT:", alg1.iloc[sample, objectColLocation].replace("dbr:",""), "- CONFUSION:", "FN")
				else:
					print("SID:", alg1.iloc[sample, idColLocation], "- SUBJECT:", alg1.iloc[sample, subjectColLocation].replace("dbr:",""), "- OID:", alg1.iloc[sample, objectIDColLocation], "- OBJECT:", alg1.iloc[sample, objectColLocation].replace("dbr:",""), "- CONFUSION:", "TN")

		print("\n\nALGORITHM 2: CALCULATING CONFUSION METRICS I.E. TRUE POSITIVE, FALSE POSITIVE, ETC...\n\n")

		for sample in range(0, alg2.shape[0]):
			if sample % alg2DistinctObjNum == 0:
				if alg2.iloc[sample, classColLocation] == 1:
					print("SID:", alg2.iloc[sample, idColLocation], "- SUBJECT:", alg2.iloc[sample, subjectColLocation].replace("dbr:",""), "- OID:", alg2.iloc[sample, objectIDColLocation], "- OBJECT:", alg2.iloc[sample, objectColLocation].replace("dbr:",""), "- CONFUSION:", "TP")
				else:
					print("SID:", alg2.iloc[sample, idColLocation], "- SUBJECT:", alg2.iloc[sample, subjectColLocation].replace("dbr:",""), "- OID:", alg2.iloc[sample, objectIDColLocation], "- OBJECT:", alg2.iloc[sample, objectColLocation].replace("dbr:",""), "- CONFUSION:", "FP")
			else:
				if alg2.iloc[sample, classColLocation] == 1:
					print("SID:", alg2.iloc[sample, idColLocation], "- SUBJECT:", alg2.iloc[sample, subjectColLocation].replace("dbr:",""), "- OID:", alg2.iloc[sample, objectIDColLocation], "- OBJECT:", alg2.iloc[sample, objectColLocation].replace("dbr:",""), "- CONFUSION:", "FN")
				else:
					print("SID:", alg2.iloc[sample, idColLocation], "- SUBJECT:", alg2.iloc[sample, subjectColLocation].replace("dbr:",""), "- OID:", alg2.iloc[sample, objectIDColLocation], "- OBJECT:", alg2.iloc[sample, objectColLocation].replace("dbr:",""), "- CONFUSION:", "TN")
		print("Done.")
	
	#WRITING THE RESULTS TO FILES
	else:
		try:
			fileComp = open(outputcomparison,'w')
		except:
			print("Error writing to the specified outputcomparison file, please check -oc argument", file=sys.stderr)
			sys.exit(1)

		#Printing Results (comparisonWrite function used for this)
		def comparisonWrite(algorithm):
			for subject, score in algorithm.items():
				fileComp.write("ID: " + str(idSubjectMapping[subject]) + " - Subject: " + str(subject.replace("dbr:","")) + " with score " + str(score) + "\n")

		fileComp.write("\nALG1 CORRECTLY PREDICTED THE FOLLOWING\n")
		comparisonWrite(correctPredAlg1)

		fileComp.write("\nALG1 INCORRECTLY PREDICTED THE FOLLOWING\n")
		comparisonWrite(incorrectPredAlg1)

		fileComp.write("\nALG2 CORRECTLY PREDICTED THE FOLLOWING\n")
		comparisonWrite(correctPredAlg2)

		fileComp.write("\nALG2 INCORRECTLY PREDICTED THE FOLLOWING\n")
		comparisonWrite(incorrectPredAlg2)

		bothCorrect = set(correctPredAlg1.keys()) & set(correctPredAlg2.keys())
		bothIncorrect = set(incorrectPredAlg1.keys()) & set(incorrectPredAlg2.keys())

		alg1Correctalg2Incorrect = set(correctPredAlg1.keys()) & set(incorrectPredAlg2.keys())
		alg2Correctalg1Incorrect = set(correctPredAlg2.keys()) & set(incorrectPredAlg1.keys())

		fileComp.write("\nINSTANCES WHERE ALG1 PREDICTED CORRECTLY BUT ALG2 DIDN'T\n")
		for instance in alg1Correctalg2Incorrect:
			fileComp.write("ID: " + str(idSubjectMapping[instance]) + " - Subject: " + str(instance.replace("dbr:","")) + "\n")

		fileComp.write("\nINSTANCES WHERE ALG2 PREDICTED CORRECTLY BUT ALG1 DIDN'T\n")
		for instance in alg2Correctalg1Incorrect:
			fileComp.write("ID: " + str(idSubjectMapping[instance]) + " - Subject: " + str(instance.replace("dbr:","")) + "\n")

		#Comparison of algorithms for when both predicted correctly
		fileComp.write("\nCOMPARISON OF ALGS WHEN BOTH PREDICTED CORRECTLY\n")
		for subject in bothCorrect:

			alg1CompSegment = alg1[alg1['sub'] == subject]
			alg2CompSegment = alg2[alg2['sub'] == subject]

			#Same prediction level, so must compare ratios between our first correct prediction, and the next prediction
			alg1Ratio = alg1CompSegment.iloc[0,scoreColLocation]/alg1CompSegment.iloc[1,scoreColLocation]
			alg2Ratio = alg2CompSegment.iloc[0,scoreColLocation]/alg2CompSegment.iloc[1,scoreColLocation]

			#The greater the ratio the better the prediction, i.e. 
			#0.5 (our prediction) 0.04 (next prediction) vs.
			#0.5 (our prediction) 0.4 (next prediction)
			if alg1Ratio > alg2Ratio:
				fileComp.write("Alg1 > Alg2 for relation: " + " ID: " + str(idSubjectMapping[subject]) + " - Subject: " + str(subject.replace("dbr:","")) + "\n")
			elif alg1Ratio < alg2Ratio:
				fileComp.write("Alg1 < Alg2 for relation: " + " ID: " + str(idSubjectMapping[subject]) + " - Subject: " + str(subject.replace("dbr:","")) + "\n")

		fileComp.write("\nCOMPARISON OF ALGS WHEN BOTH PREDICTED INCORRECTLY\n")
		#Comparison of algorithms for when both predicted incorrectly
		for subject in bothIncorrect:

			#Alg1 Analysis
			alg1CompSegment = alg1[alg1['sub'] == subject]
			alg1Sample = 0
			while alg1CompSegment.iloc[alg1Sample,classColLocation] != 1:
				alg1Sample = alg1Sample + 1

			#Alg2 Analysis
			alg2CompSegment = alg2[alg2['sub'] == subject]
			alg2Sample = 0
			while alg2CompSegment.iloc[alg2Sample,classColLocation] != 1:
				alg2Sample = alg2Sample + 1

			if alg1Sample > alg2Sample:
				fileComp.write("Alg1 < Alg2 for relation: " + " ID: " + str(idSubjectMapping[subject]) + " - Subject: " + str(subject.replace("dbr:","")) + "\n")
			elif alg1Sample < alg2Sample:
				fileComp.write("Alg1 > Alg2 for relation: " + " ID: " + str(idSubjectMapping[subject]) + " - Subject: " + str(subject.replace("dbr:","")) + "\n")
			elif alg1Sample == alg2Sample:
				#Same prediction level, so must compare ratios between highest weight prediction and this wrong preiction
				alg1Ratio = alg1CompSegment.iloc[0,scoreColLocation]/alg1CompSegment.iloc[alg1Sample,scoreColLocation]
				alg2Ratio = alg2CompSegment.iloc[0,scoreColLocation]/alg2CompSegment.iloc[alg2Sample,scoreColLocation]
				#The bigger the ratio the worse the prediction: i.e. 0.5 (first), 0.3, 0.001, 0.0001 (our prediction) vs.
				#0.5 (first), 0.2 (our prediction)
				if alg1Ratio > alg2Ratio:
					fileComp.write("Alg1 < Alg2 for relation: " + " ID: " + str(idSubjectMapping[subject]) + " - Subject: " + str(subject.replace("dbr:","")) + "\n")
				elif alg1Ratio < alg2Ratio:
					fileComp.write("Alg1 > Alg2 for relation: " + " ID: " + str(idSubjectMapping[subject]) + " - Subject: " + str(subject.replace("dbr:","")) + "\n")
		fileComp.close

		#Function to print confusion result files
		def confusionFileWrite(writeFileLoc, algorithm):
			try:
				fileConfAlg = open(writeFileLoc,"w")
			except:
				print("Error writing to the specified output file, please check -o1 argument", file=sys.stderr)
				sys.exit(1)
			
			fileConfAlg.write("sid,sub,oid,obj,conf\n")

			algorithmDistinctObjNum = len(algorithm.oid.unique())
			
			for sample in range(0, algorithm.shape[0]):
				if sample % algorithmDistinctObjNum == 0:
					if algorithm.iloc[sample, classColLocation] == 1:
						fileConfAlg.write(str(algorithm.iloc[sample, idColLocation]) + "," + str(algorithm.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(algorithm.iloc[sample, objectIDColLocation]) + "," + str(algorithm.iloc[sample, objectColLocation].replace("dbr:","")) + ",TP" + "\n")
					else:
						fileConfAlg.write(str(algorithm.iloc[sample, idColLocation]) + "," + str(algorithm.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(algorithm.iloc[sample, objectIDColLocation]) + "," + str(algorithm.iloc[sample, objectColLocation].replace("dbr:","")) + ",FP" + "\n")
				else:
					if algorithm.iloc[sample, classColLocation] == 1:
						fileConfAlg.write(str(algorithm.iloc[sample, idColLocation]) + "," + str(algorithm.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(algorithm.iloc[sample, objectIDColLocation]) + "," + str(algorithm.iloc[sample, objectColLocation].replace("dbr:","")) + ",FN" + "\n")
					else:
						fileConfAlg.write(str(algorithm.iloc[sample, idColLocation]) + "," + str(algorithm.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(algorithm.iloc[sample, objectIDColLocation]) + "," + str(algorithm.iloc[sample, objectColLocation].replace("dbr:","")) + ",TN" + "\n")

			fileConfAlg.close
			
		#Writing confusion result files
		confusionFileWrite(file1Confusion, alg1)
		confusionFileWrite(file2Confusion, alg2)

		print("Done, please check output files.")

if __name__ == "__main__":
	main()
