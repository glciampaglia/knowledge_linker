#Author: Mihai Avram - mihai.v.avram@gmail.com
#Date: 3/15/2017

import sys
import argparse
import pandas as pd
import numpy as np
import string

def main():
	parser = argparse.ArgumentParser()
	#Required parameters -a1 and -a2 which denote the paths to the two input algorithm files to be compared (print output to the console)
	parser.add_argument("-a1", "--alg1file", type=str, help="the relative or absolute location and name of the input file of the first algorithm", required=True)
	parser.add_argument("-a2", "--alg2file", type=str, help="the relative or absolute location and name of the input file of the second algorithm", required=True)
	#Optional parameters -oc, -oa1, -oa2 which denote output paths for the comparisons (print output to these files)
	parser.add_argument("-oc", "--outputcomparison", type=str, help="the relative or absolute location and name of the alg1/alg2 comparison output file (.txt extension recommended)", default="CompOutputDefault.txt")
	parser.add_argument("-oa1", "--alg1confusion", type=str, help="the relative or absolute location and name of the alg1 confusion result output file (.csv extension recommended)", default="Alg1OutConfDefault.csv")
	parser.add_argument("-oa2", "--alg2confusion", type=str, help="the relative or absolute location and name of the alg2 confusion result output file (.csv extension recommended)", default="Alg2OutConfDefault.csv")
	args = parser.parse_args()
	
	computeAndPrint(args.alg1file, args.alg2file, args.outputcomparison, args.alg1confusion, args.alg2confusion)
	
def computeAndPrint(alg1file, alg2file, outputcomparison, alg1confusion, alg2confusion):
    #Importing algorithms to be compared	
	try:
		alg1 = pd.read_csv(alg1file)
		alg2 = pd.read_csv(alg2file)
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
	if outputcomparison == "CompOutputDefault.txt" and alg1confusion == "Alg1OutConfDefault.csv" and alg2confusion == "Alg2OutConfDefault.csv":
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

		try:
			fileConfAlg1 = open(alg1confusion,"w")
		except:
			print("Error writing to the specified alg1confusion file, please check -oa1 argument", file=sys.stderr)
			sys.exit(1)
		
		fileConfAlg1.write("sid,sub,oid,obj,conf\n")

		for sample in range(0, alg1.shape[0]):
			if sample % alg1DistinctObjNum == 0:
				if alg1.iloc[sample, classColLocation] == 1:
					fileConfAlg1.write(str(alg1.iloc[sample, idColLocation]) + "," + str(alg1.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(alg1.iloc[sample, objectIDColLocation]) + "," + str(alg1.iloc[sample, objectColLocation].replace("dbr:","")) + ",TP" + "\n")
				else:
					fileConfAlg1.write(str(alg1.iloc[sample, idColLocation]) + "," + str(alg1.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(alg1.iloc[sample, objectIDColLocation]) + "," + str(alg1.iloc[sample, objectColLocation].replace("dbr:","")) + ",FP" + "\n")
			else:
				if alg1.iloc[sample, classColLocation] == 1:
					fileConfAlg1.write(str(alg1.iloc[sample, idColLocation]) + "," + str(alg1.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(alg1.iloc[sample, objectIDColLocation]) + "," + str(alg1.iloc[sample, objectColLocation].replace("dbr:","")) + ",FN" + "\n")
				else:
					fileConfAlg1.write(str(alg1.iloc[sample, idColLocation]) + "," + str(alg1.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(alg1.iloc[sample, objectIDColLocation]) + "," + str(alg1.iloc[sample, objectColLocation].replace("dbr:","")) + ",TN" + "\n")

		fileConfAlg1.close

		
		try:
			fileConfAlg2 = open(alg2confusion,"w")
		except:
			print("Error writing to the specified alg2confusion file, please check -oa2 argument", file=sys.stderr)
			sys.exit(1)
			
		fileConfAlg2.write("sid,sub,oid,obj,conf\n")

		for sample in range(0, alg2.shape[0]):
			if sample % alg2DistinctObjNum == 0:
				if alg2.iloc[sample, classColLocation] == 1:
					fileConfAlg2.write(str(alg2.iloc[sample, idColLocation]) + "," + str(alg2.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(alg2.iloc[sample, objectIDColLocation]) + "," + str(alg2.iloc[sample, objectColLocation].replace("dbr:","")) + ",TP" + "\n")
				else:
					fileConfAlg2.write(str(alg2.iloc[sample, idColLocation]) + "," + str(alg2.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(alg2.iloc[sample, objectIDColLocation]) + "," + str(alg2.iloc[sample, objectColLocation].replace("dbr:","")) + ",FP" + "\n")
			else:
				if alg2.iloc[sample, classColLocation] == 1:
					fileConfAlg2.write(str(alg2.iloc[sample, idColLocation]) + "," + str(alg2.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(alg2.iloc[sample, objectIDColLocation]) + "," + str(alg2.iloc[sample, objectColLocation].replace("dbr:","")) + ",FN" + "\n")
				else:
					fileConfAlg2.write(str(alg2.iloc[sample, idColLocation]) + "," + str(alg2.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(alg2.iloc[sample, objectIDColLocation]) + "," + str(alg2.iloc[sample, objectColLocation].replace("dbr:","")) + ",TN" + "\n")

		fileConfAlg2.close
		print("Done, please check output files.")

if __name__ == "__main__":
	main()
