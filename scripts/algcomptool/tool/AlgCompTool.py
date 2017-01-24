#Author: Mihai Avram - mihai.v.avram@gmail.com

import sys, getopt
import pandas as pd
import numpy as np
import string

    
def main(argv):
    if len(argv) < 2:
        print("Executing...\n")
        computeAndPrint(False)
    elif(argv[1] in ['--help','-h']):
        print("Enter python AlgCompTool.py to print to terminal, or python AlgCompTool.py -f to print to files")
    elif(argv[1] in ['--file','-f']):
        print("Executing...\n")
        computeAndPrint(True)


def computeAndPrint(writeToFile):
    #Importing algorithms to be compared
    alg1 = pd.read_csv("../input/presidentcouplesNODES.csv")
    alg2 = pd.read_csv("../input/presidentcouplesRSIM.csv")

    noNulls = 1
    #Alg1 checking for null values in pertinent columns
    if any(pd.isnull(alg1['sid'])) or any(pd.isnull(alg1['oid'])) or any(pd.isnull(alg1['class'])) or any(pd.isnull(alg1['score'])):
        noNulls = 0
        print("There are missing column values in the sid, oid, class, or score columns in alg1")

    #Alg2 checking for null values in pertinent columns
    if any(pd.isnull(alg2['sid'])) or any(pd.isnull(alg2['oid'])) or any(pd.isnull(alg2['class'])) or any(pd.isnull(alg2['score'])):
        noNulls = 0
        print("There are missing column values in the sid, oid, class, or score columns in alg2")
    

    #Algorithm comparison
    if noNulls:
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


        if not writeToFile:
            #Printing Results
            print("\nALG1 CORRECTLY PREDICTED THE FOLLOWING")
            for subject, score in correctPredAlg1.items():
                print("ID:", idSubjectMapping[subject], "- Subject:", subject.replace("dbr:",""), "with score", score)

            print("\nALG1 INCORRECTLY PREDICTED THE FOLLOWING")
            for subject, score in incorrectPredAlg1.items():
                print("ID:", idSubjectMapping[subject], "- Subject:", subject.replace("dbr:",""), "with score", score)   

            print("\nALG2 CORRECTLY PREDICTED THE FOLLOWING")
            for subject, score in correctPredAlg2.items():
                print("ID:", idSubjectMapping[subject], "- Subject:", subject.replace("dbr:",""), "with score", score)

            print("\nALG2 INCORRECTLY PREDICTED THE FOLLOWING")
            for subject, score in incorrectPredAlg2.items():
                print("ID:", idSubjectMapping[subject], "- Subject:", subject.replace("dbr:",""), "with score", score)


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
            fileComp = open("../output/algorithmscomparison.txt",'w')

            #Printing Results
            fileComp.write("\nALG1 CORRECTLY PREDICTED THE FOLLOWING\n")
            for subject, score in correctPredAlg1.items():
                fileComp.write("ID: " + str(idSubjectMapping[subject]) + " - Subject: " + str(subject.replace("dbr:","")) + " with score " + str(score) + "\n")

            fileComp.write("\nALG1 INCORRECTLY PREDICTED THE FOLLOWING\n")
            for subject, score in incorrectPredAlg1.items():
                fileComp.write("ID: " + str(idSubjectMapping[subject]) + " - Subject: " + str(subject.replace("dbr:","")) + " with score " + str(score) + "\n") 

            fileComp.write("\nALG2 CORRECTLY PREDICTED THE FOLLOWING\n")
            for subject, score in correctPredAlg2.items():
                fileComp.write("ID: " + str(idSubjectMapping[subject]) + " - Subject: " + str(subject.replace("dbr:","")) + " with score " + str(score) + "\n")

            fileComp.write("\nALG2 INCORRECTLY PREDICTED THE FOLLOWING\n")
            for subject, score in incorrectPredAlg2.items():
                fileComp.write("ID: " + str(idSubjectMapping[subject]) + " - Subject: " + str(subject.replace("dbr:","")) + " with score " + str(score) + "\n")

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

            fileConfAlg1 = open("../output/confusionalg1.csv","w")
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

            fileConfAlg2 = open("../output/confusionalg2.csv","w")
            fileConfAlg2.write("sid,sub,oid,obj,conf\n")

            for sample in range(0, alg2.shape[0]):
                if sample % alg2DistinctObjNum == 0:
                    if alg2.iloc[sample, classColLocation] == 1:
                        fileConfAlg2.write(str(alg1.iloc[sample, idColLocation]) + "," + str(alg1.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(alg1.iloc[sample, objectIDColLocation]) + "," + str(alg1.iloc[sample, objectColLocation].replace("dbr:","")) + ",TP" + "\n")
                    else:
                        fileConfAlg2.write(str(alg1.iloc[sample, idColLocation]) + "," + str(alg1.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(alg1.iloc[sample, objectIDColLocation]) + "," + str(alg1.iloc[sample, objectColLocation].replace("dbr:","")) + ",FP" + "\n")
                else:
                    if alg2.iloc[sample, classColLocation] == 1:
                        fileConfAlg2.write(str(alg1.iloc[sample, idColLocation]) + "," + str(alg1.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(alg1.iloc[sample, objectIDColLocation]) + "," + str(alg1.iloc[sample, objectColLocation].replace("dbr:","")) + ",FN" + "\n")
                    else:
                        fileConfAlg2.write(str(alg1.iloc[sample, idColLocation]) + "," + str(alg1.iloc[sample, subjectColLocation].replace("dbr:","")) + "," + str(alg1.iloc[sample, objectIDColLocation]) + "," + str(alg1.iloc[sample, objectColLocation].replace("dbr:","")) + ",TN" + "\n")

            fileConfAlg2.close
            print("Done.")

if __name__ == "__main__":
	main(sys.argv)
