Author: Mihai Avram - mihai.v.avram@gmail.com
Date: 3/15/2017

# DESCRIPTION:
This tool is used to compare the results of two algorithms that have subject to object relations, a score for the given relation, and the class (ground truth) of such relation. The tool compares scores for the two algorithms for the same relations and lists descriptive information about the algorithms, such as which algorithm performed better for which relations and conversely, which algorithm performed worse for which relations. The tool also prints out .csv formatted confusion information (i.e. True Positive, True Negative, False Positive, False Negative) for each algorithm specifically.

# STEPS TO CLEAN DATA:

Note: Look for input/presidentcouplesNODES.csv and input/presidentcouplesRSIM.csv for examples
	of what cleaned, proper input files should look like for the -a1 and -a2 parameters

1) Ensure the input files have non-null values in titled columns 'sid','score','sub','class','oid','obj'
2) Ensure the input files have (subject)*(object) rows/samples
3) Ensure all values in the 'sub' column for both files have the exact same named entries
	(i.e. dbr:A_Beautiful_Mind_(film) in alg1 should be exactly the same name
	and not dbresource/:A_Beautiful_Mind_(film) or A_Beautiful_Mind)
4) Ensure the output/ folder is empty as files may get created there and should not
	be over-written, if the same three specified output files are listed in multiple iterations of
	the tool running, they will be overwritten.

# HOW TO RUN TOOL:
The tool will run in Python 3, and can be executed the following way from the command line:

The following prints the usage of the tool:

`python AlgCompTool.py`

Tool run with required parameters (printing results to the terminal):

`python AlgCompTool.py -a1 [ALGORITHM1INPUTFILELOCATION] -a2 [ALGORITHM2INPUTFILELOCATION]`

Optional parameters -oc, -oa1, and -oa2 which are used to place output in output files:

`python AlgCompTool.py -a1 [ALGORITHM1INPUTFILELOCATION] -a2 [ALGORITHM2INPUTFILELOCATION] -oc [COMPARISONOUTPUTFILELOC] -oa1 [ALG1CONFUSIONOUTPUT] -oa2 [ALG2CONFUSIONOUTPUT]`

# TODO-IMPROVMENETS: 
A possible extension to this script would be to add comparison for multiple algorithms,
not just two. Another possible extension is to parallelize code for bigger data.
