import pandas as pd
import numpy as np
from docplex.mp.model import Model
import math
import li_test

INF = math.inf

class selectLiTest:

    """
        Constructor

        Args:
            matrix (dataframe): DMU's information including ONLY inputs and outputs
            inputsIncluded (list of str): inputs that can not be removed (column names of the matrix or empty list)
            y (list of str): outputs (column names of the matrix). The rest of columns are intrepreted as inputs.
            model (str): DEA measure. Only BBC is supported. 
            alpha (flaot): minimum significance level to consider a variable relevant to the model
    """
    def __init__(self, matrix, inputsIncluded, y, model="BCC", alpha=0.05):

        # Inputs that can not be removed
        self.inputsIncludedCol = inputsIncluded
        # Inputs that are considered for elimination
        self.inputsCandidateCol = list(matrix.drop(self.inputsIncludedCol + y, axis=1).columns)
        # Inputs removed
        self.inputsRemoved = []
        self.inputsRemovedPValues = []
        self.inputsIncludedPValues = []
        # Inputs currently in the model
        self.xCol = self.inputsIncludedCol + self.inputsCandidateCol

        # Outputs
        self.yCol = y

        self._check_enter_parameters(matrix, self.xCol, y)

        # DEA measure
        self.model = model

        # original matrix (will not be modified while fitting)
        self.matrixOri = matrix.loc[:, self.xCol + self.yCol]  # Order variables
        self.NOri = len(self.matrixOri)
        # copy of matrix (will be modified while fitting)
        self.matrix = self.matrixOri.copy()
        self.N = self.NOri

        # inputs and outputs indexes
        self.inputsIncluded = matrix.columns.get_indexer(self.inputsIncludedCol).tolist()    # Index var.ind in matrix
        self.inputsCandidate = matrix.columns.get_indexer(self.inputsCandidateCol).tolist()  # Index var.ind in matrix
        self.y = matrix.columns.get_indexer(self.yCol).tolist()                              # Index var. obj in matrix
        self.inputs = self.inputsCandidate + self.inputsIncluded

        # Number of inputs and outputs
        self.numInputsIncluded = len(self.inputsIncluded) 
        self.numInputsCandidate = len(self.inputsCandidate)
        self.numOutput = len(self.y)

        # Hyperparameters
        self.alpha = alpha

    'Destructor'
    def __del__(self):
        try:
            del self.N
            del self.matrix
            del self.numInputsIncluded
            del self.numInputsCandidate
            del self.numOutput
            del self.inputsIncluded
            del self.inputsCandidate
            del self.inputsIncluded
            del self.inputsCandidate
            del self.y
            del self.xCol
            del self.yCol

        except Exception:
            pass

    def _check_enter_parameters(self, matrix, x, y):
        # var. x and var. y have been procesed
        if type(x[0]) == int or type(y[0]) == int:
            self.matrix = matrix
            if any(self.matrix.dtypes == 'int64'):
                self.matrix = self.matrix.astype('float64')
            return
        else:
            self.matrix = matrix.loc[:, x + y]  # Order variables
            if any(self.matrix.dtypes == 'int64'):
                self.matrix = self.matrix.astype('float64')

        if len(matrix) == 0:
            raise EXIT("ERROR. The dataset must contain data")
        elif len(x) == 0:
            raise EXIT("ERROR. The inputs of dataset must contain data")
        elif len(y) == 0:
            raise EXIT("ERROR. The outputs of dataset must contain data")
        else:
            cols = x + y
            for col in cols:
                if col not in matrix.columns.tolist():
                    raise EXIT("ERROR. The names of the inputs or outputs are not in the dataset")

            for col in x:
                if col in y:
                    raise EXIT("ERROR. The names of the inputs and the outputs are overlapping")


    'Li test input selection algorithm'
    def fit(self):
        result_model_i = pd.DataFrame()

        # resolve total model
        self.x = self.inputsIncluded + self.inputsCandidate
        self.numInputs = len(self.x)
        result_model_i.loc[:, "total_model"] = self.DEA(self.matrix)["DEA"]

        # init inputs removed: at the beginning none
        self.inputsRemoved = []
        self.inputsRemovedPValues = []

        # stopping rule: 
        # there are no more candidate inputs to be remove
        # OR there is only one input left
        while self.numInputsCandidate != 0 and self.numInputs != 1: 
            
            pvalue_max = -INF  # highest p-value out of all the reduced models 
            self.inputsIncludedPValues = [] # all p-values of every reduced model

            for i in range(self.numInputsCandidate):

                # Resolve the model without tested variable i
                inputsCandidate = self.inputsCandidate.copy()
                inputsCandidate.remove(self.inputsCandidate[i])
                self.x = self.inputsIncluded + inputsCandidate
                self.numInputs = len(self.x)
                reduced = self.DEA(self.matrix)["DEA"] # Reduce model

                # Adapated Li test computation between total model and the current reduced model scores
                comparation = {'total': result_model_i.loc[:, "total_model"], 
                               'reduced': reduced}
                comparation = pd.DataFrame(comparation)
                pvalue_reduced = li_test.li_test(comparation, "total", "reduced", self.numInputs+1, self.numOutput)
                self.inputsIncludedPValues.append((self.inputsCandidate[i], pvalue_reduced))
                
                # check if current p-value is higher than the stored highest p-value and update accordangly
                if pvalue_reduced > pvalue_max:
                    pvalue_max = pvalue_reduced
                    x_result = self.inputsCandidate[i]
                    xCol_result = self.inputsCandidateCol[x_result]
                    result_model_i = result_model_i.copy()
                    result_model_i["reduced_model"] = reduced
                    result_model_i["pvalue"] = pvalue_reduced

            # check if highest p-value is grester than alpha
            # if it is, remove the associated variable keep iterating
            # else stop iterating: all remaining variables are significant
            if pvalue_max > self.alpha:
                self.inputsRemoved.append(xCol_result)
                self.inputsRemovedPValues.append(round(pvalue_max,5))
                self.inputsCandidate.remove(x_result)  # Borrarla definitivamente
                self.numInputsCandidate = len(self.inputsCandidate)
                result_model_i = result_model_i.drop(columns="total_model")
                result_model_i = result_model_i.rename(columns={"reduced_model": "total_model"})
                self.inputs = self.inputsIncluded + self.inputsCandidate
                self.numInputs = len(self.inputs)
            else:
                break # Stop iterating

        return result_model_i.loc[:, "total_model"] # Final DEA model without not signicant inputs 



    ######################################## 
    ################ DEA ###################
    ########################################

    'Compute output-oriented radial DEA model'
    def _scoreDEA_BCC_output(self, x, y):
        # Prepare matrix
        self.xmatrix = self.matrix.iloc[:, self.x].T  # xmatrix
        self.ymatrix = self.matrix.iloc[:, self.y].T  # ymatrix

        # create one model instance, with a name
        m = Model(name='beta_DEA')

        # by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
        beta = {0: m.continuous_var(name="beta")}

        # Constrain 2.4
        name_lambda = {i: m.continuous_var(name="l_{0}".format(i)) for i in range(self.N)}

        # Constrain 2.3
        m.add_constraint(m.sum(name_lambda[n] for n in range(self.N)) == 1)  # sum(lambda) = 1

        # Constrain 2.1 y 2.2
        for i in range(self.numInputs):
            # Constrain 2.1
            m.add_constraint(m.sum(self.xmatrix.iloc[i, j] * name_lambda[j] for j in range(self.N)) <= x[i])

        for i in range(self.numOutput):
            # Constrain 2.2
            m.add_constraint(
                m.sum(self.ymatrix.iloc[i, j] * name_lambda[j] for j in range(self.N)) >= beta[0] * y[i])

        # objetive
        m.maximize(beta[0])

        # Model Information
        # m.print_information()

        m.solve()

        # solution
        if m.solution == None:
            sol = 0
        else:
            sol = m.solution.objective_value
        return sol


    def DEA(self, matrix):
        nameCol = "DEA"
        matrix.loc[:, nameCol] = 0

        for i in range(len(matrix)):
            if self.model == "BCC":
                matrix.loc[i, nameCol] = self._scoreDEA_BCC_output(self.matrixOri.iloc[i, self.x].to_list(),
                                                           self.matrixOri.iloc[i, self.y].to_list())
            else:
                EXIT("No more models are available")

        return matrix

'Handle errrors'
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

'Handle errrors'
class EXIT(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return style.YELLOW + "\n\n" + self.message + style.RESET
