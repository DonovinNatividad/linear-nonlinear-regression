import argparse
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import os.path as osp
import sys

def calc_jacobian(X,p):
    """
    Calculate Jacobian matrix for model function a*(x**b)+c*x+d
    Input:
        X: a N-by-1 matrix (numpy array) of the input data
        p: a D-by-1 matrix (numpy array) of the parameters
           where P(0,0) is parameter "a", P(1,0) is "b", etc.
    Output:
        J: derivatives per data point and per parameter. Please represent it as an N-by-D matrix
    Useful tool:
        1. np.log
        1. ** for exponentiation
    """

    N=X.shape[0]
    D=p.shape[0]
    J=np.zeros((N,D))

    ### Your job starts here ###

    # Fill in Jacobian Matrix by Row
    a, b, c = p[0, 0], p[1, 0], p[2, 0]

    for i in range(N):
        # Grab parameters and current data value
        x = X[i, 0]

        # Now we must take the derivative with respect to each parameter
        J[i, 0], J[i, 1], J[i, 2], J[i, 3] = x**b, a * (x**b) * np.log(x), x, 1

    ### Your job ends here ###

    return J

GAUSS_NEWTON_ITERATIONS=10 #PLEASE change your iteration count here
def nonlinear_regression_gn(X, Y, initialP):
    """
    Estimate parameters for model function a*(x**b)+c*x+d
    using Gauss-Newton.
    Input:
        X: a N-by-1 matrix (numpy array) of the input data
        Y: a N-by-1 matrix (numpy array) of the label
        initialP: a D-by-1 matrix (numpy array) of the initial guess for parameters
    Output:
        p: the linear parameter vector. Please represent it as a D-by-1 matrix (numpy array).
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
        3. np.linalg.inv: for matrix inversion
        4. you may use the model_function(...) function defined below
        5. your own linear_regression(...) function (you should be able to copy and modify
           that code as _part_ of your solution here)
    """

    p=np.copy(initialP)

    ### Your job starts here ###

    for iteration in range(GAUSS_NEWTON_ITERATIONS): #PLEASE do not change this line

        # 1. Find error for current guess (the t'th parameter)
        deltaY = Y - model_function(X, p)

        # 2. Find Jacobian around current guess
        J = calc_jacobian(X, p)

        # 3. Calculate change in Guess
        jTransposed = np.transpose(J)

        # Multiply the matrices
        jMultiplied = np.matmul(jTransposed, J)

        # Take the inverse of the multiplied matrices
        invJMul = np.linalg.inv(jMultiplied)

        # Multiply them with A tranposed again
        pBeforeY = np.matmul(invJMul, jTransposed)

        # Finally, perform the final multiplication with the b vector, in this case
        # it's deltaY, the output vector of the polynomial with varying parameters
        deltaP = np.matmul(pBeforeY, deltaY)

        # 4. Make new guess
        p += deltaP


    ### Your job ends here ###
    return p

GRADIENT_DESCENT_ITERATIONS=100
LEARNING_RATE=1e-3 #WARNING, this is probably too small!
def nonlinear_regression_gd(X, Y, initialP):
    """
    Estimate parameters for model function a*(x**b)+c*x+d
    using Gradient Descent.
    Input:
        X: a N-by-1 matrix (numpy array) of the input data
        Y: a N-by-1 matrix (numpy array) of the label
        initialP: a D-by-1 matrix (numpy array) of the initial guess for parameters
    Output:
        p: the linear parameter vector. Please represent it as a D-by-1 matrix (numpy array).
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
        4. you may use the model_function(...) function defined below
        5. your own nonlinear_regression_gn(...) function (you should be able to re-use some of
           that code, refer to slide 24)
    """

    p=np.copy(initialP)

    ### Your job starts here ###
    a = LEARNING_RATE

    tolerance = 0.001

    converged = False

    for iteration in range(GRADIENT_DESCENT_ITERATIONS): #PLEASE do not change this line
        # PLEASE use LEARNING_RATE variable defined above

        # Calculate the difference between observed and predicted values
        delta_y = Y - model_function(X, p)

        # Calculate the transpose of the Jacobian matrix
        jacobian_transpose = np.transpose(calc_jacobian(X, p))

        # Compute the gradient
        G = -2 * np.matmul(jacobian_transpose, delta_y)

        # Calculate the new guess
        newp = p - a * G

        if np.all(abs(newp - p) < tolerance):
            converged = True
            break

        p = newp

    ### Your job ends here ###
    return p

##############################################################################

def model_function(X,p):
    """
    Calculate output of model function a*(x**b)+c*x+d
    Input:
        X: a N-by-1 matrix (numpy array) of the input data
        p: a D-by-1 matrix (numpy array) of the parameters
           where P(0,0) is parameter "a", P(1,0) is "b", etc.
    Output:
        Ytilde: model output values as N-by-1 matrix
    """
    return p[0,0]*(X**p[1,0])+p[2,0]*X+p[3,0]

## Data loader and data generation functions
def data_loader(args):
    """
    Output:
        X: the data matrix (numpy array) of size 1-by-N
        Y: the label matrix (numpy array) of size N-by-1
    """
    
    if not osp.isfile(r"dataset_nonlinear.npz"):
        N=30
        Ntst=math.floor(N*0.3)
        Ntrn=N-Ntst
        X=np.concatenate((data_X(Ntrn,0,1,1),data_X(Ntst+2,0,1,1,unordered=False)[1:-1])).reshape(-1,1)
        X=X+0.5
        rr=np.random.rand(4)
        a=rr[0]*0.4+0.8
        b=rr[1]*0.5+1.25
        c=rr[2]*6-3
        d=rr[3]*2-1
        print("Generate p: ", np.array([a,b,c,d]))
        Y=a*(X**b)+c*X+d
        Y=Y+np.random.normal(scale=(np.max(Y)-np.min(Y))*0.02,size=N).reshape(-1,1)
        np.savez(r"dataset_nonlinear.npz", X = X, Y = Y)
        del X, Y
    data = np.load(r"dataset_nonlinear.npz")
    X = data['X']
    Y = data['Y']
    return X, Y

def data_X(N,min,max,sigma,unordered=True):
    X=np.random.normal(size=N)*sigma+1 #spacing semi-gaussian
    X=np.abs(np.mod(X+1,2)-1)+1 #mirror
    X[0]=0
    X=np.cumsum(X) #totals
    X=(X/X[-1]*(max-min))+min #linmap
    if unordered:
        X=np.random.permutation(X)
    return X

## Displaying the results
def display_LR(p, X_trn, Y_trn, X_tst, Y_tst, save=False):
    deg = p.shape[0]-1
    x_min = np.min(X_trn)
    x_max = np.max(X_trn)
    x = np.linspace(x_min-0.05, x_max+0.05, num=1000).reshape(-1,1)
    YY = model_function(x, p)
    plt.plot(x.reshape(-1), YY.reshape(-1), color='black', linewidth=3)
    plt.scatter(X_trn.reshape(-1), Y_trn.reshape(-1), c='red')
    if(X_tst is not None):
        plt.scatter(X_tst.reshape(-1), Y_tst.reshape(-1), c='blue')
    if save:
        if args.method.startswith("grad"):
            fname="gradient_descent"
        else:
            fname="gauss_newton"
        plt.savefig(fname + '.png', format='png')
        np.savez('Results_' + fname + '.npz', p=p, X_trn=X_trn, X_tst=X_tst, Y_trn=Y_trn, Y_tst=Y_tst)
    plt.show()
    plt.close()

## auto_grader
def auto_grade(display):
    print("In auto grader!\n")
    
    p=np.array([0.5,1.5,5,-1]).reshape(-1,1)
    print("Checking calc_jacobian()...")
    J_expected=np.array([0.8538149682454624, -0.04497919266535625, 0.9, 1.0, 1.153689732987167, 0.05497918794470525, 1.1, 1.0]).reshape(-1,4)
    J=calc_jacobian(np.array([0.9,1.1]).reshape(-1,1),p)
    if J.shape!=J_expected.shape:
        print("Error, expected shape ",J_expected.shape,", instead got ",J.shape)
        return
    if (np.abs(J-J_expected)>1e-9).any():
        print("Got J=",J)
        for i in range(4):
            if (np.abs(J[:,i]-J_expected[:,i])>1e-9).any():
                print("Error in column ",i," (zero indexed)")
        return
    print("Jacobian OK")
    
    p=np.array([1,1.5,1,-1]).reshape(-1,1)
    XX = np.linspace(0.5, 1.5, num=30).reshape(-1,1)
    YY = model_function(XX, p)
    
    if not args.method.startswith("grad"):
        print("\nChecking nonlinear_regression_gn()...")
        initialP=p+0.1
        global GAUSS_NEWTON_ITERATIONS
        GAUSS_NEWTON_ITERATIONS=1
        pExpected=np.array([9.616951854092781193e-01,1.528478679323104217e+00,1.039968168648498947e+00,-1.001665398287315378e+00]).reshape(-1,1)
        pS=nonlinear_regression_gn(XX, YY, initialP)
        # np.savetxt(sys.stdout,pS)
        if pS.shape!=pExpected.shape:
            print("Error, expected shape ",pExpected.shape,", instead got ",pS.shape)
            return
        if (np.abs(pS-pExpected)>1e-6).any():
            print("Error, expected p=",pExpected,", instead got p=",pS)
            return
        print("Gauss-Newton OK")

    if not args.method.startswith("gauss"):
        print("\nChecking nonlinear_regression_gd()...")
        initialP=p+0.1
        global GRADIENT_DESCENT_ITERATIONS
        GRADIENT_DESCENT_ITERATIONS=1
        global LEARNING_RATE
        LEARNING_RATE=1e-2
        pExpected=np.array([8.729450285013207189e-01,1.557936080178939919e+00,8.926387394411837706e-01,-1.088187586270846996e+00]).reshape(-1,1)
        pS=nonlinear_regression_gd(XX, YY, initialP)
        # np.savetxt(sys.stdout,pS)
        if pS.shape!=pExpected.shape:
            print("Error, expected shape ",pExpected.shape,", instead got ",pS.shape)
            return
        if (np.abs(pS-pExpected)>1e-6).any():
            print("Error, expected p=",pExpected,", instead got p=",pS)
            return
        print("Gradient Descent OK")

## Main function
def main(args):

    if args.auto_grade:
        args.save=False
        auto_grade(args.display)
        return

    ## Loading data
    X, Y = data_loader(args) # X: the N-by-1 data matrix (numpy array); Y: the N-by-1 label vector

    ## Setup (separate to train and test)
    N = X.shape[0]  # number of data instances of X
    X_test = X[int(0.7*N):,:]
    X_trn = X[:int(0.7*N),:]
    Y_test = Y[int(0.7 * N):, :]
    Y_trn = Y[:int(0.7 * N), :]

    initialP=np.array([1,1.5,0,0]).reshape(-1,1) #Cheating here, since I know how the data is generated
    if args.method.startswith("grad"):
        print("Method: Gradient Descent")
        p=nonlinear_regression_gd(X_trn, Y_trn, initialP)
    else:
        print("Method: Gauss-Newton")
        p=nonlinear_regression_gn(X_trn, Y_trn, initialP)
    print("p: ", p.reshape(-1))

    # Evaluation
    training_error = np.mean((model_function(X_trn,p) - Y_trn) ** 2)
    print("Training mean square error: ", training_error)
    test_error = np.mean((model_function(X_test,p) - Y_test) ** 2)
    print("Test mean square error: ", test_error)

    if args.display:
        display_LR(p, X_trn, Y_trn, X_test, Y_test, save=args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running linear regression (LR)")
    parser.add_argument('--method', default="", type=str)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--auto_grade', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
