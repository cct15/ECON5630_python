### Some details about assigment2 in python

For the analysis part, I didn't find partial linear regression function in python (neither in statsmodel nor in sklearn package), so I implemented it myself.

I used the Kernel Regression in statsmodel. First I calculated the bandwidth of y and l with respect to k and investment. Then with the estimated kernel, I did a linear regression on $$Y-E(Y|k,inv)=\beta (l-E(l|k,inv))+\epsilon$$

With this method, **I got the exactly same results as those in R**.

Note: 

1. the Kernel Regression in python is faster than that in R. (Seems that Kernel Regression in python doesn't use Leave-one-out? )
2. Python has more machine learning models but less stats models. The models in python often lack detailed stats summary.
3. the regression type in the KernelReg should be 'lc' (local constant). Defalut is 'll' (local linear).
4. 
In my method:

$$(y-E(y|k,inv))=\beta (l-E(l|k,inv)) + \epsilon$$

So

$$y=\beta l +E(y|k,inv)-\beta E(l|k,inv) + \epsilon$$

It means

$$\phi (k,inv)=E(y|k,inv)-\beta E(l|k,inv)$$
