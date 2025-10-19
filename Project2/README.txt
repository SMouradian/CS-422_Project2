Samuel Mouradian
CS 422.622.1001 - Machine Learning
Professor - Dr. Emily Hand
Assignment - Project 2, README Descrption Document (Due 10.20.2025)


1. PERCEPTRON:
    i. perceptron_train(X, Y):
            This function trains the data for testing purposes. What it does is it initializes
        the weight and bias to zero (like we learned in class), and creates data samples
        and features for the model to train with. Then, it begins iterating through the
        samples up to a limit of 1,000 epochs; it creates temporary variables like [x_old]
        and [y_old] to update [w] and [b]. Those variables are used if the activation
        value ends up being less than or equal to zero. This also updates an error
        counter, which is used to register when the function makes a successful pass
        through an epoch (i.e., an epoch does not require an update of [w] and [b]).
        The [w] and [b] values are returned once the function breaks from iterating
        through the epochs (i.e., the error counter is zero).

    ii. perceptron_test(X_test, Y_test, w, b):
            This function tests the data created through the previous function. It calculates
        an activation using the weight [w] and bias [b] values, and then creates a
        prediction value using that activation value. The prediction value, like we
        learned in class, becomes positive (or negative, respectfully), and is then
        used to find the accuracy of the perceptron. Once the accuracy is calculated, it
        is returned, therefore giving an output for the first dataset [data_1.txt] and
        the second dataset [data_2.txt].



2. GRADIENT DESCENT:
    i. gradient_descent(delta_f, x_init, eta):
            This function is used to calculate gradient descent for the two datasets. We
        create a varaible for [x_init], then iterate through the data up to 1,000
        points (or until convergence). During this iteration, we are creating a gradient
        using [dela_f] and the variable for [x_init]. We then check if the gradient is near
        a minimum, which allows us to break from the iteration. Once we break, we calculate
        the new guess of x using the learning rate[x = x_init now becomes
        x = x - (eta * gradient)]. Once that is caclulated, we return the value as an output.



3. LINEAR CLASSIFIER:
    i. linear_train(X, Y, dLdw, dLdb, eta):
            This function is simiilar to what we were doing in the training function for the
        perceptron, only with a few noticeable differences. For one, we have our [n_features]
        varaible, and we initialize our weight and bias values to zero. Then, we begin
        iterating up to a range of 1,000; this allows us to create a prediction using the
        [X] values, and an error meter using [prediction] and [Y]. This allows us to create
        [dw] and [db] values, which we use to calculate the loss-variant equations for
        weight and bias (the ones we learned in class). Once we have those calculated, we
        calculate the new [w] and [b] by subtracting the previous weight value from the
        product of the learning rate and our derivative values [dw]/[db]. Once that is done,
        we return the new weight and bias values for the testing function.

    ii. linear_test(X_test, Y_test, w, b):
            This function allows us to test the accuracy of our training model. We create an
        action prediction using the [X_test] value, as well as the [w] and [b] values we
        gathered from the previous function. We then create a label prediction using that
        action prediction variable [act_pred]. Once we have all that calculated, we can
        then find the correct prediction of our model, which will give us a gague on our
        accuracy. Once that is calculated correctly, we return the value as an output for
        the data.