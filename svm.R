library(dplyr)
#library(mlr)
library(caret)
library(e1071)
library(caTools)
library(rlist)

#custom SVM
customSVM <- list(type = "Classification", library = "e1071", loop = NULL)
customSVM$parameters <- data.frame(parameter = c("cost", "gamma"), class = rep("numeric", 2),
                                  label = c("cost", "gamma"))
customSVM$grid <- function(x, y, len = NULL, search = "grid") {}
customSVM$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  svm(x, y, gamma=param$gamma, cost=param$cost, ...)
}
customSVM$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customSVM$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customSVM$sort <- function(x) x[order(x[,1]),]
customSVM$levels <- function(x) x$classes

#import the dataset and cleaning
leaf <- read.csv("leaf.csv", header = FALSE,
                 col.names = c("Class", "Speciment_n°", "Eccentricity", "Aspect_Ratio",
                               "Elongation", "Solidity", "Stochastic_Convexity",
                               "Isoperimetric_Factor", "Maximal_Indentation_Depth",
                               "Lobedness", "Average_Intensity", "Average_Contrast",
                               "Smoothness", " Third_Moment", "Uniformity",
                               "Entropy"))
set.seed(65283)
leaf <- leaf[,-2]
leaf$Class <- as.factor(leaf$Class)
#setting the test and train datasets, the later used for 
#CV and tuning
sp <- sample.split(leaf$Class, SplitRatio = 0.7)
Train_svm <- subset(leaf, sp == TRUE)
Test_svm <- subset(leaf, sp == FALSE)

shuffled <- Train_svm[sample(nrow(Train_svm)),]

K <- 5
accuracy=rep(0,K)
best_gamma=rep(0,K)
best_cost=rep(0,K)
n <- nrow(shuffled)
dimX <- dim(shuffled[,-1])[2]
for(i in 1:K) {
  indexes <- ((i-1)*round(1/K*n) + 1):(i*round(1/K*n))
  if(exists("train_custom") && exists("test_custom")){
    rm(train_custom)
    rm(test_custom)
  }
  train_custom <- shuffled[-indexes,]
  test_custom <- shuffled[indexes,]
  print(nrow(train_custom) + nrow(test_custom) == nrow(shuffled))
  control <- trainControl(method="cv", number=4)
  tunegrid <- expand.grid(.gamma=c((1/dimX)^(-5:5)),
                          .cost=c(10^(-2:0), 2:10))
  custom <- train(Class~., data=train_custom, method=customSVM, metric="Accuracy",
                  tuneGrid=tunegrid, trControl=control)
  best_gamma[i] <- custom$bestTune$gamma
  best_cost[i] <- custom$bestTune$cost
  svm <- svm(Class~., data = train_custom, gamma = best_gamma[i],
             cost = best_cost[i])
  prediction <- predict(svm, test_custom[,-1], type="class")
  confMat <- table(test_custom$Class, prediction)
  accuracy[i] <- sum(diag(confMat))/sum(confMat)
}
print(accuracy)
print(best_gamma)
print(best_cost)

vect_of_matrix <- vector(mode = "list", length = K)
best_acc <- rep(0, K)
for(i in 1:K) {
  svm <- svm(Class~., data = Train_svm, gamma = best_gamma[i],
                     cost = best_cost[i])
  pr <- predict(svm, Test_svm[,-1], type="class")
  vect_of_matrix[[i]]<- table(pr, Test_svm$Class)
  best_acc[i] <- sum(diag(vect_of_matrix[[i]]))/sum(vect_of_matrix[[i]])
  print(paste("Accuracy with gamma = ", best_gamma[i], " cost = ", best_cost[i], ": ",
              best_acc[i]))
  
}

