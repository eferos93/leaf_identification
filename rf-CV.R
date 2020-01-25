library(dplyr)
library(randomForest)
library(caret)
library(mlbench)
library(e1071)
library(caTools)
leaf <- read.csv("leaf.csv", header = FALSE,
                 col.names = c("Class", "Speciment_n°", "Eccentricity", "Aspect_Ratio",
                               "Elongation", "Solidity", "Stochastic_Convexity",
                               "Isoperimetric_Factor", "Maximal_Indentation_Depth",
                               "Lobedness", "Average_Intensity", "Average_Contrast",
                               "Smoothness", " Third_Moment", "Uniformity",
                               "Entropy"))
leaf <- leaf[,-2]
leaf$Class <- as.factor(leaf$Class)

set.seed(4838649)
spl <- sample.split(leaf$Class, SplitRatio = 0.7)
Train_rf <- subset(leaf, spl == TRUE)
Test_rf <- subset(leaf, spl == FALSE)
shuffled <- Train_rf[sample(nrow(Train_rf)),]

#custom RF
customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("ntree", "mtry"), class = rep("numeric", 2),
                                  label = c("ntree", "mtry"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, ntree=param$ntree, mtry=param$mtry, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

# train model-----------------------------------------------------------------------
K <- 5
accuracy=rep(0,K)
best_mtry=rep(0,K)
best_ntree=rep(0,K)
n <- nrow(shuffled)
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
  tunegrid <- expand.grid(.ntree=c(500, 1000, 1500, 2000, 2500), .mtry=c(1:14))
  custom <- train(Class~., data=train_custom, method=customRF, metric="Accuracy",
                  tuneGrid=tunegrid, trControl=control)
  best_mtry[i] <- custom$bestTune$mtry
  best_ntree[i] <- custom$bestTune$ntree
  randomF <- randomForest(Class~., data = train_custom, mtry = best_mtry[i],
                          ntree = best_ntree[i])
  prediction <- predict(randomF, test_custom[,-1], type="class")
  confMat <- table(test_custom$Class, prediction)
  accuracy[i] <- sum(diag(confMat))/sum(confMat)
}
print( accuracy)
print(best_mtry)
print(best_ntree)

conf_matrixes <- rep(0, K)
for(i in 1:K) {
  rf <- randomForest(Class~., data = Train_rf, mtry = best_mtry[i],
                     ntree = best_ntree[i])
  pr <- predict(rf, Test_rf[,-1], type="class")
  conf_matrixes[i] <- confusionMatrix(data = pr, reference = Test_rf$Class)
  print(paste("Accuracy with mtry = ", best_mtry[i], " ntree = ", best_ntree[i], ": ",
              conf_matrixes[i]$overall[1]))
  
}
