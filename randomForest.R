library(randomForest)
library(caret)
library(mlbench)
set.seed(3000)
spl <- sample.split(leaf$Class, SplitRatio = 0.7)
Train <- subset(leaf, spl == TRUE)
Test <- subset(leaf, spl == FALSE)


rf <- rfcv(Train[,-1], Train$Class, cv.fold = 5)
with(rf, plot(n.var, error.cv, type="b", col="red"))
r <- randomForest(Class~., data=Train, mtry=2)
p <- predict(r, Test)
cf <- confusionMatrix(p, Test$Class)
# Create model with default paramters
control <- trainControl(method="repeatedcv", number=4, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(leaf)-1)
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(Class~., data=Train, method="rf", metric=metric,
                    tuneGrid=tunegrid, trControl=control)
print(rf_default)

# Random Search
control <- trainControl(method="repeatedcv", number=4, repeats=3, search="random")
set.seed(seed)
mtry <- sqrt(ncol(leaf)-1)
rf_random <- train(Class~., data=Train, method="rf", 
                   metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)

#Grid search
control <- trainControl(method="repeatedcv", number=10, repeats=1, search="grid")
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- train(Class~., data=Train, method="rf", metric=metric,
                       tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)
# mtry = 7 seems to be optimal 0.7734


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
  indexes <- ((i-1)*round(1/K*n) + 1):i*round(1/K*n)
  print(paste("i=", i, "i_min=", (i-1)*round(1/K*n) + 1, " I_max=", i*round(1/K*n)))
  rm(train_custom)
  rm(test_custom)
  train_custom <- shuffled[-indices,] 
  test_custom <- shuffled[indices,]
  print(nrow(train_custom) + nrow(test_custom) == n)
  control <- trainControl(method="cv", number=4)
  tunegrid <- expand.grid(.ntree=c(1000, 1500, 2000, 2500), .mtry=c(1:15))
  custom <- train(Class~., data=train_custom, method=customRF, metric="Accuracy",
                  tuneGrid=tunegrid, trControl=control)
  best_mtry[i] <- custom$bestTune$mtry
  best_ntree[i] <- custom$bestTune$ntree
  randomF <- randomForest(Class~., data = train_custom, mtry = best_mtry[i],
                          ntree = best_ntree[i])
  prediction <- predict(randomF, test_custom)
  confMat <- table(test_custom$Class, prediction)
  accuracy[i] <- sum(diag(confMat))/sum(confMat)
}
print(accuracy)
print(best_mtry)
print(best_ntree)
#---------------------------------------------------------------------------------
rf_CV <- randomForest(Class~., data = Train)
predict_RFCV <- predict(rf_CV, Test)
confMatrixRFCV <- table(Test$Class, predict_RFCV)
print(paste("Accuracy: ", sum(diag(confMatrixRFCV))/sum(confMatrixRFCV)))


rf_CV <- randomForest(Class~., data = Train, mtry=2)
predict_RFCV <- predict(rf_CV, Test)
confMatrixRFCV <- table(Test$Class, predict_RFCV)
print(paste("Accuracy: ", sum(diag(confMatrixRFCV))/sum(confMatrixRFCV)))
#acc = 0.8452

rf_CV <- randomForest(Class~., data = Train, mtry=7, ntree=2500)
predict_RFCV <- predict(rf_CV, Test)
confMatrixRFCV <- table(Test$Class, predict_RFCV)
print(paste("Accuracy: ", sum(diag(confMatrixRFCV))/sum(confMatrixRFCV)))
#acc=0.80

rf_CV <- randomForest(Class~., data = Train, mtry=14, ntree=2500)
predict_RFCV <- predict(rf_CV, Test)
confMatrixRFCV <- table(Test$Class, predict_RFCV)
print(paste("Accuracy: ", sum(diag(confMatrixRFCV))/sum(confMatrixRFCV)))
# acc = 0.82
#------------------------------------------------------------------------------------------
n <- nrow(leaf)
nTrees <- c(1,10,25,50,75,100,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000)
errRates <- c()
shuffled <- leaf[sample(n),]
#i <- 2
#indexes <- ((i-1)*round(1/10*n) + 1):i*round(1/10*n)
#train <- shuffled[-indexes,]
#test <- shuffled[indexes,]
for(numbOfTrees in nTrees) {
  accuracy <- rep(0,10)
  #Cross Validation
  for (i in 1:K) {
    indexes <- ((i-1)*round(1/K*n) + 1):i*round(1/K*n)
    train <- shuffled[-indexes,]
    test <- shuffled[indexes,]
    #train
    
    #rf <- randomForest(Species ~ ., train)
    #less influent vars
    rf <- randomForest(as.factor(Class) ~ ., train, ntree = numbOfTrees)
    #most influent vars
    #rf <- randomForest(Species ~ Petal.Length + Petal.Width, train, ntree = numbOfTrees)
    #prediction
    predicted <- predict(rf, test)
    confusionM <- table(test$Class, predicted)
    accuracy[i] <- sum(diag(confusionM))/sum(confusionM)
    errRateTemp <- rf[["err.rate"]]
    errRates[i] <- errRateTemp[numbOfTrees][1]
  }
  
  print(paste("The accuracy of RF for predicting species with ntree=",
              numbOfTrees," is :", mean(accuracy)))
  print(paste("The OOB train error rate is: ", mean(errRates)))
}
# 1. investigate variable importance with RF
rf <- randomForest(as.factor(Class) ~ ., data = shuffled, importance = TRUE)
predictForest <- predict(rf, shuffled)
confusionM <- table(shuffled$Class, predictForest)
print(paste("Accuracy: ", sum(diag(confusionM))/sum(confusionM)))
importance(rf, scale = TRUE)
