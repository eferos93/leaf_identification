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

set.seed(65283)

Train_rf <- read.csv("Train.csv", header = TRUE)
Test_rf <- read.csv("Test.csv", header = TRUE)
Train_rf <- Train_rf[,-1]
Test_rf <- Test_rf[,-1]
Train_rf$Class <- Train_rf$Class %>% as.factor
Test_rf$Class <- Test_rf$Class %>% as.factor
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
mtry=rep(0,K)
ntree=rep(0,K)
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
  mtry[i] <- custom$bestTune$mtry
  ntree[i] <- custom$bestTune$ntree
  randomF <- randomForest(Class~., data = train_custom, mtry = mtry[i],
                          ntree = ntree[i])
  prediction <- predict(randomF, test_custom[,-1], type="class")
  confMat <- table(test_custom$Class, prediction)
  accuracy[i] <- sum(diag(confMat))/sum(confMat)
}
avg_acc <- mean(accuracy)
sd_acc <- sd(accuracy)
write(paste("CV mean:", avg_acc, "; CV sd:", sd_acc), append = FALSE, file = "RF_result.txt")
write(c("Accuracies ", accuracy), append = TRUE, file = "RF_result.txt")
write(c("Best mtrys ", mtry), append = TRUE, file = "RF_result.txt")
write(c("Best ntrees ", ntree), append = TRUE, file = "RF_result.txt")

vect_of_matrix <- vector(mode = "list", length = K)
accuracies <- rep(0, K)

for(i in 1:K) {
  rf <- randomForest(Class~., data = Train_rf, mtry = mtry[i],
                     ntree = ntree[i])
  pr <- predict(rf, Test_rf[,-1], type="class")
  vect_of_matrix[[i]]<- table(pr, Test_rf$Class)
  accuracies[i] <- sum(diag(vect_of_matrix[[i]]))/sum(vect_of_matrix[[i]])
  write(paste("Accuracy with mtry = ", mtry[i], " ntree = ", ntree[i], ": ",
              accuracies[i]), append = TRUE, file = "RF_result.txt")
  
}

avg_acc_test <- accuracies %>% mean
sd_acc_test <- accuracies %>%  sd
classes <- leaf$Class %>% unique %>% sort %>% as.vector
best_acc <- accuracies %>% max
index <- match(best_acc, accuracies)
conf_matrix_best <- vect_of_matrix[[index]]
best_ntree <- ntree[index]
best_mtry <- mtry[index]

write(paste("Final test\n\tAvg:", avg_acc_test, " sd:", sd_acc_test), append = TRUE, file = "RF_result.txt")
write(paste("Best Accuracy:", best_acc, "with parameters ntree =", best_ntree, "and mtry =", best_mtry),
      append = TRUE, file = "RF_result.txt")

for (j in 1:30) {
  fp <- conf_matrix_best[j,-j] %>% sum
  tn <- conf_matrix_best[-j,-j] %>% diag %>% sum
  fpr <- fp/(fp+tn)
  
  fn <- conf_matrix_best[-j, j] %>% sum
  tp <- conf_matrix_best[j,j] 
  
  fnr <- fn/(fn+tp)
  
  write(paste("Class", classes[j], "\n\tFP=", fp, "TN=", tn, "FPR=", fpr, "\n\tFN=", fn, "TP=", tp, "FNR=", fnr),
        append = TRUE, file = "RF_result.txt")
  
}