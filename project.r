library(dplyr)
library(rpart)
library(caret)
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
Train_dt <- read.csv("Train.csv", header = TRUE)
Test_dt <- read.csv("Test.csv", header = TRUE)
Train_dt <- Train_dt[,-1]
Test_dt <- Test_dt[,-1]
Train_dt$Class <- Train_dt$Class %>% as.factor
Test_dt$Class <- Test_dt$Class %>% as.factor
shuffled <- Train_dt[sample(nrow(Train_dt)),]

K <- 5
accuracy <- rep(0,K)
cps <- rep(0,K)
for (i in 1:K) {
  # These indices indicate the interval of the test set
  indexes <- (((i-1) * round((1/K)*nrow(shuffled))) + 1):((i*round((1/K) * nrow(shuffled))))
  #take all the rows execpt those between 1:indices
  if(exists("train") && exists("test")){
    rm(train)
    rm(test)
  }
  train <- shuffled[-indexes,]
  #take all the rows with indices = 1:indices
  test <- shuffled[indexes,]
  print(nrow(train) + nrow(test) == nrow(shuffled))
  numFolds <- trainControl(method = "cv", number = 4)
  cpGrid <- expand.grid(.cp = seq(0.01, 0.5, 0.01))
  
  train_dt <- train(Class~., data = train, method = "rpart", metric="Accuracy",
                    trControl = numFolds, tuneGrid = cpGrid)
  cps[i] <- train_dt$bestTune$cp
  
  tree <- rpart(Class ~ ., train, method="class", cp = cps[i])
  pred <- predict(tree, test[,-1],type="class")
  confusionM <- table(test$Class, pred)
  accuracy[i] <- sum(diag(confusionM))/sum(confusionM)
}


write(paste("CV mean:", mean(accuracy), "; CV sd:", sd(accuracy)), append = FALSE, file = "DT_result.txt")
write(c("Accuracies ", accuracy), append = TRUE, file = "DT_result.txt")
write(c("Best cps ", cps), append = TRUE, file = "DT_result.txt")

accuracies <- rep(0, K)
for(i in 1:K) {
  rf <- rpart(Class~., data = Train_dt, method="class", cp = cps[i]) 
  pred_dt <- predict(rf, Test_dt[,-1], type="class")
  cf_dt <- table(Test_dt$Class, pred_dt)
  accuracies[i] <- sum(diag(cf_dt))/sum(cf_dt)
  write(paste("Acc with cp = ", cps[i], ": ", sum(diag(cf_dt))/sum(cf_dt)), append = TRUE, file = "DT_result.txt")
}
write(paste("Mean acc:", mean(accuracies), "sd:", sd(accuracies)))
